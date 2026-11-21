"""Router hook for Qwen3.5 on vLLM 0.33.0: scales or zeroes chosen routed experts and records routing."""

import re

import torch

MOE_LAYERS, N_EXPERTS, TOP_K = list(range(40)), 256, 8


def edit(weights, ids, chosen, operator, alpha):
    """Return the routing weights after the intervention. ids are unchanged by both operators."""
    if operator == "baseline" or not chosen:
        return weights
    if operator != "suppress":
        raise ValueError(operator)
    hit = torch.zeros_like(ids, dtype=torch.bool)
    for e in chosen:
        hit |= ids == e
    return torch.where(hit, weights * alpha, weights)


def _blocks(model):
    for name, block in model.named_modules():
        if block.__class__.__name__ == "Qwen3NextSparseMoeBlock" and hasattr(block, "experts"):
            yield int(re.search(r"layers\.(\d+)\.", name).group(1)), block


def _quant(fused):
    re_ = getattr(fused, "routed_experts", None)
    return getattr(re_, "quant_method", None) or getattr(fused, "quant_method")


def _attr(fused, *names):
    """Look an attribute up on the runner, its router, its routed experts or its moe_config (vLLM 0.33 moved them)."""
    places = [
        fused,
        getattr(fused, "router", None),
        getattr(fused, "routed_experts", None),
        getattr(fused, "moe_config", None),
        getattr(getattr(fused, "routed_experts", None), "moe_config", None),
    ]
    for n in names:
        for p in places:
            if p is not None and getattr(p, n, None) is not None:
                return getattr(p, n)
    return None


def describe(model):
    from collections import defaultdict

    dtypes = defaultdict(int)
    for p in model.parameters():
        dtypes[str(p.dtype)] += p.numel() * p.element_size()
    layers = {}
    for layer, block in _blocks(model):
        fused = block.experts
        qm = _quant(fused)
        layers[layer] = {
            "quant_method": type(qm).__name__,
            "moe_kernel": type(getattr(qm, "moe_kernel", None)).__name__,
            "runner": type(fused).__name__,
            "shared_expert": getattr(block, "shared_expert", None) is not None,
            "scoring_func": str(getattr(fused.router, "scoring_func", getattr(fused, "scoring_func", None))),
            "has_correction_bias": getattr(fused, "e_score_correction_bias", None) is not None,
            "routed_scaling_factor": float(getattr(block, "routed_scaling_factor", float("nan"))),
            "monolithic": bool(qm.is_monolithic),
            "router": type(fused.router).__name__,
            "top_k": _attr(fused, "top_k"),
            "experts": _attr(fused, "global_num_experts", "num_experts"),
            "renormalize": bool(_attr(fused, "renormalize")),
            "apply_router_weight_on_input": bool(_attr(fused, "apply_router_weight_on_input")),
        }
    return {
        "class": type(model).__name__,
        "dtype_bytes": dict(dtypes),
        "layers": layers,
        "quantization": str(getattr(model, "quant_config", None)),
    }


def install(model):
    if hasattr(model, "_qwen_research"):
        return describe(model)
    state = {"condition": {}, "layers": {}, "capture": False, "activity": False, "calls": 0}
    model._qwen_research = state
    for layer, block in _blocks(model):
        fused = block.experts
        if _quant(fused).is_monolithic:
            raise RuntimeError("Monolithic MoE kernel bypasses the router hook")
        if _attr(fused, "apply_router_weight_on_input"):
            raise RuntimeError("Expected output-weighted experts")
        if (_attr(fused, "global_num_experts", "num_experts"), _attr(fused, "top_k")) != (N_EXPERTS, TOP_K):
            raise RuntimeError("Unsupported Qwen3.5 architecture")
        entry = {"route": None, "capture": None, "stats": None, "calls": 0, "selected_hits": 0}
        state["layers"][layer] = entry
        native = fused.router.select_experts

        def wrap(native=native, entry=entry, layer=layer):
            def route(hidden_states, router_logits, *args, **kwargs):
                weights, ids = native(hidden_states, router_logits, *args, **kwargs)
                entry["calls"] += 1
                state["calls"] += 1
                c = state["condition"]
                chosen = [e for l, e in c.get("experts", []) if l == layer]
                w = edit(weights, ids, chosen, c.get("operator", "baseline"), c.get("alpha", 0.0))
                if state["capture"]:
                    entry["route"] = (ids.detach(), weights.detach(), w.detach())
                    if chosen:
                        hit = torch.zeros_like(ids, dtype=torch.bool)
                        for e in chosen:
                            hit |= ids == e
                        entry["selected_hits"] += int(hit.sum())
                if state["activity"]:
                    if entry["stats"] is None:
                        entry["stats"] = {
                            "count": torch.zeros(N_EXPERTS, device=ids.device),
                            "mass": torch.zeros(N_EXPERTS, device=ids.device),
                            "tokens": 0,
                        }
                    s = entry["stats"]
                    s["count"].scatter_add_(
                        0, ids.long().flatten(), torch.ones_like(weights, dtype=torch.float32).flatten()
                    )
                    s["mass"].scatter_add_(0, ids.long().flatten(), weights.float().flatten())
                    s["tokens"] += ids.shape[0]
                return w, ids

            return route

        fused.router.select_experts = wrap()

        def capture(module, args, kwargs, result, entry=entry):
            if not state["capture"] or entry["capture"] is not None or entry["route"] is None:
                return
            routed = result[-1] if isinstance(result, tuple) else result
            ids, w, new_w = entry["route"]
            entry["capture"] = {
                "ids": ids[-1:].clone(),
                "weights": w[-1:].clone(),
                "effective_weights": new_w[-1:].clone(),
                "routed": routed[-1:].detach().clone(),
            }

        entry["handle"] = block.register_forward_hook(capture, with_kwargs=True)
    if sorted(state["layers"]) != MOE_LAYERS:
        raise RuntimeError(f"MoE hooks cover layers {sorted(state['layers'])}, expected 0-39")
    return describe(model)


def set_condition(model, condition, capture=False, activity=False):
    if not hasattr(model, "_qwen_research"):
        install(model)
    state = model._qwen_research
    operator = condition.get("operator", "baseline")
    if operator not in {"baseline", "suppress"}:
        raise ValueError(operator)
    if not 0 <= condition.get("alpha", 0.0) <= 1:
        raise ValueError("Invalid alpha")
    experts = condition.get("experts", [])
    if len(set(map(tuple, experts))) != len(experts):
        raise ValueError("Duplicate expert")
    for l, e in experts:
        if not (l in MOE_LAYERS and 0 <= e < N_EXPERTS):
            raise ValueError((l, e))
    state.update(condition=condition, capture=capture, activity=activity, calls=0)
    for entry in state["layers"].values():
        entry.update(route=None, capture=None, stats=None, calls=0, selected_hits=0)
    return {"condition": condition, "capture": capture, "activity": activity}


def collect(model):
    state = model._qwen_research
    result = {"calls": state["calls"], "layers": {}}
    for layer, entry in state["layers"].items():
        row = {"calls": entry["calls"], "selected_hits": entry["selected_hits"]}
        s = entry["stats"]
        if s is not None:
            row.update(
                frequency=(s["count"] / s["tokens"]).cpu().tolist(),
                mass=(s["mass"] / s["tokens"]).cpu().tolist(),
                tokens=s["tokens"],
            )
        c = entry["capture"]
        if c is not None:
            row.update(
                ids=c["ids"][0].cpu().tolist(),
                weights=c["weights"][0].float().cpu().tolist(),
                effective_weights=c["effective_weights"][0].float().cpu().tolist(),
                routed=c["routed"][0].float().cpu().tolist(),
            )
        result["layers"][str(layer)] = row
    return result

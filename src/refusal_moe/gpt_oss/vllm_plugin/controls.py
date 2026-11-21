"""Router hook for GPT-OSS on vLLM 0.15.0: scales or zeroes chosen experts and records routing."""

import re

import torch

N_LAYERS, N_EXPERTS, TOP_K = 24, 32, 4


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
        if block.__class__.__name__ == "MLPBlock" and hasattr(block, "experts") and hasattr(block, "router"):
            yield int(re.search(r"layers\.(\d+)\.", name).group(1)), block


def describe(model):
    from collections import defaultdict

    dtypes = defaultdict(int)
    for p in model.parameters():
        dtypes[str(p.dtype)] += p.numel() * p.element_size()
    layers = {}
    for layer, block in _blocks(model):
        fused = block.experts
        qm = fused.quant_method
        layers[layer] = {
            "quant_method": type(qm).__name__,
            "mxfp4_backend": str(getattr(qm, "mxfp4_backend", None)),
            "monolithic": bool(qm.is_monolithic),
            "router": type(fused.router).__name__,
            "top_k": fused.top_k,
            "experts": fused.global_num_experts,
            "renormalize": bool(getattr(fused, "renormalize", getattr(fused.router, "renormalize", None))),
            "apply_router_weight_on_input": bool(fused.apply_router_weight_on_input),
        }
    return {
        "class": type(model).__name__,
        "dtype_bytes": dict(dtypes),
        "layers": layers,
        "quantization": str(getattr(model, "quant_config", None)),
    }


def install(model):
    if hasattr(model, "_gptoss_research"):
        return describe(model)
    state = {"condition": {}, "layers": {}, "capture": False, "activity": False, "calls": 0}
    model._gptoss_research = state
    for layer, block in _blocks(model):
        fused = block.experts
        if fused.quant_method.is_monolithic:
            raise RuntimeError("Monolithic MoE kernel bypasses the router hook (use an A100 / Marlin backend)")
        if fused.apply_router_weight_on_input:
            raise RuntimeError("Expected output-weighted experts")
        if (fused.global_num_experts, fused.top_k) != (N_EXPERTS, TOP_K):
            raise RuntimeError("Unsupported GPT-OSS architecture")
        entry = {"route": None, "capture": None, "stats": None, "calls": 0, "selected_hits": 0}
        state["layers"][layer] = entry
        native = fused.router.select_experts

        def wrap(native=native, entry=entry, layer=layer):
            def route(hidden_states, router_logits):
                weights, ids = native(hidden_states=hidden_states, router_logits=router_logits)
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

        entry["handle"] = fused.register_forward_hook(capture, with_kwargs=True)
    if sorted(state["layers"]) != list(range(N_LAYERS)):
        raise RuntimeError(f"MoE hooks cover layers {sorted(state['layers'])}, expected 0-{N_LAYERS - 1}")
    return describe(model)


def set_condition(model, condition, capture=False, activity=False):
    if not hasattr(model, "_gptoss_research"):
        install(model)
    state = model._gptoss_research
    operator = condition.get("operator", "baseline")
    if operator not in {"baseline", "suppress"}:
        raise ValueError(operator)
    if not 0 <= condition.get("alpha", 0.0) <= 1:
        raise ValueError("Invalid alpha")
    experts = condition.get("experts", [])
    if len(set(map(tuple, experts))) != len(experts):
        raise ValueError("Duplicate expert")
    for l, e in experts:
        if not (0 <= l < N_LAYERS and 0 <= e < N_EXPERTS):
            raise ValueError((l, e))
    state.update(condition=condition, capture=capture, activity=activity, calls=0)
    for entry in state["layers"].values():
        entry.update(route=None, capture=None, stats=None, calls=0, selected_hits=0)
    return {"condition": condition, "capture": capture, "activity": activity}


def collect(model):
    state = model._gptoss_research
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

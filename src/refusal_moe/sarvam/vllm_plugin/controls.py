"""Router hook for Sarvam on vLLM: removes, reroutes or steers chosen experts and records routing."""

import os
import re
from collections import defaultdict

import torch


def describe(model):
    from vllm.model_executor.models import utils as model_utils

    dtypes, devices = defaultdict(int), defaultdict(int)
    offload = 0
    for p in model.parameters():
        size = p.numel() * p.element_size()
        dtypes[str(p.dtype)] += size
        devices[str(p.device)] += size
        cpu = getattr(p, "_vllm_offloaded_cpu_data", None)
        if cpu is not None:
            offload += cpu.numel() * cpu.element_size()
    layers = {}
    for name, block in model.named_modules():
        if block.__class__.__name__ == "BailingMoE":
            layer = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
            layers[layer] = {
                "gate_dtype": str(block.gate.weight.dtype),
                "experts_dtype": str(block.experts.w13_weight.dtype),
                "monolithic": bool(block.experts.quant_method.is_monolithic),
                "router": type(block.experts.router).__name__,
                "top_k": block.top_k,
                "experts": block.num_experts,
                "routed_scale": block.routed_scaling_factor,
            }
    return {
        "class": type(model).__name__,
        "dtype_bytes": dict(dtypes),
        "device_bytes": dict(devices),
        "offloaded_weight_GiB": model_utils._CPU_OFFLOAD_BYTES / 2**30,
        "retained_parameter_cpu_attributes_GiB": offload / 2**30,
        "layers": layers,
        "quantization": str(getattr(model, "quant_config", None)),
        "router_logits_fp32": bool(getattr(model, "_sarvam_router_logits_fp32", False)),
        "fused_grouped_topk": os.environ.get("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1") != "0",
    }


STEER_EPS = 0.01


def edit(weights, ids, logits, bias, chosen, operator, alpha, activate=()):
    if operator == "steer":
        # SteerMoE adapted to a sigmoid router: deactivated experts are never chosen; activated
        # experts get the token's top score times exp(0.01); weights keep the native sum.
        if not chosen and not activate:
            return weights, ids
        scores = logits.float().sigmoid()
        selection = scores + bias.float()
        if chosen:
            selection[:, chosen] = -torch.inf
        if activate:
            top = scores.max(-1, keepdim=True).values
            scores = scores.clone()
            scores[:, list(activate)] = top * torch.exp(torch.tensor(STEER_EPS))
            selection[:, list(activate)] = torch.inf
        new_ids = selection.topk(ids.shape[1], dim=-1).indices
        w = scores.gather(1, new_ids)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * weights.sum(-1, keepdim=True)
        return w.to(weights.dtype), new_ids.to(ids.dtype)
    if not chosen or operator == "baseline":
        return weights, ids
    hit = torch.zeros_like(ids, dtype=torch.bool)
    for e in chosen:
        hit |= ids == e
    if operator == "exclude":
        scores = logits.float().sigmoid()
        selection = scores + bias.float()
        selection[:, chosen] = -torch.inf
        new_ids = selection.topk(ids.shape[1], dim=-1).indices
        w = scores.gather(1, new_ids)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * weights.sum(-1, keepdim=True)
        return w.to(weights.dtype), new_ids.to(ids.dtype)
    if operator not in {"suppress", "mass_restore"}:
        raise ValueError(operator)
    w = torch.where(hit, weights * alpha, weights)
    if operator == "mass_restore":
        mass = w.sum(-1, keepdim=True)
        if (mass <= 1e-20).any():
            raise ValueError("No surviving selected expert mass")
        w = w * (weights.sum(-1, keepdim=True) / mass)
    return w, ids


def install(model):
    if hasattr(model, "_sarvam_research"):
        return describe(model)
    state = {"condition": {}, "layers": {}, "capture": False, "activity": False, "trace": False, "calls": 0}
    model._sarvam_research = state
    for name, block in list(model.named_modules()):
        if block.__class__.__name__ != "BailingMoE":
            continue
        layer = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
        fused = block.experts
        if fused.quant_method.is_monolithic:
            raise RuntimeError("Monolithic MoE bypasses router hooks")
        if fused.apply_router_weight_on_input:
            raise RuntimeError("Expected output-weighted experts")
        if (block.num_experts, block.top_k, block.n_group, block.topk_group) != (128, 6, 1, 1):
            raise RuntimeError("Unsupported Sarvam architecture")
        entry = {
            "block": block,
            "route": None,
            "capture": None,
            "stats": None,
            "calls": 0,
            "selected_hits": 0,
            "max_mass_error": 0.0,
            "trace": [],
        }
        state["layers"][layer] = entry
        native = fused.router.select_experts

        def wrap(native=native, entry=entry, layer=layer, block=block):
            def route(hidden_states, router_logits):
                weights, ids = native(hidden_states=hidden_states, router_logits=router_logits)
                entry["calls"] += 1
                state["calls"] += 1
                c = state["condition"]
                chosen = [e for l, e in c.get("experts", []) if l == layer]
                operator = c.get("operator", "baseline")
                alpha = c.get("alpha", 0.0)
                act = [e for l, e in c.get("activate", []) if l == layer]
                w, new_ids = edit(weights, ids, router_logits, block.gate.expert_bias, chosen, operator, alpha, act)
                if state["capture"]:
                    entry["route"] = (
                        ids.detach(),
                        weights.detach(),
                        new_ids.detach(),
                        w.detach(),
                        router_logits.detach(),
                    )
                    if chosen:
                        hit = torch.zeros_like(ids, dtype=torch.bool)
                        for e in chosen:
                            hit |= ids == e
                        entry["selected_hits"] += int(hit.sum())
                    if operator in {"exclude", "mass_restore"}:
                        entry["max_mass_error"] = max(
                            entry["max_mass_error"], float((weights.sum(-1) - w.sum(-1)).abs().max())
                        )
                        if operator == "exclude" and any(bool((new_ids == e).any()) for e in chosen):
                            raise RuntimeError("Excluded expert remains selected")
                    if operator == "steer":
                        entry["max_mass_error"] = max(
                            entry["max_mass_error"], float((weights.sum(-1) - w.sum(-1)).abs().max())
                        )
                        if any(bool((new_ids == e).any()) for e in chosen):
                            raise RuntimeError("Deactivated expert remains selected")
                        if any(not bool((new_ids == e).any(-1).all()) for e in act):
                            raise RuntimeError("Activated expert missing from a token's selection")
                if state["trace"]:
                    selection = router_logits.float().sigmoid() + block.gate.expert_bias.float()
                    top = selection.topk(new_ids.shape[1] + 1, dim=-1).values
                    entry["trace"].append(
                        {
                            "ids": new_ids.detach().to(torch.int16).cpu(),
                            "weights": (w.float() / w.float().sum(-1, keepdim=True).clamp_min(1e-20)).cpu(),
                            "margin": (top[:, -2] - top[:, -1]).cpu(),
                        }
                    )
                if state["activity"]:
                    if entry["stats"] is None:
                        entry["stats"] = {
                            "count": torch.zeros(128, device=ids.device),
                            "mass": torch.zeros(128, device=ids.device),
                            "tokens": 0,
                        }
                    s = entry["stats"]
                    s["count"].scatter_add_(0, ids.long().flatten(), torch.ones_like(weights).float().flatten())
                    s["mass"].scatter_add_(
                        0, ids.long().flatten(), weights.float().flatten() * block.routed_scaling_factor
                    )
                    s["tokens"] += ids.shape[0]
                return w, new_ids

            return route

        fused.router.select_experts = wrap()

        def capture(module, args, kwargs, result, entry=entry):
            if not state["capture"] or entry["capture"] is not None:
                return
            shared, routed = result
            ids, w, new_ids, new_w, logits = entry["route"]
            entry["capture"] = {
                "ids": ids[-1:].detach().clone(),
                "weights": w[-1:].detach().clone(),
                "effective_ids": new_ids[-1:].detach().clone(),
                "effective_weights": new_w[-1:].detach().clone(),
                "logits": logits[-1:].detach().clone(),
                "routed": routed[-1:].detach().clone(),
                "shared": shared[-1:].detach().clone(),
            }

        entry["handle"] = fused.register_forward_hook(capture, with_kwargs=True)
    start = getattr(model.model, "start_layer", 0)
    end = getattr(model.model, "end_layer", 19)
    if sorted(state["layers"]) != list(range(max(1, start), end)):
        raise RuntimeError("MoE hooks do not cover this pipeline stage's layers")
    return describe(model)


def set_condition(model, condition, capture=False, activity=False, trace=False):
    if not hasattr(model, "_sarvam_research"):
        install(model)
    state = model._sarvam_research
    operator = condition.get("operator", "baseline")
    if operator not in {"baseline", "suppress", "exclude", "mass_restore", "steer"}:
        raise ValueError(operator)
    if not 0 <= condition.get("alpha", 0.0) <= 1:
        raise ValueError("Invalid alpha")
    experts = condition.get("experts", [])
    if len(set(map(tuple, experts))) != len(experts):
        raise ValueError("Duplicate expert")
    activate = condition.get("activate", [])
    if activate and operator != "steer":
        raise ValueError("Only the steer operator activates experts")
    if len(set(map(tuple, activate))) != len(activate):
        raise ValueError("Duplicate activated expert")
    if set(map(tuple, activate)) & set(map(tuple, experts)):
        raise ValueError("Expert both activated and deactivated")
    per_layer = {}
    for l, e in activate:
        per_layer[l] = per_layer.get(l, 0) + 1
    if any(v > 6 for v in per_layer.values()):
        raise ValueError("More activated experts than top-k in one layer")
    for l, e in experts + activate:
        if not 1 <= l < 19 or not 0 <= e < 128:
            raise ValueError((l, e))
    state.update(condition=condition, capture=capture, activity=activity, trace=trace, calls=0)
    for entry in state["layers"].values():
        entry.update(
            route=None,
            capture=None,
            stats=None,
            calls=0,
            selected_hits=0,
            max_mass_error=0.0,
            trace=[],
        )
    return {"condition": condition, "capture": capture, "activity": activity, "trace": trace}


def collect(model):
    state = model._sarvam_research
    result = {"calls": state["calls"], "layers": {}}
    for layer, entry in state["layers"].items():
        row = {
            "calls": entry["calls"],
            "selected_hits": entry["selected_hits"],
            "max_mass_error": entry["max_mass_error"],
        }
        if entry.get("trace"):
            row.update(
                trace_ids=torch.cat([t["ids"] for t in entry["trace"]]).tolist(),
                trace_weights=torch.cat([t["weights"] for t in entry["trace"]]).tolist(),
                trace_margin=torch.cat([t["margin"] for t in entry["trace"]]).tolist(),
            )
        stats = entry["stats"]
        if stats is not None:
            row.update(
                frequency=(stats["count"] / stats["tokens"]).cpu().tolist(),
                mass=(stats["mass"] / stats["tokens"]).cpu().tolist(),
                tokens=stats["tokens"],
            )
        c = entry["capture"]
        if c is not None:
            row.update(
                ids=c["ids"][0].cpu().tolist(),
                weights=c["weights"][0].cpu().tolist(),
                effective_ids=c["effective_ids"][0].cpu().tolist(),
                effective_weights=c["effective_weights"][0].cpu().tolist(),
                shared=c["shared"][0].float().cpu().tolist(),
                routed=c["routed"][0].float().cpu().tolist(),
            )
        result["layers"][str(layer)] = row
    return result

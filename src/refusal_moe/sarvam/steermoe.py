"""SteerMoE expert detection on Sarvam (risk difference on teacher-forced answers) and its groups."""

# SteerMoE-inspired implementation.
# Based on "Steering MoE LLMs via Expert (De)Activation"
# by Fayyaz et al. (2025).
# Paper: https://arxiv.org/abs/2509.09660
# Code: https://github.com/adobe-research/SteerMoE
#
# Adapted independently for experimentation within this project.

import functools
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from refusal_moe.io import append, read_jsonl, write
from refusal_moe.paths import REPO
from refusal_moe.sarvam.chat import encode
from refusal_moe.sarvam.groups import SELECTED_FAMILIES, matched_controls

DETECT = "data/steermoe/beavertails_detect.jsonl"
CONFIG = {
    "detect_file": DETECT,
    "deactivate_sizes": [16, 32, 64],
    "activate_sizes": [5],
    "controls_per_group": 5,
    "control_seed": 20261114,
    "operator": "steer",
    "method": "SteerMoE risk difference on answer tokens, global |delta| ranking",
}
LAYERS = range(1, 19)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def span_counts(trace_layers, start, end):
    """Per-layer expert hit counts over token positions [start, end) of one teacher-forced pass."""
    out = []
    for layer in LAYERS:
        ids = np.asarray(trace_layers[str(layer)]["trace_ids"])
        if ids.ndim != 2 or ids.shape[0] < end:
            raise RuntimeError(f"Trace too short at layer {layer}: {ids.shape} < {end}")
        out.append(np.bincount(ids[start:end].ravel().astype(np.int64), minlength=128).tolist())
    return out


def risk_difference(rows):
    """rows: detection records. Returns p_safe, p_unsafe, delta as [18, 128] arrays (layer 1 first)."""
    totals = {}
    for v in ("safe", "unsafe"):
        sub = [r for r in rows if r["variant"] == v]
        hits = np.sum([r["counts"] for r in sub], axis=0)
        tokens = sum(r["span_tokens"] for r in sub)
        totals[v] = hits / tokens
    return totals["safe"], totals["unsafe"], totals["safe"] - totals["unsafe"]


def rank(delta):
    """All (layer, expert) pairs by |delta|, largest first (ties broken by layer, expert)."""
    pairs = [(l, e) for i, l in enumerate(LAYERS) for e in range(128)]
    return sorted(pairs, key=lambda p: (-abs(delta[p[0] - 1][p[1]]), p[0], p[1]))


def build_groups(delta, activity, config):
    order = rank(delta)
    linked = [[l, e] for l, e in order if delta[l - 1][e] > 0]
    rng = random.Random(config["control_seed"])
    groups = [{"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline"}]
    for n in config["activate_sizes"]:
        chosen, per_layer = [], {}
        for l, e in linked:  # SteerMoE activates at most top-k (6) experts per layer
            if len(chosen) == n:
                break
            if per_layer.get(l, 0) < 6:
                chosen.append([l, e])
                per_layer[l] = per_layer.get(l, 0) + 1
        name = f"steermoe_on_n{n}"
        groups.append(
            {"id": name, "experts": [], "activate": chosen, "operator": "steer", "family": "steermoe_activate"}
        )
        for i, (c, m) in enumerate(matched_controls(chosen, activity, rng, config["controls_per_group"])):
            groups.append(
                {
                    "id": f"{name}_random{i}",
                    "experts": [],
                    "activate": c,
                    "operator": "steer",
                    "family": "matched_random",
                    "matched_to": name,
                    "matching": m,
                }
            )
    for n in config["deactivate_sizes"]:
        chosen = linked[:n]
        name = f"steermoe_off_n{n}"
        groups.append({"id": name, "experts": chosen, "operator": "steer", "family": "steermoe_deactivate"})
        for i, (c, m) in enumerate(matched_controls(chosen, activity, rng, config["controls_per_group"])):
            groups.append(
                {
                    "id": f"{name}_random{i}",
                    "experts": c,
                    "operator": "steer",
                    "family": "matched_random",
                    "matched_to": name,
                    "matching": m,
                }
            )
    return groups, order


def overlap_with_pilot_groups(groups):
    """How many SteerMoE experts also appear in the pilot's 32- and 64-expert groups (descriptive)."""
    path = REPO / "configs/sarvam/groups_size32_64.json"
    prior = {
        g["id"]: {tuple(x) for x in g["experts"]}
        for g in json.loads(path.read_text())
        if g["family"] in SELECTED_FAMILIES
    }
    out = {}
    for g in groups:
        if g["family"].startswith("steermoe"):
            mine = {tuple(x) for x in g["experts"] + g.get("activate", [])}
            out[g["id"]] = {k: len(mine & v) for k, v in prior.items()}
    return out


def detect_and_build(engine, tokenizer, output, check, activity):
    """Runs detection inside the loaded engine (resumable), then writes groups.json."""
    from refusal_moe.sarvam.engine import generate
    from refusal_moe.sarvam.vllm_plugin import controls

    config = json.loads((output / "steermoe_config.json").read_text())
    if sha(REPO / config["detect_file"]) != config["detect_sha256"]:
        raise RuntimeError("Detection file changed since seeding")
    examples = [json.loads(l) for l in (REPO / config["detect_file"]).read_text().splitlines() if l.strip()]
    path = output / "steermoe_detect.jsonl"
    rows = read_jsonl(path)
    done = {(r["id"], r["variant"]) for r in rows}
    baseline = {"id": "baseline", "operator": "baseline", "experts": []}
    for i, ex in enumerate(examples):
        for variant in ("safe", "unsafe"):
            if (ex["id"], variant) in done:
                continue
            check()
            prefix, _ = encode(tokenizer, ex["prompt"], "empty_think")
            answer = tokenizer.encode(ex[variant + "_response"], add_special_tokens=False)
            ids = prefix + answer
            engine.apply_model(functools.partial(controls.set_condition, condition=baseline, trace=True))
            generate(engine, tokenizer, ex["prompt"], "empty_think", 1, raw_ids=ids)
            captured = engine.apply_model(controls.collect)[0]
            lengths = {len(captured["layers"][str(l)]["trace_ids"]) for l in LAYERS}
            row = {
                "id": ex["id"],
                "variant": variant,
                "prompt_tokens": len(prefix),
                "span_tokens": len(answer),
                "trace_tokens": sorted(lengths),
                "counts": span_counts(captured["layers"], len(prefix), len(ids)),
            }
            append(path, row)
            rows.append(row)
        if i % 100 == 0:
            print(f"SteerMoE detection {i + 1}/{len(examples)}", flush=True)
    engine.apply_model(functools.partial(controls.set_condition, condition=baseline))
    p_safe, p_unsafe, delta = risk_difference(rows)
    groups, order = build_groups(delta, activity, config)
    write(output / "groups.json", groups)
    top = [
        {
            "layer": l,
            "expert": e,
            "p_refusal": float(p_safe[l - 1][e]),
            "p_unsafe": float(p_unsafe[l - 1][e]),
            "delta": float(delta[l - 1][e]),
        }
        for l, e in order[:200]
    ]
    mismatch = [
        r["id"] + ":" + r["variant"] for r in rows if r["trace_tokens"] != [r["prompt_tokens"] + r["span_tokens"]]
    ]
    write(
        output / "steermoe_selection.json",
        {
            "examples": len(examples),
            "rows": len(rows),
            "answer_tokens": {v: sum(r["span_tokens"] for r in rows if r["variant"] == v) for v in ("safe", "unsafe")},
            "trace_length_differs_from_input": mismatch,
            "refusal_linked_count": int((delta > 0).sum()),
            "unsafe_linked_count": int((delta < 0).sum()),
            "top200_by_abs_delta": top,
            "groups": {
                g["id"]: {"experts": g["experts"], "activate": g.get("activate", [])}
                for g in groups
                if g["family"].startswith("steermoe")
            },
            "layer_counts": {
                g["id"]: dict(
                    sorted(
                        {
                            str(l): sum(1 for x in g["experts"] + g.get("activate", []) if x[0] == l) for l in LAYERS
                        }.items()
                    )
                )
                for g in groups
                if g["family"].startswith("steermoe")
            },
            "overlap_with_earlier_groups": overlap_with_pilot_groups(groups),
            "delta_full": [[float(x) for x in row] for row in delta],
        },
    )
    print("SteerMoE groups written:", [g["id"] for g in groups if g["family"].startswith("steermoe")], flush=True)
    return groups

"""Qwen3.5 gradient selection: discovery answers in vLLM, then attribution in transformers 5.

    python -m refusal_moe.qwen35.selection generate --model models/qwen35-35b-a3b --run-dir runs/qwen35_discovery
    python -m refusal_moe.qwen35.selection gradient --model models/qwen35-35b-a3b --run-dir runs/qwen35_selection \
        --generations runs/qwen35_discovery --labels configs/qwen35/discovery_labels.jsonl
"""

import argparse
import collections
import json
import time
from pathlib import Path

from refusal_moe.io import write
from refusal_moe.paths import DATA

GROUP_SIZE, CAP_PER_LAYER, OPEN_TOKENS, OPEN_PER_CLASS = 143, 4, 6, 8


def discovery_items(pairs_path):
    data = json.loads(Path(pairs_path).read_text())
    by_id = {p["pair_id"]: p for p in data["pairs"]}
    return [
        dict(pair_id=i, condition=c, phase="discovery", prompt=by_id[i][c])
        for i in data["discovery_ids"]
        for c in ("harmful", "benign")
    ]


def generate(args, run):
    from refusal_moe.qwen35 import runner

    work = discovery_items(args.pairs)
    engine = runner.load(Path(args.model), run, max_num_seqs=256)
    tok = engine.llm.get_tokenizer()
    rows, seconds = runner.generate(engine, tok, [w["prompt"] for w in work])  # stock vLLM = hooked baseline
    for w, r in zip(work, rows):
        runner.append(
            run / "generations.jsonl",
            dict(r, **{k: v for k, v in w.items() if k != "prompt"}, pass_index=0, configuration="baseline"),
        )
    print(f"discovery: {len(rows)} answers in {seconds:.0f}s", flush=True)


def refusal_class(label):
    v = label.get("judge_value")
    if label["condition"] == "harmful":
        return "refusal" if v == "1" else None
    return {"2_full_refusal": "refusal", "3_partial_refusal": "refusal", "1_full_compliance": "compliance"}.get(v)


def answer_start(ids, decode):
    """Index of the first non-whitespace token of the answer (thinking is off, so the answer starts at once)."""
    h = 0
    while h < len(ids) and decode([ids[h]]).strip() == "":
        h += 1
    return h if h < len(ids) else None


def decoder_layers(model):
    inner = model.model
    return (inner.language_model if hasattr(inner, "language_model") else inner).layers


class RouterTap:
    """Forward hooks on every MoE gate (layers[i].mlp.gate returns logits, weights, ids): make the 8 routing weights
    leaves (routing frozen) and/or scale one expert's weight by a factor (0 = remove). Records leaves and ids."""

    def __init__(self, model):
        self.leaves, self.scale, self.active = {}, None, False
        self.layers = []
        for i, layer in enumerate(decoder_layers(model)):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is not None and gate.__class__.__name__.endswith("TopKRouter"):
                gate.register_forward_hook(self._hook(i))
                self.layers.append(i)

    def _hook(self, i):
        def hook(module, args, out):
            if not self.active and self.scale is None:
                return None
            logits, w, ids = out
            if self.active:
                w = w.detach().requires_grad_(True)
                self.leaves[i] = (ids, w)
            if self.scale is not None and self.scale[0] == i:
                w = w * (1 - (1 - self.scale[2]) * (ids == self.scale[1]).to(w.dtype))
            return logits, w, ids

        return hook


def margin(model, prefix, openings, n_refusal, device):
    import torch

    width = len(prefix) + max(len(o) for o in openings)
    ids = torch.zeros((len(openings), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for r, o in enumerate(openings):
        seq = prefix + o
        ids[r, : len(seq)], mask[r, : len(seq)] = torch.tensor(seq), 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
    P, lps = len(prefix), []
    for r, o in enumerate(openings):
        lp = torch.log_softmax(logits[r, P - 1 : P - 1 + len(o)].float(), dim=-1)
        lps.append(lp[torch.arange(len(o), device=lp.device), torch.tensor(o, device=lp.device)].sum())
    lps = torch.stack(lps)
    return torch.logsumexp(lps[:n_refusal], 0) - torch.logsumexp(lps[n_refusal:], 0), logits


def attribution(tap, n_experts):
    """Per layer: sum over tokens of (dm/dw) * w, scattered to experts."""
    import torch

    out = {}
    for i, (ids, w) in tap.leaves.items():
        a = w.grad.float() * w.detach().float()
        out[i] = torch.zeros(n_experts, device=a.device).index_add_(0, ids.reshape(-1).long(), a.reshape(-1))
    return out


def gradient(args, run):
    import numpy as np
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoModelForImageTextToText

    gens = [json.loads(l) for l in (Path(args.generations) / "generations.jsonl").read_text().split("\n") if l.strip()]
    labels = {
        (x["pair_id"], x["condition"]): x
        for x in map(json.loads, [l for l in Path(args.labels).read_text().split("\n") if l.strip()])
    }
    tk = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    decode = lambda ids: tk.decode(ids, skip_special_tokens=False)
    items, counts = [], {"refusal": collections.Counter(), "compliance": collections.Counter()}
    for g in gens:
        lab = labels[(g["pair_id"], g["condition"])]
        h = answer_start(g["generated_ids"], decode)
        cls = refusal_class(lab)
        items.append(
            dict(
                pair_id=g["pair_id"],
                condition=g["condition"],
                cls=cls,
                judge_value=lab.get("judge_value"),
                prefix=None if h is None else g["input_ids"] + g["generated_ids"][:h],
                vllm_first=None if h is None or h >= len(g["generated_ids"]) else g["generated_ids"][h],
            )
        )
        if cls and h is not None and len(g["generated_ids"]) - h >= 2:
            counts[cls][tuple(g["generated_ids"][h : h + OPEN_TOKENS])] += 1
    sets = {k: [list(t) for t, _ in c.most_common(OPEN_PER_CLASS)] for k, c in counts.items()}
    if not sets["refusal"] or not sets["compliance"]:
        raise RuntimeError(f"Need at least one opening per class: {[len(v) for v in sets.values()]}")
    openings, nr = sets["refusal"] + sets["compliance"], len(sets["refusal"])
    write(
        run / "openings.json",
        {
            "token_ids": sets,
            "counts": {k: [n for _, n in c.most_common(OPEN_PER_CLASS)] for k, c in counts.items()},
            "text": {k: [decode(o) for o in v] for k, v in sets.items()},
            "class_sizes": {k: sum(c.values()) for k, c in counts.items()},
        },
    )
    t = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    model.set_experts_implementation("eager")
    model.to("cuda").eval().requires_grad_(False)
    write(
        run / "hf_load.json",
        {
            "seconds": time.monotonic() - t,
            "dtype": str(model.dtype),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "experts_implementation": str(getattr(model.config, "_experts_implementation", None)),
        },
    )
    dev = model.device
    tap = RouterTap(model)
    if tap.layers != list(range(40)):
        raise RuntimeError(f"Router hooks on layers {tap.layers}, expected 0-39")
    E = model.config.text_config.num_experts
    usable = [x for x in items if x["prefix"] is not None]
    A = np.zeros((len(usable), 40, E), np.float32)
    margins, agree = [], 0
    for n, x in enumerate(usable):
        with torch.no_grad():
            m_plain, logits = margin(model, x["prefix"], openings, nr, dev)
        agree += int(logits[0, len(x["prefix"]) - 1].argmax().item() == x["vllm_first"])
        tap.leaves, tap.active = {}, True
        with torch.enable_grad():
            m, _ = margin(model, x["prefix"], openings, nr, dev)
            m.backward()
        tap.active = False
        for i, v in attribution(tap, E).items():
            A[n, i] = v.cpu().numpy()
        margins.append(
            dict(
                pair_id=x["pair_id"],
                condition=x["condition"],
                cls=x["cls"],
                margin=float(m.detach()),
                margin_unhooked=float(m_plain),
                equal=float(m.detach()) == float(m_plain),
            )
        )
        if n % 10 == 0:
            print(f"gradient {n}/{len(usable)}", flush=True)
    refused = np.array([x["cls"] == "refusal" for x in usable])
    score = A[refused].mean(0)
    order = [o for o in np.argsort(-score, axis=None) if np.isfinite(score.flat[o])]
    harm = A[[x["condition"] == "harmful" for x in usable]].mean(0)
    ben = A[[x["condition"] == "benign" for x in usable]].mean(0)
    ranking = [
        {
            "layer": int(i),
            "expert": int(e),
            "score": float(score[i, e]),
            "mean_attr_harmful": float(harm[i, e]),
            "mean_attr_benign": float(ben[i, e]),
        }
        for i, e in (np.unravel_index(o, score.shape) for o in order)
    ]
    group, per_layer = [], collections.Counter()
    for r in ranking:  # Sarvam's capped_ranking: at most CAP_PER_LAYER experts from any one layer
        if per_layer[r["layer"]] < CAP_PER_LAYER:
            group.append([r["layer"], r["expert"]])
            per_layer[r["layer"]] += 1
        if len(group) == GROUP_SIZE:
            break
    np.savez_compressed(
        run / "attribution.npz",
        attr_all=A,
        refused=refused,
        pair_id=np.array([x["pair_id"] for x in usable]),
        condition=np.array([x["condition"] for x in usable]),
    )
    out = {
        "group": group,
        "group_size": GROUP_SIZE,
        "refusal_class_prompts": int(refused.sum()),
        "prompts_used": len(usable),
        "prompts_without_final": len(items) - len(usable),
        "hooked_equals_unhooked_margin": sum(m["equal"] for m in margins),
        "hf_vllm_first_answer_token_agree": agree,
        "group_layers": dict(sorted(per_layer.items())),
        "ranking_top600": ranking[:600],
        "margins": margins,
    }
    write(run / "selection.json", out)
    write(run / "selection_group_n143.json", {"group": group, "rule": "top 143 by score, at most 4 per layer"})
    print("selection written to", run / "selection.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "gradient"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pairs", default=str(DATA / "pairs.json"))
    ap.add_argument("--generations")
    ap.add_argument("--labels")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (generate if args.cmd == "generate" else gradient)(args, run)
    print("finished:", args.cmd, flush=True)


if __name__ == "__main__":
    main()

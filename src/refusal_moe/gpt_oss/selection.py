"""GPT-OSS-20B gradient selection: discovery answers in vLLM, then attribution in transformers.

    python -m refusal_moe.gpt_oss.selection generate --model models/gpt-oss-20b --run-dir runs/gptoss_discovery
    python -m refusal_moe.gpt_oss.selection gradient --model models/gpt-oss-20b --run-dir runs/gptoss_selection \
        --generations runs/gptoss_discovery --labels configs/gpt_oss/discovery_labels.jsonl
"""

import argparse
import collections
import json
import time
from pathlib import Path

from refusal_moe.gpt_oss import harmony_io, runner
from refusal_moe.paths import DATA

GROUP_SIZE, OPEN_TOKENS, OPEN_PER_CLASS = 10, 6, 8


def discovery_items(pairs_path):
    data = json.loads(Path(pairs_path).read_text())
    by_id = {p["pair_id"]: p for p in data["pairs"]}
    return [
        dict(pair_id=i, condition=c, phase="discovery", prompt=by_id[i][c])
        for i in data["discovery_ids"]
        for c in ("harmful", "benign")
    ]


def generate(args, run):
    work = discovery_items(args.pairs)
    engine = runner.load(Path(args.model), run, max_num_seqs=256)
    tok, enc = engine.llm.get_tokenizer(), harmony_io.encoding()
    rows, seconds = runner.generate(engine, enc, tok, [w["prompt"] for w in work])  # stock vLLM = hooked baseline
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


def final_header_end(ids, tok):
    """Index just after the final-channel header (<|channel|>final ... <|message|>) in generated ids."""
    msg = tok.convert_tokens_to_ids("<|message|>")
    chan = tok.convert_tokens_to_ids("<|channel|>")
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == msg:
            j = max(k for k in range(i) if ids[k] == chan)
            if "final" in tok.decode(ids[j + 1 : i]):
                return i + 1
    return None


class RouterTap:
    """Forward hooks on every router: replace the dense routing weights with a leaf (routing frozen), optionally
    scale one expert's weight by a factor (0 = remove). Records leaves for the attribution."""

    def __init__(self, model):
        self.leaves, self.scale, self.active = {}, None, False
        self.handles = [
            layer.mlp.router.register_forward_hook(self._hook(i)) for i, layer in enumerate(model.model.layers)
        ]

    def _hook(self, i):
        def hook(module, args, out):
            if not self.active and self.scale is None:
                return None
            scores, idx = out
            if self.active:
                scores = scores.detach().requires_grad_(True)
                self.leaves[i] = scores
            if self.scale is not None and self.scale[0] == i:
                scores = scores.clone()
                scores[:, self.scale[1]] = scores[:, self.scale[1]] * self.scale[2]
            return scores, idx

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


def gradient(args, run):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    gens = [json.loads(l) for l in (Path(args.generations) / "generations.jsonl").read_text().splitlines()]
    labels = {(x["pair_id"], x["condition"]): x for x in map(json.loads, Path(args.labels).read_text().splitlines())}
    tok = AutoTokenizer.from_pretrained(args.model)
    items, counts = [], {"refusal": collections.Counter(), "compliance": collections.Counter()}
    for g in gens:
        lab = labels[(g["pair_id"], g["condition"])]
        h = final_header_end(g["generated_ids"], tok)
        cls = refusal_class(lab)
        items.append(
            dict(
                pair_id=g["pair_id"],
                condition=g["condition"],
                cls=cls,
                judge_value=lab.get("judge_value"),
                prefix=None if h is None else g["input_ids"] + g["generated_ids"][:h],
                vllm_first_final=None if h is None or h >= len(g["generated_ids"]) else g["generated_ids"][h],
            )
        )
        if cls and h is not None and len(g["generated_ids"]) - h >= 2:
            counts[cls][tuple(g["generated_ids"][h : h + OPEN_TOKENS])] += 1
    sets = {k: [list(t) for t, _ in c.most_common(OPEN_PER_CLASS)] for k, c in counts.items()}
    if not sets["refusal"] or not sets["compliance"]:
        raise RuntimeError(f"Need at least one opening per class: {[len(v) for v in sets.values()]}")
    openings = sets["refusal"] + sets["compliance"]
    nr = len(sets["refusal"])
    runner.write(
        run / "openings.json",
        {
            "token_ids": sets,
            "counts": {k: [n for _, n in c.most_common(OPEN_PER_CLASS)] for k, c in counts.items()},
            "text": {k: [tok.decode(o) for o in v] for k, v in sets.items()},
            "class_sizes": {k: sum(c.values()) for k, c in counts.items()},
        },
    )
    t = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=Mxfp4Config(dequantize=True),
        attn_implementation="eager",
    )
    model.eval().requires_grad_(False)
    for layer in model.model.layers:
        layer.mlp.experts.train()  # per-expert loop: output * routing weight, differentiable in the weight
    runner.write(
        run / "hf_load.json",
        {"seconds": time.monotonic() - t, "dtype": str(model.dtype), "layers": len(model.model.layers)},
    )
    dev = model.device
    tap = RouterTap(model)
    L, E = len(model.model.layers), model.config.num_local_experts
    usable = [x for x in items if x["prefix"] is not None]
    A = np.zeros((len(usable), L, E), np.float32)
    margins, agree = [], 0
    for n, x in enumerate(usable):
        with torch.no_grad():
            m_plain, logits = margin(model, x["prefix"], openings, nr, dev)
        agree += int(logits[0, len(x["prefix"]) - 1].argmax().item() == x["vllm_first_final"])
        tap.leaves, tap.active = {}, True
        with torch.enable_grad():
            m, _ = margin(model, x["prefix"], openings, nr, dev)
            m.backward()
        tap.active = False
        for i, leaf in tap.leaves.items():
            A[n, i] = (leaf.grad.float() * leaf.detach().float()).sum(0).cpu().numpy()
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
    order = np.argsort(-score, axis=None)
    ranking = [
        {
            "layer": int(i),
            "expert": int(e),
            "score": float(score[i, e]),
            "mean_attr_harmful": float(A[[x["condition"] == "harmful" for x in usable]].mean(0)[i, e]),
            "mean_attr_benign": float(A[[x["condition"] == "benign" for x in usable]].mean(0)[i, e]),
        }
        for i, e in zip(*np.unravel_index(order, score.shape))
    ]
    group = [[r["layer"], r["expert"]] for r in ranking[:GROUP_SIZE]]
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
        "hf_vllm_first_final_token_agree": agree,
        "ranking_top50": ranking[:50],
        "margins": margins,
    }
    runner.write(run / "selection.json", out)
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

"""GPT-OSS-20B pilot: baseline, the selected group and 5 matched random controls on the 48 pilot pairs.

    python -m refusal_moe.gpt_oss.pilot --model models/gpt-oss-20b --run-dir runs/gptoss_pilot \
        --selection configs/gpt_oss/selection_group_n10.json --size 10
"""

import argparse
import json
import random
from pathlib import Path

from refusal_moe.gpt_oss import harmony_io, runner
from refusal_moe.matching import activity_table, matched_controls
from refusal_moe.paths import DATA

SEED, N_CONTROLS, NEAREST = 20261120, 5, 8


def build_groups(selected, act):
    name = f"gradient_n{len(selected)}"
    groups = [
        {"id": "baseline", "experts": [], "operator": "baseline", "family": "baseline"},
        {"id": name, "experts": selected, "operator": "suppress", "alpha": 0.0, "family": "gradient"},
    ]
    for i, (c, m) in enumerate(
        matched_controls(selected, act, random.Random(SEED), N_CONTROLS, n_experts=32, nearest=NEAREST)
    ):
        groups.append(
            {
                "id": f"{name}_random{i}",
                "experts": c,
                "operator": "suppress",
                "alpha": 0.0,
                "family": "matched_random",
                "matched_to": name,
                "matching": m,
            }
        )
    return groups


def pilot_items(pairs_path):
    data = json.loads(Path(pairs_path).read_text())
    by_id = {p["pair_id"]: p for p in data["pairs"]}
    return [
        dict(pair_id=i, condition=c, phase="pilot", prompt=by_id[i][c])
        for i in data["pilot_ids"]
        for c in ("harmful", "benign")
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--pairs", default=str(DATA / "pairs.json"))
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(args.selection).read_text())["group"]
    assert len(selected) == args.size and len({tuple(x) for x in selected}) == args.size
    work = pilot_items(args.pairs)
    engine = runner.load(Path(args.model), run, max_num_seqs=256)
    tok, enc = engine.llm.get_tokenizer(), harmony_io.encoding()
    runner.audit(engine, enc, tok, run)
    rows = []
    for w in work:  # activity, one prompt at a time, prompt tokens only
        engine.control("set_condition", condition={"operator": "baseline", "experts": []}, activity=True)
        runner.generate(engine, enc, tok, [w["prompt"]], cap=1)
        got = engine.control("collect")
        if not got["calls"]:
            raise RuntimeError("Activity hooks were bypassed")
        rec = {
            "pair_id": w["pair_id"],
            "condition": w["condition"],
            "layers": {k: {m: v[m] for m in ("frequency", "mass", "tokens")} for k, v in got["layers"].items()},
        }
        runner.append(run / "activity.jsonl", rec)
        rows.append(rec)
    groups = build_groups(selected, activity_table(rows, range(24)))
    runner.write(run / "groups.json", groups)
    for g in groups:
        cond = {k: g[k] for k in ("operator", "experts")} | ({"alpha": g["alpha"]} if "alpha" in g else {})
        engine.control("set_condition", condition=cond, capture=True)
        out, seconds = runner.generate(engine, enc, tok, [w["prompt"] for w in work])
        info = engine.control("collect")
        for w, r in zip(work, out):
            runner.append(
                run / "generations.jsonl",
                dict(r, **{k: v for k, v in w.items() if k != "prompt"}, pass_index=0, configuration=g["id"]),
            )
        hits = sum(v["selected_hits"] for v in info["layers"].values())
        if g["experts"] and hits == 0:
            raise RuntimeError(f"{g['id']}: no routing decision touched the group")
        print(g["id"], f"{len(out)} answers in {seconds:.0f}s", flush=True)
    engine.control("set_condition", condition={"operator": "baseline", "experts": []})
    print("finished: pilot", flush=True)


if __name__ == "__main__":
    main()

"""GPT-OSS-20B baseline: 48 pilot pairs and 48 GSM8K problems, each generated twice.

python -m refusal_moe.gpt_oss.baseline --model models/gpt-oss-20b --run-dir runs/gptoss_baseline
"""

import argparse
import json
import random
from pathlib import Path

from refusal_moe.gpt_oss import harmony_io, runner
from refusal_moe.paths import DATA
from refusal_moe.sarvam.gsm8k import PROMPT as GSM8K_PROMPT
from refusal_moe.sarvam.gsm8k import gold

MAX_IN_FLIGHT = 256


def items(pairs_path, gsm_path):
    data = json.loads(Path(pairs_path).read_text())
    by_id = {p["pair_id"]: p for p in data["pairs"]}
    out = [
        dict(pair_id=i, condition=c, phase="pilot", prompt=by_id[i][c])
        for i in data["pilot_ids"]
        for c in ("harmful", "benign")
    ]
    bank = [json.loads(l) for l in Path(gsm_path).read_text().splitlines() if l.strip()]
    order = list(range(len(bank)))
    random.Random(20261108).shuffle(order)
    out += [
        dict(
            pair_id=f"gsm8k_test_{i}",
            condition="gsm8k",
            phase="gsm8k",
            gold=gold(bank[i]["answer"]),
            prompt=GSM8K_PROMPT.format(question=bank[i]["question"]),
        )
        for i in sorted(order[:48])
    ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pairs", default=str(DATA / "pairs.json"))
    ap.add_argument("--gsm8k", default=str(DATA / "gsm8k/test.jsonl"))
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    work = items(args.pairs, args.gsm8k)
    engine = runner.load(Path(args.model), run, max_num_seqs=MAX_IN_FLIGHT)
    tok, enc = engine.llm.get_tokenizer(), harmony_io.encoding()
    runner.audit(engine, enc, tok, run)  # installs the hooks; refuses to continue on failure
    engine.control("set_condition", condition={"operator": "baseline", "experts": []})
    for n in range(2):
        rows, seconds = runner.generate(engine, enc, tok, [w["prompt"] for w in work])
        rows = [
            dict(r, **{k: v for k, v in w.items() if k != "prompt"}, pass_index=n, configuration="baseline")
            for w, r in zip(work, rows)
        ]
        for r in rows:
            runner.append(run / "generations.jsonl", r)
        print(f"pass {n}: {len(rows)} answers in {seconds:.0f}s", flush=True)
    print("finished: baseline", flush=True)


if __name__ == "__main__":
    main()

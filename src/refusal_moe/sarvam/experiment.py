"""Runs every expert-group configuration of a Sarvam run on fixed batches, resumably."""

import functools
import hashlib
import json
import random
import time

import numpy as np

from refusal_moe.io import append, read_jsonl, write
from refusal_moe.paths import DATA, REPO
from refusal_moe.sarvam.chat import encode
from refusal_moe.sarvam.groups import RANKING, activity_table, make_groups
from refusal_moe.sarvam.gsm8k import PROMPT, gold
from refusal_moe.sarvam.vllm_plugin import controls


def condition(engine, group, capture=False, activity=False):
    return engine.apply_model(
        functools.partial(controls.set_condition, condition=group, capture=capture, activity=activity)
    )


def margin(engine, tokenizer, text, group):
    """Teacher-forced log-probability margin between a fixed refusal and a fixed compliance opening."""
    from refusal_moe.sarvam.engine import generate

    condition(engine, group)
    prefix, _ = encode(tokenizer, text, "empty_think")
    scores = []
    for opening in ["I cannot help with that request.", "Here is a helpful answer."]:
        suffix = tokenizer.encode(opening, add_special_tokens=False)
        result = generate(engine, tokenizer, text, "empty_think", 1, prompt_logprobs=0, raw_ids=prefix + suffix)
        values = result["input_token_logprobs"][len(prefix) :]
        if len(values) != len(suffix) or any(x is None for x in values):
            raise RuntimeError("Incomplete teacher-forced continuation log probabilities")
        scores.append(float(np.mean(values)))
    return scores[0] - scores[1]


def run_experiment(engine, tokenizer, args, output, deadline, stopped):
    from refusal_moe.sarvam.engine import generate, generate_many

    baseline = {"id": "baseline", "operator": "baseline", "experts": []}
    bank = json.loads((DATA / "pairs.json").read_text())
    by_id = {r["pair_id"]: r for r in bank["pairs"]}
    discovery = [by_id[u] for u in bank["discovery_ids"][:8]]
    pilot = [by_id[u] for u in bank["pilot_ids"]]
    fresh = args.prompt_bank
    if fresh:
        rows = {}
        for line in (REPO / fresh).read_text().splitlines():
            r = json.loads(line)
            rows.setdefault(r["pair_id"], {"pair_id": r["pair_id"]})[r["condition"]] = r["prompt"]
        pilot = list(rows.values())
        if any(set(r) != {"pair_id", "harmful", "benign"} for r in pilot):
            raise ValueError("Bank rows must form harmful/benign pairs")
    prior = json.loads(RANKING.read_text())["ranking"]
    completed = {r["key"]: r for r in read_jsonl(output / "generations.jsonl")}

    def check():
        if stopped() or time.monotonic() >= deadline:
            raise TimeoutError

    def request_batch(items, regime, cap, group, **meta):
        """items: [(key,text,extra_meta)], generated together as one fixed batch."""
        if all(k in completed for k, _, _ in items):
            return [completed[k] for k, _, _ in items]
        check()
        condition(engine, group)
        caps = [x.get("_cap", cap) for _, _, x in items]
        records, elapsed = generate_many(engine, tokenizer, [t for _, t, _ in items], regime, caps)
        batch_id = hashlib.sha256("|".join(k.split("/", 2)[-1] for k, _, _ in items).encode()).hexdigest()[:12]
        out = []
        for (key, text, extra), r, c in zip(items, records, caps):
            if key in completed:
                out.append(completed[key])
                continue
            m = dict(meta, **{k: v for k, v in extra.items() if k != "_cap"})
            r.update(
                key=key,
                prompt=text,
                regime=regime,
                max_new_tokens=c,
                configuration=group["id"],
                operator=group["operator"],
                experts=group["experts"],
                alpha=group.get("alpha", 0),
                batch_id=batch_id,
                batch_position=len(out),
                **m,
                **({"activate": group["activate"]} if group.get("activate") else {}),
            )
            append(output / "generations.jsonl", r)
            completed[key] = r
            out.append(r)
        print(
            f"batch {group['id']} x{len(items)} ({meta.get('phase')}): {sum(r['output_tokens'] for r in records)} tokens "
            f"in {elapsed:.1f}s, capped {sum(r['token_cap_reached'] for r in records)}",
            flush=True,
        )
        return out

    try:
        activity_rows = read_jsonl(output / "activity.jsonl")
        seen = {r["uid"] for r in activity_rows}
        for row in discovery:
            for label in ["harmful", "benign"]:
                uid = row["pair_id"] + ":" + label
                if uid in seen:
                    continue
                check()
                condition(engine, baseline, activity=True)
                generate(engine, tokenizer, row[label], "empty_think", 1)
                captured = engine.apply_model(controls.collect)[0]
                if not captured["calls"]:
                    raise RuntimeError("Activity hooks were bypassed")
                record = {"uid": uid, "condition": label, "layers": captured["layers"]}
                append(output / "activity.jsonl", record)
                activity_rows.append(record)
        activity = activity_table(activity_rows)
        prompts = [row[label] for row in discovery[:2] for label in ["harmful", "benign"]]
        basefile = output / "calibration_baseline.json"
        if basefile.exists():
            base = json.loads(basefile.read_text())
        else:
            base = []
            for text in prompts:
                check()
                base.append(margin(engine, tokenizer, text, baseline))
            write(basefile, base)
        exact = read_jsonl(output / "calibration.jsonl")
        seen = {tuple(r["expert"]) for r in exact}
        for candidate in prior[:16]:
            expert = [candidate["layer"], candidate["expert"]]
            if tuple(expert) in seen:
                continue
            group = {"id": "calibration", "operator": "suppress", "experts": [expert]}
            delta = []
            for text, value in zip(prompts, base):
                check()
                delta.append(margin(engine, tokenizer, text, group) - value)
            row = {"expert": expert, "delta": delta}
            append(output / "calibration.jsonl", row)
            exact.append(row)
            print("calibration", expert, delta, flush=True)
        if (output / "groups.json").exists():
            groups = json.loads((output / "groups.json").read_text())
        elif (output / "steermoe_config.json").exists():
            from refusal_moe.sarvam.steermoe import detect_and_build

            groups = detect_and_build(engine, tokenizer, output, check, activity)
        else:
            groups = make_groups(activity, exact, output)
        by_name = {g["id"]: g for g in groups}
        first = ["baseline", "L8E29_suppress", "historical_gradient_n4"]
        first += [f"historical_gradient_n4_random{i}" for i in range(5)]
        first += [
            "historical_n4_exclude",
            "historical_n4_mass_restore",
            "historical_n4_half",
            "L8E29_exclude",
            "L8E29_mass_restore",
        ]
        first = [name for name in first if name in by_name]
        rest = [g for g in groups if g["id"] not in first]
        rest.sort(key=lambda g: (g.get("priority", 1), len(g["experts"]), g.get("matched_to", g["id"]), g["id"]))
        schedule = [by_name[name] for name in first] + rest
        prompts_in_order = [(row, label) for row in pilot for label in ["harmful", "benign"]]
        size = args.batch_size
        if size % 2:
            raise ValueError("Batch size must be even so pairs stay in one batch")
        blocks = [prompts_in_order[i : i + size] for i in range(0, len(prompts_in_order), size)]
        combined = args.gsm8k_with_pilot
        gitems = []
        if args.gsm8k_problems:
            gbank = [json.loads(l) for l in (DATA / "gsm8k/test.jsonl").read_text().splitlines() if l.strip()]
            order = list(range(len(gbank)))
            random.Random(20261108).shuffle(order)
            chosen = [(f"gsm8k_test_{i}", gbank[i]) for i in sorted(order[: args.gsm8k_problems])]
            gitems = [
                (
                    pid,
                    PROMPT.format(question=row["question"]),
                    {"condition": "gsm8k", "pair_id": pid, "gold": gold(row["answer"])},
                )
                for pid, row in chosen
            ]
        if combined and (len(blocks) != 1 or not gitems):
            raise ValueError("--gsm8k-with-pilot needs exactly one pilot batch and GSM8K problems")

        def gsm_combined(group):
            """GSM8K items appended after the pilot batch, with their own phase and cap."""
            return [
                (f"gsm8k/{group['id']}/{pid}", text, dict(x, phase="gsm8k", _cap=args.gsm8k_cap))
                for pid, text, x in gitems
            ]

        write(
            output / "schedule.json",
            {
                "configurations": [g["id"] for g in schedule],
                "pilot_pair_ids": [r["pair_id"] for r in pilot],
                "precision": "BF16 weights, FP32 router logits, no quantization",
                "batch_size": size,
                "pilot_cap": args.pilot_cap,
                "batches": [
                    [f"{row['pair_id']}:{label}" for row, label in b]
                    + ([pid for pid, _, _ in gitems] if combined else [])
                    for b in blocks
                ],
                "gsm8k_in_pilot_batch": combined,
                "gsm8k_cap": args.gsm8k_cap,
            },
        )
        for block_index, block in enumerate(blocks):
            for group in schedule:
                check()
                if group.get("pilot", True) is False:
                    continue
                request_batch(
                    [
                        (
                            f"pilot/{group['id']}/{row['pair_id']}:{label}",
                            row[label],
                            {"condition": label, "pair_id": row["pair_id"], "block": block_index},
                        )
                        for row, label in block
                    ]
                    + (gsm_combined(group) if combined else []),
                    "empty_think",
                    args.pilot_cap,
                    group,
                    phase="pilot",
                )
        if gitems and not combined:
            gblocks = [gitems[i : i + size] for i in range(0, len(gitems), size)]
            for group in schedule:
                for gb in gblocks:
                    request_batch(
                        [(f"gsm8k/{group['id']}/{pid}", text, x) for pid, text, x in gb],
                        "empty_think",
                        args.gsm8k_cap,
                        group,
                        phase="gsm8k",
                    )
        return "queue_complete"
    except TimeoutError:
        return "budget_reached"

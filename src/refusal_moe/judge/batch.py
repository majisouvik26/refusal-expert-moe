"""Judge a run's answers with gpt-5-nano through the OpenAI Batch API (HEx-PHI and XSTest rubrics).

    python -m refusal_moe.judge.batch build --run runs/sarvam_fresh_test [--phases pilot]
    python -m refusal_moe.judge.batch drive --run runs/sarvam_fresh_test --poll 120

Each request: the rubric file as the system message, the prompt and answer as the user message.
Replies that do not parse are sent once more. Labels: <run>_judge/judge_labels.jsonl.
Needs OPENAI_API_KEY in the environment.
"""

import argparse
import json
import time
from pathlib import Path

from refusal_moe.io import read_jsonl
from refusal_moe.judge.common import (
    OUTCOME,
    RUBRICS,
    answer_text,
    blind_id,
    collect_records,
    judge_content,
    load_policies,
    parse_value,
    sha256_text,
)

MODEL, EFFORT, MAX_TOKENS, RETRY_MAX_TOKENS = "gpt-5-nano", "low", 4000, 8000
CHUNK = 2000
DONE_STATES = {"completed", "failed", "expired", "cancelled"}


def body(policy, content, max_tokens):
    return {
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "max_completion_tokens": max_tokens,
        "messages": [{"role": "system", "content": policy}, {"role": "user", "content": content}],
    }


class BatchJob:
    """Request chunks, batch ids and statuses for one judged run, kept in <work>/state.json."""

    def __init__(self, work):
        self.work = Path(work)
        self.state_path = self.work / "state.json"

    def load(self):
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {"chunks": {}}

    def save(self, state):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state_path)

    def submit(self, client):
        s = self.load()
        busy = any(ch["status"] == "submitted" for ch in s["chunks"].values())
        for name, ch in sorted(s["chunks"].items()):
            if ch["status"] not in ("built", "retry") or (ch["status"] == "retry" and busy):
                continue
            f = client.files.create(file=(self.work / f"{name}.requests.jsonl").open("rb"), purpose="batch")
            b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
            ch.update(
                status="submitted",
                input_file_id=f.id,
                batch_id=b.id,
                submitted_at=time.time(),
                attempts=ch.get("attempts", 0) + 1,
            )
            self.save(s)
            print(f"submitted {name}: {b.id}", flush=True)

    def status(self, client):
        s = self.load()
        for name, ch in sorted(s["chunks"].items()):
            if ch["status"] != "submitted":
                continue
            b = client.batches.retrieve(ch["batch_id"])
            ch.update(
                api_status=b.status,
                counts=b.request_counts.model_dump() if b.request_counts else None,
                output_file_id=b.output_file_id,
                error_file_id=b.error_file_id,
            )
            if b.status in DONE_STATES:
                ch["errors"] = [e.model_dump() for e in (b.errors.data if b.errors else [])]
                ch["status"] = "finished" if b.status == "completed" else "retry"
                print(f"{name}: {b.status} {ch['counts']} {ch['errors'][:1]}", flush=True)
        self.save(s)

    def collect(self, client):
        s = self.load()
        for name, ch in sorted(s["chunks"].items()):
            if ch["status"] != "finished":
                continue
            if ch.get("output_file_id"):
                (self.work / f"{name}.output.jsonl").write_bytes(client.files.content(ch["output_file_id"]).read())
            if ch.get("error_file_id"):
                (self.work / f"{name}.errors.jsonl").write_bytes(client.files.content(ch["error_file_id"]).read())
            ch["status"] = "collected"
            self.save(s)
            print(f"collected {name}", flush=True)


def paths(run):
    run = Path(run).resolve()
    return run, run.parent / f"{run.name}_judge"


def build(a):
    run, out = paths(a.run)
    job = BatchJob(out / "batch")
    if job.state_path.exists():
        raise SystemExit(f"{job.state_path} exists; refusing to rebuild over a live job")
    job.work.mkdir(parents=True, exist_ok=True)
    records = collect_records(run, set(a.phases.split(",")))
    policies = load_policies()
    rows, requests = [], []
    for r in records:
        cid = blind_id(r["key"])
        rows.append(
            {
                "custom_id": cid,
                "source": r["_source"],
                "key": r["key"],
                "condition": r["condition"],
                "pair_id": r.get("pair_id"),
                "configuration": r.get("configuration"),
                "token_cap_reached": r.get("token_cap_reached"),
                "regime": r.get("regime"),
                "max_new_tokens": r.get("max_new_tokens"),
                "empty": not answer_text(r),
            }
        )
        if answer_text(r):
            requests.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body(policies[r["condition"]], judge_content(r), MAX_TOKENS),
                }
            )
    assert len({x["custom_id"] for x in rows}) == len(rows)
    with (job.work / "rows.jsonl").open("w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    state = {
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "run": str(run),
        "rubric_sha256": {c: sha256_text(p) for c, p in policies.items()},
        "total_requests": len(requests),
        "empty_by_rule": sum(x["empty"] for x in rows),
        "chunks": {},
    }
    for i in range(0, len(requests), CHUNK):
        name = f"chunk_{i // CHUNK:03d}"
        with (job.work / f"{name}.requests.jsonl").open("w") as f:
            for q in requests[i : i + CHUNK]:
                f.write(json.dumps(q) + "\n")
        state["chunks"][name] = {"requests": len(requests[i : i + CHUNK]), "status": "built"}
    job.save(state)
    print(
        f"{len(rows)} answers: {len(requests)} requests in {len(state['chunks'])} chunks, "
        f"{state['empty_by_rule']} empty (labelled by rule)",
        flush=True,
    )


def outputs(work, conditions):
    """custom_id -> parsed reply; retries override first attempts."""
    got = {}
    for out in sorted(work.glob("chunk_*.output.jsonl")) + sorted(work.glob("retry_*.output.jsonl")):
        for line in read_jsonl(out):
            resp = line.get("response") or {}
            b = resp.get("body") or {}
            if resp.get("status_code") != 200 or not b.get("choices"):
                continue
            raw = b["choices"][0]["message"].get("content") or ""
            value, err = parse_value(raw, conditions[line["custom_id"]])
            u = b.get("usage") or {}
            got[line["custom_id"]] = {
                "value": value,
                "err": err,
                "raw": raw,
                "model": b.get("model"),
                "attempt": "retry" if out.name.startswith("retry_") else "primary",
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "finish_reason": b["choices"][0].get("finish_reason"),
            }
    return got


def add_retry(job):
    """One retry chunk for every request without a parsed value; returns its size."""
    s = job.load()
    if any(n.startswith("retry_") for n in s["chunks"]):
        return 0
    rows = {x["custom_id"]: x for x in read_jsonl(job.work / "rows.jsonl")}
    got = outputs(job.work, {k: v["condition"] for k, v in rows.items()})
    sent = {q["custom_id"]: q for req in sorted(job.work.glob("chunk_*.requests.jsonl")) for q in read_jsonl(req)}
    redo = [q for cid, q in sent.items() if got.get(cid, {}).get("value") is None]
    if not redo:
        return 0
    with (job.work / "retry_000.requests.jsonl").open("w") as f:
        for q in redo:
            q["body"]["max_completion_tokens"] = RETRY_MAX_TOKENS
            f.write(json.dumps(q) + "\n")
    s["chunks"]["retry_000"] = {"requests": len(redo), "status": "built"}
    job.save(s)
    print(f"retry chunk: {len(redo)} requests", flush=True)
    return len(redo)


def merge(job, out):
    s = job.load()
    rows = read_jsonl(job.work / "rows.jsonl")
    got = outputs(job.work, {x["custom_id"]: x["condition"] for x in rows})
    tok = {"prompt": 0, "completion": 0, "reasoning": 0}
    unparsed = 0
    with (out / "judge_labels.jsonl").open("w") as f:
        for x in rows:
            base = {
                "source": x["source"],
                "key": x["key"],
                "blind_id": x["custom_id"],
                "pair_id": x["pair_id"],
                "condition": x["condition"],
                "rubric": RUBRICS[x["condition"]]["name"],
                "configuration": x["configuration"],
                "regime": x["regime"],
                "max_new_tokens": x["max_new_tokens"],
                "token_cap_reached": x["token_cap_reached"],
                "rubric_sha256": s["rubric_sha256"][x["condition"]],
            }
            if x["empty"]:
                row = dict(
                    base,
                    judge_value=None,
                    outcome="no_final_answer",
                    label_source="rule:no_final_answer",
                    judge_attempt="rule",
                    parse_error=None,
                    final_raw=None,
                )
            else:
                g = got.get(x["custom_id"]) or {}
                for k in tok:
                    tok[k] += g.get(f"{k}_tokens") or 0
                unparsed += g.get("value") is None
                row = dict(
                    base,
                    judge_value=g.get("value"),
                    outcome=OUTCOME.get(g.get("value")) if g.get("value") else None,
                    label_source=MODEL,
                    judge_model=g.get("model"),
                    judge_attempt=g.get("attempt", "missing"),
                    parse_error=g.get("err", "no_reply"),
                    final_raw=g.get("raw"),
                    finish_reason=g.get("finish_reason"),
                    reasoning_tokens=g.get("reasoning_tokens"),
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "judge_config.json").write_text(
        json.dumps(
            {
                "judge_model": MODEL,
                "reasoning_effort": EFFORT,
                "max_completion_tokens": [MAX_TOKENS, RETRY_MAX_TOKENS],
                "rubric_sha256": s["rubric_sha256"],
                "run": s["run"],
            },
            indent=1,
        )
        + "\n"
    )
    summary = {
        "rows": len(rows),
        "empty_by_rule": sum(x["empty"] for x in rows),
        "unparsed_after_retry": unparsed,
        "tokens": tok,
    }
    (out / "merge_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def drive(a):
    from openai import OpenAI

    _, out = paths(a.run)
    job = BatchJob(out / "batch")
    client = OpenAI()
    while True:
        job.submit(client)
        job.status(client)
        job.collect(client)
        chunks = job.load()["chunks"].values()
        if all(ch["status"] == "collected" for ch in chunks) and not add_retry(job):
            break
        print(
            time.strftime("%H:%M:%S"),
            {
                st: sum(ch["status"] == st for ch in job.load()["chunks"].values())
                for st in ("built", "submitted", "retry", "collected")
            },
            flush=True,
        )
        time.sleep(a.poll)
    merge(job, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True, help="Generation run directory")
    b.add_argument("--phases", default="pilot", help="Comma-separated generation phases to judge")
    d = sub.add_parser("drive")
    d.add_argument("--run", required=True)
    d.add_argument("--poll", type=int, default=120)
    m = sub.add_parser("merge")
    m.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    elif a.cmd == "drive":
        drive(a)
    else:
        _, out = paths(a.run)
        merge(BatchJob(out / "batch"), out)


if __name__ == "__main__":
    main()

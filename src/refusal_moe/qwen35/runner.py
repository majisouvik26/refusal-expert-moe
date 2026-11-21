"""Qwen3.5-35B-A3B on vLLM 0.33.0 (BF16, text only, thinking off): engine loading, generation and router-hook audit."""

import json
import os
import time

from refusal_moe.io import write
from refusal_moe.qwen35 import chat_io

REVISION = "2508b2437e382c605eb0fc52b83f5545960f6b4a"
CAP = 2048
SEED = 20261120


class Engine:
    def __init__(self, llm):
        self.llm = llm

    def control(self, fn, **options):
        return self.llm.collective_rpc("research_control", kwargs={"operation": fn, "options": options}, timeout=120)[0]


def load(model_dir, run, max_num_seqs):
    from vllm import LLM

    settings = dict(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype="bfloat16",
        tensor_parallel_size=1,
        language_model_only=True,
        moe_backend="triton",
        gdn_prefill_backend="triton",
        gpu_memory_utilization=0.90,
        max_model_len=1024 + CAP + 64,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        worker_extension_cls="refusal_moe.qwen35.vllm_plugin.worker.ResearchWorker",
        seed=SEED,
        generation_config="vllm",
    )
    write(
        run / "engine_settings.json",
        dict(settings, environment={k: os.environ.get(k) for k in ["VLLM_ATTENTION_BACKEND"]}),
    )
    t = time.monotonic()
    engine = Engine(LLM(**settings))
    write(run / "load_time.json", {"seconds": time.monotonic() - t})
    desc = engine.control("describe")  # hooks are installed later, after the stock reference answer
    write(run / "model_representation.json", desc)
    if sorted(map(int, desc["layers"])) != list(range(40)):
        raise RuntimeError("Not all 40 MoE layers found")
    for row in desc["layers"].values():
        if row["monolithic"] or row["top_k"] != 8 or row["experts"] != 256 or not row["renormalize"]:
            raise RuntimeError(f"Unexpected MoE layer: {row}")
    return engine


def stop_ids(tok):
    return sorted({tok.convert_tokens_to_ids(t) for t in chat_io.STOP_TOKENS})


def generate(engine, tok, prompts, cap=CAP):
    """One fixed batch, greedy. Batch membership affects numerics, so callers keep it fixed."""
    from vllm import SamplingParams

    ids = [chat_io.render(tok, p) for p in prompts]
    params = SamplingParams(
        temperature=0, max_tokens=cap, stop_token_ids=stop_ids(tok), skip_special_tokens=False, logprobs=1
    )
    t = time.monotonic()
    results = engine.llm.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False)
    seconds = time.monotonic() - t
    rows = []
    for p, i, r in zip(prompts, ids, results):
        out = r.outputs[0]
        tokens = list(out.token_ids)
        raw = tok.decode(tokens, skip_special_tokens=False)
        parsed = chat_io.parse(raw)
        rows.append(
            dict(
                prompt=p,
                input_ids=i,
                prompt_tokens=len(i),
                generated_ids=tokens,
                output_tokens=len(tokens),
                raw_text=raw,
                final_answer=parsed["final"],
                analysis=parsed["analysis"],
                has_final=parsed["has_final"],
                final_ended=parsed["ended"],
                finish_reason=out.finish_reason,
                capped=out.finish_reason == "length",
                token_logprobs=[float(s[t].logprob) for t, s in zip(tokens, out.logprobs or [])],
                batch_size=len(prompts),
                batch_seconds=seconds,
            )
        )
    return rows, seconds


def audit(engine, tok, run):
    import numpy as np

    prompt = "Reply with only the number: 2 + 3 ="
    gen = lambda: generate(engine, tok, [prompt], cap=24)[0][0]

    def request(condition, capture=True, activity=False):
        engine.control("set_condition", condition=condition, capture=capture, activity=activity)
        out = gen()
        info = engine.control("collect")
        if not info["calls"]:
            raise RuntimeError("Router hooks were bypassed")
        return out, info

    stock = gen()  # stock vLLM: hooks not yet installed
    write(run / "model_representation_hooked.json", engine.control("install"))
    base, base_info = request({"operator": "baseline", "experts": []}, activity=True)
    report = {
        "prompt": prompt,
        "rendered_prompt": tok.decode(stock["input_ids"], skip_special_tokens=False),
        "stock_raw": stock["raw_text"],
        "stock_answer": chat_io.parse(stock["raw_text"]),
        "baseline_tokens_equal": stock["generated_ids"] == base["generated_ids"],
        "baseline_logprobs_equal": stock["token_logprobs"] == base["token_logprobs"],
        "layers_hooked": len(base_info["layers"]),
        "checks": {},
    }
    for layer in (3, 36):
        row = base_info["layers"][str(layer)]
        active = row["ids"][0]
        inactive = min(range(256), key=lambda e: row["frequency"][e])
        inactive_used = row["frequency"][inactive] > 0
        ex = lambda e, a: {"operator": "suppress", "experts": [[layer, e]], "alpha": a}
        inac, inac_info = request(ex(inactive, 0.0))
        one, one_info = request(ex(active, 1.0))
        zero, zero_info = request(ex(active, 0.0))
        half, half_info = request(ex(active, 0.5))
        b = np.array(base_info["layers"][str(layer)]["routed"])
        z = np.array(zero_info["layers"][str(layer)]["routed"])
        h = np.array(half_info["layers"][str(layer)]["routed"])
        norm = max(np.linalg.norm(b), 1e-8)
        report["checks"][str(layer)] = {
            "active_expert": active,
            "inactive_expert": inactive,
            "inactive_expert_used_in_prompt": inactive_used,
            "inactive_tokens_equal": inactive_used or inac["generated_ids"] == stock["generated_ids"],
            "inactive_selected_hits": inac_info["layers"][str(layer)]["selected_hits"],
            "alpha_one_tokens_equal": one["generated_ids"] == stock["generated_ids"],
            "alpha_one_logprobs_equal": one["token_logprobs"] == stock["token_logprobs"],
            "zero_selected_hits": zero_info["layers"][str(layer)]["selected_hits"],
            "zero_relative_change": float(np.linalg.norm(b - z) / norm),
            "half_midpoint_relative_error": float(np.linalg.norm(h - (b + z) / 2) / max(np.linalg.norm(b - z), 1e-8)),
            "zero_answer": chat_io.parse(zero["raw_text"])["final"],
        }
    c = report["checks"].values()
    report["passed"] = bool(
        report["baseline_tokens_equal"]
        and report["baseline_logprobs_equal"]
        and report["layers_hooked"] == 40
        and all(
            x["inactive_tokens_equal"]
            and x["alpha_one_tokens_equal"]
            and x["alpha_one_logprobs_equal"]
            and x["zero_selected_hits"] > 0
            and x["zero_relative_change"] > 0
            and x["half_midpoint_relative_error"] < 0.05
            for x in c
        )
    )
    write(run / "audit.json", report)
    engine.control("set_condition", condition={"operator": "baseline"})
    if not report["passed"]:
        raise RuntimeError("Qwen3.5 intervention audit failed; study blocked")
    print("audit passed", json.dumps(report["checks"]), flush=True)
    return report

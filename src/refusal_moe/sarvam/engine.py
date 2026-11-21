"""Sarvam-30B on vLLM (BF16, FP32 routers): load, audit the router hook, then run an experiment.

python -m refusal_moe.sarvam.engine --run-dir runs/sarvam_pilot --batch-size 24 --pilot-cap 2048
"""

import argparse
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from refusal_moe.io import append, write
from refusal_moe.paths import MODELS, REPO
from refusal_moe.sarvam.chat import REVISION, encode, parse_answer, quality

STOP = False
PLACEMENTS = {"single_gpu": 1, "two_gpu": 2}
SOURCE_FILES = [
    "src/refusal_moe/io.py",
    "src/refusal_moe/sarvam/chat.py",
    "src/refusal_moe/sarvam/engine.py",
    "src/refusal_moe/sarvam/experiment.py",
    "src/refusal_moe/sarvam/groups.py",
    "src/refusal_moe/sarvam/gsm8k.py",
    "src/refusal_moe/sarvam/steermoe.py",
    "src/refusal_moe/sarvam/vllm_plugin/__init__.py",
    "src/refusal_moe/sarvam/vllm_plugin/controls.py",
    "src/refusal_moe/sarvam/vllm_plugin/model.py",
    "src/refusal_moe/sarvam/vllm_plugin/worker.py",
    "data/pairs.json",
    "data/sarvam/historical_ranking.json",
]


def stop(signum, frame):
    global STOP
    STOP = True
    print("Stop requested; saving after the current vLLM request.", flush=True)


def validate_gpu(placement):
    import torch

    want = PLACEMENTS[placement]
    if torch.cuda.device_count() != want:
        raise RuntimeError(f"{placement} needs exactly {want} visible GPU(s)")
    devices = []
    for i in range(want):
        p = torch.cuda.get_device_properties(i)
        devices.append({"logical_gpu": i, "name": p.name, "total_memory_GiB": p.total_memory / 2**30})
    if placement == "single_gpu" and devices[0]["total_memory_GiB"] < 75:
        raise RuntimeError("single_gpu needs one GPU with at least 80 GB (the BF16 weights are about 60 GiB)")
    return {"devices": devices, "torch_cuda": torch.version.cuda}


def merge_pipeline_traces(results):
    merged = {"calls": 0, "layers": {}}
    for result in results:
        if set(merged["layers"]) & set(result["layers"]):
            raise RuntimeError("Duplicate layers in pipeline traces")
        merged["calls"] += result["calls"]
        merged["layers"].update(result["layers"])
    if set(merged["layers"]) != {str(i) for i in range(1, 19)}:
        raise RuntimeError("Pipeline traces do not cover all 18 MoE layers")
    return merged


class ResearchEngine:
    """vLLM engine whose workers expose the routing controls over collective RPC."""

    def __init__(self, engine):
        self.engine = engine

    def generate(self, *args, **kwargs):
        return self.engine.generate(*args, **kwargs)

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    def close(self):
        self.engine.llm_engine.engine_core.shutdown()

    def apply_model(self, fn):
        operation = fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
        options = fn.keywords if isinstance(fn, functools.partial) else {}
        results = self.engine.collective_rpc(
            "research_control", kwargs={"operation": operation, "options": options}, timeout=120
        )
        return [merge_pipeline_traces(results)] if operation == "collect" else results


def load(model, placement, output, overrides=None):
    from vllm import LLM

    from refusal_moe.sarvam.vllm_plugin import controls

    receipt = json.loads((model / "download_receipt.json").read_text())
    if receipt["revision"] != REVISION:
        raise RuntimeError("Wrong checkpoint revision")
    pipeline = placement == "two_gpu"
    if pipeline and not os.environ.get("VLLM_PP_LAYER_PARTITION"):
        raise RuntimeError("two_gpu needs VLLM_PP_LAYER_PARTITION, for example 13,6")
    settings = dict(
        model=str(model),
        tokenizer=str(model),
        trust_remote_code=True,
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=2 if pipeline else 1,
        cpu_offload_gb=0,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        worker_extension_cls="refusal_moe.sarvam.vllm_plugin.worker.ResearchWorker",
        disable_custom_all_reduce=True,
        seed=20261106,
        generation_config="vllm",
    )
    allowed = {"max_num_seqs", "max_num_batched_tokens", "max_model_len", "gpu_memory_utilization"}
    if overrides:
        if set(overrides) - allowed:
            raise ValueError(f"Cannot override {set(overrides) - allowed}")
        settings.update(overrides)
    env = {
        k: os.environ.get(k)
        for k in ["SARVAM_ROUTER_LOGITS_FP32", "VLLM_PP_LAYER_PARTITION", "VLLM_USE_FUSED_MOE_GROUPED_TOPK"]
    }
    write(output / "engine_settings.json", dict(settings, environment=env))
    if pipeline:
        settings["distributed_executor_backend"] = "mp"
    t = time.monotonic()
    engine = ResearchEngine(LLM(**settings))
    write(output / "load_time.json", {"seconds": time.monotonic() - t})
    details = engine.apply_model(controls.describe)
    write(output / "model_representation.json", details)
    want_fp32 = os.environ.get("SARVAM_ROUTER_LOGITS_FP32") == "1"
    if any(d["router_logits_fp32"] != want_fp32 for d in details):
        raise RuntimeError("Router-logit precision differs from SARVAM_ROUTER_LOGITS_FP32 on some worker")
    for detail in details:
        if detail["quantization"] != "None" or any(
            k not in {"torch.bfloat16", "torch.float32"} for k in detail["dtype_bytes"]
        ):
            raise RuntimeError("Expected BF16 weights with native FP32 router parameters, no quantization")
        if detail["offloaded_weight_GiB"] != 0:
            raise RuntimeError("Unexpected CPU weight offload")
    return engine


def stop_ids(tokenizer):
    stops = {tokenizer.eos_token_id}
    for token in ["<|end_of_turn|>", "<eos>"]:
        value = tokenizer.convert_tokens_to_ids(token)
        if value is not None and value != tokenizer.unk_token_id:
            stops.add(value)
    return sorted(x for x in stops if x is not None)


def generate(engine, tokenizer, text, regime, cap, prompt_logprobs=None, raw_ids=None):
    """One greedy request; raw_ids teacher-forces a given token sequence."""
    from vllm import SamplingParams

    ids, rendered = encode(tokenizer, text, regime) if raw_ids is None else (raw_ids, "teacher_forced")
    stops = stop_ids(tokenizer)
    params = SamplingParams(
        temperature=0,
        max_tokens=cap,
        stop_token_ids=stops,
        skip_special_tokens=False,
        logprobs=1,
        prompt_logprobs=prompt_logprobs,
    )
    start = time.monotonic()
    result = engine.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
    elapsed = time.monotonic() - start
    answer = result.outputs[0]
    tokens = list(answer.token_ids)
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    clean = raw
    for tid in stops:
        clean = clean.replace(tokenizer.decode([tid]), "")
    final, has_final = parse_answer(clean, regime)
    lp = [float(step[token].logprob) for token, step in zip(tokens, answer.logprobs or [])]
    record = dict(
        raw_text=raw,
        final_answer=final,
        input_ids=ids,
        rendered_prompt=rendered,
        generated_ids=tokens,
        prompt_tokens=len(ids),
        output_tokens=len(tokens),
        seconds=elapsed,
        tokens_per_second=len(tokens) / max(elapsed, 1e-8),
        eos_reached=answer.finish_reason == "stop",
        interrupted=False,
        token_logprobs=lp,
        **quality(raw, final, has_final, answer.finish_reason == "length"),
    )
    if prompt_logprobs is not None:
        record["input_token_logprobs"] = [
            None if step is None else float(step[token].logprob) for token, step in zip(ids, result.prompt_logprobs)
        ]
    return record


def generate_many(engine, tokenizer, texts, regime, cap):
    """Greedy generation for one fixed batch; cap may be one value or one per prompt."""
    from vllm import SamplingParams

    encoded = [encode(tokenizer, t, regime) for t in texts]
    stops = stop_ids(tokenizer)
    caps = list(cap) if isinstance(cap, (list, tuple)) else [cap] * len(texts)
    if len(caps) != len(texts):
        raise ValueError("One cap per prompt")
    params = [
        SamplingParams(temperature=0.0, max_tokens=c, stop_token_ids=stops, skip_special_tokens=False, logprobs=1)
        for c in caps
    ]
    start = time.monotonic()
    results = engine.generate([{"prompt_token_ids": ids} for ids, _ in encoded], params, use_tqdm=False)
    elapsed = time.monotonic() - start
    records = []
    for (ids, rendered), result in zip(encoded, results):
        answer = result.outputs[0]
        tokens = list(answer.token_ids)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        clean = raw
        for tid in stops:
            clean = clean.replace(tokenizer.decode([tid]), "")
        final, has_final = parse_answer(clean, regime)
        lp = [float(step[token].logprob) for token, step in zip(tokens, answer.logprobs or [])]
        records.append(
            dict(
                raw_text=raw,
                final_answer=final,
                input_ids=ids,
                rendered_prompt=rendered,
                generated_ids=tokens,
                prompt_tokens=len(ids),
                output_tokens=len(tokens),
                batch_seconds=elapsed,
                seconds=elapsed,
                tokens_per_second=len(tokens) / max(elapsed, 1e-8),
                batch_size=len(texts),
                eos_reached=answer.finish_reason == "stop",
                interrupted=False,
                token_logprobs=lp,
                temperature=0.0,
                **quality(raw, final, has_final, answer.finish_reason == "length"),
            )
        )
    return records, elapsed


def audit(engine, tokenizer, output):
    """No-op, inactive-expert, alpha-one, suppression and mass checks on layer 8; blocks the run on failure."""
    from refusal_moe.sarvam.vllm_plugin import controls

    prompt = "Reply with only the number: 2 + 3 ="
    native = generate(engine, tokenizer, prompt, "empty_think", 8)
    engine.apply_model(controls.install)

    def request(condition):
        engine.apply_model(functools.partial(controls.set_condition, condition=condition, capture=True))
        response = generate(engine, tokenizer, prompt, "empty_think", 8)
        captured = engine.apply_model(controls.collect)[0]
        if not captured["calls"]:
            raise RuntimeError("Router hooks were bypassed")
        return response, captured

    noop, base = request({"operator": "baseline", "experts": []})
    active = base["layers"]["8"]["ids"][0]
    engine.apply_model(functools.partial(controls.set_condition, condition={"operator": "baseline"}, activity=True))
    generate(engine, tokenizer, prompt, "empty_think", 8)
    frequency = engine.apply_model(controls.collect)[0]["layers"]["8"]["frequency"]
    inactive = next(e for e, value in enumerate(frequency) if value == 0)
    inactive_response, inactive_trace = request({"operator": "suppress", "experts": [[8, inactive]]})
    identity, _ = request({"operator": "suppress", "experts": [[8, active]], "alpha": 1})
    suppressed, supp_trace = request({"operator": "suppress", "experts": [[8, active]]})
    excluded, exclude_trace = request({"operator": "exclude", "experts": [[8, active]]})
    restored, restore_trace = request({"operator": "mass_restore", "experts": [[8, active]]})
    b = np.array(base["layers"]["8"]["routed"])
    s = np.array(supp_trace["layers"]["8"]["routed"])
    report = {
        "prompt": prompt,
        "active_expert": active,
        "inactive_expert": inactive,
        "no_op_tokens_equal": native["generated_ids"] == noop["generated_ids"],
        "no_op_logprobs_equal": native["token_logprobs"] == noop["token_logprobs"],
        "inactive_tokens_equal": native["generated_ids"] == inactive_response["generated_ids"],
        "inactive_logprobs_equal": native["token_logprobs"] == inactive_response["token_logprobs"],
        "inactive_selected_hits": inactive_trace["layers"]["8"]["selected_hits"],
        "alpha_one_tokens_equal": native["generated_ids"] == identity["generated_ids"],
        "alpha_one_logprobs_equal": native["token_logprobs"] == identity["token_logprobs"],
        "suppression_block_relative_change": float(np.linalg.norm(b - s) / max(np.linalg.norm(b), 1e-8)),
        "shared_prefix_contribution_unchanged": base["layers"]["8"]["shared"] == supp_trace["layers"]["8"]["shared"],
        "exclude_mass_error": exclude_trace["layers"]["8"]["max_mass_error"],
        "restored_mass_error": restore_trace["layers"]["8"]["max_mass_error"],
        "baseline_answer": native["final_answer"],
    }
    report["passed"] = (
        all(
            report[k]
            for k in [
                "no_op_tokens_equal",
                "no_op_logprobs_equal",
                "inactive_tokens_equal",
                "inactive_logprobs_equal",
                "alpha_one_tokens_equal",
                "alpha_one_logprobs_equal",
                "shared_prefix_contribution_unchanged",
            ]
        )
        and report["inactive_selected_hits"] == 0
        and report["suppression_block_relative_change"] > 0
        and max(report["exclude_mass_error"], report["restored_mass_error"]) < 1e-5
    )
    write(output / "audit.json", report)
    write(
        output / "audit_generations.json",
        {
            "native": native,
            "no_op": noop,
            "inactive": inactive_response,
            "alpha_one": identity,
            "suppressed": suppressed,
            "excluded": excluded,
            "restored_mass": restored,
        },
    )
    engine.apply_model(functools.partial(controls.set_condition, condition={"operator": "baseline"}))
    if not report["passed"]:
        raise RuntimeError("Router-hook audit failed; run blocked")
    print("Audit passed:", report, flush=True)
    return report


def identity(args):
    files = SOURCE_FILES + ([args.prompt_bank] if args.prompt_bank else [])
    return {
        "files": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in files},
        "placement": args.placement,
        "revision": REVISION,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default=str(MODELS / "sarvam-30b"))
    p.add_argument("--placement", choices=list(PLACEMENTS), default="single_gpu")
    p.add_argument("--hours", type=float, default=0.5, help="Time budget; a rerun resumes the same run")
    p.add_argument("--batch-size", type=int, default=24, help="Prompts per fixed batch")
    p.add_argument("--pilot-cap", type=int, default=1500, help="Token cap for pilot answers")
    p.add_argument("--gsm8k-problems", type=int, default=0, help="GSM8K problems per configuration (0 = off)")
    p.add_argument("--gsm8k-cap", type=int, default=1024)
    p.add_argument(
        "--gsm8k-with-pilot",
        action="store_true",
        help="Put the GSM8K problems in the same fixed batch as the single pilot batch",
    )
    p.add_argument("--prompt-bank", help="JSONL pair bank (repo-relative) that replaces the 48 pilot pairs")
    args = p.parse_args()
    if not 0 < args.hours <= 24:
        p.error("--hours must be in (0, 24]")
    output = Path(args.run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    deadline = started + args.hours * 3600
    packages = {k: importlib.metadata.version(k) for k in ["torch", "vllm", "transformers", "numpy"]}
    if (output / "packages.json").exists() and json.loads((output / "packages.json").read_text()) != packages:
        raise RuntimeError("Package versions changed; use a new run directory")
    write(output / "packages.json", packages)
    write(output / "hardware.json", validate_gpu(args.placement))
    write(
        output / "protocol.json",
        dict(vars(args), revision=REVISION, precision="BF16 + native FP32 routers; no quantization"),
    )
    ident = identity(args)
    if (output / "identity.json").exists() and json.loads((output / "identity.json").read_text()) != ident:
        raise RuntimeError("Code, data, or settings changed; use a new run directory")
    write(output / "identity.json", ident)
    status, engine = "failed", None
    overrides = {
        "max_num_seqs": args.batch_size + (args.gsm8k_problems if args.gsm8k_with_pilot else 0),
        "max_num_batched_tokens": 4096,
        "max_model_len": max(4096, 1024 + args.pilot_cap + 64),
    }
    try:
        engine = load(Path(args.model), args.placement, output, overrides)
        tokenizer = engine.get_tokenizer()
        audit(engine, tokenizer, output)
        from refusal_moe.sarvam.experiment import run_experiment

        status = run_experiment(engine, tokenizer, args, output, deadline, lambda: STOP)
    finally:
        append(
            output / "sessions.jsonl", {"status": status, "seconds": time.monotonic() - started, "hours": args.hours}
        )
        if engine is not None:
            engine.close()
    print("Session finished:", status, flush=True)


if __name__ == "__main__":
    main()

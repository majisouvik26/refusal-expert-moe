"""Run one named Sarvam experiment (configs/sarvam/experiments.json): seed if needed, then the engine.

    python -m refusal_moe.sarvam.launch fresh-test --run-dir runs/sarvam_fresh_test [--placement single_gpu]
Rerunning the same command resumes an interrupted run.
"""

import argparse
import json
import sys

from refusal_moe.paths import CONFIGS, MODELS, REPO
from refusal_moe.sarvam import engine, seeds

EXPERIMENTS = json.loads((CONFIGS / "sarvam/experiments.json").read_text())


def launch(experiment, run_dir, placement="single_gpu", model=None):
    spec = EXPERIMENTS[experiment]
    run = (REPO / run_dir).resolve()
    if spec["seed"] and not ((run / "groups.json").exists() or (run / "steermoe_config.json").exists()):
        sys.argv = ["seeds", spec["seed"], "--to", str(run)]
        seeds.main()
    sys.argv = [
        "engine",
        "--run-dir",
        str(run),
        "--placement",
        placement,
        "--model",
        model or str(MODELS / "sarvam-30b"),
        *spec["args"],
    ]
    engine.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(EXPERIMENTS))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--placement", choices=list(engine.PLACEMENTS), default="single_gpu")
    ap.add_argument("--model")
    a = ap.parse_args()
    launch(a.experiment, a.run_dir, a.placement, a.model)


if __name__ == "__main__":
    main()

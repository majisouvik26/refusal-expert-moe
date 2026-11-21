"""Run the GPT-OSS-20B pipeline on Modal (one A100-80GB).

    modal run scripts/modal_gpt_oss.py::download
    modal run --detach scripts/modal_gpt_oss.py::launch --task pilot-n10 --name gpt_oss_run
    modal run scripts/modal_gpt_oss.py::progress --name gpt_oss_run
    modal run scripts/modal_gpt_oss.py::fetch --name gpt_oss_run          # copies the run to runs/

Tasks: baseline, discovery, selection, pilot-n10, pilot-n32.
Uses your active Modal profile (or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET). The weight download reads HF_TOKEN
from the Modal secret named by HF_SECRET_NAME (default "huggingface"), or from a local HF_TOKEN.
"""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CODE = "/root/repo"
PYTHON = "/opt/venv/bin/python"
MODEL = "gpt-oss-20b"
MODEL_DIR = f"/models/{MODEL}"
TASKS = {
    "baseline": ["refusal_moe.gpt_oss.baseline"],
    "discovery": ["refusal_moe.gpt_oss.selection", "generate"],
    "selection": ["refusal_moe.gpt_oss.selection", "gradient", "--labels", "configs/gpt_oss/discovery_labels.jsonl"],
    "pilot-n10": [
        "refusal_moe.gpt_oss.pilot",
        "--size",
        "10",
        "--selection",
        "configs/gpt_oss/selection_group_n10.json",
    ],
    "pilot-n32": [
        "refusal_moe.gpt_oss.pilot",
        "--size",
        "32",
        "--selection",
        "configs/gpt_oss/selection_group_n32.json",
    ],
}
ENV = {"CUDA_VISIBLE_DEVICES": "0", "VLLM_MXFP4_USE_MARLIN": "1", "TIKTOKEN_RS_CACHE_DIR": "/tmp/harmony"}


def command_for(task, run, source):
    """selection reads the discovery run named by --source from the runs volume."""
    module, *args = TASKS[task]
    extra = ["--generations", f"/runs/{source}"] if task == "selection" else []
    return [module, *args, "--model", MODEL_DIR, "--run-dir", run, *extra]


app = modal.App("refusal-moe-gpt_oss")
weights = modal.Volume.from_name("refusal-moe-weights", create_if_missing=True)
runs = modal.Volume.from_name("refusal-moe-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.12.23")
    .add_local_file(REPO / "requirements/vllm015.txt", "/opt/requirements.txt", copy=True)
    .run_commands(
        "uv venv /opt/venv --python /usr/local/bin/python3.12 && "
        "VIRTUAL_ENV=/opt/venv uv pip install -r /opt/requirements.txt"
    )
    .add_local_file(REPO / "pyproject.toml", f"{CODE}/pyproject.toml", copy=True)
    .add_local_dir(REPO / "src", f"{CODE}/src", copy=True, ignore=["**/__pycache__"])
    .add_local_dir(REPO / "configs", f"{CODE}/configs", copy=True)
    .add_local_dir(REPO / "data", f"{CODE}/data", copy=True)
    .run_commands(f"VIRTUAL_ENV=/opt/venv uv pip install --no-deps -e {CODE}")
)


def hf_secret():
    if modal.is_local() and os.environ.get("HF_TOKEN"):
        return modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
    return modal.Secret.from_name(os.environ.get("HF_SECRET_NAME", "huggingface"))


@app.function(image=image, volumes={"/models": weights}, secrets=[hf_secret()], timeout=3 * 3600, cpu=4, memory=16384)
def download_weights():
    subprocess.run(
        [PYTHON, "-m", "refusal_moe.download", MODEL, "--to", MODEL_DIR],
        cwd=CODE,
        check=True,
        env=dict(os.environ, HF_HOME="/tmp/hf"),
    )
    weights.commit()


@app.function(
    image=image, gpu="A100-80GB", volumes={"/models": weights, "/runs": runs}, timeout=3 * 3600, cpu=8, memory=98304
)
def run_task(task, name, source=""):
    if not (Path(MODEL_DIR) / "download_receipt.json").exists():
        raise RuntimeError("Weights missing: run ::download first")
    env = dict(
        os.environ,
        HF_HOME="/tmp/hf",
        TMPDIR="/tmp/work",
        VLLM_RPC_BASE_PATH="/tmp/ipc",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_USAGE_STATS_ENABLED="0",
        DO_NOT_TRACK="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        **ENV,
    )
    for d in ["/tmp/hf", "/tmp/work", "/tmp/ipc"]:
        os.makedirs(d, exist_ok=True)
    run, stored = Path(CODE) / "runs" / name, Path("/runs") / name
    if stored.exists():
        raise RuntimeError(f"{name} already exists in the volume; use a new run name")
    run.parent.mkdir(parents=True, exist_ok=True)
    if stored.exists():
        shutil.copytree(stored, run, ignore=shutil.ignore_patterns(".run.lock"))
    run.mkdir(parents=True, exist_ok=True)

    def mirror():
        shutil.copytree(run, stored, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".run.lock"))
        runs.commit()

    done = threading.Event()

    def committer():
        while not done.wait(300):
            try:
                mirror()
            except Exception as e:
                print("volume mirror failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    command = [PYTHON, "-u", "-m", *command_for(task, str(run), source)]
    with (run / "console.log").open("a") as log:
        log.write(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
        p = subprocess.Popen(command, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        status = p.wait()
        log.write(f"=== {task} exit {status}\n")
    done.set()
    mirror()
    return status


@app.function(volumes={"/runs": runs}, timeout=600)
def read_progress(name):
    run = Path("/runs") / name
    files = sorted(f.name for f in run.iterdir()) if run.exists() else []
    tail = (run / "console.log").read_text()[-2500:] if (run / "console.log").exists() else ""
    return files, tail


@app.function(volumes={"/runs": runs}, timeout=1800)
def pack(name):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(Path("/runs") / name, arcname=name)
    return buf.getvalue()


@app.local_entrypoint()
def download():
    download_weights.remote()


@app.local_entrypoint()
def launch(task: str, name: str, source: str = ""):
    if task not in TASKS:
        raise SystemExit(f"Unknown task {task}; choose from {sorted(TASKS)}")
    print("spawned", run_task.spawn(task, name, source).object_id)


@app.local_entrypoint()
def progress(name: str):
    files, tail = read_progress.remote(name)
    print(files)
    print(tail)


@app.local_entrypoint()
def fetch(name: str):
    (REPO / "runs").mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(pack.remote(name)), mode="r:gz") as tar:
        tar.extractall(REPO / "runs", filter="data")
    print("fetched to", REPO / "runs" / name)

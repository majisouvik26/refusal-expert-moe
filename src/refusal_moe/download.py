"""Download a pinned checkpoint from the Hugging Face Hub and write a sha256 receipt.

    python -m refusal_moe.download sarvam-30b --to models/sarvam-30b
Uses HF_TOKEN from the environment when set.
"""

import argparse
import hashlib
import json
from pathlib import Path

from refusal_moe.paths import MODELS

CHECKPOINTS = {
    "sarvam-30b": {
        "repo": "sarvamai/sarvam-30b",
        "revision": "071ae95e933605ca1104a6b4524a36a98488efa4",
        "allow": ["*.json", "*.py", "*.safetensors", "*.model", "*.jinja", "*.txt"],
    },
    "gpt-oss-20b": {
        "repo": "openai/gpt-oss-20b",
        "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "allow": [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "LICENSE",
            "USAGE_POLICY",
        ],
    },
    "qwen35-35b-a3b": {
        "repo": "Qwen/Qwen3.5-35B-A3B",
        "revision": "2508b2437e382c605eb0fc52b83f5545960f6b4a",
        "allow": None,
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, target):
    from huggingface_hub import snapshot_download

    spec = CHECKPOINTS[name]
    target = Path(target)
    receipt = target / "download_receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    snapshot_download(
        spec["repo"], revision=spec["revision"], local_dir=str(target), allow_patterns=spec["allow"], max_workers=4
    )
    files = {
        str(f.relative_to(target)): {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
        for f in sorted(target.rglob("*"))
        if f.is_file() and ".cache" not in f.parts
    }
    out = {"model": spec["repo"], "revision": spec["revision"], "files": files}
    receipt.write_text(json.dumps(out, indent=1) + "\n")
    print(
        f"{spec['repo']}@{spec['revision'][:12]}: {len(files)} files, "
        f"{sum(v['bytes'] for v in files.values()) / 2**30:.1f} GiB in {target}"
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=list(CHECKPOINTS))
    ap.add_argument("--to", help="Target directory (default models/<model>)")
    a = ap.parse_args()
    download(a.model, a.to or MODELS / a.model)


if __name__ == "__main__":
    main()

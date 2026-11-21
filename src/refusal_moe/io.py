"""Crash-safe JSON and JSONL helpers shared by every pipeline."""

import json
import os
from pathlib import Path


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def append(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """Read JSONL; a partial last line left by a hard kill is truncated, any other corruption raises."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result, valid_bytes = [], 0
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise
            with path.open("r+b") as f:
                f.truncate(valid_bytes)
            break
        valid_bytes += len(line)
    return result

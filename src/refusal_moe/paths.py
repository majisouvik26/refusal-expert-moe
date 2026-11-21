"""Repository paths; run commands from the repo root or set REFUSAL_MOE_ROOT."""

import os
from pathlib import Path

REPO = Path(os.environ.get("REFUSAL_MOE_ROOT", ".")).resolve()
DATA = REPO / "data"
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"
MODELS = REPO / "models"

# ───────── src/utils/path_utils.py ────────────────────────────────────────────
"""
Utility paths that work no matter where you launch Python from.
"""

from pathlib import Path

# project root = two levels above *this* file  (src/utils/… → project-root)
ROOT_DIR   = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

PLOTS_DIR      = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS_DIR     = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

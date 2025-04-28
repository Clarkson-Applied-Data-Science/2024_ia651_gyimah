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

# ---------------------------------------------------------------------
# helper: give me the path of the *real* latest pickle for a stage
# ---------------------------------------------------------------------
def latest_checkpoint(stage: str) -> Path | None:
    """
    Returns Path('checkpoints/<stage>_<timestamp>.pkl') or None
    depending on whether the text pointer exists.
    """
    ptr = CHECKPOINT_DIR / f"{stage}_latest.txt"
    if ptr.exists():
        return Path(ptr.read_text().strip())
    return None

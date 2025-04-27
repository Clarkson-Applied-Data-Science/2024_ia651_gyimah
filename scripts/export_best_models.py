# ───────── scripts/post_training_utils.py ────────────────────────────────────
"""
Export best pipelines from a hold-out checkpoint to models/*.joblib

Usage:
    python scripts/post_training_utils.py checkpoints/holdout_evaluation_*.pkl
"""
from pathlib import Path
import sys, joblib, pickle
from utils.path_utils import MODELS_DIR

def main(chk_path: str):
    chk_path = Path(chk_path)
    if not chk_path.exists():
        raise FileNotFoundError(chk_path)

    with open(chk_path, "rb") as fh:
        holdout = pickle.load(fh)

    for arm, bundle in holdout.items():
        pipe  = bundle["pipeline"]
        fname = MODELS_DIR / f"{pipe.named_steps['clf'].__class__.__name__}_{arm}.joblib"
        joblib.dump(pipe, fname)
        print(f"✔  exported → {fname}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("give the .pkl checkpoint as single argument"); sys.exit(1)
    main(sys.argv[1])

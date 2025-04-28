#!/usr/bin/env python

import sys
import pathlib
import joblib
import __main__

# ─────────────────────────────────────────────────────────────────────────────
# 1) Make sure project root (the folder containing `src/`) is on sys.path:
# ─────────────────────────────────────────────────────────────────────────────
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# 2) Inject classes so that unpickling finds them:
# ─────────────────────────────────────────────────────────────────────────────
from src.genetic_variant_classifier2 import (
    PathwayFeatureExtractor,
    DataPreprocessor
)
__main__.PathwayFeatureExtractor = PathwayFeatureExtractor
__main__.DataPreprocessor        = DataPreprocessor
# also expose the helper used in FunctionTransformer
__main__._concat_sequences       = DataPreprocessor._concat_sequences

# ─────────────────────────────────────────────────────────────────────────────
# 3) Now imports work as if you ran "python -m src.…" from PROJECT_ROOT
# ─────────────────────────────────────────────────────────────────────────────
from src.genetic_variant_classifier2 import GeneticVariantAnalysis
from src.utils.post_training_utils import plot_confusion

# ─────────────────────────────────────────────────────────────────────────────
# 4) Configuration: paths & names
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = project_root / "data"   / "raw_variant_data.csv"
MODEL_PATHS = {
    "pop_aware":    project_root / "models" / "best_pop_aware_binary.pkl",
    "nonpop_aware": project_root / "models" / "best_nonpop_aware_binary.pkl",
}
OUTPUT_DIR  = project_root / "plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Load & preprocess once (this will resume from checkpoints, so it's fast)
# ─────────────────────────────────────────────────────────────────────────────
analysis = GeneticVariantAnalysis(str(DATA_PATH))
df_pop   = analysis.run_exploratory_analysis()

# ─────────────────────────────────────────────────────────────────────────────
# 6) Stratified split
# ─────────────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te, pop_tr, pop_te = (
    analysis.preprocessor
            .population_stratified_split(df_pop, test_size=0.20, random_state=42)
)

# ─────────────────────────────────────────────────────────────────────────────
# 7) For each saved model, predict on X_te and plot its confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
for mode, mpath in MODEL_PATHS.items():
    if not mpath.exists():
        print(f"[!] Model not found: {mpath}; skipping {mode}")
        continue

    pipe   = joblib.load(str(mpath))
    y_pred = pipe.predict(X_te)

    title = f"LogisticRegression (binary) — {mode.replace('_', ' ').title()}"
    # plot_confusion signature is: (y_true, y_pred, class_labels, title, out_dir)
    fig_path = plot_confusion(
        y_true       = y_te,
        y_pred       = y_pred,
        class_labels = ["non_pathogenic", "pathogenic"],
        title        = title,
        out_dir      = str(OUTPUT_DIR)
    )
    print(f"Wrote confusion matrix for {mode} → {fig_path}")

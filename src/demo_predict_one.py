"""
Demo: load the *population-aware* RandomForest pipeline saved by your
Hold-out run and score one variant row.
"""
import joblib
import pandas as pd
from post_training_utils import score_with_acmg_hybrid

# Path to the pipeline stored inside the hold-out checkpoint
ckpt = joblib.load("checkpoints/holdout_evaluation_YYYYMMDD_HHMMSS.pkl")
pipe = ckpt["population_aware"]["pipeline"]

# ---- fake variant row ----------------------------------------------------
example = pd.DataFrame({
    "gene": ["G6PD"],
    "chromosome": ["X"],
    "position": [154535341],
    "allele": ["A"],
    "allele_freq": [0.0003],
    "consequence_type": ["missense_variant"],
    "population_name": ["African Caribbean"],
    "region": ["exonic"],
    # + include every feature column pipeline expects
})

acmg_rules = ["PM2", "PP1"]          # toy example
out = score_with_acmg_hybrid(example, pipe, acmg_rules)
print(out)

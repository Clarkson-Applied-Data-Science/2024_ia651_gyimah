"""
Utility helpers that run *after* a model/pipeline has been trained.
Just `import post_training_utils as ptu` wherever you need them.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------------------------------------------ #
# A. Confusion-matrix plotting
# ------------------------------------------------------------------ #
def plot_confusion(y_true, y_pred, class_labels, title, out_dir="plots"):
    """
    Saves a PNG confusion-matrix (normalised by true-class) and
    returns the figure object (so caller may inline it in a notebook).
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)),
                          normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    fig_path = Path(out_dir) / f"{title.lower().replace(' ', '_')}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path

# ------------------------------------------------------------------ #
# B. ACMG + ML hybrid score
# ------------------------------------------------------------------ #
def score_with_acmg_hybrid(row, fitted_pipeline, acmg_rules,
                           alpha=0.7, min_score=-20, max_score=20):
    """
    row            – one-row DataFrame (or Series) containing the variant
    fitted_pipeline – *already trained* pipeline (preproc + clf)
    acmg_rules     – list of ACMG/AMP criteria strings for *this* variant
    α              – weight of ML prob in final score (0–1)
    Returns: dict(prob_pathogenic, acmg_raw, acmg_norm, hybrid_score)
    """
    # --- ML probability ----------------------------------------------------
    prob_path = fitted_pipeline.predict_proba(row)[0, 1]

    # --- ACMG raw score ----------------------------------------------------
    from acmg_scoring import ACMGPopulationIntegration   # your existing class
    api = ACMGPopulationIntegration()
    pop = row.get("population_name", None)
    acmg_raw = api.calculate_score(acmg_rules, pop)

    # --- normalise ACMG to [0,1] ------------------------------------------
    acmg_norm = (acmg_raw - min_score) / (max_score - min_score)
    acmg_norm = np.clip(acmg_norm, 0, 1)

    # --- hybrid ------------------------------------------------------------
    hybrid = alpha * prob_path + (1-alpha) * acmg_norm
    return dict(prob_pathogenic=prob_path,
                acmg_raw=acmg_raw,
                acmg_norm=acmg_norm,
                hybrid_score=hybrid)

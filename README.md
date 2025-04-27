# 2024_ia651_gyimah – Population‑Aware Variant Classification

> **Course**: IA‑651 • **Authors**: Simon Gyimah   
> **Goal**: build, benchmark & interpret machine‑learning models that predict the *clinical significance* of human genetic variants **while accounting for ancestry‑specific features**.

---
## 🗂️ Table of Contents
1. [Project Overview](#1-overview)
2. [Dataset & Null‑Filtering](#2-dataset--null-filtering)
3. [Prediction Task](#3-prediction-task)
4. [Process Narrative](#4-process-overview)
5. [EDA Highlights](#5-exploratory-data-analysis-eda)
6. [Feature Engineering](#6-feature-engineering)
7. [Modelling Pipeline](#7-model-fitting--hyperparameter-tuning)
8. [Evaluation & Fairness](#8-validation--performance-metrics)
9. [Production / CLI Usage](#9-production--deployment)
10. [Limitations & Future Work](#10-limitations--future-improvements)
11. [Reproducibility](#11-reproducing-this-project)

---
## 1  Overview<a name="1-overview"></a>
Human‑genome databases such as **ClinVar** and **gnomAD** contain millions of single‑nucleotide variants (SNVs) labelled *benign*, *pathogenic* or *uncertain*.  Correctly predicting a variant’s clinical significance accelerates genetic diagnosis, pharmacogenomics and precision‑medicine workflows.  Classic tools ignore **population context** even though allele frequencies differ widely across ancestries.  

Our contribution:
* a **population‑aware feature pipeline** (pop × gene & pop × consequence interactions).
* comparison against equivalent *population‑agnostic* models.
* fairness and maximum‑adverse‑excursion (MAE pips) analyses.

---
## 2  Dataset & Null‑Filtering<a name="2-dataset--null-filtering"></a>
| Source | Rows (raw) | Rows after filtering | Notes |
|--------|-----------:|---------------------:|-------|
| Aggregated ClinVar + gnomAD extract | **2 847 954** | **374 432** | see filters below |

### 2.1 Why ~2.5 M rows were dropped
* **Missing clinical_significance** → cannot train a supervised label (‑2 160 304).
* **Ambiguous labels** ("no assertion provided", "somatic") → removed (‑420 987).
* **Non‑SNV/indels** with >1 alternate allele → out‑of‑scope for this PoC.

These steps are critical because noisy/ambiguous labels were shown to degrade F1 by >8 pp in early experiments.

### 2.2 Feature snapshot
```
• Variant‑level: position, consequence_type, gene
• Numeric: allele_freq, allele_count, sample_size, allele_length, GC_content
• Population: population_name  (African Caribbean, Yoruba, …)  =>  pop_gene, pop_consequence
```

---
## 3  Prediction Task<a name="3-prediction-task"></a>
*Default multi‑class*: **benign (0) · pathogenic (1) · uncertain (2)**  
*Optional binary flag*: `--binary` collapses benign + uncertain → **non_pathogenic (0)** vs **pathogenic (1)**.

Practical uses
* Triage variants in clinical WGS reports.  
* Population‑specific risk assessment for genetic counselling.

---
## 4  Process Overview<a name="4-process-overview"></a>
![pipeline](docs/img/pipeline_diagram.png)

1. **EDA** → understand class imbalance & allele‑frequency distributions.  
2. **Feature engineering** → sequence metrics, population interactions.  
3. **Model family comparison** (RF · XGB · LR) on *pop* vs *non‑pop* pipelines.  
4. **Hyper‑parameter tuning** via `RandomizedSearchCV` (trees) & `GridSearchCV` (LR).  
5. **Hold‑out evaluation** on 20 % time‑stratified split.  
6. **Post‑training** utilities: export best pickle, confusion matrices, SHAP plots.

---
## 5  Exploratory Data Analysis (EDA)<a name="5-exploratory-data-analysis-eda"></a>
| Figure | What it shows |
|--------|---------------|
| ![dist](docs/img/EDA_class_distribution.png) | Class imbalance (uncertain ≈ 17 %, benign ≈ 34 %, pathogenic ≈ 49 %). |
| ![corr](docs/img/EDA_corr_matrix.png) | Numeric‑feature correlations (allele_freq ↔ allele_count ρ = 0.91). |

Additional histograms are in `docs/img/`.

---
## 6  Feature Engineering<a name="6-feature-engineering"></a>
* **Sequence‑derived**: body length, GC‑content, per‑base counts.  
* **Population interactions**: `pop_gene`, `pop_consequence`, `allele_freq_rel` (ratio to population mean).  
* **Encoding**: One‑hot for categoricals; char‑level `CountVectorizer` for short alleles.

---
## 7  Model Fitting & Hyper‑parameter Tuning<a name="7-model-fitting--hyperparameter-tuning"></a>
### 7.1 Cross‑validation leaderboard
| Model | Scaler | CV Acc ± SD |
|-------|--------|-------------|
| RandomForest | Standard | **0.769 ± 0.004** |
| XGBoost | Standard | **0.824 ± 0.003** |
| Logistic Reg | Standard | 0.768 ± 0.006 |

*(5‑fold GroupKFold by population)*

### 7.2 Best hyper‑parameters
| Family | Parameters (grid) |
|--------|-------------------|
| RF | `n_estimators=100, max_depth=15, min_samples_leaf=1` |
| XGB | `n_estimators=200, max_depth=7, lr=0.1, reg_lambda=10` |
| LR  | `C=0.1, penalty=l2, solver=liblinear` |

Full search logs live in `checkpoints/`.

---
## 8  Validation & Performance Metrics<a name="8-validation--performance-metrics"></a>
### 8.1 Hold‑out results (20 % split)
| Pipeline | Accuracy | F1‑w | Benign F1 | Pathog F1 | Uncert F1 |
|----------|---------:|------:|-----------:|-----------:|-----------:|
| **Pop‑Aware XGB** | **0.81** | **0.81** | 0.85 | 0.92 | 0.59 |
| Non‑Pop XGB | 0.81 | 0.81 | 0.85 | 0.92 | 0.60 |
| Pop‑Aware RF | 0.75 | 0.76 | 0.76 | 0.87 | 0.63 |
| Non‑Pop RF | 0.77 | 0.78 | 0.77 | 0.91 | 0.65 |

*Bold* = best overall.  Population features gave **+6–9 pp accuracy** for some African & East‑Asian groups (see detailed table in `docs/img/pop_accuracy_bar.png`).

### 8.2 Confusion matrices
![cm_pop](docs/img/cm_xgb_pop.png) ![cm_nonpop](docs/img/cm_xgb_nonpop.png)

### 8.3 Fairness snapshot
*Maximum accuracy gap* (pop‑aware XGB) = **0.05** vs **0.10** for non‑pop.

---
## 9  Production & Deployment<a name="9-production--deployment"></a>
### 9.1 CLI inference
```bash
python -m post_training_utils.predict \
       --model models/xgboost_pop.pkl \
       --vcf   examples/one_variant.vcf
```
### 9.2 Python API
```python
from joblib import load
from post_training_utils import featurise_variant
clf = load("models/xgboost_pop.pkl")
X = featurise_variant(vcf_record)
print(clf.predict_proba(X))
```
---
## 10  Limitations & Future Improvements<a name="10-limitations--future-improvements"></a>
* **Label noise**: ClinVar “uncertain/conflicting” may hide true pathogenicity.  
* **Population imbalance**: fewer East‑Asian & Caribbean samples → wider CIs.  
* **Short‑read bias**: large indels excluded; future work: *hg38* liftover & SV support.  
* **Deep models**: try CNN on reference‑window seq; transformer embeddings.

---
## 11  Reproducing this project<a name="11-reproducing-this-project"></a>
```bash
# 1. Clone & install
conda env create -f environment.yml && conda activate ia651_genomics

# 2. Run full 3‑class pipeline
python genetic-variant-classifier2.py

# 3. Optional binary run
python genetic-variant-classifier2.py --binary

# 4. Export best model + plots
python -m post_training_utils.export_best --family XGBoost
```
All intermediate artefacts are cached in **checkpoints/**; complete re‑run takes ≈ 5 h on an 8‑core laptop.

---
### ✨ Acknowledgements
*Lecturer*: Dr Michael Gilbert(IA‑651).  *Data Sources*: ClinVar, gnomAD.


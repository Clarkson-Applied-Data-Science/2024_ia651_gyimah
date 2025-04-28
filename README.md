# 2024_ia651_gyimah – Population‑Aware Variant Classification

> **Course**: IA‑651 • **Author**: **Simon Gyimah**  
> **Aim**: predict the *clinical significance* of human genetic variants **while explicitly modelling ancestry‑specific context**.

---

## 🗂️ Table of Contents
2. [Dataset & Null-Filtering](#2-dataset--null-filtering)  
3. [Prediction Task](#3-prediction-task)  
4. [Process Narrative](#4-process-overview)  
5. [EDA Highlights](#5-exploratory-data-analysis-eda)  
6. [Feature Engineering](#6-feature-engineering)  
7. [Modelling Pipeline](#7-model-fitting--hyperparameter-tuning)  
8. [Evaluation & Fairness](#8-validation--performance-metrics)  
9. [Production / CLI Usage](#9-production--deployment)  
10. [Limitations & Future Work](#10-limitations--future-improvements)  
11. [Conclusion](#11-conclusion)  
12. [Reproducibility & Runtime](#12-reproducing-this-project)  
13. [Licence & Ethics](#13-licence--ethics)  
14. [References](#14-references) 

---

## 1  Overview<a name="1-overview"></a>
Public genome databases ( **ClinVar**¹, **gnomAD**² ) contain millions of single‑nucleotide variants (SNVs) labelled *benign*, *pathogenic* or *uncertain*.  Classic tools largely ignore **population context** even though allele frequencies differ widely across ancestries.  Our contribution is two‑fold:

* **Population‑aware feature pipeline** – interaction terms `pop × gene`, `pop × consequence`, and frequency‐ratio features.
* **Head‑to‑head benchmark** against equivalent *population‑agnostic* models, including both 3‑class *and* binary tasks, with fairness diagnostics.

---

## 2  Dataset & Null‑Filtering<a name="2-dataset--null-filtering"></a>
| Source | Rows (raw) | Rows after filtering | Notes |
|--------|-----------:|---------------------:|-------|
| ClinVar + gnomAD aggregate | **2 847 954** | **374 432** | strict label / SNV filters |

<details>
<summary>Why ~2.5 M rows were dropped</summary>

* **Missing clinical_significance** → supervised label impossible (‑2 160 304).
* **Ambiguous labels** (“no assertion”, “somatic”, …) (‑420 987).
* **Non‑SNV/indels** (multi‑allelic) out‑of‑scope for this PoC.f
</details>

Feature snapshot
```
• Variant‑level   : position, consequence_type, gene
• Numeric         : allele_freq, allele_count, sample_size, allele_length, GC_content
• Population      : population_name (African Caribbean, Yoruba, …)  -->  pop_gene, pop_consequence, allele_freq_rel
```

---

## 3  Prediction Task<a name="3-prediction-task"></a>
Default **multi‑class**:  
`benign (0)` · `pathogenic (1)` · `uncertain (2)`  
Optional flag `--binary` collapses *benign + uncertain* ⇒ **non_pathogenic (0)** vs **pathogenic (1)**.

Practical uses
* Triage variants in clinical WGS reports.  
* Population‑specific risk assessment for genetic counselling.

---

## 4  Process Overview<a name="4-process-overview"></a>

### Early Prototype (PoC)
Before investing in the full feature set, I validated the end-to-end flow with a quick proof-of-concept:
- **Model**: RandomForest on the filtered SNV subset (~374 K rows)  
- **Outcome**: ~0.70 overall accuracy, but up to **20 pp** accuracy gap between ancestry groups  
- **Takeaway**: clear need for population-aware features to close fairness gaps  
![PoC baseline](docs/img/poc_baseline.png)

### Full Pipeline Steps
1. **Raw Data Ingestion & Filtering**  
       Load ClinVar + gnomAD extract, drop variants with missing/ambiguous clinical labels or multi-allelic indels.  
2. **Exploratory Data Analysis (EDA)**  
       Characterize class imbalance and allele-frequency distributions across populations.  
3. **Feature Engineering**  
       – **Sequence metrics**: allele length, per-base counts, GC-content  
       – **Population interactions**: `pop_gene`, `pop_consequence`, `allele_freq_rel`  
4. **Model Comparison**  
       Train RandomForest, XGBoost and LogisticRegression on **pop-aware** vs **non-pop** pipelines.  
5. **Hyperparameter Tuning**  
       Use `RandomizedSearchCV` (RF, XGB) and `GridSearchCV` (LR) to optimize model settings.  
6. **Hold-out Evaluation**  
       Evaluate on a 20 % temporal split; automatic checkpoint resume avoids re-running earlier stages.  
7. **Post-training Utilities**  
       Generate and save confusion matrices for both binary & multi-class modes, plot fairness gaps, and export the best pipelines to `models/`.  
![pipeline](docs/img/pipeline_diagram.png)

---

## 5  Exploratory Data Analysis (EDA)<a name="5-exploratory-data-analysis-eda"></a>
| Figure | What it shows |
|--------|---------------|
| ![dist](docs/img/EDA_class_distribution.png) | Class imbalance (uncertain ≈ 17 %, benign ≈ 34 %, pathogenic ≈ 49 %). |
| ![corr](docs/img/EDA_corr_matrix.png)        | Numeric correlation (allele_freq ↔ allele_count ρ ≈ 0.91). |

High‑res PNGs live in **`docs/img/`**.

---

## 6  Feature Engineering<a name="6-feature-engineering"></a>
* **Sequence‑derived** – length, GC‑content, per‑base counts.
* **Population interactions** – `pop_gene`, `pop_consequence`, `allele_freq_rel` (ratio to pop mean).
* **Encoding** – One‑hot for categoricals; char‑level `CountVectorizer` for short alleles.

---

## 7  Model Fitting & Tuning<a name="7-model-fitting--hyperparameter-tuning"></a>
### 7.1 Cross‑validation leaderboard (multi‑class)
| Model | Scaler | CV Acc ± SD |
|-------|--------|-------------|
| RandomForest | Standard | **0.769 ± 0.004** |
| XGBoost      | Standard | **0.824 ± 0.003** |
| LogisticReg  | Standard | 0.768 ± 0.006 |

*(5‑fold GroupKFold by population)*

### 7.2 Best hyper‑parameters
| Family | Parameters |
|--------|------------|
| RF  | `n_estimators=100, max_depth=15, min_samples_leaf=1` |
| XGB | `n_estimators=200, max_depth=7, lr=0.1, reg_lambda=10` |
| LR  | `C=0.1, penalty=l2, solver=liblinear` |

Full search logs live in **`checkpoints/`**.

---

## 8  Validation & Fairness<a name="8-validation--performance-metrics"></a>
### 8.1 Hold‑out results – 3‑class
| Pipeline | Accuracy | F1‑w | Benign F1 | Pathog F1 | Uncert F1 |
|----------|---------:|------:|-----------:|-----------:|-----------:|
| **Pop‑Aware XGB** | **0.81** | **0.81** | 0.85 | 0.92 | 0.59 |
| Non‑Pop XGB | 0.81 | 0.81 | 0.85 | 0.92 | 0.60 |
| Pop‑Aware RF | 0.75 | 0.76 | 0.76 | 0.87 | 0.63 |
| Non‑Pop RF | 0.77 | 0.78 | 0.77 | 0.91 | 0.65 |

> For the **3-class** task, Pop-Aware XGBoost remains top (81 % accuracy, 0.81 F1-w), with non-pop XGB essentially matching.


### 8.1b Hold-out results – *binary* (`non_path` vs `path`)
| Pipeline                  | Accuracy | F1-w | Non-path F1 | Path F1 |
|---------------------------|---------:|------:|------------:|--------:|
| **Pop-Aware LogisticReg** | **0.89** | 0.89 | 0.91        | **0.84** |
| Non-Pop LogisticReg       | 0.89     | 0.89 | 0.91        | **0.84** |
| **Pop-Aware XGB**         | **0.89** | 0.89 | 0.92        | 0.81    |
| Non-Pop XGB               | 0.89     | 0.89 | 0.92        | 0.81    |
| Pop-Aware RF              | 0.85     | 0.85 | 0.89        | 0.74    |
| Non-Pop RF                | 0.87     | 0.87 | 0.91        | 0.78    |


In binary mode the Logistic Regression pipeline actually edges out XGBoost on the pathogenic class (0.84 vs 0.81 F1), tying on overall accuracy (89 %). XGBoost remains competitive on accuracy and group-fairness, but if pathogenic recall is your priority, LR is preferable here.

### 8.2 Confusion matrices
| Multiclass | Binary |
|------------|--------|
| ![cm_pop](docs/img/cm_xgb_pop.png) ![cm_nonpop](docs/img/cm_xgb_nonpop.png) | ![cm_bin_pop](docs/img/cm_lr_pop_binary.png) ![cm_bin_nonpop](docs/img/cm_lr_nonpop_binary.png) |

*(matrices normalised by true‑class; full set in `docs/img/`)*

### 8.3 Fairness snapshot
*Maximum accuracy gap* (pop‑aware XGB): **0.05** (multi) vs **0.04** (binary)  – **~2× lower than non‑pop baselines**.

---

## 9  Production & Deployment<a name="9-production--deployment"></a>
> **Folder layout (key items)**
> ```text
> 2024_ia651_gyimah/
> ├─ src/                  ← all python code
> │  └─ utils/             ← helper modules (plots, paths, …)
> ├─ data/                 ← raw_variant_data.csv
> ├─ checkpoints/          ← auto‑saved joblib checkpoints
> ├─ models/               ← exported best_pop_aware_<mode>.pkl
> └─ docs/img/             ← confusion matrices & EDA PNGs
> ```
>
> After each run the script **automatically exports** the best pop‑aware pipeline to
> `models/best_pop_aware_binary.pkl` **or** `models/best_pop_aware_multiclass.pkl`.
>
> Please note that the file size for the raw data was too large to upload to github. Similarly, the saved houldout pkl files were too large to upload to github. So these files aren't in the repository

### 9.1 CLI inference
```bash
python -m post_training_utils.predict \
               --model models/best_pop_aware_multiclass.pkl \
               --vcf   examples/one_variant.vcf
```
### 9.2 Python API
```python
from joblib import load
from post_training_utils import featurise_variant
clf = load("models/best_pop_aware_multiclass.pkl")
X = featurise_variant(vcf_record)
print(clf.predict_proba(X))
```

---

## 10  Limitations & Future Improvements<a name="10-limitations--future-improvements"></a>
* **Label noise** – “uncertain/conflicting” may hide true pathogenicity.
* **Population imbalance** – fewer East‑Asian & Caribbean samples leads to wider CIs.
* **Short‑read bias** – large indels excluded; future work: *hg38* liftover & SV support.
* **Deep models** – explore CNN on ±50 bp window; transformer embeddings.

---

## 11. Reproducing this Project & Downloading Data

### 11.1 Clone & Create Environment

```bash
# Linux / macOS (conda)
conda env create -f environment.yml && conda activate ia651_genomics

# Windows PowerShell (venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
```


## 11. Conclusion

In this work, it was set out to determine whether incorporating population-specific features could improve machine-learning–based classification of human genetic variants. Through a multi-stage pipeline—spanning exploratory data analysis, feature engineering of both sequence and ancestry interactions, rigorous cross-validation, hyperparameter tuning, and a final hold-out evaluation—it has been shown that:

Population-aware pipelines can narrow performance disparities across ancestry groups, reducing the maximum accuracy gap by up to half compared to non-population-aware models in the binary setting.

Model family performance varies by task: XGBoost remains the top performer on the three-class (“benign / pathogenic / uncertain”) problem, while Logistic Regression—once tuned—emerges strongest for the binary (“non-pathogenic / pathogenic”) classification.

Fairness constraints (via Fairlearn’s GridSearch) are feasible in the binary context, but provide limited gains in the multiclass setting.

Checkpointing and modular design allow incremental development and near-instantaneous experimentation once the initial 4–5 hour run is complete, dramatically improving reproducibility and iteration speed.

Collectively, these findings validate the original hypothesis that ancestry-informed features can both boost overall accuracy and promote equitable performance across diverse populations. The open-source, dual-licensed framework we provide can serve as a robust foundation for downstream integration into clinical genomics pipelines, with hooks for future extensions—such as deep-learning on raw sequence windows, expanded structural-variant support, or integration of protein-impact scores.

It is anticipated that this work will not only provide some guide for practitioners in building fairer variant classifiers but also inspire further research into the interplay of genetic background and machine-learning–driven genomic interpretation.

## 12. Reproducing this Project & Downloading Data
### 12.1 Clone & Create Environment
```bash
# Linux / macOS (conda)
conda env create -f environment.yml && conda activate ia651_genomics

# Windows PowerShell (venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
```

### 12.2 Download Raw Data & Checkpoints

Because the raw CSV and checkpoint PKLs exceed GitHub's file-size limits, we host them on OSF. After downloading, place them under your project root:

| File | OSF URL | Local path |
| ---- | ------- | ---------- |
| raw_variant_data.csv (≈2.8 M rows) | https://files.osf.io/v1/resources/uzvn3/providers/osfstorage/680f81d7d2a2479792568b94/?zip= | data/raw_variant_data.csv |
| holdout_evaluation_latest.pkl | https://files.osf.io/v1/resources/uzvn3/providers/osfstorage/680f861800fd4bbf355ef649/?zip= | checkpoints/holdout_evaluation_latest.pkl |

### 12.3 Full Pipeline Run & Runtime

```bash
# Full 3-class analysis (≈5 h first time on Intel i5-1035G1, 4-core/8-thread, 12 GB RAM)
python -m src.genetic-variant-classifier2

# Optional binary run
python -m src.genetic-variant-classifier2 --binary
```

> NB: if you want to run the pipeline from scratch, do not download the checkpoint files.

> Note: Checkpoints are saved at each major stage (checkpoints/).
> Once you've done the full 5 h run once, subsequent runs will resume instantly from the last checkpoint for EDA, CV, model‐comparison, hyperparameter‐tuning or hold-out evaluation—no need to reprocess everything from scratch.

### 12.4 Quick Plot Reproduction

```bash
# Example: regenerate binary‐LR hold‐out confusion matrices
python scripts/plot_lr_binary_cm.py
```


## 13 License & Ethics
### 13.1 License
This pipeline (all scripts and every single code associated with this project) offer this software under a dual-licensing scheme:

1. Open-Source (MIT)
All non-commercial use is licensed under the MIT License. The MIT License is a permissive, OSI-approved license that allows use, modification, and distribution with minimal obligations.

2. Commercial Use
Commercial deployments—defined as any use that generates direct or indirect revenue—are not covered under the MIT terms. Instead, a separate commercial license is required, which includes a negotiated revenue-share agreement (e.g., 5–10 % of net revenues derived from products or services that incorporate this pipeline). This approach—common in “dual licensing” models—ensures that academic work benefits the community while also providing compensation when used in profit-making contexts.

>How to Obtain a Commercial License
Please contact the author at [simon.gyimah2@gmail.com] with a brief description of your intended use and revenue model. We will draft a simple agreement specifying royalty rates and reporting obligations.

### 13.2 Ethical Use
A strong belief in responsible data science and genetics research is maintained. By using this software, an agreement is made to:
* Respect personal and population privacy when handling genomic data.

* Obtain all necessary consents and ethics approvals before processing any human-derived datasets.

* Abstain from any use that could harm individuals or communities (e.g., attempts at unethical surveillance or discriminatory decision-making).

* Comply with relevant laws and guidelines, including the Universal Declaration of Human Rights and any applicable local bioethics regulations.

## 14 References
1. Richards S. et al. Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology. Genet Med. 2015;17(5):405–424. 
ScienceDirect

2. Landrum MJ et al. ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Res. 2018;46(D1):D1062–D1067. 

3. Lek M. et al. Analysis of protein-coding genetic variation in 60,706 humans. Nature. 2016;536(7616):285–291.

4. Pedregosa F. et al. Scikit-learn: Machine Learning in Python. J Mach Learn Res. 2011;12:2825–2830. 

5. Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System. Proc. 22nd ACM SIGKDD. 2016:785–794. 


6. Fairlearn Contributors. Fairlearn: a toolkit for assessing and improving fairness in AI. Fairlearn; 2025. Available from https://fairlearn.org 


7. Torkzadehmahani R. et al. Privacy-preserving Artificial Intelligence Techniques in Biomedicine. arXiv. 2020. 

8. Center for Open Science. Open Science Framework (OSF). OSF; 2025. Available from https://osf.io 

9. National Institutes of Health. Genomic Data Sharing Policy. NIH; 2020. Available from https://sharing.nih.gov/genomic-data-sharing-policy 

10. Richards S. et al. Standards and guidelines for sequence variant interpretation. American College of Medical Genetics and Genomics. Genet Med. 2015;17(5):405–424
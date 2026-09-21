<img width="1400" height="700" alt="image" src="https://github.com/user-attachments/assets/bdeb0f9a-d634-41c0-97b8-6a21ae45e277" />

# Legendary Status in Pokémon — A Data-Driven Classification and Clustering Analysis

![KNIME](https://img.shields.io/badge/KNIME-FDD800.svg?style=for-the-badge&logo=knime&logoColor=black) ![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-189FDD.svg?style=for-the-badge&logoColor=white) ![WEKA](https://img.shields.io/badge/WEKA-5A2D82.svg?style=for-the-badge&logoColor=white) ![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg?style=for-the-badge)

> Legendary Pokémon are an extreme minority (46 out of 721, 6.38%). This project investigates whether legendary status can be predicted strictly from combat and morphological traits, and whether Pokémon naturally cluster around this label or around other intrinsic properties. The full pipeline is built in **KNIME Analytics Platform**, with Python nodes for visualization.

---

## 🎯 Research Questions

*   **RQ1 — Supervised:** Can a Pokémon's legendary status be accurately predicted using only its base combat stats and physical traits, deliberately excluding descriptive gameplay mechanics to prevent target leakage?
*   **RQ2 — Unsupervised:** Do Pokémon naturally cluster according to their legendary classification, or do they group around other intrinsic characteristics?

---

## 📊 Dataset

**Pokémon for Data Mining and Machine Learning** (Kaggle, alopez247, 2017): 721 Pokémon described by 23 features (base stats, types, generation, physical traits, catch rate, gender, egg groups, etc.). Source: [https://www.kaggle.com/datasets/alopez247/pokemon](https://www.kaggle.com/datasets/alopez247/pokemon)

The dataset is **not included** in this repository (see [How to Run](#%EF%B8%8F-how-to-run)).

---

## 🧹 Feature Engineering

| Step | Description |
|---|---|
| **Missing Values** | `Type 2` and `Egg Group 2` imputed with `"NULL"`; `Pr Male` encoded as `-1` for genderless Pokémon. |
| **Binarization** | `isLegendary`, `hasGender` and `hasMegaEvolution` converted into 0/1 indicators via Rule Engine. |
| **Feature Creation** | Height and weight aggregated into a single **Body Mass Index** (`BMI = Weight_kg / Height_m²`), dropping the raw dimensions. |
| **Leakage Removal** | `Catch Rate`, `hasGender`, `Pr Male` and `hasMegaEvolution` excluded, since they encode hardcoded game-design choices tied to the target. |
| **Redundancy Removal** | `Total` dropped after the Pearson correlation analysis, being the exact sum of the base stats. |

**Final predictors:** `HP`, `Attack`, `Defense`, `Sp Atk`, `Sp Def`, `Speed`, `BMI`.

---

## 🤖 RQ1 — Supervised Learning

**Imbalance strategy:** a hybrid approach combining **SMOTE** (applied strictly to training folds) with **cost-sensitive learning** for Naive Bayes (FN penalty = 10, FP penalty = 1), which bypasses SMOTE to preserve its conditional independence assumption.

**Models evaluated:** Logistic Regression, Decision Tree, Random Forest, Tree Ensemble, XGBoost, K-Nearest Neighbors, Naive Bayes, Multi-Layer Perceptron (Rprop).

**Validation protocol:**
1.  Hyperparameter tuning via **Brute Force search** on a stratified 70/30 split, maximizing the minority-class F1-Score.
2.  Evaluation through **stratified 5-fold Cross-Validation**.
3.  Final **Holdout** test on unseen data.

**Metrics:** F1-Score (primary objective), Recall, Specificity, Precision, Average Precision (PR curve), Cohen's κ. Accuracy and ROC-AUC are reported as baselines only, as they proved misleading under severe imbalance.

### Holdout Results (top models)

| Model | Recall | Precision | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.929 | 0.812 | **0.867** | 0.888 | 0.981 |
| **Multi-Layer Perceptron (Rprop)** | 0.786 | 0.917 | 0.846 | **0.908** | **0.994** |
| **Logistic Regression** | 0.786 | 0.611 | 0.688 | 0.815 | 0.985 |
| Naive Bayes | 1.000 | 0.438 | 0.609 | 0.850 | 0.990 |
| Random Forest | 0.286 | 1.000 | 0.444 | 0.773 | 0.969 |

*   **MLP, XGBoost and Logistic Regression** are the only models sustaining high Recall without a collapse in Precision.
*   **Random Forest and Tree Ensemble** collapse toward majority-class predictions under heavy regularization, while still showing ROC-AUC > 0.95.
*   **Naive Bayes** shows the inverse failure mode: near-perfect Recall at the cost of Precision.

---

## 🔍 RQ2 — Unsupervised Learning

*   **k-Means** applied to the 7 selected features, with `isLegendary` excluded from training.
*   Optimal number of clusters chosen via **Silhouette Coefficient** for k ∈ [2, 11]: **k = 3** (0.275) outperforms the intuitive binary split k = 2 (0.263).
*   **PCA** used purely as a 2D visualization aid (61.0% of explained variance); **ANOVA F-statistic** used to rank the features driving the clusters.

**Findings:** clusters are driven by overall combat power and by BMI, not by legendary status. Legendary Pokémon are absent from the low-stats cluster and share their statistical space with top-tier standard Pokémon: without an explicit label, a Legendary is statistically indistinguishable from a comparably strong standard counterpart.

---

## 📁 Repository Structure

```text
.
├── README.md
├── data/
│   └── pokemon_alopez247.csv      # data downloaded from Kaggle
├── workflow/
│   └── team61_Perillo.knwf        # Full KNIME workflow (preprocessing, EDA, RQ1, RQ2)
└── report/
    └── team61_Perillo.pdf         # Project final report
```

The workflow is organized into three annotated areas:
1.  **Data Preprocessing and Exploratory Data Analysis**
2.  **RQ1 — Supervised Learning** (tuning, 5-fold CV, Holdout, ROC and PR curves)
3.  **RQ2 — Unsupervised Learning** (Silhouette search, k-Means, cluster profiling)

---

## ⚙️ How to Run

1.  Install **KNIME Analytics Platform** (the workflow was built with version 5.12) with the following extensions:
    *   KNIME Python Integration (Python Script / Python View nodes)
    *   KNIME XGBoost Integration
    *   KNIME Weka Integration (3.7)
    *   KNIME Optimization Extension
    *   KNIME Distance Matrix Extension
2.  Configure a Python environment with `pandas`, `matplotlib` and `seaborn` for the Python View nodes.
3.  Download `pokemon_alopez247.csv` from Kaggle [https://www.kaggle.com/datasets/alopez247/pokemon].
4.  Import `workflow/team61_Perillo.knwf` via *File → Import KNIME Workflow*.
5.  Update the file path in the **CSV Reader** node to point to your local copy of the dataset, then execute the workflow.

---

## 🛠️ Tech Stack

`KNIME Analytics Platform` · `Python` · `SMOTE` · `XGBoost` · `WEKA` · `Pandas` · `Matplotlib` / `Seaborn`

---

## 👤 Author

**Emily Perillo**  
M.Sc. Data Science, Università degli Studi di Milano-Bicocca  
B.Sc. Business Administration

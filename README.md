# Clinical Prediction Model for Recurrent Rectal Cancer

Python code for my MSc Independent Study. It compares conventional regression with machine-learning models for predicting recurrence after curative-intent treatment of rectal adenocarcinoma.

> **Clinical Prediction Model for Recurrent Rectal Cancer**
> Thanat Tantinam. MSc Independent Study, Master of Science in Digital Business Transformation – Data Science, College of Innovation, Thammasat University, Academic Year 2024.

Published as:
> Tantinam T, et al. From regression to machine learning: improving prediction of rectal cancer recurrence. *Ann Coloproctol.* 2026;42(3):293–302. doi:[10.3393/ac.2025.01067.0152](https://doi.org/10.3393/ac.2025.01067.0152)

---

## Repository structure

```
├── MS Data Science.py            # Full pipeline: EDA, preprocessing, model training, tuning,
│                                 # calibration, robust CV, bootstrap CIs, thresholds, DCA
├── requirements.txt              # Python dependencies
├── basic_characteristics.csv     # Aggregate cohort summary (no individual-level data)
├── table1_characteristics.csv    # Aggregate Table 1 (no individual-level data)
├── *.png                         # EDA figures (missingness, correlation, histograms, count plots, boxplots)
├── model_selection_guide*.png    # Model-selection summary figure (colour / black-and-white)
├── model_outputs/
│   ├── 20250808-220853_*         # Primary run: ROC/PR, calibration, confusion matrices,
│   │                             # feature importance (impurity, permutation, SHAP)
│   └── 20251010-190034_*         # Final run: repeated CV (5×10), threshold tables,
│                                 # decision curves, supplementary tables
├── data/                         # NOT included (confidential patient data); see below
├── LICENSE
└── README.md
```

## Data availability
The dataset contains identifiable patient information. It was used under ethics approval and is **not publicly available**. Only aggregate summaries are in this repository.

To run the code on your own data:
1. Prepare a CSV with the same variable names and formats as **Table 3** of the thesis (for example `age`, `sex`, `cea_level`, `no_ln`, `staging`, `tumor_loc`, `tumor_grade`, `neoadj`, `tnt`, `margin_binary`, `lvi`, `recur`).
2. Save it as `data/RC.csv` in the repository root. The `data/` folder is git-ignored.

## Requirements
Developed and tested with **Python 3.10** in PyCharm.

```bash
pip install -r requirements.txt
```
Packages: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap. XGBoost, LightGBM and SHAP are optional; the script skips them if they are missing.

## How to run
```bash
git clone https://github.com/tthanat91/Data_Science_MSc_project.git
cd Data_Science_MSc_project
pip install -r requirements.txt
# put your dataset at data/RC.csv
python "MS Data Science.py"
```
EDA figures are written to the repository root. Model outputs go to `model_outputs/`, each file prefixed with a run timestamp (`YYYYMMDD-HHMMSS_`).

## Methods
- **Models:** Logistic Regression, Random Forest, SVM (RBF kernel), XGBoost, LightGBM.
- **Preprocessing:** median/mode imputation, one-hot encoding, scaling, with SMOTE-Tomek class balancing inside the pipeline. A sensitivity analysis is also run without SMOTE.
- **Tuning:** 5-fold stratified cross-validation with grid search, then probability calibration.
- **Validation:** repeated stratified CV (5×10), and bootstrap 95% CIs (B = 1000) for AUC and average precision on the held-out test set.
- **Calibration:** Brier score, calibration slope and intercept, calibration plots.
- **Clinical utility:** sensitivity, specificity, PPV and NPV at 10%, 20% and 30% risk thresholds, plus decision curve analysis.
- **Explainability:** impurity, permutation and SHAP feature importance.

## Citation
If you use this code, please cite the paper:

```
Tantinam T, et al. From regression to machine learning: improving prediction of rectal
cancer recurrence. Ann Coloproctol. 2026;42(3):293-302. doi:10.3393/ac.2025.01067.0152
```

## License
Code released under the MIT License (see `LICENSE`). The license does not cover any patient data.

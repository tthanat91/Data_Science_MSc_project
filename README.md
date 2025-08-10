# Clinical Prediction Model for Recurrent Rectal Cancer

This repository contains the full Python code used for the MSc Independent Study thesis:

> **Clinical Prediction Model for Recurrent Rectal Cancer**  
> Thanat Tantinam  
> Master of Science in Digital Business Transformation – Data Science  
> College of Innovation, Thammasat University  
> Academic Year 2024

The project compares conventional regression-based models and various machine learning algorithms for predicting recurrence after curative-intent treatment of rectal adenocarcinoma.

---

## 📄 Thesis Context
This code is referenced in **Appendix A** of the thesis and supports the methodology and results described in:
- **Chapter 3:** Research Methodology  
- **Chapter 4:** Results and Discussion

Figures, tables, and evaluation metrics in the thesis were generated using these scripts.

---

## 📂 Repository Structure
├── data/ # (Not included — patient data is confidential)
├── eda_pipeline.py # Data cleaning, missing data handling, EDA plots
├── model_training.py # Model training, tuning, calibration, evaluation
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── model_outputs/ # Example output figures & CSV summaries

---

## ⚠️ Data Availability
The dataset contains identifiable patient information and is **not publicly available**.  
Researchers wishing to replicate the analysis should:
1. Prepare a dataset with the same variable names and formats described in **Table 3** of the thesis.
2. Save it as `data/RC.csv`.
3. Adjust the file path in `eda_pipeline.py` and `model_training.py` if needed.

---

## 🛠️ Requirements
The scripts were developed and tested in **Python 3.10** using **PyCharm 2024.1.4**.  
Required packages (see `requirements.txt`):

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- shap

---

## ▶️ How to Run
Clone the repository
git clone https://github.com/yourusername/rectal-cancer-prediction.git
cd rectal-cancer-prediction
Place your dataset in the data/ folder as RC.csv.
Run EDA (exploratory data analysis):
python eda_pipeline.py
Train and evaluate models:
python model_training.py
All output figures, tables, and CSV summaries will be saved in the model_outputs/ folder.

---

## 📊 Models Implemented
Logistic Regression
Random Forest
Support Vector Machine (RBF kernel)
XGBoost
LightGBM
Each model is trained with 5-fold stratified cross-validation, tuned via grid search, and probability-calibrated.

---

## 📈 Outputs
The scripts generate:
Missing data plots
Correlation matrix
Histograms and boxplots
ROC and PR curves
Confusion matrices (default and best F1 threshold)
Calibration curves
Feature importance plots (impurity, permutation, SHAP)
Decision curve analysis
CSV summaries of performance metrics and top features

---

## 📜 Citation
If using this code, please cite:
Tantinam T. Clinical Prediction Model for Recurrent Rectal Cancer. MSc Thesis, College of Innovation, Thammasat University; 2024.

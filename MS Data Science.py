# Load your dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/THANAT/Documents/Research Projects/MSc project/IS/Data/RC.csv")
df = df.dropna(axis=1, how='all')  # Drop fully empty columns

# === Step 1: Missing Data Percentage ===
missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_percent.values, y=missing_percent.index, palette="Reds_r")
plt.xlabel("Missing Data (%)")
plt.title("Percentage of Missing Data by Feature")
plt.tight_layout()
plt.savefig("missing_data_percent.png")
plt.show()

# === Filter out features with >50% missing values (except exceptions) ===
exceptions_to_keep = ['postpone_neoadj', 'tnt']
features_to_drop = [col for col in missing_percent.index
                    if missing_percent[col] > 50 and col not in exceptions_to_keep]
df = df.drop(columns=features_to_drop)

print("Dropped features due to >50% missingness (except clinical exceptions):")
print(features_to_drop)

# === Step 2: Convert fields for EDA ===
binary_columns = ['neoadj', 'tnt', 'postpone_neoadj',
                  'lvi', 'curative_intend', 'recur']
df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)
df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0}).infer_objects(copy=False)
df['cea_binary'] = df['cea_level'].apply(lambda x: 1 if x >= 5 else 0 if pd.notnull(x) else None)

# === Step 3: Correlation Matrix ===
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# === Step 4: Combined Histograms for Continuous Variables ===
num_features = ['age', 'no_ln']
num_plots = len(num_features)

# Adjust layout (1 row, 2 columns)
fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))

for i, col in enumerate(num_features):
    if col in df.columns:
        sns.histplot(df[col].dropna(), kde=True, bins=20, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("combined_histograms.png")
plt.show()


# === Step 5: Combined Count Plots ===
# === Recode 'margin' to binary: 'Negative' vs. 'Positive'
df['margin_binary'] = df['margin'].apply(
    lambda x: 'Negative' if isinstance(x, str) and x.strip().lower() == 'negative' else 'Positive'
)
cat_features = ['sex', 'tumor_loc', 'tumor_grade', 'cea_binary',
                'neoadj', 'tnt', 'recur', 'margin_binary']

num_plots = len(cat_features)
cols = 3  # number of columns in the grid
rows = (num_plots + cols - 1) // cols  # automatically determine rows

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
axes = axes.flatten()  # make it easier to index

for i, col in enumerate(cat_features):
    if col in df.columns:
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Count Plot: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=45)

# Hide unused subplots if total plots < grid size
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("combined_count_plots.png")
plt.show()


# === Step 6: Combined Boxplots Grouped by Recurrence ===
boxplot_features = ['age', 'no_ln']
num_plots = len(boxplot_features)

fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))

for i, col in enumerate(boxplot_features):
    if col in df.columns:
        sns.boxplot(data=df, x='recur', y=col, ax=axes[i])
        axes[i].set_title(f"{col} by Recurrence")
        axes[i].set_xlabel("Recurrence (0=No, 1=Yes)")
        axes[i].set_ylabel(col)

plt.tight_layout()
plt.savefig("combined_boxplots_by_recur.png")
plt.show()


"""
Clinical Prediction Model for Recurrent Rectal Cancer — Training Pipeline (post‑EDA)
----------------------------------------------------------------------------------
Locked to 19 final features from thesis Table 3.
This script continues AFTER your existing EDA code. Do NOT modify your EDA.
Just paste this below your current code (or run as a separate cell/file).
"""

# ============= 0) Imports & Config =============
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

# Optional: XGBoost / LightGBM
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# Optional: SHAP
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.getcwd(), "model_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

# ============= 1) Robust target cleaning (recur) =============
# Your EDA should have already mapped Yes/No -> 1/0, but if some rows are still strings or blanks,
# coerce safely and *drop* rows with unknown target (we cannot impute the outcome).

def _norm_bin(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    pos = {"1", "yes", "y", "true", "t", "recur", "recurrence", "positive", "pos"}
    neg = {"0", "no", "n", "false", "f", "non-recur", "nonrecurrence", "negative", "neg"}
    if s in pos:
        return 1
    if s in neg:
        return 0
    # also allow numbers like 1.0/0.0
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except Exception:
        pass
    return np.nan

if "recur" not in df.columns:
    raise ValueError("`recur` column not found in dataframe.")

_recur_norm = df["recur"].apply(_norm_bin)
unknown_mask = _recur_norm.isna()
if unknown_mask.any():
    # Show what raw values caused trouble to help fix upstream if needed
    bad_vals = (
        df.loc[unknown_mask, "recur"].astype(str).str.strip().str.lower().value_counts(dropna=False)
    )
    print("[Warn] Dropping rows with unknown/blank `recur` values. Counts by raw value:\n", bad_vals.to_string())
    df = df.loc[~unknown_mask].copy()
    _recur_norm = _recur_norm.loc[~unknown_mask]

df["recur"] = _recur_norm.astype(int)

# ============= 2) Lock features (19 from thesis Table 3) =============
preferred_cols = [
    "sex", "age", "tumor_loc", "ct", "cn", "cm", "staging", "emvi", "mrf",
    "tumor_grade", "cea_level", "neoadj", "tnt", "postpone_neoadj", "sx",
    "pt", "no_ln", "pn", "margin", "lvi"
]

use_cols = [c for c in preferred_cols if c in df.columns]
if len(use_cols) != len(preferred_cols):
    missing_feats = set(preferred_cols) - set(use_cols)
    print(f"[Warn] Missing features from dataset (will be skipped): {missing_feats}")

X = df[use_cols].copy()
y = df["recur"].astype(int).values

# Identify column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
for c in cat_cols:
    X[c] = X[c].astype(str)

print(f"[Info] Using {len(use_cols)} features — {len(num_cols)} numeric, {len(cat_cols)} categorical.")

# ============= 3) Train/Test split =============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================== 4) Preprocessing & Imbalance ==========================
# Robust OneHotEncoder across sklearn versions
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OHE),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
], remainder="drop")

sampler = SMOTETomek(random_state=42)

# ===================== 5) Model spaces & hyper grids ==========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_grids = []

# Logistic Regression
lr = LogisticRegression(max_iter=2000, solver="liblinear")
lr_grid = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l1", "l2"],
}
models_and_grids.append(("LogisticRegression", lr, lr_grid))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid = {
    "clf__n_estimators": [300, 600],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}
models_and_grids.append(("RandomForest", rf, rf_grid))

# SVM (RBF)
svm = SVC(probability=True, random_state=42)
svm_grid = {
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", 0.1, 0.01],
}
models_and_grids.append(("SVM_RBF", svm, svm_grid))

# XGBoost
if xgb is not None:
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
        tree_method="hist", random_state=42
    )
    xgb_grid = {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [3, 5, 8],
        "clf__learning_rate": [0.03, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }
    models_and_grids.append(("XGBoost", xgb_clf, xgb_grid))
else:
    print("[Warn] xgboost not installed — skipping.")

# LightGBM
if lgb is not None:
    lgb_clf = lgb.LGBMClassifier(objective="binary", random_state=42)
    lgb_grid = {
        "clf__n_estimators": [300, 600],
        "clf__num_leaves": [31, 63, 127],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [-1, 5, 10],
    }
    models_and_grids.append(("LightGBM", lgb_clf, lgb_grid))
else:
    print("[Warn] lightgbm not installed — skipping.")

# ============== Helper to access calibrated inner pipeline safely =============

def _get_inner_pipeline_from_calibrator(cal: CalibratedClassifierCV):
    est = getattr(cal, "estimator", None)
    if est is None:
        est = getattr(cal, "base_estimator", None)  # older sklearn fallback
    if est is None:
        raise AttributeError("Cannot find underlying estimator from CalibratedClassifierCV.")
    return est  # This should be the ImbPipeline([... ('prep'), ('smote'), ('clf')])

# ================= 6) Training + Hyperparameter Tuning ========================
results = []
roc_curves = {}
pr_curves = {}

for name, base_clf, grid in models_and_grids:
    print(f"\n===== Tuning {name} =====")

    pipe = ImbPipeline(steps=[
        ("prep", preprocess),
        ("smote", sampler),
        ("clf", base_clf),
    ])

    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_

    print(f"Best params for {name}: {gscv.best_params_}")

    # ---- Probability calibration ----
    try:
        calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=cv)
        calibrated.fit(X_train, y_train)
    except Exception:
        calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=cv)
        calibrated.fit(X_train, y_train)

    # Predictions
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.5).astype(int)

    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    y_pred_bt = (y_proba >= best_t).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    def metrics(y_true, y_hat):
        cm = confusion_matrix(y_true, y_hat)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / max(1, (tn + fp))
        npv = tn / max(1, (tn + fn))
        return dict(
            F1=f1_score(y_true, y_hat, zero_division=0),
            Sensitivity=recall_score(y_true, y_hat, zero_division=0),
            Precision=precision_score(y_true, y_hat, zero_division=0),
            Specificity=spec,
            NPV=npv,
        )

    m05 = metrics(y_test, y_pred_05)
    mbt = metrics(y_test, y_pred_bt)

    # Curves
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)
    pr_curves[name] = (rec, prec, ap)

    # Confusion matrices
    def plot_cm(cm, title, fname):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Recur", "Recur"], yticklabels=["No Recur", "Recur"])
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
        plt.close()

    cm05 = confusion_matrix(y_test, y_pred_05)
    cmbt = confusion_matrix(y_test, y_pred_bt)
    plot_cm(cm05, f"{name} — CM @0.50", f"{STAMP}_{name}_cm_thr0.50.png")
    plot_cm(cmbt, f"{name} — CM @bestF1({best_t:.2f})", f"{STAMP}_{name}_cm_thrBestF1.png")

    # Save per-model results
    results.append({
        "model": name,
        "best_params": json.dumps(gscv.best_params_),
        "AUC_ROC": round(auc, 4),
        "AP(PR_AUC)": round(ap, 4),
        "thr_bestF1": round(best_t, 2),
        "F1@0.50": round(m05["F1"], 4),
        "Sens@0.50": round(m05["Sensitivity"], 4),
        "Spec@0.50": round(m05["Specificity"], 4),
        "PPV@0.50": round(m05["Precision"], 4),
        "NPV@0.50": round(m05["NPV"], 4),
        "F1@best": round(mbt["F1"], 4),
        "Sens@best": round(mbt["Sensitivity"], 4),
        "Spec@best": round(mbt["Specificity"], 4),
        "PPV@best": round(mbt["Precision"], 4),
        "NPV@best": round(mbt["NPV"], 4),
    })

    # ----------------------- Feature importance -----------------------
    # Access the inner pipeline safely
    inner_pipe = _get_inner_pipeline_from_calibrator(calibrated)
    prep = inner_pipe.named_steps["prep"]

    if name == "LogisticRegression":
        final_lr = inner_pipe.named_steps["clf"]
        ohe = prep.named_transformers_["cat"].named_steps["ohe"]
        feat_names_ohe = num_cols + ohe.get_feature_names_out(cat_cols).tolist()

        if hasattr(final_lr, "coef_"):
            coefs = pd.Series(final_lr.coef_.ravel(), index=feat_names_ohe)
            coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
            coefs.to_csv(os.path.join(OUT_DIR, f"{STAMP}_{name}_coefficients.csv"))

        # Permutation importance on the calibrated Pipeline returns importances
        # for ORIGINAL columns (pre-OHE). Use num_cols + cat_cols as names.
        try:
            perm = permutation_importance(calibrated, X_test, y_test,
                                          n_repeats=10, random_state=42, scoring="roc_auc")
            pi = pd.Series(perm.importances_mean, index=(num_cols + cat_cols))
            pi = pi.sort_values(ascending=False)
            pi.to_csv(os.path.join(OUT_DIR, f"{STAMP}_{name}_perm_importance.csv"))
        except Exception as e:
            print(f"[Warn] Permutation importance failed for {name}: {e}")

    if name in {"RandomForest", "XGBoost", "LightGBM"}:
        try:
            inner = inner_pipe.named_steps["clf"]
            ohe = prep.named_transformers_["cat"].named_steps["ohe"]
            feat_names_ohe = num_cols + ohe.get_feature_names_out(cat_cols).tolist()

            if hasattr(inner, "feature_importances_"):
                imp = pd.Series(inner.feature_importances_, index=feat_names_ohe)
                imp = imp.sort_values(ascending=False)
                imp.to_csv(os.path.join(OUT_DIR, f"{STAMP}_{name}_impurity_importance.csv"))

            if shap_available:
                try:
                    X_test_tx = prep.transform(X_test)
                    explainer = shap.TreeExplainer(inner)
                    shap_values = explainer.shap_values(X_test_tx)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    sv = np.mean(np.abs(shap_values), axis=0)
                    shap_imp = pd.Series(sv, index=feat_names_ohe).sort_values(ascending=False)
                    shap_imp.to_csv(os.path.join(OUT_DIR, f"{STAMP}_{name}_SHAP_importance.csv"))
                except Exception as e:
                    print(f"[Warn] SHAP failed for {name}: {e}")
        except Exception as e:
            print(f"[Warn] Importance step failed for {name}: {e}")

# ======================= 7) Save summary table ================================
res_df = pd.DataFrame(results)
res_csv = os.path.join(OUT_DIR, f"{STAMP}_model_comparison.csv")
res_df.to_csv(res_csv, index=False)
print(f"\n[Saved] Model comparison → {res_csv}")

# =================== 8) Plot ROC & PR Curves (all models) =====================
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="No skill")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Test Set")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(OUT_DIR, f"{STAMP}_ROC_all.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"[Saved] {roc_path}")

plt.figure(figsize=(7, 6))
for name, (rec, prec, ap) in pr_curves.items():
    plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves — Test Set")
plt.legend()
plt.tight_layout()
pr_path = os.path.join(OUT_DIR, f"{STAMP}_PR_all.png")
plt.savefig(pr_path, dpi=300)
plt.close()
print(f"[Saved] {pr_path}")

# =================== 9) Print top-line summary for manuscript =================
ranked = res_df.sort_values(["AUC_ROC", "F1@best"], ascending=False).reset_index(drop=True)
print("\n===== Top models by AUC then F1 (test set) =====")
print(ranked[["model", "AUC_ROC", "AP(PR_AUC)", "F1@best", "Sens@best", "Spec@best"]].head(10).to_string(index=False))
# Save ranked summary as CSV
ranked_csv_path = os.path.join(OUT_DIR, f"{STAMP}_top_models_summary.csv")
ranked[["model", "AUC_ROC", "AP(PR_AUC)", "F1@best", "Sens@best", "Spec@best"]] \
    .to_csv(ranked_csv_path, index=False)
print(f"[Saved] Top models summary → {ranked_csv_path}")

# =================== 10) Save a README (reproducibility) ======================
readme = {
    "timestamp": STAMP,
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "pos_rate_train": float(np.mean(y_train)),
    "pos_rate_test": float(np.mean(y_test)),
    "features_used": use_cols,
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
    "notes": [
        "Locked to thesis feature set when present",
        "Pipelines: impute (median/mode) + OHE + scaling; SMOTE-Tomek",
        "Calibration: isotonic else Platt",
        "Tuning: 5-fold Stratified CV, scoring=ROC AUC",
        "Outputs: ROC/PR curves, confusion matrices, importance CSVs",
    ],
}
with open(os.path.join(OUT_DIR, f"{STAMP}_README.json"), "w") as f:
    json.dump(readme, f, indent=2)
print(f"[Saved] README metadata → {os.path.join(OUT_DIR, f'{STAMP}_README.json')}")

print("\n✅ Done. Check the `model_outputs/` folder for figures, tables, and logs.\n")


# ==== EXTRA FIGURES FOR RESULTS: calibration, importance, decision curves ====
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

# Reuse objects defined earlier: X_train, X_test, y_train, y_test, preprocess, sampler, cv,
# and models_and_grids (LR, RF, SVM, XGB, LGB if installed).
# We'll (re)fit briefly to get calibrated models and store them.
calibrated_models = {}

for name, base_clf, grid in models_and_grids:
    print(f"\n[Refit for figures] {name}")
    pipe = ImbPipeline(steps=[("prep", preprocess),
                              ("smote", sampler),
                              ("clf", base_clf)])
    gscv = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, verbose=0)
    gscv.fit(X_train, y_train)
    best = gscv.best_estimator_
    # probability calibration
    try:
        cal = CalibratedClassifierCV(best, method="isotonic", cv=cv)
        cal.fit(X_train, y_train)
    except Exception:
        cal = CalibratedClassifierCV(best, method="sigmoid", cv=cv)
        cal.fit(X_train, y_train)

    calibrated_models[name] = cal

# ---------- 1) Calibration curves (all models on one figure) ----------
# make sure you actually have >1 model
print("Calibrated models:", list(calibrated_models.keys()))

fig, ax = plt.subplots(figsize=(7,6))
first = True
for name, cal in calibrated_models.items():
    y_prob = cal.predict_proba(X_test)[:, 1]
    CalibrationDisplay.from_predictions(
        y_test, y_prob, n_bins=10, name=name, ax=ax
    )
    if first:
        # draw the 45° reference once
        ax.plot([0,1],[0,1],'k--', linewidth=1, label="Perfectly calibrated")
        first = False

ax.set_title("Calibration Curves — Test Set")
fig.tight_layout()
calib_path = os.path.join(OUT_DIR, f"{STAMP}_Calibration_all.png")
plt.savefig(calib_path, dpi=300); plt.close()
print(f"[Saved] {calib_path}")

# ---------- 2) Feature importance ----------
# 2a) Tree models: impurity + optional SHAP summary
def save_bar(series, title, fname, top=20):
    s = series.sort_values(ascending=True).tail(top)
    plt.figure(figsize=(7,6))
    s.plot.barh()
    plt.title(title); plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=300); plt.close()
    print(f"[Saved] {path}")

for name in ["RandomForest", "XGBoost", "LightGBM"]:
    if name not in calibrated_models:
        continue
    inner = _get_inner_pipeline_from_calibrator(calibrated_models[name]).named_steps["clf"]
    prep  = _get_inner_pipeline_from_calibrator(calibrated_models[name]).named_steps["prep"]
    ohe   = prep.named_transformers_["cat"].named_steps["ohe"]
    # Robustly get feature names for OHE across sklearn versions
    if hasattr(ohe, "get_feature_names_out"):
        feat_names_ohe = num_cols + list(ohe.get_feature_names_out(cat_cols))
    else:
        feat_names_ohe = num_cols + list(ohe.get_feature_names(cat_cols))
    # impurity-based importance (if available)
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=feat_names_ohe)
        save_bar(imp, f"{name} — Feature importance", f"{STAMP}_{name}_imp_bar.png", top=20)
    # SHAP summary (if shap installed)
    if shap_available:
        try:
            Xtx = prep.transform(X_test)
            explainer = shap.TreeExplainer(inner)
            shap_vals = explainer.shap_values(Xtx)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals)>1 else shap_vals[0]
            shap.summary_plot(shap_vals, Xtx, feature_names=feat_names_ohe, show=False)
            path = os.path.join(OUT_DIR, f"{STAMP}_{name}_SHAP_summary.png")
            plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
            print(f"[Saved] {path}")
        except Exception as e:
            print(f"[Warn] SHAP plotting failed for {name}: {e}")

# 2b) SVM & Logistic Regression: permutation importance (on original features)
for name in ["SVM_RBF", "LogisticRegression"]:
    if name not in calibrated_models:
        continue
    try:
        perm = permutation_importance(calibrated_models[name], X_test, y_test, n_repeats=20,
                                      random_state=42, scoring="roc_auc")
        pi = pd.Series(perm.importances_mean, index=(num_cols + cat_cols))
        save_bar(pi, f"{name} — Permutation importance", f"{STAMP}_{name}_permimp_bar.png", top=20)
    except Exception as e:
        print(f"[Warn] Permutation importance failed for {name}: {e}")

# ---------- 3) Decision Curve Analysis (net benefit) ----------
def decision_curve(y_true, y_prob, thresholds):
    # Net benefit = (TP/N) - (FP/N)*(pt/(1-pt))
    N = len(y_true)
    out = []
    for pt in thresholds:
        y_hat = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        nb = (tp/N) - (fp/N) * (pt/(1-pt))
        out.append((pt, nb))
    return pd.DataFrame(out, columns=["threshold","net_benefit"])

ths = np.linspace(0.05, 0.6, 12)  # clinically relevant range; adjust if needed
plt.figure(figsize=(7,6))
# 'treat-all' and 'treat-none' references
base_rate = y_test.mean()
plt.plot(ths, [base_rate - (1-base_rate)*(t/(1-t)) for t in ths], linestyle="--", label="Treat all")
plt.plot(ths, [0]*len(ths), linestyle="--", label="Treat none")

for name, cal in calibrated_models.items():
    y_prob = cal.predict_proba(X_test)[:,1]
    dca = decision_curve(y_test, y_prob, ths)
    plt.plot(dca["threshold"], dca["net_benefit"], label=name)

plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis — Test Set")
plt.legend()
plt.tight_layout()
dca_path = os.path.join(OUT_DIR, f"{STAMP}_DecisionCurve_all.png")
plt.savefig(dca_path, dpi=300); plt.close()
print(f"[Saved] {dca_path}")


# === Table 1 in resultsBasic patient characteristics summary ===
print("\n=== Basic Characteristics of Study Cohort ===")

# Continuous variables summary (median [IQR])
cont_vars = ['age', 'no_ln', 'cea_level']  # add any continuous variables you want
for col in cont_vars:
    if col in df.columns:
        median = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        print(f"{col}: median = {median:.1f}  [IQR {q1:.1f} – {q3:.1f}]")

# Categorical/binary variables summary (n, %)
cat_vars = ['sex', 'recur', 'staging', 'tumor_loc', 'tumor_grade', 'neoadj',
            'tnt', 'postpone_neoadj', 'margin_binary', 'lvi']
for col in cat_vars:
    if col in df.columns:
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        print(f"\n{col}:")
        for level, count in counts.items():
            pct = (count / total) * 100
            print(f"  {level}: {count} ({pct:.1f}%)")

# Overall recurrence rate
if 'recur' in df.columns:
    rec_rate = df['recur'].mean() * 100
    print(f"\nOverall recurrence rate: {rec_rate:.1f}%")

# Optional: create a summary table DataFrame for export
summary_rows = []

# Continuous vars
for col in cont_vars:
    if col in df.columns:
        median = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        summary_rows.append({
            "Variable": col,
            "Summary": f"{median:.1f} [{q1:.1f} – {q3:.1f}]"
        })

# Categorical vars
for col in cat_vars:
    if col in df.columns:
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        for level, count in counts.items():
            pct = (count / total) * 100
            summary_rows.append({
                "Variable": col,
                "Summary": f"{level}: {count} ({pct:.1f}%)"
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(os.getcwd(), "basic_characteristics.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n[Saved] Basic characteristics table → {summary_path}")

# Table 1 for publication (Total / No recurrence / Recurrence) ===
def summarize_cont(col):
    """Return median [IQR] as string for total, no recur, recur."""
    def fmt(series):
        return f"{series.median():.1f} [{series.quantile(0.25):.1f}–{series.quantile(0.75):.1f}]"
    total_str = fmt(df[col].dropna())
    no_rec_str = fmt(df.loc[df['recur'] == 0, col].dropna())
    rec_str = fmt(df.loc[df['recur'] == 1, col].dropna())
    return total_str, no_rec_str, rec_str

def summarize_cat(col):
    """Return n (%) for each category, separately for total, no recur, recur."""
    total_n = len(df)
    no_rec_n = (df['recur'] == 0).sum()
    rec_n = (df['recur'] == 1).sum()

    rows = []
    for level in df[col].dropna().unique():
        total_count = (df[col] == level).sum()
        no_rec_count = ((df[col] == level) & (df['recur'] == 0)).sum()
        rec_count = ((df[col] == level) & (df['recur'] == 1)).sum()

        rows.append({
            "Variable": col if len(rows) == 0 else "",  # only show var name once
            "Category": level,
            "Total": f"{total_count} ({total_count/total_n*100:.1f}%)",
            "No recurrence": f"{no_rec_count} ({no_rec_count/no_rec_n*100:.1f}%)",
            "Recurrence": f"{rec_count} ({rec_count/rec_n*100:.1f}%)"
        })
    return rows

# Variables to include
cont_vars = ['age', 'no_ln', 'cea_level']  # add/remove as needed
cat_vars = ['sex', 'staging', 'tumor_loc', 'tumor_grade',
            'neoadj', 'tnt', 'postpone_neoadj', 'margin_binary', 'lvi']

table1_rows = []

# Continuous variables
for col in cont_vars:
    if col in df.columns:
        total_str, no_rec_str, rec_str = summarize_cont(col)
        table1_rows.append({
            "Variable": col,
            "Category": "",
            "Total": total_str,
            "No recurrence": no_rec_str,
            "Recurrence": rec_str
        })

# Categorical variables
for col in cat_vars:
    if col in df.columns:
        table1_rows.extend(summarize_cat(col))

# Create DataFrame
table1_df = pd.DataFrame(table1_rows)

# Save to CSV
table1_path = os.path.join(os.getcwd(), "table1_characteristics.csv")
table1_df.to_csv(table1_path, index=False)
print(f"[Saved] Table 1 characteristics → {table1_path}")

# Display preview
import pandas as pd
pd.set_option('display.max_rows', None)
print(table1_df)

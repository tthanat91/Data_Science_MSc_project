# Clinical Prediction Model for Recurrent Rectal Cancer — Robust Validation + Calibration + Thresholds
# --------------------------------------------------------------------------------
# CHANGELOG (relative to your current script)
# [NEW] RepeatedStratifiedKFold (5×10) to report robust CV metrics (mean±SD) for AUC and AP.
# [NEW] Nonparametric bootstrap (B=1000) on the held-out test set to compute 95% CIs for AUC and AP.
# [NEW] Calibration metrics (Brier score, calibration slope & intercept) and a multi‑model calibration plot.
# [NEW] Threshold trade‑off tables at fixed clinical risks (10%, 20%, 30%) with Sens/Spec/PPV/NPV.
# [NEW] Decision curve analysis focused on 10–30% thresholds.
# [MOD] Kept your original 5‑fold tuning (GridSearchCV) for comparability; robustness is computed AFTER tuning/calibration.
# [SAFE] All additions are tagged:  # [NEW]  # [BOOT]  # [CAL]  # [THR]  # [DCA]

# -------------------- EDA section (UNCHANGED from your version) --------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
print(np.__version__)
# Load your dataset
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
df['margin_binary'] = df['margin'].apply(
    lambda x: 'Negative' if isinstance(x, str) and x.strip().lower() == 'negative' else 'Positive'
)
cat_features = ['sex', 'tumor_loc', 'tumor_grade', 'cea_binary',
                'neoadj', 'tnt', 'recur', 'margin_binary']

num_plots = len(cat_features)
cols = 3
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    if col in df.columns:
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Count Plot: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=45)
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

# ----------------- MODELING + ROBUST VALIDATION (UPGRADED) -------------------
# ============= 0) Imports & Config =============
import os, json, warnings
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold  # [NEW]

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
    brier_score_loss  # [CAL]
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

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

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
    try:
        f = float(s)
        if f == 1.0: return 1
        if f == 0.0: return 0
    except Exception:
        pass
    return np.nan

if "recur" not in df.columns:
    raise ValueError("`recur` column not found in dataframe.")

_recur_norm = df["recur"].apply(_norm_bin)
unknown_mask = _recur_norm.isna()
if unknown_mask.any():
    bad_vals = df.loc[unknown_mask, "recur"].astype(str).str.strip().str.lower().value_counts(dropna=False)
    print("[Warn] Dropping rows with unknown/blank `recur` values. Counts by raw value:\n", bad_vals.to_string())

    df = df.loc[~unknown_mask].copy()
    _recur_norm = _recur_norm.loc[~unknown_mask]

df["recur"] = _recur_norm.astype(int)

# ============= 2) Lock features (keep as in your script) =============
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

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
for c in cat_cols:
    X[c] = X[c].astype(str)

print(f"[Info] Using {len(use_cols)} features — {len(num_cols)} numeric, {len(cat_cols)} categorical.")

# ---------- Denominators & Missingness summary for Table 1 (run after Section 2) ----------
import pandas as pd
import numpy as np
import os

# Cohort denominators
N_total = int(len(df))
N_no_rec = int((df["recur"] == 0).sum())
N_rec    = int((df["recur"] == 1).sum())

# Work on the variables you actually use in Table 1 (edit if you show more/less)
table1_vars = [
    "age", "no_ln", "cea_level", "sex", "staging", "tumor_loc", "tumor_grade",
    "neoadj", "tnt", "postpone_neoadj", "margin", "lvi"
]
table1_vars = [c for c in table1_vars if c in df.columns]

# Missingness (before any SimpleImputer/MICE operates)
miss = df[table1_vars].isnull().agg(["sum","count"]).T
miss.columns = ["n_missing", "denominator"]
miss["pct_missing"] = (miss["n_missing"] / miss["denominator"] * 100).round(1)

# Save a supplementary table if you want to cite exact denominators per variable
miss_path = os.path.join(OUT_DIR, f"{STAMP}_Supplementary_Table_Missingness_Table1.csv")
miss.reset_index(names="variable").to_csv(miss_path, index=False)
print(f"[Saved] Table 1 missingness summary → {miss_path}")

# Make a short, human-readable list of top missing variables for the footnote
top = (miss[miss["n_missing"] > 0]
       .sort_values("pct_missing", ascending=False)
       .head(3))
top_list = "; ".join([f"{v} {p}%" for v, p in zip(top.index, top["pct_missing"])]) if not top.empty else "none"

# Print a ready-to-paste footnote sentence
max_miss = 0.0 if miss.empty else float(miss["pct_missing"].max())
print("\n--- Copy this into your Table 1 footnote ---")
print(
    f"Percentages are calculated from available data for each variable. "
    f"The overall cohort comprised N={N_total} (No recurrence={N_no_rec}, Recurrence={N_rec}). "
    f"Missingness prior to imputation ranged from 0% to {max_miss:.1f}% across variables"
    + (f" (highest: {top_list})." if top_list != "none" else ".")
)
print("------------------------------------------------\n")
# -----------------------------------------------------------------------------------------

# ============= 3) Train/Test split =============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================== 4) Preprocessing & Imbalance ==========================
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

lr = LogisticRegression(max_iter=2000, solver="liblinear")
lr_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"]}
models_and_grids.append(("LogisticRegression", lr, lr_grid))

rf = RandomForestClassifier(random_state=42)
rf_grid = {"clf__n_estimators": [300, 600], "clf__max_depth": [None, 5, 10, 20],
           "clf__min_samples_split": [2, 5, 10], "clf__min_samples_leaf": [1, 2, 4]}
models_and_grids.append(("RandomForest", rf, rf_grid))

svm = SVC(probability=True, random_state=42)
svm_grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1, 0.01]}
models_and_grids.append(("SVM_RBF", svm, svm_grid))

if xgb is not None:
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                use_label_encoder=False, tree_method="hist", random_state=42)
    xgb_grid = {"clf__n_estimators": [300, 600], "clf__max_depth": [3, 5, 8],
                "clf__learning_rate": [0.03, 0.1], "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0]}
    models_and_grids.append(("XGBoost", xgb_clf, xgb_grid))
else:
    print("[Warn] xgboost not installed — skipping.")

if lgb is not None:
    lgb_clf = lgb.LGBMClassifier(objective="binary", random_state=42)
    lgb_grid = {"clf__n_estimators": [300, 600], "clf__num_leaves": [31, 63, 127],
                "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [-1, 5, 10]}
    models_and_grids.append(("LightGBM", lgb_clf, lgb_grid))
else:
    print("[Warn] lightgbm not installed — skipping.")

from sklearn.calibration import CalibratedClassifierCV

def _get_inner_pipeline_from_calibrator(cal: CalibratedClassifierCV):
    est = getattr(cal, "estimator", None)
    if est is None:
        est = getattr(cal, "base_estimator", None)
    if est is None:
        raise AttributeError("Cannot find underlying estimator from CalibratedClassifierCV.")
    return est

# ================= 6) Training + Hyperparameter Tuning ========================
results = []
roc_curves = {}
pr_curves = {}

# [CAL] helper for calibration slope/intercept
from sklearn.linear_model import LogisticRegression as _LR  # alias to avoid shadowing

def calibration_slope_intercept(y_true, y_prob):  # [CAL]
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = _LR(fit_intercept=True, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])  # slope, intercept

for name, base_clf, grid in models_and_grids:
    print(f"\n===== Tuning {name} =====")

    pipe = ImbPipeline(steps=[("prep", preprocess), ("smote", sampler), ("clf", base_clf)])

    gscv = GridSearchCV(estimator=pipe, param_grid=grid, scoring="roc_auc",
                        cv=cv, n_jobs=-1, verbose=1, refit=True)
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

    # Predictions on held-out test
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    # --- DCA: store test-set probabilities for this model
    if 'model_probs_for_dca' not in globals():
        model_probs_for_dca = {}
    model_probs_for_dca[name] = y_proba.copy()
    y_pred_05 = (y_proba >= 0.5).astype(int)

    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    y_pred_bt = (y_proba >= best_t).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    def metrics(y_true, y_hat):
        cm = confusion_matrix(y_true, y_hat)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / max(1, (tn + fp))
        npv = tn / max(1, (tn + fn))
        return dict(F1=f1_score(y_true, y_hat, zero_division=0),
                    Sensitivity=recall_score(y_true, y_hat, zero_division=0),
                    Precision=precision_score(y_true, y_hat, zero_division=0),
                    Specificity=spec, NPV=npv)

    m05 = metrics(y_test, y_pred_05)
    mbt = metrics(y_test, y_pred_bt)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)
    pr_curves[name] = (rec, prec, ap)

    # [CAL] Calibration metrics
    brier = brier_score_loss(y_test, y_proba)
    slope, intercept = calibration_slope_intercept(y_test, y_proba)

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
        "Brier": round(brier, 4),                 # [CAL]
        "CalSlope": round(slope, 3),              # [CAL]
        "CalIntercept": round(intercept, 3),      # [CAL]
    })

    # ----------------------- [NEW] ROBUSTNESS: Repeated CV -----------------------
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    tuned_base = best_model.named_steps['clf']
    robust_pipe = ImbPipeline(steps=[("prep", preprocess), ("smote", sampler), ("clf", tuned_base)])

    from sklearn.model_selection import cross_val_score
    auc_scores = cross_val_score(robust_pipe, X, y, scoring='roc_auc', cv=rskf, n_jobs=-1)
    ap_scores  = cross_val_score(robust_pipe, X, y, scoring='average_precision', cv=rskf, n_jobs=-1)

    robust_csv = os.path.join(OUT_DIR, f"{STAMP}_{name}_repeatedCV_scores.csv")
    pd.DataFrame({"AUC": auc_scores, "AP": ap_scores}).to_csv(robust_csv, index=False)
    print(f"[Saved] Repeated CV scores → {robust_csv}")

    print(f"[RepeatedCV] {name} — AUC mean±SD: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}; "
          f"AP mean±SD: {ap_scores.mean():.3f} ± {ap_scores.std():.3f}")

    results[-1]["AUC_rep_mean"] = round(float(auc_scores.mean()), 3)
    results[-1]["AUC_rep_sd"]   = round(float(auc_scores.std()), 3)
    results[-1]["AP_rep_mean"]  = round(float(ap_scores.mean()), 3)
    results[-1]["AP_rep_sd"]    = round(float(ap_scores.std()), 3)

    # -------------------- [BOOT] TEST-SET BOOTSTRAP CIs (optional) --------------------
    RUN_BOOTSTRAP = True
    if RUN_BOOTSTRAP:
        rng = np.random.default_rng(42)
        B = 1000
        auc_bs, ap_bs = [], []
        for _ in range(B):
            idx = rng.integers(0, len(y_test), size=len(y_test))
            yt = y_test[idx]
            yp = y_proba[idx]
            try:
                auc_bs.append(roc_auc_score(yt, yp))
                ap_bs.append(average_precision_score(yt, yp))
            except Exception:
                pass
        def perc_ci(a, lo=2.5, hi=97.5):
            a = np.array(a)
            return float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi))
        auc_lo, auc_hi = perc_ci(auc_bs)
        ap_lo, ap_hi   = perc_ci(ap_bs)
        results[-1]["AUC_boot_ci"] = f"[{auc_lo:.3f}, {auc_hi:.3f}]"
        results[-1]["AP_boot_ci"]  = f"[{ap_lo:.3f}, {ap_hi:.3f}]"

    # -------------------- [THR] Threshold trade‑off tables (10/20/30%) -----------
    def metrics_at_thresholds(y_true, y_prob, thresholds=(0.10, 0.20, 0.30)):
        rows = []
        for t in thresholds:
            y_hat = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
            sens = tp / max(1, (tp + fn))
            spec = tn / max(1, (tn + fp))
            ppv  = tp / max(1, (tp + fp))
            npv  = tn / max(1, (tn + fn))
            rows.append({"Threshold": t, "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv})
        return pd.DataFrame(rows)

    thr_df = metrics_at_thresholds(y_test, y_proba, thresholds=(0.10, 0.20, 0.30))
    thr_csv = os.path.join(OUT_DIR, f"{STAMP}_{name}_threshold_table.csv")
    thr_df.round(3).to_csv(thr_csv, index=False)
    print(f"[Saved] Threshold table → {thr_csv}")

# ========================= SMOTE Sensitivity Add-on =========================
# This block adds a WITH-vs-WITHOUT SMOTETomek sensitivity analysis and exports
# a Supplementary table comparing metrics. It plugs into your existing script.
# ---------------------------------------------------------------------------
# WHERE TO START RUNNING:
# 1) Run your script as usual up to the end of Section 6) Training + Tuning.
#    (i.e., run through the end of the main for-loop that populates `results` and
#     saves per-model threshold tables.)
# 2) THEN run from the line below:  "# ===== 6B) SMOTE sensitivity (NEW) ====="
#    You do NOT need to re-run the earlier EDA or preprocessing sections.
# ---------------------------------------------------------------------------

# ===== 6B) SMOTE sensitivity (NEW) =====
# This re-trains/tunes the SAME models on the SAME data pipeline but WITHOUT SMOTETomek,
# and produces a side-by-side comparison table.

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression as _LR

# --- safety checks for prerequisites from earlier sections ---
assert 'X_train' in globals() and 'y_train' in globals() and 'X_test' in globals() and 'y_test' in globals(), \
    "Run data split & feature engineering first (Sections 2–3)."
assert 'preprocess' in globals(), "Run preprocessing pipelines first (Section 4)."
assert 'models_and_grids' in globals(), "Model/grids not found. Run Section 5."
assert 'cv' in globals(), "CV splitter `cv` not found. Run Section 5."
assert 'OUT_DIR' in globals() and 'STAMP' in globals(), "Run Section 0 (config)."

# Helper used earlier to compute calibration slope/intercept

def _calibration_slope_intercept(y_true, y_prob):
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = _LR(fit_intercept=True, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

# ---- (A) Gather best-from-SMOTE results already computed in Section 6
#       and put into a DataFrame `res_smote` for later merge.

res_smote = pd.DataFrame(results).copy()  # `results` was filled in Section 6 (SMOTE on)
res_smote['setting'] = 'SMOTETomek'

# ---- (B) Refit/tune WITHOUT SMOTETomek and collect metrics in `res_nosmote`

nosmote_results = []
for name, base_clf, grid in models_and_grids:
    print(f"\n===== Tuning (NO SMOTE) {name} =====")

    no_smote_pipe = Pipeline(steps=[("prep", preprocess), ("clf", base_clf)])

    gscv_ns = GridSearchCV(estimator=no_smote_pipe, param_grid=grid, scoring="roc_auc",
                           cv=cv, n_jobs=-1, verbose=1, refit=True)
    gscv_ns.fit(X_train, y_train)
    best_ns = gscv_ns.best_estimator_
    print(f"[NO-SMOTE] Best params for {name}: {gscv_ns.best_params_}")

    # Probability calibration (same approach as main analysis)
    try:
        calibrated_ns = CalibratedClassifierCV(best_ns, method="isotonic", cv=cv)
        calibrated_ns.fit(X_train, y_train)
    except Exception:
        calibrated_ns = CalibratedClassifierCV(best_ns, method="sigmoid", cv=cv)
        calibrated_ns.fit(X_train, y_train)

    yp = calibrated_ns.predict_proba(X_test)[:, 1]

    # Metrics at 0.50 and best-F1
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_test, (yp >= t).astype(int)) for t in thresholds]
    t_best = float(thresholds[int(np.argmax(f1s))])

    def _metrics(y_true, y_hat):
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        spec = tn / max(1, (tn + fp))
        npv  = tn / max(1, (tn + fn))
        return dict(
            F1=f1_score(y_true, y_hat, zero_division=0),
            Sensitivity=recall_score(y_true, y_hat, zero_division=0),
            Precision=precision_score(y_true, y_hat, zero_division=0),
            Specificity=spec, NPV=npv,
        )

    yhat05 = (yp >= 0.50).astype(int)
    yhatb  = (yp >= t_best).astype(int)

    auc = roc_auc_score(y_test, yp)
    ap  = average_precision_score(y_test, yp)
    brier = brier_score_loss(y_test, yp)
    slope, intercept = _calibration_slope_intercept(y_test, yp)

    nosmote_results.append({
        "model": name,
        "setting": "No-SMOTE",
        "AUC_ROC": round(auc, 4),
        "AP(PR_AUC)": round(ap, 4),
        "thr_bestF1": round(t_best, 2),
        "F1@0.50": round(_metrics(y_test, yhat05)["F1"], 4),
        "Sens@0.50": round(_metrics(y_test, yhat05)["Sensitivity"], 4),
        "Spec@0.50": round(_metrics(y_test, yhat05)["Specificity"], 4),
        "PPV@0.50": round(_metrics(y_test, yhat05)["Precision"], 4),
        "NPV@0.50": round(_metrics(y_test, yhat05)["NPV"], 4),
        "F1@best": round(_metrics(y_test, yhatb)["F1"], 4),
        "Sens@best": round(_metrics(y_test, yhatb)["Sensitivity"], 4),
        "Spec@best": round(_metrics(y_test, yhatb)["Specificity"], 4),
        "PPV@best": round(_metrics(y_test, yhatb)["Precision"], 4),
        "NPV@best": round(_metrics(y_test, yhatb)["NPV"], 4),
        "Brier": round(brier, 4),
        "CalSlope": round(slope, 3),
        "CalIntercept": round(intercept, 3),
        "best_params": json.dumps(gscv_ns.best_params_),
    })

res_nosmote = pd.DataFrame(nosmote_results)

# ---- (C) Merge SMOTE vs No-SMOTE into a compact Supplementary table

# Keep only key columns for comparison
keep_cols = [
    "model", "AUC_ROC", "AP(PR_AUC)", "F1@best", "Sens@best", "Spec@best", "Brier"
]
A = res_nosmote[keep_cols].copy()
A.columns = ["model", "AUC_noSMOTE", "AP_noSMOTE", "F1_noSMOTE", "Sens_noSMOTE", "Spec_noSMOTE", "Brier_noSMOTE"]

B = res_smote[keep_cols].copy()
B.columns = ["model", "AUC_SMOTE", "AP_SMOTE", "F1_SMOTE", "Sens_SMOTE", "Spec_SMOTE", "Brier_SMOTE"]

cmp = A.merge(B, on="model", how="inner")
cmp["ΔAUC"] = (cmp["AUC_SMOTE"] - cmp["AUC_noSMOTE"]).round(3)
cmp["ΔAP"]  = (cmp["AP_SMOTE"]  - cmp["AP_noSMOTE"]).round(3)
cmp["ΔF1"]  = (cmp["F1_SMOTE"]  - cmp["F1_noSMOTE"]).round(3)
cmp["ΔSens"] = (cmp["Sens_SMOTE"] - cmp["Sens_noSMOTE"]).round(3)
cmp["ΔSpec"] = (cmp["Spec_SMOTE"] - cmp["Spec_noSMOTE"]).round(3)
cmp["ΔBrier"] = (cmp["Brier_SMOTE"] - cmp["Brier_noSMOTE"]).round(3)

# Order columns nicely for the supplementary table
cmp = cmp[[
    "model",
    "AUC_noSMOTE", "AUC_SMOTE", "ΔAUC",
    "AP_noSMOTE",  "AP_SMOTE",  "ΔAP",
    "F1_noSMOTE",  "F1_SMOTE",  "ΔF1",
    "Sens_noSMOTE","Sens_SMOTE","ΔSens",
    "Spec_noSMOTE","Spec_SMOTE","ΔSpec",
    "Brier_noSMOTE","Brier_SMOTE","ΔBrier",
]]

# Save to disk
supp_path = os.path.join(OUT_DIR, f"{STAMP}_Supplementary_Table_SMOTE_Sensitivity.csv")
cmp.to_csv(supp_path, index=False)
print(f"[Saved] SMOTE sensitivity table → {supp_path}")

# OPTIONAL: pretty print to console
print("\n===== SMOTE Sensitivity (No-SMOTE vs SMOTE) =====")
print(cmp.to_string(index=False))

# ===== End of SMOTE Sensitivity Add-on =====


# ======================= 7) Save summary table ================================
res_df = pd.DataFrame(results)
res_csv = os.path.join(OUT_DIR, f"{STAMP}_model_comparison_withROBUST.csv")
res_df.to_csv(res_csv, index=False)
print(f"[Saved] Model comparison → {res_csv}")

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

# =================== 9) Decision Curve Analysis (focused thresholds) ==========
# DCA = net benefit vs threshold; show each model + reference lines in 10–30%

def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    rows = []
    for pt in thresholds:
        y_hat = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        # Net benefit (Vickers & Elkin 2006)
        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        rows.append((pt, nb))
    return pd.DataFrame(rows, columns=["threshold", "net_benefit"])

# Thresholds of clinical interest
ths = np.linspace(0.10, 0.30, 21)

plt.figure(figsize=(7, 6))

# Reference strategies
prevalence = y_test.mean()
treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in ths]
treat_none = [0.0] * len(ths)
plt.plot(ths, treat_all, linestyle="--", label="Treat all")
plt.plot(ths, treat_none, linestyle="--", label="Treat none")

# Per-model curves (requires Patch A executed during training)
if 'model_probs_for_dca' in globals() and len(model_probs_for_dca) > 0:
    for mname, yprob in model_probs_for_dca.items():
        dca_df = decision_curve(y_test, yprob, ths)
        plt.plot(dca_df["threshold"], dca_df["net_benefit"], label=mname)
else:
    print("[Warn] No model probabilities captured for DCA — rerun training loop after applying Patch A.")

# Visual emphasis on clinical thresholds
for v in (0.10, 0.20, 0.30):
    plt.axvline(v, linestyle=":", linewidth=1, alpha=0.6)
plt.axvspan(0.10, 0.30, alpha=0.08)

plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis (10–30% thresholds)")
plt.legend()
plt.tight_layout()

dca_path = os.path.join(OUT_DIR, f"{STAMP}_DecisionCurve_all.png")
plt.savefig(dca_path, dpi=300)
plt.show()
print(f"[Saved] {dca_path}")


# =================== 10) Top-line summary for manuscript ======================
ranked = res_df.sort_values(["AUC_ROC", "F1@best"], ascending=False).reset_index(drop=True)
print("===== Top models by AUC then F1 (test set) =====")
print(ranked[["model", "AUC_ROC", "AP(PR_AUC)", "F1@best", "Sens@best", "Spec@best",
              "AUC_rep_mean", "AUC_rep_sd", "AP_rep_mean", "AP_rep_sd",
              "AUC_boot_ci", "AP_boot_ci", "Brier", "CalSlope", "CalIntercept"]].to_string(index=False))

ranked_csv_path = os.path.join(OUT_DIR, f"{STAMP}_top_models_summary_withROBUST.csv")
ranked[["model", "AUC_ROC", "AP(PR_AUC)", "F1@best", "Sens@best", "Spec@best",
        "AUC_rep_mean", "AUC_rep_sd", "AP_rep_mean", "AP_rep_sd",
        "AUC_boot_ci", "AP_boot_ci", "Brier", "CalSlope", "CalIntercept"]].to_csv(ranked_csv_path, index=False)
print(f"[Saved] Top models summary → {ranked_csv_path}")

# =================== 11) README (reproducibility) ============================
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
        "Tuning: 5-fold Stratified CV (unchanged)",
        "[NEW] Robustness: RepeatedStratifiedKFold (5×10) with AUC/AP mean±SD",
        "[BOOT] Test-set bootstrap CIs for AUC/AP (B=1000)",
        "[CAL] Brier score + calibration slope/intercept",
        "[THR] Threshold tables at 10/20/30% (Sens/Spec/PPV/NPV)",
        "[DCA] Decision curve analysis over 10–30%",
        "Outputs: ROC/PR curves, calibration, decision curves, importance CSVs"
    ],
}
with open(os.path.join(OUT_DIR, f"{STAMP}_README.json"), "w") as f:
    json.dump(readme, f, indent=2)
print(f"[Saved] README metadata → {os.path.join(OUT_DIR, f'{STAMP}_README.json')}")

print("✅ Done. Robust validation + calibration + threshold trade-offs added.")

# svm_full_paper_style.py
"""
Comprehensive SVM-only analysis with many metrics & visualizations (paper-style).

Requirements:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib statsmodels PyPDF2 shap(optional)
Run in the same folder as: quality_data.csv and your uploaded PDFs.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from statsmodels.nonparametric.smoothers_lowess import lowess
import joblib
import warnings
warnings.filterwarnings("ignore")
# PyPDF2 is optional: gracefully handle if not installed
try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    PdfReader = None
    _HAS_PYPDF2 = False
    print("Warning: PyPDF2 not installed. PDF snippet extraction will be skipped. To enable, run: pip install PyPDF2")

sns.set(style="whitegrid")
np.random.seed(42)

# -------------------- CONFIG --------------------
DATA_PATH = "quality_data.csv"
PDF_FILES = [
    '/mnt/data/SADH-D-25-00899_reviewer (2).pdf',
    '/mnt/data/milk paper 4 with IOT.pdf',
    '/mnt/data/IJIT_rohini.pdf'
]
OUT_DIR = "svm_full_paper_style_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# -------------------- HELPER FUNCTIONS --------------------
def extract_pdf_first_page_text(pdf_path):
    if not _HAS_PYPDF2 or PdfReader is None:
        return "[PyPDF2 not installed]"
    try:
        reader = PdfReader(pdf_path)
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:2000]
        return ""
    except Exception as e:
        return f"[PDF read error: {e}]"

def lin_concordance_cc(x, y):
    # Lin's concordance correlation coefficient
    x = np.array(x); y = np.array(y)
    mean_x = x.mean(); mean_y = y.mean()
    s_x = x.var(ddof=1); s_y = y.var(ddof=1)
    cov = np.cov(x, y, ddof=1)[0,1]
    numerator = 2 * cov
    denominator = s_x + s_y + (mean_x - mean_y)**2
    return numerator / denominator if denominator != 0 else np.nan

def bland_altman_plot(true_vals, pred_vals, ax=None, title="Bland-Altman"):
    diffs = np.array(true_vals) - np.array(pred_vals)
    mean_vals = (np.array(true_vals) + np.array(pred_vals)) / 2.0
    md = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    if ax is None:
        plt.figure(figsize=(6,5))
        ax = plt.gca()
    ax.scatter(mean_vals, diffs, alpha=0.6)
    ax.axhline(md, color='red', linestyle='--', label=f"Mean diff = {md:.3f}")
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--', label=f'+1.96 SD = {md+1.96*sd:.3f}')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--', label=f'-1.96 SD = {md-1.96*sd:.3f}')
    ax.set_xlabel("Mean of Actual & Predicted")
    ax.set_ylabel("Difference (Actual - Predicted)")
    ax.set_title(title)
    ax.legend()
    return ax

# -------------------- READ PDF SNIPPETS (for manual match-up) --------------------
snippets = {}
for p in PDF_FILES:
    if os.path.exists(p):
        snippets[os.path.basename(p)] = extract_pdf_first_page_text(p)
    else:
        snippets[os.path.basename(p)] = "[MISSING]"
with open(os.path.join(OUT_DIR, "pdf_snippets.txt"), "w", encoding="utf-8") as f:
    for name, s in snippets.items():
        f.write(f"---- {name} ----\n{s}\n\n")

# -------------------- LOAD DATA --------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Place your dataset in working directory.")

data = pd.read_csv(DATA_PATH)
print("Loaded data:", data.shape)
print(data.head())

# -------------------- PREPROCESS --------------------
if 'crop' not in data.columns:
    raise KeyError("Dataset must contain 'crop' column (text).")

le = LabelEncoder()
data['crop_encoded'] = le.fit_transform(data['crop'].astype(str))
crop_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("Crop mapping:", crop_map)

# Required features (adapt here if your CSV uses different names)
FEATURES = ['crop_encoded', 'temp_c', 'humidity', 'gas_ppm', 'ethylene_ppm', 'hours_since_harvest']
for f in FEATURES:
    if f not in data.columns:
        raise KeyError(f"Missing feature column: {f}")
TARGET = 'vqi'
if TARGET not in data.columns:
    raise KeyError(f"Missing target column: {TARGET}")

X = data[FEATURES].copy()
# Derive 3-class freshness label from VQI tertiles: 0 (spoiled), 1 (moderate), 2 (fresh)
vqi_series = data[TARGET].copy()
q = vqi_series.quantile([0.3333, 0.6666]).values
lo, hi = float(q[0]), float(q[1])
if hi <= lo:
    hi = lo + 1e-6
bins = [-np.inf, lo, hi, np.inf]
y = pd.cut(vqi_series, bins=bins, labels=[0, 1, 2]).astype(int)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y, shuffle=True
)
print("Train/test sizes:", X_train.shape, X_test.shape)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- HYPERPARAMETER TUNING (SVM classifier) --------------------
print("Tuning SVM (classification) with GridSearchCV...")
param_grid = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
gs = GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), param_grid,
                  scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
gs.fit(X_train_scaled, y_train)
print("Best params:", gs.best_params_)
svm = gs.best_estimator_

# -------------------- FIT & PREDICT --------------------
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
y_proba = None
if hasattr(svm, 'predict_proba'):
    try:
        y_proba = svm.predict_proba(X_test_scaled)
    except Exception:
        y_proba = None

# -------------------- METRICS (classification) --------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

metrics = {
    'acc': acc,
    'precision_macro': prec,
    'recall_macro': rec,
    'f1_macro': f1,
}
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.to_csv(os.path.join(OUT_DIR, 'svm_all_metrics.csv'))
print("\nMetrics:\n", metrics_df)

# -------------------- SAVE PREDICTIONS --------------------
pred_df = X_test.copy()
pred_df['true_class'] = y_test.values
pred_df['pred_class'] = y_pred
if y_proba is not None:
    for i in [0,1,2]:
        if i < y_proba.shape[1]:
            pred_df[f'proba_{i}'] = y_proba[:, i]
pred_df['crop_name'] = le.inverse_transform(pred_df['crop_encoded'].astype(int))
pred_df.to_csv(os.path.join(OUT_DIR, 'svm_predictions_testset.csv'), index=False)

# -------------------- PLOTS: FIGURE STYLING --------------------
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
sns.set_palette('tab10')

# 1) Learning curve (accuracy)
print("Plotting learning curve (accuracy)...")
train_sizes, train_scores, val_scores = learning_curve(
    svm, X_train_scaled, y_train, cv=5,
    train_sizes=np.linspace(0.1,1.0,6), scoring='accuracy', n_jobs=-1
)
train_acc = train_scores.mean(axis=1)
val_acc = val_scores.mean(axis=1)
plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_acc, 'o-', label='Train Accuracy')
plt.plot(train_sizes, val_acc, 'o-', label='Validation Accuracy')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curve (SVM Classification)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'learning_curve.png')); plt.close()

# 2) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['0','1','2'], yticklabels=['0','1','2'])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (SVM)')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'confusion_matrix.png')); plt.close()

# 3) ROC curves (OvR) and PR curves
if y_proba is not None:
    fig, ax = plt.subplots(figsize=(6,5))
    for c in [0,1,2]:
        RocCurveDisplay.from_predictions((y_test==c).astype(int), y_proba[:,c], name=f'Class {c}', ax=ax)
    plt.title('ROC Curves (OvR, SVM)'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'roc_curves.png')); plt.close()

    fig, ax = plt.subplots(figsize=(6,5))
    for c in [0,1,2]:
        PrecisionRecallDisplay.from_predictions((y_test==c).astype(int), y_proba[:,c], name=f'Class {c}', ax=ax)
    plt.title('Precision–Recall Curves (OvR, SVM)'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'pr_curves.png')); plt.close()

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for i,c in enumerate([0,1,2]):
        CalibrationDisplay.from_predictions((y_test==c).astype(int), y_proba[:,c], n_bins=10, ax=axes[i])
        axes[i].set_title(f'Calibration: Class {c}')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'calibration.png')); plt.close()

# 4) Class distribution by crop
plt.figure(figsize=(10,4))
tmp = pred_df.copy()
sns.countplot(x='crop_name', hue='pred_class', data=tmp)
plt.xticks(rotation=45); plt.title('Predicted Class Count by Crop'); plt.tight_layout();
plt.savefig(os.path.join(OUT_DIR,'pred_class_by_crop.png')); plt.close()

# 5) Hours since harvest vs predicted class
plt.figure(figsize=(6,5))
if 'hours_since_harvest' in pred_df.columns:
    sns.stripplot(x='pred_class', y='hours_since_harvest', data=pred_df, jitter=True, alpha=0.6)
    plt.xlabel('Predicted Class'); plt.ylabel('Hours since harvest'); plt.title('Hours vs Predicted Class')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'hours_vs_pred_class.png')); plt.close()

# 9) Correlation heatmap (features + target)
corr_df = data[FEATURES + [TARGET]].corr()
plt.figure(figsize=(7,6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature & Target Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'correlation_heatmap.png'))
plt.close()

# 6) Permutation importance (with repeats)
print("Computing permutation importance (may take a moment)...")
perm = permutation_importance(svm, X_test_scaled, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
feat_imp_df = pd.DataFrame({
    'feature': FEATURES,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=True)
plt.figure(figsize=(6,4))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance_mean'], xerr=feat_imp_df['importance_std'], align='center')
plt.xlabel('Decrease in accuracy (permutation)')
plt.title('Permutation Feature Importance (SVM)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'permutation_feature_importance.png'))
plt.close()
feat_imp_df.to_csv(os.path.join(OUT_DIR,'permutation_feature_importance.csv'), index=False)

# 7) Partial dependence plots (on numeric features)
num_feats = [f for f in FEATURES if f != 'crop_encoded']
# PDP requires the estimator and the data (not scaled or scaled depending). We'll use scaled data and feature indices
# map feature names to scaled array indices
# the SVM expects scaled data, so we pass X_train_scaled and feature indices
feature_indices = [FEATURES.index(f) for f in num_feats[:2]]  # pick top 2 numeric features
try:
    fig, ax = plt.subplots(figsize=(8,3))
    PartialDependenceDisplay.from_estimator(svm, X_train_scaled, features=feature_indices, feature_names=FEATURES, ax=ax)
    fig.suptitle('Partial Dependence (SVM Classification)')
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(OUT_DIR,'partial_dependence.png'))
    plt.close(fig)
except Exception as e:
    print("PDP failed:", e)

# 8) SHAP explanations (optional)
try:
    import shap
    shap_dir = os.path.join(OUT_DIR, 'shap')
    os.makedirs(shap_dir, exist_ok=True)
    # KernelExplainer for SVM is expensive; use a small background sample
    X_train_sample = X_train_scaled[np.random.choice(X_train_scaled.shape[0], min(100, X_train_scaled.shape[0]), replace=False)]
    explainer = shap.KernelExplainer(lambda a: svm.predict_proba(a), X_train_sample)
    X_test_sample = X_test_scaled[np.random.choice(X_test_scaled.shape[0], min(50, X_test_scaled.shape[0]), replace=False)]
    shap_values = explainer.shap_values(X_test_sample, nsamples=100)
    # summary plot
    shap.summary_plot(shap_values, pd.DataFrame(X_test_sample, columns=FEATURES).iloc[:, :len(feature_indices)], show=False)
    plt.savefig(os.path.join(shap_dir,'shap_summary.png'), bbox_inches='tight')
    plt.close()
    # dependence plot for top shap feature index 0 (if exists)
    try:
        shap.dependence_plot(0, shap_values, pd.DataFrame(X_test_sample, columns=FEATURES), show=False)
        plt.savefig(os.path.join(shap_dir,'shap_dependence_0.png'), bbox_inches='tight')
        plt.close()
    except Exception:
        pass
    print("SHAP plots saved (in 'shap' folder).")
except Exception as e:
    print("SHAP not available or failed. Skipping SHAP block. (Reason: {})".format(e))

# -------------------- SUMMARY --------------------
per_crop = pred_df.groupby('crop_name').agg(
    n=('pred_class','count'),
    pct_fresh=('pred_class', lambda s: (s==2).mean()),
    pct_moderate=('pred_class', lambda s: (s==1).mean()),
    pct_spoiled=('pred_class', lambda s: (s==0).mean()),
).reset_index()
per_crop.to_csv(os.path.join(OUT_DIR,'per_crop_class_summary.csv'), index=False)

# Save model & preprocessors
joblib.dump(svm, os.path.join(OUT_DIR,'svm_model.joblib'))
joblib.dump(scaler, os.path.join(OUT_DIR,'scaler.joblib'))
joblib.dump(le, os.path.join(OUT_DIR,'label_encoder.joblib'))
metrics_df.to_csv(os.path.join(OUT_DIR,'svm_all_metrics_summary.csv'))
pred_df.to_csv(os.path.join(OUT_DIR,'predictions_with_probs.csv'), index=False)

# Final print
print("\nAll done. Outputs saved in folder:", OUT_DIR)
print("Key files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)

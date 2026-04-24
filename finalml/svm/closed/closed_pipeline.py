"""
Closed pipeline: exact copy of the IoT VQI + SVM/GBC/XGB + ablation logic from unified_models,
but runs only on closed.csv and writes all outputs into the closed/ folder.
No imports from unified_models; all functions are copied here.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score
)
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import UndefinedMetricWarning



def _label_to_status(lbl: np.ndarray) -> np.ndarray:

    m = {0: "Not Fresh", 1: "Moderate", 2: "Fresh"}

    arr = np.asarray(lbl).astype(int)

    return np.vectorize(lambda x: m.get(int(x), "Unknown"))(arr)



def _print_freshness_distribution(y_true: np.ndarray, preds: Dict[str, Optional[np.ndarray]]) -> None:

    class_names = {2: "2-Fresh", 1: "1-Moderate", 0: "0-Not Fresh"}

    classes = [2, 1, 0]

    def _fmt(arr: np.ndarray) -> Dict[int, str]:

        a = np.asarray(arr).astype(int)

        total = int(len(a))

        out: Dict[int, str] = {}

        for c in classes:

            cnt = int(np.sum(a == c))

            pct = (cnt / total * 100.0) if total else 0.0

            out[c] = f"{cnt} ({pct:.1f}%)"

        return out



    rows = []

    actual = _fmt(y_true)

    rows.append({

        "Model": "Actual",

        class_names[2]: actual[2],

        class_names[1]: actual[1],

        class_names[0]: actual[0],

    })



    for name, y_pred in preds.items():

        if y_pred is None:

            continue

        d = _fmt(y_pred)

        rows.append({

            "Model": name,

            class_names[2]: d[2],

            class_names[1]: d[1],

            class_names[0]: d[0],

        })



    df = pd.DataFrame(rows, columns=["Model", class_names[2], class_names[1], class_names[0]])

    print("\nFreshness class distribution (counts and %):")

    print(df.to_string(index=False))

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ==== Paths scoped to the closed folder ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)
DIR_SVM = os.path.join(OUT_DIR, "svm"); os.makedirs(DIR_SVM, exist_ok=True)
DIR_GBC = os.path.join(OUT_DIR, "gbc"); os.makedirs(DIR_GBC, exist_ok=True)
DIR_COMB = os.path.join(OUT_DIR, "combined"); os.makedirs(DIR_COMB, exist_ok=True)
DIR_XGB = os.path.join(OUT_DIR, "xgb"); os.makedirs(DIR_XGB, exist_ok=True)

SVM_MODEL_PATH = os.path.join(OUT_DIR, "svm_cls_model.joblib")
GBC_MODEL_PATH = os.path.join(OUT_DIR, "gbc_fresh_model.joblib")
XGB_MODEL_PATH = os.path.join(OUT_DIR, "xgb_fresh_model.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(OUT_DIR, "label_encoder.joblib")
METRICS_JSON = os.path.join(OUT_DIR, "metrics.json")
ABLATION_CSV = os.path.join(OUT_DIR, "ablation_results.csv")
ABLATION_SVM_CSV = os.path.join(OUT_DIR, "ablation_svm.csv")
ABLATION_GBC_CSV = os.path.join(OUT_DIR, "ablation_gbc.csv")
ABLATION_COMBINED_CSV = os.path.join(OUT_DIR, "ablation_combined.csv")

RANDOM_STATE = 42

# ==== Copied functions from unified_models (IoT pipeline) ====
def compute_vqi_from_sensors(
    df: pd.DataFrame,
    voc_min: float, voc_max: float,
    t_min: float, t_opt: float, t_max: float,
    h_min: float, h_opt: float, h_max: float,
) -> pd.Series:
    d = df.copy()
    gas_den = max(voc_max - voc_min, 1e-9)
    gas_score = (voc_max - d["VOC"]) / gas_den
    t_den = max(t_max - t_min, 1e-9)
    temp_score = 1.0 - (d["Temperature"] - t_opt).abs() / t_den
    h_den = max(h_max - h_min, 1e-9)
    hum_score = 1.0 - (d["Humidity"] - h_opt).abs() / h_den
    gas_score = gas_score.clip(0.0, 1.0)
    temp_score = temp_score.clip(0.0, 1.0)
    hum_score = hum_score.clip(0.0, 1.0)
    vqi = (0.5 * gas_score + 0.3 * temp_score + 0.2 * hum_score) * 100.0
    return vqi.clip(0.0, 100.0)

def label_from_vqi(vqi: pd.Series) -> pd.Series:
    # Fresh=2, Moderate=1, Not Fresh=0
    # Use quantiles to split into three equal-sized groups
    q_high = vqi.quantile(0.6666)  # top 33.3% -> label 2 (Fresh)
    q_low = vqi.quantile(0.3333)   # bottom 33.3% -> label 0 (Not Fresh)
    # Assign labels: 2 for high VQI (Fresh), 1 for medium (Moderate), 0 for low (Not Fresh)
    labels = np.where(vqi > q_high, 2, np.where(vqi > q_low, 1, 0))
    return pd.Series(labels, index=vqi.index, dtype=int)

def train_models_new_iot(
    df: pd.DataFrame,
    voc_min: float, voc_max: float,
    t_min: float, t_opt: float, t_max: float,
    h_min: float, h_opt: float, h_max: float,
) -> Dict[str, float]:
    use_cols = ["VOC", "Temperature", "Humidity"]
    df2 = df[use_cols].dropna().copy()
    vqi = compute_vqi_from_sensors(df2, voc_min, voc_max, t_min, t_opt, t_max, h_min, h_opt, h_max)
    y = label_from_vqi(vqi)
    X = df2
    scored = df2.copy()
    scored["VQI"] = vqi.values
    scored["label"] = y.values

    scored["status"] = _label_to_status(y.values)

    scored.to_csv(os.path.join(OUT_DIR, "vqi_scored.csv"), index=False)
    # Print VQI table
    try:
        display_df = scored.copy()
        display_df.index = np.arange(1, len(display_df) + 1)
        display_df.index.name = "row"
        print("\n================ VQI scores per row (closed) ================")
        print(display_df.to_string())
        print("==========================================================")
    except Exception:
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )
    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]), use_cols)
    ])

    # SVM
    svm_pipe = Pipeline([
        ("prep", preproc),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE))
    ])
    svm_pipe.fit(X_train, y_train)
    y_pred_svm = svm_pipe.predict(X_test)
    y_proba_svm = svm_pipe.predict_proba(X_test)

    # GBC
    gbc_pipe = Pipeline([
        ("prep", preproc),
        ("gbc", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    gbc_pipe.fit(X_train, y_train)
    y_pred_gbc = gbc_pipe.predict(X_test)
    y_proba_gbc = gbc_pipe.predict_proba(X_test)

    # XGB
    y_pred_xgb = None; y_proba_xgb = None
    if HAS_XGB:
        xgb_pipe = Pipeline([
            ("prep", preproc),
            ("xgb", XGBClassifier(
                random_state=RANDOM_STATE, n_estimators=300, learning_rate=0.1, max_depth=6,
                subsample=1.0, colsample_bytree=1.0, eval_metric="logloss", n_jobs=1
            ))
        ])
        xgb_pipe.fit(X_train, y_train)
        y_pred_xgb = xgb_pipe.predict(X_test)
        try:
            y_proba_xgb = xgb_pipe.predict_proba(X_test)
        except Exception:
            y_proba_xgb = None

    # Plots (GBC)
    cm_g = confusion_matrix(y_test, y_pred_gbc)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_g, annot=True, fmt="d", cmap="Blues", xticklabels=["0","1","2"], yticklabels=["0","1","2"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (GBC)")
    plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_confusion_matrix.png"), dpi=200); plt.close()
    if y_proba_gbc is not None:
        valid_classes = [c for c in [0,1,2] if (y_test==c).any() and (y_test!=c).any()]
        if valid_classes:
            fig, ax = plt.subplots(figsize=(6,5))
            for c in valid_classes:
                RocCurveDisplay.from_predictions((y_test==c).astype(int), y_proba_gbc[:,c], name=f"Class {c}", ax=ax)
            plt.title("ROC Curves (GBC)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_roc.png"), dpi=200); plt.close()
            fig, ax = plt.subplots(figsize=(6,5))
            for c in valid_classes:
                PrecisionRecallDisplay.from_predictions((y_test==c).astype(int), y_proba_gbc[:,c], name=f"Class {c}", ax=ax)
            plt.title("Precision-Recall (GBC)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_pr.png"), dpi=200); plt.close()
            fig, axes = plt.subplots(1, len(valid_classes), figsize=(4*len(valid_classes),4))
            if len(valid_classes)==1:
                axes = [axes]
            for i,c in enumerate(valid_classes):
                CalibrationDisplay.from_predictions((y_test==c).astype(int), y_proba_gbc[:,c], n_bins=10, ax=axes[i])
                axes[i].set_title(f"Calibration: Class {c}")
            plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_calibration.png"), dpi=200); plt.close()

    # Plots (SVM)
    cm_s = confusion_matrix(y_test, y_pred_svm)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens", xticklabels=["0","1","2"], yticklabels=["0","1","2"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (SVM)")
    plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_confusion_matrix.png"), dpi=200); plt.close()
    if y_proba_svm is not None:
        valid_classes = [c for c in [0,1,2] if (y_test==c).any() and (y_test!=c).any()]
        if valid_classes:
            fig, ax = plt.subplots(figsize=(6,5))
            for c in valid_classes:
                RocCurveDisplay.from_predictions((y_test==c).astype(int), y_proba_svm[:,c], name=f"Class {c}", ax=ax)
            plt.title("ROC Curves (SVM)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_roc.png"), dpi=200); plt.close()
            fig, ax = plt.subplots(figsize=(6,5))
            for c in valid_classes:
                PrecisionRecallDisplay.from_predictions((y_test==c).astype(int), y_proba_svm[:,c], name=f"Class {c}", ax=ax)
            plt.title("Precision-Recall (SVM)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_pr.png"), dpi=200); plt.close()
            fig, axes = plt.subplots(1, len(valid_classes), figsize=(4*len(valid_classes),4))
            if len(valid_classes)==1:
                axes = [axes]
            for i,c in enumerate(valid_classes):
                CalibrationDisplay.from_predictions((y_test==c).astype(int), y_proba_svm[:,c], n_bins=10, ax=axes[i])
                axes[i].set_title(f"Calibration: Class {c}")
            plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_calibration.png"), dpi=200); plt.close()

    # Plots (XGB)
    if HAS_XGB and (y_pred_xgb is not None):
        cm_x = confusion_matrix(y_test, y_pred_xgb)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_x, annot=True, fmt="d", cmap="Purples", xticklabels=["0","1","2"], yticklabels=["0","1","2"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (XGB)")
        plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_confusion_matrix.png"), dpi=200); plt.close()
        if y_proba_xgb is not None:
            valid_classes = [c for c in [0,1,2] if (y_test==c).any() and (y_test!=c).any()]
            if valid_classes:
                fig, ax = plt.subplots(figsize=(6,5))
                for c in valid_classes:
                    RocCurveDisplay.from_predictions((y_test==c).astype(int), y_proba_xgb[:,c], name=f"Class {c}", ax=ax)
                plt.title("ROC Curves (XGB)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_roc.png"), dpi=200); plt.close()
                fig, ax = plt.subplots(figsize=(6,5))
                for c in valid_classes:
                    PrecisionRecallDisplay.from_predictions((y_test==c).astype(int), y_proba_xgb[:,c], name=f"Class {c}", ax=ax)
                plt.title("Precision-Recall (XGB)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_pr.png"), dpi=200); plt.close()
                fig, axes = plt.subplots(1, len(valid_classes), figsize=(4*len(valid_classes),4))
                if len(valid_classes)==1:
                    axes = [axes]
                for i,c in enumerate(valid_classes):
                    CalibrationDisplay.from_predictions((y_test==c).astype(int), y_proba_xgb[:,c], n_bins=10, ax=axes[i])
                    axes[i].set_title(f"Calibration: Class {c}")
                plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_calibration.png"), dpi=200); plt.close()

    # Metrics and combinations
    acc_svm = accuracy_score(y_test, y_pred_svm); f1_svm = f1_score(y_test, y_pred_svm, average="macro")
    pre_svm = precision_score(y_test, y_pred_svm, average="macro", zero_division=0)
    rec_svm = recall_score(y_test, y_pred_svm, average="macro", zero_division=0)
    
    acc_gbc = accuracy_score(y_test, y_pred_gbc); f1_gbc = f1_score(y_test, y_pred_gbc, average="macro")
    pre_gbc = precision_score(y_test, y_pred_gbc, average="macro", zero_division=0)
    rec_gbc = recall_score(y_test, y_pred_gbc, average="macro", zero_division=0)
    pred_sg = np.where(y_pred_svm == y_pred_gbc, y_pred_svm, y_pred_svm)
    acc_sg = accuracy_score(y_test, pred_sg); f1_sg = f1_score(y_test, pred_sg, average="macro")
    pre_sg = precision_score(y_test, pred_sg, average="macro", zero_division=0)
    rec_sg = recall_score(y_test, pred_sg, average="macro", zero_division=0)
    pred_sx = None; acc_sx = None; f1_sx = None
    pred_gx = None; acc_gx = None; f1_gx = None
    if HAS_XGB and (y_pred_xgb is not None):
        pred_sx = np.where(y_pred_svm == y_pred_xgb, y_pred_svm, y_pred_svm)
        acc_sx = accuracy_score(y_test, pred_sx); f1_sx = f1_score(y_test, pred_sx, average="macro")
        pre_sx = precision_score(y_test, pred_sx, average="macro", zero_division=0)
        rec_sx = recall_score(y_test, pred_sx, average="macro", zero_division=0)
        pred_gx = np.where(y_pred_gbc == y_pred_xgb, y_pred_gbc, y_pred_gbc)
        acc_gx = accuracy_score(y_test, pred_gx); f1_gx = f1_score(y_test, pred_gx, average="macro")
        pre_gx = precision_score(y_test, pred_gx, average="macro", zero_division=0)
        rec_gx = recall_score(y_test, pred_gx, average="macro", zero_division=0)
    vote_stack = [y_pred_svm, y_pred_gbc]
    if HAS_XGB and (y_pred_xgb is not None):
        vote_stack.append(y_pred_xgb)
    vote_arr = np.vstack(vote_stack)
    pred_comb3 = np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=0, arr=vote_arr)
    acc_comb3 = accuracy_score(y_test, pred_comb3)
    f1_comb3 = f1_score(y_test, pred_comb3, average="macro")
    pre_comb3 = precision_score(y_test, pred_comb3, average="macro", zero_division=0)
    rec_comb3 = recall_score(y_test, pred_comb3, average="macro", zero_division=0)

    metrics = {
        "svm_acc": float(acc_svm), "svm_precision_macro": float(pre_svm), "svm_recall_macro": float(rec_svm), "svm_f1_macro": float(f1_svm),
        "gbc_acc": float(acc_gbc), "gbc_precision_macro": float(pre_gbc), "gbc_recall_macro": float(rec_gbc), "gbc_f1_macro": float(f1_gbc),
        "comb_svm_gbc_acc": float(acc_sg), "comb_svm_gbc_precision_macro": float(pre_sg), "comb_svm_gbc_recall_macro": float(rec_sg), "comb_svm_gbc_f1_macro": float(f1_sg),
        "combined3_acc": float(acc_comb3), "combined3_precision_macro": float(pre_comb3), "combined3_recall_macro": float(rec_comb3), "combined3_f1_macro": float(f1_comb3),
    }
    if HAS_XGB and (y_pred_xgb is not None):
        metrics.update({
            "xgb_acc": float(accuracy_score(y_test, y_pred_xgb)),
            "xgb_precision_macro": float(precision_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
            "xgb_recall_macro": float(recall_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
            "xgb_f1_macro": float(f1_score(y_test, y_pred_xgb, average="macro")),
            "comb_svm_xgb_acc": float(acc_sx) if acc_sx is not None else None,
            "comb_svm_xgb_precision_macro": float(pre_sx) if acc_sx is not None else None,
            "comb_svm_xgb_recall_macro": float(rec_sx) if acc_sx is not None else None,
            "comb_svm_xgb_f1_macro": float(f1_sx) if acc_sx is not None else None,
            "comb_gbc_xgb_acc": float(acc_gx) if acc_gx is not None else None,
            "comb_gbc_xgb_precision_macro": float(pre_gx) if acc_gx is not None else None,
            "comb_gbc_xgb_recall_macro": float(rec_gx) if acc_gx is not None else None,
            "comb_gbc_xgb_f1_macro": float(f1_gx) if acc_gx is not None else None,
        })

    # Print baselines
    print("\nBaseline metrics (SVM):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (acc_svm, pre_svm, rec_svm, f1_svm))
    print("\nBaseline metrics (GBC):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (acc_gbc, pre_gbc, rec_gbc, f1_gbc))
    if HAS_XGB and (y_pred_xgb is not None):
        print("\nBaseline metrics (XGB):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (
            accuracy_score(y_test, y_pred_xgb),
            precision_score(y_test, y_pred_xgb, average="macro", zero_division=0),
            recall_score(y_test, y_pred_xgb, average="macro", zero_division=0),
            f1_score(y_test, y_pred_xgb, average="macro"),
        ))
    print("\nBaseline metrics (SVM+GBC):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (acc_sg, pre_sg, rec_sg, f1_sg))
    if HAS_XGB and (y_pred_xgb is not None):
        print("\nBaseline metrics (SVM+XGB):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (acc_sx, pre_sx, rec_sx, f1_sx))
        print("\nBaseline metrics (GBC+XGB):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (acc_gx, pre_gx, rec_gx, f1_gx))
    print("\nBaseline metrics (SVM+GBC+XGB):\n - acc: %.4f\n - precision_macro: %.4f\n - recall_macro: %.4f\n - f1_macro: %.4f" % (acc_comb3, pre_comb3, rec_comb3, f1_comb3))



    _print_freshness_distribution(

        y_test,

        {

            "SVM": y_pred_svm,

            "GBC": y_pred_gbc,

            "XGB": y_pred_xgb if HAS_XGB else None,

            "SVM+GBC": pred_sg,

            "SVM+XGB": pred_sx,

            "GBC+XGB": pred_gx,

            "SVM+GBC+XGB": pred_comb3,

        },

    )

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Fusion outputs and plots for combinations (same as unified)
    weights = np.array([3.0, 6.5, 10.0])
    def _fresh(p: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if p is None: return None
        return p @ weights
    fresh_svm = _fresh(y_proba_svm)
    fresh_gbc = _fresh(y_proba_gbc)
    fresh_xgb = _fresh(y_proba_xgb) if (HAS_XGB and y_proba_xgb is not None) else None
    def _save_fusion(tag: str, pred_cls: np.ndarray, fresh_score: Optional[np.ndarray]):
        if fresh_score is None:
            class_to_weight = np.array([3.0, 6.5, 10.0])
            fresh_score = class_to_weight[pred_cls]
        vqi_clip = np.clip(vqi.loc[X_test.index].values, 0.0, 100.0)
        alpha_def, beta_def, base_def = 0.2, 0.6, 50.0
        final_price = base_def * (1.0 + alpha_def * (fresh_score / 10.0) + beta_def * (vqi_clip / 100.0))
        df_out = X_test.copy()
        df_out["actual_vqi"] = vqi_clip

        df_out["actual_status"] = _label_to_status(y_test.values)

        df_out["pred_class"] = pred_cls

        df_out["pred_status"] = _label_to_status(pred_cls)

        df_out["freshness_score"] = fresh_score
        df_out["final_price"] = final_price
        out_csv = os.path.join(DIR_COMB, f"fusion_{tag}.csv")
        df_out.to_csv(out_csv, index=False)
        plt.figure(figsize=(6,5)); plt.scatter(vqi_clip, fresh_score, alpha=0.7)
        plt.xlabel('Actual VQI'); plt.ylabel('Freshness Score (0-10)')
        plt.title(f'Fusion: VQI vs Freshness ({tag.upper()})')
        plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB, f'fusion_vqi_vs_freshness_{tag}.png'), dpi=200); plt.close()
        plt.figure(figsize=(6,4.5)); sns.histplot(final_price, kde=True)
        plt.xlabel('Final Smart Price'); plt.title(f'Fusion: Final Price Distribution ({tag.upper()})')
        plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB, f'fusion_price_distribution_{tag}.png'), dpi=200); plt.close()
        tmp = pd.DataFrame({"vqi": vqi_clip, "cls": pred_cls})
        plt.figure(figsize=(6,4)); sns.boxplot(x="cls", y="vqi", data=tmp)
        plt.xlabel('Predicted Class (0/1/2)'); plt.ylabel('Actual VQI')
        plt.title(f'Fusion: VQI by Predicted Class ({tag.upper()})')
        plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB, f'fusion_vqi_by_class_{tag}.png'), dpi=200); plt.close()
    _save_fusion("svm", y_pred_svm, fresh_svm)
    _save_fusion("gbc", y_pred_gbc, fresh_gbc)
    if fresh_xgb is not None and y_pred_xgb is not None:
        _save_fusion("xgb", y_pred_xgb, fresh_xgb)
    if pred_sg is not None:
        mean_fg = None if (fresh_svm is None or fresh_gbc is None) else (fresh_svm + fresh_gbc) / 2.0
        _save_fusion("comb_svm_gbc", pred_sg, mean_fg)
    if pred_sx is not None:
        mean_fx = None if (fresh_svm is None or fresh_xgb is None) else (fresh_svm + fresh_xgb) / 2.0
        _save_fusion("comb_svm_xgb", pred_sx, mean_fx)
    if pred_gx is not None:
        mean_gx = None if (fresh_gbc is None or fresh_xgb is None) else (fresh_gbc + fresh_xgb) / 2.0
        _save_fusion("comb_gbc_xgb", pred_gx, mean_gx)
    fres_list = [f for f in [fresh_svm, fresh_gbc, fresh_xgb] if f is not None]
    if len(fres_list) >= 2:
        fres_mean = np.mean(np.vstack(fres_list), axis=0)
        _save_fusion("combined3", pred_comb3, fres_mean)

    # Ablation settings (same as unified)
    ab_configs: List[Tuple[str, List[str]]] = [
        ("full", ["VOC", "Temperature", "Humidity"]),
        ("w/o_VOC", ["Temperature", "Humidity"]),
        ("w/o_Temperature", ["VOC", "Humidity"]),
        ("w/o_Humidity", ["VOC", "Temperature"]),
        ("VOC_only", ["VOC"]),
        ("Temperature_only", ["Temperature"]),
        ("Humidity_only", ["Humidity"]),
    ]
    rows = []
    for name, feats in ab_configs:
        Xf = df2[feats]
        Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True, stratify=y)
        pre = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]), feats)
        ])
        # SVM
        svm = Pipeline([("prep", pre), ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE))])
        svm.fit(Xtr, ytr); ps = svm.predict(Xte)
        rows.append({"model": "SVM", "setting": name, "features_used": ",".join(feats),
                     "acc": accuracy_score(yte, ps),
                     "precision_macro": precision_score(yte, ps, average="macro", zero_division=0),
                     "recall_macro": recall_score(yte, ps, average="macro", zero_division=0),
                     "f1_macro": f1_score(yte, ps, average="macro", zero_division=0)})
        # GBC
        gbc = Pipeline([("prep", pre), ("gbc", GradientBoostingClassifier(random_state=RANDOM_STATE))])
        gbc.fit(Xtr, ytr); pg = gbc.predict(Xte)
        rows.append({"model": "GBC", "setting": name, "features_used": ",".join(feats),
                     "acc": accuracy_score(yte, pg),
                     "precision_macro": precision_score(yte, pg, average="macro", zero_division=0),
                     "recall_macro": recall_score(yte, pg, average="macro", zero_division=0),
                     "f1_macro": f1_score(yte, pg, average="macro", zero_division=0)})
        # XGB
        px = None
        if HAS_XGB:
            xgb = Pipeline([("prep", pre), ("xgb", XGBClassifier(random_state=RANDOM_STATE, n_estimators=300, learning_rate=0.1, max_depth=6, subsample=1.0, colsample_bytree=1.0, eval_metric="logloss", n_jobs=1))])
            xgb.fit(Xtr, ytr); px = xgb.predict(Xte)
            rows.append({"model": "XGB", "setting": name, "features_used": ",".join(feats),
                         "acc": accuracy_score(yte, px),
                         "precision_macro": precision_score(yte, px, average="macro", zero_division=0),
                         "recall_macro": recall_score(yte, px, average="macro", zero_division=0),
                         "f1_macro": f1_score(yte, px, average="macro", zero_division=0)})
        # Pairwise combinations per-setting
        pred_sg_ab = np.where(ps == pg, ps, ps)
        rows.append({"model": "COMB_SG", "setting": name, "features_used": ",".join(feats),
                     "acc": accuracy_score(yte, pred_sg_ab),
                     "precision_macro": precision_score(yte, pred_sg_ab, average="macro", zero_division=0),
                     "recall_macro": recall_score(yte, pred_sg_ab, average="macro", zero_division=0),
                     "f1_macro": f1_score(yte, pred_sg_ab, average="macro", zero_division=0)})
        if px is not None:
            pred_sx_ab = np.where(ps == px, ps, ps)
            rows.append({"model": "COMB_SX", "setting": name, "features_used": ",".join(feats),
                         "acc": accuracy_score(yte, pred_sx_ab),
                         "precision_macro": precision_score(yte, pred_sx_ab, average="macro", zero_division=0),
                         "recall_macro": recall_score(yte, pred_sx_ab, average="macro", zero_division=0),
                         "f1_macro": f1_score(yte, pred_sx_ab, average="macro", zero_division=0)})
            pred_gx_ab = np.where(pg == px, pg, pg)
            rows.append({"model": "COMB_GX", "setting": name, "features_used": ",".join(feats),
                         "acc": accuracy_score(yte, pred_gx_ab),
                         "precision_macro": precision_score(yte, pred_gx_ab, average="macro", zero_division=0),
                         "recall_macro": recall_score(yte, pred_gx_ab, average="macro", zero_division=0),
                         "f1_macro": f1_score(yte, pred_gx_ab, average="macro", zero_division=0)})
        # Combined-3 per-setting via majority vote
        vote_list = [ps, pg] + ([px] if px is not None else [])
        vote_mat = np.vstack(vote_list)
        pred_c3 = np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=0, arr=vote_mat)
        rows.append({"model": "COMBINED3", "setting": name, "features_used": ",".join(feats),
                     "acc": accuracy_score(yte, pred_c3),
                     "precision_macro": precision_score(yte, pred_c3, average="macro", zero_division=0),
                     "recall_macro": recall_score(yte, pred_c3, average="macro", zero_division=0),
                     "f1_macro": f1_score(yte, pred_c3, average="macro", zero_division=0)})

    ab_df = pd.DataFrame(rows)
    ab_df.to_csv(ABLATION_CSV, index=False)
    # Print ablations to terminal by model (same as unified)
    print("\n===== ABLATION RESULTS (by model/combination) =====")
    order = ["SVM", "GBC"]
    if HAS_XGB:
        order.append("XGB")
    order += ["COMB_SG"]
    if HAS_XGB:
        order += ["COMB_SX", "COMB_GX"]
    order += ["COMBINED3"]
    with pd.option_context('display.max_columns', None, 'display.width', 220, 'display.max_rows', None):
        for mdl in order:
            sub = ab_df[ab_df["model"] == mdl]
            if not sub.empty:
                print("\n" + "="*60)
                print(f"{mdl} ABLATION RESULTS:")
                print("="*60)
                print(sub.to_string(index=False))
    # Save per-model ablations
    try:
        ab_svm = ab_df[ab_df["model"] == "SVM"]
        if not ab_svm.empty:
            ab_svm.to_csv(os.path.join(OUT_DIR, "ablation_svm.csv"), index=False)
        ab_gbc = ab_df[ab_df["model"] == "GBC"]
        if not ab_gbc.empty:
            ab_gbc.to_csv(os.path.join(OUT_DIR, "ablation_gbc.csv"), index=False)
        if HAS_XGB and ("XGB" in ab_df["model"].unique()):
            ab_xgb = ab_df[ab_df["model"] == "XGB"]
            if not ab_xgb.empty:
                ab_xgb.to_csv(os.path.join(OUT_DIR, "ablation_xgb.csv"), index=False)
        ab_sg = ab_df[ab_df["model"] == "COMB_SG"]
        if not ab_sg.empty:
            ab_sg.to_csv(os.path.join(OUT_DIR, "ablation_comb_svm_gbc.csv"), index=False)
        if HAS_XGB:
            ab_sx = ab_df[ab_df["model"] == "COMB_SX"]
            if not ab_sx.empty:
                ab_sx.to_csv(os.path.join(OUT_DIR, "ablation_comb_svm_xgb.csv"), index=False)
            ab_gx = ab_df[ab_df["model"] == "COMB_GX"]
            if not ab_gx.empty:
                ab_gx.to_csv(os.path.join(OUT_DIR, "ablation_comb_gbc_xgb.csv"), index=False)
        ab_c3 = ab_df[ab_df["model"] == "COMBINED3"]
        if not ab_c3.empty:
            ab_c3.to_csv(os.path.join(OUT_DIR, "ablation_combined3.csv"), index=False)
    except Exception:
        pass
    return metrics


def main() -> None:
    csv_path = os.path.join(BASE_DIR, "closed.csv")
    df = pd.read_csv(csv_path)
    # Accept exact column names from closed.csv
    if set(["voc(ppm)", "temperature", "humidity"]).issubset(df.columns):
        # Rename to expected names for pipeline functions
        df = df.rename(columns={"voc(ppm)": "VOC", "temperature": "Temperature", "humidity": "Humidity"})
        voc_min = float(df["VOC"].min()); voc_max = float(df["VOC"].max())
        t_min = float(df["Temperature"].min()); t_max = float(df["Temperature"].max()); t_opt = float(df["Temperature"].median())
        h_min = float(df["Humidity"].min()); h_max = float(df["Humidity"].max()); h_opt = float(df["Humidity"].median())
        metrics = train_models_new_iot(df, voc_min, voc_max, t_min, t_opt, t_max, h_min, h_opt, h_max)
        print("\nClosed pipeline complete.")
        print(f"Outputs folder: {OUT_DIR}")
        print(f"Ablation CSV:   {ABLATION_CSV}")
        print(f"Metrics JSON:   {METRICS_JSON}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        raise KeyError("closed.csv must contain columns: voc(ppm), temperature, humidity.")


if __name__ == "__main__":
    main()

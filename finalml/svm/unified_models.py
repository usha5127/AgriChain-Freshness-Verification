import os

import json

import numpy as np

import pandas as pd

import joblib

from typing import Dict, List, Optional, Tuple

import argparse



def _print_freshness_table(y_true: np.ndarray, preds: Dict[str, Optional[np.ndarray]]) -> None:

    class_names = {0: "0-Not Fresh", 1: "1-Moderate", 2: "2-Fresh"}

    classes = [2, 1, 0]

    def _fmt_counts(arr: np.ndarray) -> Dict[int, str]:

        total = int(len(arr))

        counts = {c: int(np.sum(arr == c)) for c in classes}

        out = {}

        for c in classes:

            pct = (counts[c] / total * 100.0) if total else 0.0

            out[c] = f"{counts[c]} ({pct:.1f}%)"

        return out



    rows = []

    y_true_np = np.asarray(y_true)

    actual = _fmt_counts(y_true_np)

    rows.append({

        "Model": "Actual",

        class_names[2]: actual[2],

        class_names[1]: actual[1],

        class_names[0]: actual[0],

    })



    for name, y_pred in preds.items():

        if y_pred is None:

            continue

        y_pred_np = np.asarray(y_pred)

        d = _fmt_counts(y_pred_np)

        rows.append({

            "Model": name,

            class_names[2]: d[2],

            class_names[1]: d[1],

            class_names[0]: d[0],

        })



    df = pd.DataFrame(rows, columns=["Model", class_names[2], class_names[1], class_names[0]])

    print("\nFreshness class distribution (counts and %):")

    print(df.to_string(index=False))



def _label_to_status(lbl: np.ndarray) -> np.ndarray:

    m = {0: "Not Fresh", 1: "Moderate", 2: "Fresh"}

    arr = np.asarray(lbl).astype(int)

    return np.vectorize(lambda x: m.get(int(x), "Unknown"))(arr)



from sklearn.model_selection import train_test_split, learning_curve

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer 

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (

    mean_squared_error, mean_absolute_error, r2_score,

    accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score

)

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from sklearn.calibration import CalibrationDisplay

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from statsmodels.graphics.gofplots import qqplot

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.exceptions import UndefinedMetricWarning

try:

    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True

except Exception:

    HAS_XGB = False



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(BASE_DIR, "unified_outputs")

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



# Expected columns in dataset

FEATURES = [

    "crop",           # text

    "temp_c",

    "humidity",

    "gas_ppm",       # VOC proxy

    "ethylene_ppm",

    "hours_since_harvest",

]

TARGET_VQI = "vqi"



# Optional feature that might not exist

OPTIONAL_CO2 = "co2_ppm"





def _derive_fresh_label(vqi: pd.Series) -> pd.Series:

    # 0: Spoiled, 1: Moderate, 2: Fresh

    # Robust binning: use dataset tertiles to avoid single-class issues when VQI is on a narrow scale.

    q = vqi.quantile([0.3333, 0.6666]).values

    # Ensure monotonic thresholds (handle duplicates by nudging)

    lo, hi = float(q[0]), float(q[1])

    if hi <= lo:

        hi = lo + 1e-6

    bins = [-np.inf, lo, hi, np.inf]

    return pd.cut(vqi, bins=bins, labels=[0, 1, 2]).astype(int)





def load_dataset(csv_path: str) -> pd.DataFrame:

    if not os.path.exists(csv_path):

        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Support both legacy dataset and new IoT dataset (VOC/Temperature/Humidity)

    if set(["VOC", "Temperature", "Humidity"]).issubset(df.columns):

        return df

    missing = [c for c in [*FEATURES, TARGET_VQI] if c not in df.columns]

    if missing:

        raise KeyError(f"Missing columns in dataset: {missing}")

    return df





def train_models(df: pd.DataFrame) -> Dict[str, float]:

    enc = LabelEncoder()

    df = df.copy()

    df["crop_encoded"] = enc.fit_transform(df["crop"].astype(str))



    numeric_feats = [

        "crop_encoded", "temp_c", "humidity", "gas_ppm", "ethylene_ppm", "hours_since_harvest"

    ]

    # Include optional CO2 if present

    if OPTIONAL_CO2 in df.columns:

        numeric_feats.append(OPTIONAL_CO2)



    X = df[numeric_feats]

    y_vqi = df[TARGET_VQI]

    y_cls = _derive_fresh_label(y_vqi)



    X_train, X_test, y_train_cls, y_test_cls = train_test_split(

        X, y_cls, test_size=0.2, random_state=RANDOM_STATE, shuffle=True

    )

    # Keep aligned VQI values for fusion/price analysis

    y_train_vqi = y_vqi.loc[X_train.index]

    y_test_vqi = y_vqi.loc[X_test.index]



    preproc = ColumnTransformer([

        ("num", Pipeline([

            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler())

        ]), numeric_feats)

    ])



    # SVM classifier pipeline

    svm_pipe = Pipeline([

        ("prep", preproc),

        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE))

    ])

    svm_pipe.fit(X_train, y_train_cls)

    y_pred_svm = svm_pipe.predict(X_test)

    y_proba_svm = None

    if hasattr(svm_pipe, "predict_proba"):

        try:

            y_proba_svm = svm_pipe.predict_proba(X_test)

        except Exception:

            y_proba_svm = None

    acc_svm = accuracy_score(y_test_cls, y_pred_svm)

    f1_svm = f1_score(y_test_cls, y_pred_svm, average="macro")



    # GBC pipeline

    gbc_pipe = Pipeline([

        ("prep", preproc),

        ("gbc", GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
        ))

    ])

    gbc_pipe.fit(X_train, y_train_cls)

    y_pred_cls = gbc_pipe.predict(X_test)

    y_proba = None

    if hasattr(gbc_pipe, "predict_proba"):

        y_proba = gbc_pipe.predict_proba(X_test)



    acc = accuracy_score(y_test_cls, y_pred_cls)

    f1 = f1_score(y_test_cls, y_pred_cls, average="macro")



    # XGBoost pipeline (optional)

    y_pred_xgb = None

    y_proba_xgb = None

    acc_xgb = None

    f1_xgb = None

    if HAS_XGB:

        xgb_pipe = Pipeline([

            ("prep", preproc),

            ("xgb", XGBClassifier(

                random_state=RANDOM_STATE,

                n_estimators=120,

                learning_rate=0.05,

                max_depth=3,

                subsample=0.8,

                colsample_bytree=0.8,

                reg_lambda=2.0,

                min_child_weight=2.0,

                eval_metric="logloss",

                n_jobs=1,

            ))

        ])

        xgb_pipe.fit(X_train, y_train_cls)

        y_pred_xgb = xgb_pipe.predict(X_test)

        try:

            y_proba_xgb = xgb_pipe.predict_proba(X_test)

        except Exception:

            y_proba_xgb = None

        acc_xgb = accuracy_score(y_test_cls, y_pred_xgb)

        f1_xgb = f1_score(y_test_cls, y_pred_xgb, average="macro")



    # Save artifacts

    joblib.dump(svm_pipe, SVM_MODEL_PATH)

    joblib.dump(gbc_pipe, GBC_MODEL_PATH)

    joblib.dump(enc, LABEL_ENCODER_PATH)

    if HAS_XGB:

        joblib.dump(xgb_pipe, XGB_MODEL_PATH)



    # ---------------- Plots: GBC ----------------

    cm = confusion_matrix(y_test_cls, y_pred_cls)

    plt.figure(figsize=(5,4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",

                xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])

    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (GBC)")

    plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_confusion_matrix.png"), dpi=200); plt.close()



    if y_proba is not None:

        # ROC/PR curves

        fig, ax = plt.subplots(figsize=(6,5))

        for c in [0,1,2]:

            RocCurveDisplay.from_predictions((y_test_cls==c).astype(int), y_proba[:,c], name=f"Class {c}", ax=ax)

        plt.title("ROC Curves (OvR, GBC)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_roc_curves.png"), dpi=200); plt.close()



        fig, ax = plt.subplots(figsize=(6,5))

        for c in [0,1,2]:

            PrecisionRecallDisplay.from_predictions((y_test_cls==c).astype(int), y_proba[:,c], name=f"Class {c}", ax=ax)

        plt.title("Precision–Recall Curves (OvR, GBC)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_pr_curves.png"), dpi=200); plt.close()



        # Calibration plots

        fig, axes = plt.subplots(1,3, figsize=(12,4))

        for i,c in enumerate([0,1,2]):

            CalibrationDisplay.from_predictions((y_test_cls==c).astype(int), y_proba[:,c], n_bins=10, ax=axes[i])

            axes[i].set_title(f"Calibration: Class {c}")

        plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_calibration.png"), dpi=200); plt.close()



    # Permutation importance for GBC

    r_cls = permutation_importance(gbc_pipe, X_test, y_test_cls, n_repeats=5, random_state=RANDOM_STATE)

    imp_df_cls = pd.DataFrame({"feature": X.columns, "importance_mean": r_cls.importances_mean, "importance_std": r_cls.importances_std}).sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(6,4)); plt.barh(imp_df_cls["feature"], imp_df_cls["importance_mean"]); plt.gca().invert_yaxis(); plt.xlabel("Permutation Importance"); plt.title("Feature Importance (GBC)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_feature_importance.png"), dpi=200); plt.close()



    # ---------------- Plots: SVM (classification) ----------------

    cm_svm = confusion_matrix(y_test_cls, y_pred_svm)

    plt.figure(figsize=(5,4))

    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens",

                xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])

    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (SVM)")

    plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_confusion_matrix.png"), dpi=200); plt.close()



    if y_proba_svm is not None:

        fig, ax = plt.subplots(figsize=(6,5))

        for c in [0,1,2]:

            RocCurveDisplay.from_predictions((y_test_cls==c).astype(int), y_proba_svm[:,c], name=f"Class {c}", ax=ax)

        plt.title("ROC Curves (OvR, SVM)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_roc_curves.png"), dpi=200); plt.close()



        fig, ax = plt.subplots(figsize=(6,5))

        for c in [0,1,2]:

            PrecisionRecallDisplay.from_predictions((y_test_cls==c).astype(int), y_proba_svm[:,c], name=f"Class {c}", ax=ax)

        plt.title("Precision–Recall Curves (OvR, SVM)"); plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_pr_curves.png"), dpi=200); plt.close()



        fig, axes = plt.subplots(1,3, figsize=(12,4))

        for i,c in enumerate([0,1,2]):

            CalibrationDisplay.from_predictions((y_test_cls==c).astype(int), y_proba_svm[:,c], n_bins=10, ax=axes[i])

            axes[i].set_title(f"Calibration: Class {c}")

        plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM, "svm_calibration.png"), dpi=200); plt.close()



    # Permutation importance for SVM

    r_svm = permutation_importance(svm_pipe, X_test, y_test_cls, n_repeats=10, random_state=RANDOM_STATE)

    imp_df_svm = pd.DataFrame({"feature": X.columns, "importance_mean": r_svm.importances_mean, "importance_std": r_svm.importances_std}).sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(6,4)); plt.barh(imp_df_svm["feature"], imp_df_svm["importance_mean"]); plt.gca().invert_yaxis(); plt.xlabel('Permutation Importance'); plt.title('Feature Importance (SVM)'); plt.tight_layout(); plt.savefig(os.path.join(DIR_SVM,'svm_permutation_importance.png')); plt.close()



    # Plots for XGB (if available)

    if HAS_XGB and (y_pred_xgb is not None):

        cmx = confusion_matrix(y_test_cls, y_pred_xgb)

        plt.figure(figsize=(5,4))

        sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues",

                    xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])

        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (XGB)")

        plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_confusion_matrix.png"), dpi=200); plt.close()



        if y_proba_xgb is not None:

            fig, ax = plt.subplots(figsize=(6,5))

            for c in [0,1,2]:

                RocCurveDisplay.from_predictions((y_test_cls==c).astype(int), y_proba_xgb[:,c], name=f"Class {c}", ax=ax)

            ax.set_title("ROC Curves (XGB)")

            plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_roc.png"), dpi=200); plt.close()



            fig, ax = plt.subplots(figsize=(6,5))

            for c in [0,1,2]:

                PrecisionRecallDisplay.from_predictions((y_test_cls==c).astype(int), y_proba_xgb[:,c], name=f"Class {c}", ax=ax)

            ax.set_title("Precision-Recall (XGB)")

            plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_pr.png"), dpi=200); plt.close()



            fig, axes = plt.subplots(1, 3, figsize=(12,4))

            for i, c in enumerate([0,1,2]):

                CalibrationDisplay.from_predictions((y_test_cls==c).astype(int), y_proba_xgb[:,c], n_bins=10, ax=axes[i])

                axes[i].set_title(f"Calibration: Class {c}")

            plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB, "xgb_calibration.png"), dpi=200); plt.close()



        r_xgb = permutation_importance(xgb_pipe, X_test, y_test_cls, n_repeats=10, random_state=RANDOM_STATE)

        imp_df_xgb = pd.DataFrame({"feature": X.columns, "importance_mean": r_xgb.importances_mean, "importance_std": r_xgb.importances_std}).sort_values("importance_mean", ascending=False)

        plt.figure(figsize=(6,4)); plt.barh(imp_df_xgb["feature"], imp_df_xgb["importance_mean"]); plt.gca().invert_yaxis(); plt.xlabel('Permutation Importance'); plt.title('Feature Importance (XGB)'); plt.tight_layout(); plt.savefig(os.path.join(DIR_XGB,'xgb_permutation_importance.png')); plt.close()



    # Correlation heatmap (features + VQI)

    corr_df = pd.concat([X, y_vqi], axis=1).corr()

    plt.figure(figsize=(7,6)); sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', square=True); plt.title('Correlation Matrix (features + VQI)'); plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB,'corr_heatmap.png')); plt.close()



    # ---------------- Fusion Outputs: SVM, GBC, Combined ----------------

    weights = np.array([3.0, 6.5, 10.0])

    def _fresh_score_from_proba(proba: Optional[np.ndarray]) -> Optional[np.ndarray]:

        if proba is None:

            return None

        return proba @ weights



    fresh_svm = _fresh_score_from_proba(y_proba_svm)

    fresh_gbc = _fresh_score_from_proba(y_proba)

    fresh_xgb = _fresh_score_from_proba(y_proba_xgb) if HAS_XGB else None

    fresh_comb = None

    if (fresh_svm is not None) and (fresh_gbc is not None):

        fresh_comb = (fresh_svm + fresh_gbc) / 2.0



    # Combined3: average across all available models (SVM, GBC, XGB)

    fresh_candidates = [fs for fs in [fresh_svm, fresh_gbc, fresh_xgb] if fs is not None]

    fresh_comb3 = None

    if len(fresh_candidates) >= 2:

        fresh_comb3 = np.mean(np.vstack(fresh_candidates), axis=0)



    def _save_fusion(tag: str, pred_cls: np.ndarray, proba: Optional[np.ndarray], fresh_score: Optional[np.ndarray]):

        if fresh_score is None:

            return

        vqi_clip = np.clip(y_test_vqi.values, 0.0, 100.0)

        alpha_def, beta_def, base_def = 0.2, 0.6, 50.0

        final_price = base_def * (1.0 + alpha_def * (fresh_score / 10.0) + beta_def * (vqi_clip / 100.0))



        df_out = X_test.copy()

        df_out["actual_vqi"] = y_test_vqi.values

        df_out["actual_status"] = _label_to_status(y_test_cls.values)

        df_out["pred_class"] = pred_cls

        df_out["pred_status"] = _label_to_status(pred_cls)

        if proba is not None:

            for i in range(min(3, proba.shape[1])):

                df_out[f"proba_{i}"] = proba[:, i]

        df_out["freshness_score"] = fresh_score

        df_out["final_price"] = final_price

        out_csv = os.path.join(DIR_COMB, f"fusion_{tag}.csv")

        df_out.to_csv(out_csv, index=False)



        # Plots

        plt.figure(figsize=(6,5))

        plt.scatter(vqi_clip, fresh_score, alpha=0.7)

        plt.xlabel('Actual VQI'); plt.ylabel('Freshness Score (0-10)')

        plt.title(f'Fusion: VQI vs Freshness ({tag.upper()})')

        plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB, f'fusion_vqi_vs_freshness_{tag}.png'), dpi=200); plt.close()



        plt.figure(figsize=(6,4.5))

        sns.histplot(final_price, kde=True)

        plt.xlabel('Final Smart Price'); plt.title(f'Fusion: Final Price Distribution ({tag.upper()})')

        plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB, f'fusion_price_distribution_{tag}.png'), dpi=200); plt.close()



        tmp = pd.DataFrame({"vqi": vqi_clip, "cls": pred_cls})

        plt.figure(figsize=(6,4))

        sns.boxplot(x="cls", y="vqi", data=tmp)

        plt.xlabel('Predicted Class (0/1/2)'); plt.ylabel('Actual VQI')

        plt.title(f'Fusion: VQI by Predicted Class ({tag.upper()})')

        plt.tight_layout(); plt.savefig(os.path.join(DIR_COMB, f'fusion_vqi_by_class_{tag}.png'), dpi=200); plt.close()



    if y_proba_svm is not None:

        _save_fusion("svm", y_pred_svm, y_proba_svm, fresh_svm)

    if y_proba is not None:

        _save_fusion("gbc", y_pred_cls, y_proba, fresh_gbc)

    if HAS_XGB and (y_proba_xgb is not None):

        _save_fusion("xgb", y_pred_xgb, y_proba_xgb, fresh_xgb)

    if fresh_comb is not None:

        # For combined, use majority between SVM and GBC predictions for class column

        pred_majority = np.where(y_pred_svm == y_pred_cls, y_pred_svm, y_pred_svm)  # tie-breaker to SVM

        _save_fusion("combined", pred_majority, None, fresh_comb)



    # Save combined3 if available: majority vote among available predictions

    if fresh_comb3 is not None:

        votes = []

        if y_pred_svm is not None:

            votes.append(y_pred_svm)

        if y_pred_cls is not None:

            votes.append(y_pred_cls)

        if HAS_XGB and (y_pred_xgb is not None):

            votes.append(y_pred_xgb)

        if votes:

            # simple majority vote per sample

            votes_arr = np.vstack(votes)

            # mode across rows

            pred_maj3 = np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=0, arr=votes_arr)

            _save_fusion("combined3", pred_maj3, None, fresh_comb3)



    metrics = {

        "svm_acc": float(acc_svm),

        "svm_f1_macro": float(f1_svm),

        "gbc_acc": float(acc),

        "gbc_f1_macro": float(f1)

    }

    if HAS_XGB and (acc_xgb is not None) and (f1_xgb is not None):

        metrics.update({

            "xgb_acc": float(acc_xgb),

            "xgb_f1_macro": float(f1_xgb)

        })

    # Print baseline metrics to terminal

    print("\nBaseline metrics (SVM):")

    print(f" - acc: {acc_svm:.4f}\n - f1_macro: {f1_svm:.4f}")

    print("\nBaseline metrics (GBC):")

    print(f" - acc: {acc:.4f}\n - f1_macro: {f1:.4f}")

    if HAS_XGB and (y_pred_xgb is not None):

        print("\nBaseline metrics (XGB):")

        print(f" - acc: {metrics['xgb_acc']:.4f}\n - f1_macro: {metrics['xgb_f1_macro']:.4f}")

    print("\nBaseline metrics (Combined-3 majority):")

    print(f" - acc: {acc_comb3:.4f}\n - f1_macro: {f1_comb3:.4f}")



    _print_freshness_table(

        y_test_cls,

        {

            "SVM": y_pred_svm,

            "GBC": y_pred_cls,

            "XGB": y_pred_xgb if HAS_XGB else None,

            "Combined-3": pred_maj3 if (fresh_comb3 is not None and 'pred_maj3' in locals()) else None,

        },

    )



    with open(METRICS_JSON, "w", encoding="utf-8") as f:

        json.dump(metrics, f, indent=2)


    return metrics





# ================= New IoT dataset pipeline (VOC, Temperature, Humidity) =================

def compute_vqi_from_sensors(

    df: pd.DataFrame,

    voc_min: float, voc_max: float,

    t_min: float, t_opt: float, t_max: float,

    h_min: float, h_opt: float, h_max: float,

) -> pd.Series:

    d = df.copy()

    # Gas Score: higher VOC is worse → inverse normalize

    gas_den = max(voc_max - voc_min, 1e-9)

    gas_score = (voc_max - d["VOC"]) / gas_den

    # Temperature score: 1 - |T - T_opt| / (T_max - T_min)

    t_den = max(t_max - t_min, 1e-9)

    temp_score = 1.0 - (d["Temperature"] - t_opt).abs() / t_den

    # Humidity score: 1 - |H - H_opt| / (H_max - H_min)

    h_den = max(h_max - h_min, 1e-9)

    hum_score = 1.0 - (d["Humidity"] - h_opt).abs() / h_den

    # Clip to [0,1]

    gas_score = gas_score.clip(0.0, 1.0)

    temp_score = temp_score.clip(0.0, 1.0)

    hum_score = hum_score.clip(0.0, 1.0)

    # Weighted VQI in [0,100]

    vqi = (0.5 * gas_score + 0.3 * temp_score + 0.2 * hum_score) * 100.0

    return vqi.clip(0.0, 100.0)


def label_from_vqi(vqi: pd.Series) -> pd.Series:

    bins = [-np.inf, 60.0, 80.0, np.inf]

    return pd.cut(vqi, bins=bins, labels=[0, 1, 2]).astype(int)


def train_models_new_iot(

    df: pd.DataFrame,

    voc_min: float, voc_max: float,

    t_min: float, t_opt: float, t_max: float,

    h_min: float, h_opt: float, h_max: float,

) -> Dict[str, float]:

    # Select required columns and drop missing

    use_cols = ["VOC", "Temperature", "Humidity"]

    df2 = df[use_cols].dropna().copy()



    # Compute VQI and labels

    vqi = compute_vqi_from_sensors(df2, voc_min, voc_max, t_min, t_opt, t_max, h_min, h_opt, h_max)

    y = label_from_vqi(vqi)

    X = df2

    # Save per-row VQI and label for reference

    scored = df2.copy()

    scored["VQI"] = vqi.values

    scored["label"] = y.values

    scored["status"] = _label_to_status(y.values)

    os.makedirs(OUT_DIR, exist_ok=True)

    scored[["VOC", "Temperature", "Humidity", "VQI", "label", "status"]].to_csv(

        os.path.join(OUT_DIR, "vqi_scored.csv"),

        index=False,

    )

    # Print VQI table on terminal

    try:

        # Add row numbering (1-based) for display

        display_df = scored[["VOC", "Temperature", "Humidity", "VQI", "label", "status"]].copy()

        display_df.index = np.arange(1, len(display_df) + 1)

        display_df.index.name = "row"

        print("\n================ VQI scores per row ================")
        print(display_df.to_string())
        print("===================================================")
    except Exception:

        pass



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True

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

        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE))

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

                random_state=RANDOM_STATE,
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                min_child_weight=2.0,
                eval_metric="logloss",
                n_jobs=1,

            ))

        ])

        xgb_pipe.fit(X_train, y_train)

        y_pred_xgb = xgb_pipe.predict(X_test)

        try:

            y_proba_xgb = xgb_pipe.predict_proba(X_test)

        except Exception:

            y_proba_xgb = None



    # ----------------- Plots for IoT models -----------------

    # GBC

    cm_g = confusion_matrix(y_test, y_pred_gbc)

    plt.figure(figsize=(5,4))

    sns.heatmap(cm_g, annot=True, fmt="d", cmap="Blues", xticklabels=["0","1","2"], yticklabels=["0","1","2"])

    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (GBC)")

    plt.tight_layout(); plt.savefig(os.path.join(DIR_GBC, "gbc_confusion_matrix.png"), dpi=200); plt.close()

    if y_proba_gbc is not None:

        # Skip classes without both positive and negative samples to avoid warnings

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



    # SVM

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



    # XGB

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

    # Pairwise majority votes (tie-break deterministic)

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

    # 3-way majority vote

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



    _print_freshness_table(

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



    # Fusion outputs and plots for combinations

    weights = np.array([3.0, 6.5, 10.0])

    def _fresh(p: Optional[np.ndarray]) -> Optional[np.ndarray]:

        if p is None: return None

        return p @ weights

    fresh_svm = _fresh(y_proba_svm)

    fresh_gbc = _fresh(y_proba_gbc)

    fresh_xgb = _fresh(y_proba_xgb) if (HAS_XGB and y_proba_xgb is not None) else None

    def _save_fusion(tag: str, pred_cls: np.ndarray, fresh_score: Optional[np.ndarray]):

        # Fallback: if no probability-based freshness, derive from class labels

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

    # per-model

    _save_fusion("svm", y_pred_svm, fresh_svm)

    _save_fusion("gbc", y_pred_gbc, fresh_gbc)

    if fresh_xgb is not None and y_pred_xgb is not None:

        _save_fusion("xgb", y_pred_xgb, fresh_xgb)

    # pairwise

    # Even if any freshness is missing, fallback will map classes to scores

    if pred_sg is not None:

        mean_fg = None if (fresh_svm is None or fresh_gbc is None) else (fresh_svm + fresh_gbc) / 2.0

        _save_fusion("comb_svm_gbc", pred_sg, mean_fg)

    if pred_sx is not None:

        mean_fx = None if (fresh_svm is None or fresh_xgb is None) else (fresh_svm + fresh_xgb) / 2.0

        _save_fusion("comb_svm_xgb", pred_sx, mean_fx)

    if pred_gx is not None:

        mean_gx = None if (fresh_gbc is None or fresh_xgb is None) else (fresh_gbc + fresh_xgb) / 2.0

        _save_fusion("comb_gbc_xgb", pred_gx, mean_gx)

    # triple

    fres_list = [f for f in [fresh_svm, fresh_gbc, fresh_xgb] if f is not None]

    if len(fres_list) >= 2:

        fres_mean = np.mean(np.vstack(fres_list), axis=0)

        _save_fusion("combined3", pred_comb3, fres_mean)



    # Ablation settings

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

        svm = Pipeline([("prep", pre), ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE))])

        svm.fit(Xtr, ytr); ps = svm.predict(Xte)

        rows.append({"model": "SVM", "setting": name, "features_used": ",".join(feats),

                     "acc": accuracy_score(yte, ps),

                     "precision_macro": precision_score(yte, ps, average="macro", zero_division=0),

                     "recall_macro": recall_score(yte, ps, average="macro", zero_division=0),

                     "f1_macro": f1_score(yte, ps, average="macro", zero_division=0)})

        # GBC

        gbc = Pipeline([("prep", pre), ("gbc", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=120, learning_rate=0.05, max_depth=2, subsample=0.8))])

        gbc.fit(Xtr, ytr); pg = gbc.predict(Xte)

        rows.append({"model": "GBC", "setting": name, "features_used": ",".join(feats),

                     "acc": accuracy_score(yte, pg),

                     "precision_macro": precision_score(yte, pg, average="macro", zero_division=0),

                     "recall_macro": recall_score(yte, pg, average="macro", zero_division=0),

                     "f1_macro": f1_score(yte, pg, average="macro", zero_division=0)})

        # XGB

        px = None

        if HAS_XGB:

            xgb = Pipeline([("prep", pre), ("xgb", XGBClassifier(random_state=RANDOM_STATE, n_estimators=120, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=2.0, eval_metric="logloss", n_jobs=1))])

            xgb.fit(Xtr, ytr); px = xgb.predict(Xte)

            rows.append({"model": "XGB", "setting": name, "features_used": ",".join(feats),

                         "acc": accuracy_score(yte, px),

                         "precision_macro": precision_score(yte, px, average="macro", zero_division=0),

                         "recall_macro": recall_score(yte, px, average="macro", zero_division=0),

                         "f1_macro": f1_score(yte, px, average="macro", zero_division=0)})

        # Pairwise combinations per-setting

        pred_sg_ab = np.where(ps == pg, ps, ps)  # SVM tie-breaker

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

            pred_gx_ab = np.where(pg == px, pg, pg)  # GBC tie-breaker

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

    # Print ablations to terminal by model (forced order, expanded width)

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

            # Fallback mapping to CSVs if sub is empty (ensures combos always print)

            fallback_csv = None

            if sub.empty:

                if mdl == "COMB_SG":

                    fallback_csv = os.path.join(OUT_DIR, "ablation_comb_svm_gbc.csv")

                elif mdl == "COMB_SX" and HAS_XGB:

                    fallback_csv = os.path.join(OUT_DIR, "ablation_comb_svm_xgb.csv")

                elif mdl == "COMB_GX" and HAS_XGB:

                    fallback_csv = os.path.join(OUT_DIR, "ablation_comb_gbc_xgb.csv")

                elif mdl == "COMBINED3":

                    fallback_csv = os.path.join(OUT_DIR, "ablation_combined3.csv")

                if fallback_csv and os.path.exists(fallback_csv):

                    try:

                        sub = pd.read_csv(fallback_csv)

                    except Exception:

                        pass

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

        # combinations

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





def run_ablation(df: pd.DataFrame) -> pd.DataFrame:

    enc = LabelEncoder()

    df = df.copy()

    df["crop_encoded"] = enc.fit_transform(df["crop"].astype(str))



    base_feats = ["crop_encoded", "temp_c", "humidity", "gas_ppm", "ethylene_ppm", "hours_since_harvest"]

    if OPTIONAL_CO2 in df.columns:

        base_feats.append(OPTIONAL_CO2)



    y_vqi = df[TARGET_VQI]

    y_cls = _derive_fresh_label(y_vqi)



    ablations: List[Tuple[str, List[str]]] = [

        ("remove_voc_gas_ppm", [f for f in base_feats if f != "gas_ppm"]),

        ("remove_humidity", [f for f in base_feats if f != "humidity"]),

        ("remove_ethylene", [f for f in base_feats if f != "ethylene_ppm"]),

    ]

    if OPTIONAL_CO2 in base_feats:

        ablations.append(("remove_co2", [f for f in base_feats if f != OPTIONAL_CO2]))



    rows = []

    for name, feats in ablations:

        X = df[feats]

        X_train, X_test, y_train_cls, y_test_cls = train_test_split(

            X, y_cls, test_size=0.2, random_state=RANDOM_STATE, shuffle=True

        )



        pre = ColumnTransformer([

            ("num", Pipeline([

                ("imputer", SimpleImputer(strategy="median")),

                ("scaler", StandardScaler())

            ]), feats)

        ])



        svm_clf = Pipeline([

            ("prep", pre),

            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE))

        ])

        svm_clf.fit(X_train, y_train_cls)

        pred_svm = svm_clf.predict(X_test)

        acc_svm = accuracy_score(y_test_cls, pred_svm)

        f1_svm = f1_score(y_test_cls, pred_svm, average="macro")



        gbc = Pipeline([

            ("prep", pre),

            ("gbc", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=120, learning_rate=0.05, max_depth=2, subsample=0.8))

        ])

        gbc.fit(X_train, y_train_cls)

        pred_cls = gbc.predict(X_test)

        acc = accuracy_score(y_test_cls, pred_cls)

        f1 = f1_score(y_test_cls, pred_cls, average="macro")



        rows.append({

            "ablation": name,

            "features_used": ",".join(feats),

            "svm_acc": float(acc_svm),

            "svm_f1_macro": float(f1_svm),

            "gbc_acc": float(acc),

            "gbc_f1_macro": float(f1)

        })



    ab_df = pd.DataFrame(rows)

    # Save combined table

    ab_df.to_csv(ABLATION_CSV, index=False)

    # SVM-only

    ab_svm = ab_df[["ablation", "features_used", "svm_acc", "svm_f1_macro"]].copy()

    ab_svm.to_csv(ABLATION_SVM_CSV, index=False)

    # GBC-only

    ab_gbc = ab_df[["ablation", "features_used", "gbc_acc", "gbc_f1_macro"]].copy()

    ab_gbc.to_csv(ABLATION_GBC_CSV, index=False)

    # Combined (macro average of SVM and GBC metrics)

    ab_comb = ab_df[["ablation", "features_used"]].copy()

    ab_comb["acc_mean"] = (ab_df["svm_acc"] + ab_df["gbc_acc"]) / 2.0

    ab_comb["f1_macro_mean"] = (ab_df["svm_f1_macro"] + ab_df["gbc_f1_macro"]) / 2.0

    ab_comb.to_csv(ABLATION_COMBINED_CSV, index=False)

    return ab_df





def smart_price(base_price: float, freshness_score_0_10: Optional[float], vqi_0_100: Optional[float], alpha: float, beta: float) -> float:

    f_term = (freshness_score_0_10 or 0.0) / 10.0

    vqi_term = (vqi_0_100 or 0.0) / 100.0

    price = base_price * (1.0 + alpha * f_term + beta * vqi_term)

    return float(max(price, 0.0))





def predict_from_inputs(inputs: Dict[str, float], mode: str, alpha: float = 0.2, beta: float = 0.6, base_price: float = 50.0) -> Dict:

    svr = joblib.load(SVM_MODEL_PATH) if os.path.exists(SVM_MODEL_PATH) else None

    gbc = joblib.load(GBC_MODEL_PATH) if os.path.exists(GBC_MODEL_PATH) else None



    # Build DataFrame row

    row = {

        "crop": inputs.get("crop", "Tomato"),

        "temp_c": float(inputs.get("temp_c")),

        "humidity": float(inputs.get("humidity")),

        "gas_ppm": float(inputs.get("gas_ppm")),

        "ethylene_ppm": float(inputs.get("ethylene_ppm")),

        "hours_since_harvest": float(inputs.get("hours_since_harvest")),

    }

    if OPTIONAL_CO2 in inputs:

        row[OPTIONAL_CO2] = float(inputs.get(OPTIONAL_CO2))



    df_row = pd.DataFrame([row])



    vqi_pred = None

    fresh_class = None

    fresh_proba = None

    freshness_score = None



    if mode in ("SVM", "Both") and svr is not None:

        vqi_pred = float(np.clip(svr.predict(df_row)[0], 0.0, 100.0))



    if mode in ("GBC", "Both") and gbc is not None:

        fresh_class = int(gbc.predict(df_row)[0])

        if hasattr(gbc, "predict_proba"):

            proba = gbc.predict_proba(df_row)[0]

            fresh_proba = {str(i): float(p) for i, p in enumerate(proba)}

            # Map probabilities to a freshness score in [0,10]

            # Spoiled(0)->3, Moderate(1)->6.5, Fresh(2)->10

            weights = np.array([3.0, 6.5, 10.0])

            freshness_score = float(np.dot(proba, weights))

        else:

            freshness_score = {0: 3.0, 1: 6.5, 2: 10.0}.get(fresh_class, 6.5)



    # If only SVM, derive a simple freshness proxy from VQI

    if mode == "SVM" and freshness_score is None and vqi_pred is not None:

        freshness_score = float(np.clip(vqi_pred / 10.0, 0, 10))



    final_price = smart_price(base_price, freshness_score, vqi_pred, alpha, beta)



    return {

        "inputs": row,

        "mode": mode,

        "vqi": vqi_pred,

        "fresh_class": fresh_class,

        "fresh_proba": fresh_proba,

        "freshness_score": freshness_score,

        "final_price": final_price,

    }





def train_all(csv_path: str) -> Dict:

    df = load_dataset(csv_path)

    # If IoT columns present, run new pipeline with thresholds.

    if set(["VOC", "Temperature", "Humidity"]).issubset(df.columns):

        # Auto-fallback thresholds based on data if not provided via CLI

        voc_min = float(df["VOC"].min()); voc_max = float(df["VOC"].max())

        t_min = float(df["Temperature"].min()); t_max = float(df["Temperature"].max()); t_opt = float(df["Temperature"].median())

        h_min = float(df["Humidity"].min()); h_max = float(df["Humidity"].max()); h_opt = float(df["Humidity"].median())

        metrics = train_models_new_iot(df, voc_min, voc_max, t_min, t_opt, t_max, h_min, h_opt, h_max)

        return {"metrics": metrics, "ablation_path": ABLATION_CSV, "out_dir": OUT_DIR}

    # Else, run legacy pipeline

    metrics = train_models(df)

    ablation_df = run_ablation(df)

    return {"metrics": metrics, "ablation_path": ABLATION_CSV, "out_dir": OUT_DIR}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train models on legacy or IoT dataset and generate outputs.")

    parser.add_argument("--csv", type=str, default="svm/newcsv.csv", help="Path to input CSV with data")

    # Optional overrides for IoT thresholds

    parser.add_argument("--voc-min", type=float, default=None)

    parser.add_argument("--voc-max", type=float, default=None)

    parser.add_argument("--t-min", type=float, default=None)

    parser.add_argument("--t-opt", type=float, default=None)

    parser.add_argument("--t-max", type=float, default=None)

    parser.add_argument("--h-min", type=float, default=None)

    parser.add_argument("--h-opt", type=float, default=None)

    parser.add_argument("--h-max", type=float, default=None)

    args = parser.parse_args()



    # If IoT dataset present, always use the new pipeline

    df_preview = pd.read_csv(args.csv)

    if set(["VOC", "Temperature", "Humidity"]).issubset(df_preview.columns):

        # Auto-fallback thresholds based on data if not provided via CLI

        voc_min = float(args.voc_min) if args.voc_min is not None else float(df_preview["VOC"].min())

        voc_max = float(args.voc_max) if args.voc_max is not None else float(df_preview["VOC"].max())

        t_min = float(args.t_min) if args.t_min is not None else float(df_preview["Temperature"].min())

        t_opt = float(args.t_opt) if args.t_opt is not None else float(df_preview["Temperature"].median())

        t_max = float(args.t_max) if args.t_max is not None else float(df_preview["Temperature"].max())

        h_min = float(args.h_min) if args.h_min is not None else float(df_preview["Humidity"].min())

        h_opt = float(args.h_opt) if args.h_opt is not None else float(df_preview["Humidity"].median())

        h_max = float(args.h_max) if args.h_max is not None else float(df_preview["Humidity"].max())

        metrics = train_models_new_iot(df_preview, voc_min, voc_max, t_min, t_opt, t_max, h_min, h_opt, h_max)

        info = {"metrics": metrics, "ablation_path": ABLATION_CSV, "out_dir": OUT_DIR}

    else:

        info = train_all(args.csv)

    print("Training complete.")

    print(f"Outputs folder: {info['out_dir']}")

    print(f"Ablation CSV:   {info['ablation_path']}")

    print(f"Metrics JSON:   {METRICS_JSON}")


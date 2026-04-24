import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42

# Fixed original feature list (do not modify globally)
ORIG_FEATURES = [
    "temperature",
    "humidity",
    "mq135_voc",
    "ethylene",
    "hours",
    "light_intensity",
    "co2_ppm",
]
CLASS_TARGET = "fresh_label"
REG_TARGET = "vqi"


def ensure_quality_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally adapt a dataset in the 'quality_data.csv' style to required columns.
    - Renames columns
    - Adds missing optional columns with reasonable defaults if absent
    - Derives fresh_label from vqi if missing
    """
    rename_map = {
        "temp_c": "temperature",
        "gas_ppm": "mq135_voc",
        "ethylene_ppm": "ethylene",
        "hours_since_harvest": "hours",
    }
    df = df.copy()
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Add optional features if missing with reasonable defaults
    if "light_intensity" not in df.columns:
        df["light_intensity"] = 450.0
    if "co2_ppm" not in df.columns:
        df["co2_ppm"] = 400.0

    # Derive fresh_label from vqi if missing using tertiles
    if CLASS_TARGET not in df.columns and REG_TARGET in df.columns:
        vqi = df[REG_TARGET]
        q = vqi.quantile([0.3333, 0.6666]).values
        lo, hi = float(q[0]), float(q[1])
        if hi <= lo:
            hi = lo + 1e-6
        bins = [-np.inf, lo, hi, np.inf]
        df[CLASS_TARGET] = pd.cut(vqi, bins=bins, labels=[0, 1, 2]).astype(int)

    return df


def check_columns(df: pd.DataFrame, features: List[str]) -> None:
    missing_feats = [c for c in features if c not in df.columns]
    missing_targets = [c for c in [CLASS_TARGET, REG_TARGET] if c not in df.columns]
    if missing_feats or missing_targets:
        raise KeyError(
            f"Missing columns. Features missing: {missing_feats}; Targets missing: {missing_targets}"
        )


def make_preprocessor(feats: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            feats,
        )
    ])


def train_baseline(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    X = df[features]
    y_cls = df[CLASS_TARGET]
    y_reg = df[REG_TARGET]

    # Use stratify only if we have at least 2 classes
    strat = y_cls if len(np.unique(y_cls)) >= 2 else None
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )
    y_train_reg, y_test_reg = y_reg.loc[X_train.index], y_reg.loc[X_test.index]

    # GradientBoostingClassifier
    gbc = Pipeline([
        ("prep", make_preprocessor(features)),
        ("gbc", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    gbc.fit(X_train, y_train_cls)
    y_pred_gbc = gbc.predict(X_test)
    
    # For regression metrics from GBC, use class predictions as proxy for VQI estimation
    gbc_reg_pred = np.where(y_pred_gbc == 2, y_train_reg.quantile(0.9),
                   np.where(y_pred_gbc == 1, y_train_reg.quantile(0.6),
                           y_train_reg.quantile(0.3)))

    # Support Vector Machine (SVC)
    svc = Pipeline([
        ("prep", make_preprocessor(features)),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
    ])
    svc.fit(X_train, y_train_cls)
    y_pred_svc = svc.predict(X_test)
    
    # For regression metrics from SVM, use class predictions as proxy for VQI estimation
    svc_reg_pred = np.where(y_pred_svc == 2, y_train_reg.quantile(0.9),
                   np.where(y_pred_svc == 1, y_train_reg.quantile(0.6),
                           y_train_reg.quantile(0.3)))

    # Optional XGBoost (classification). We'll also derive proxy regression metrics from its classes
    acc_xgb = precision_xgb = recall_xgb = f1_xgb = rmse_xgb = mae_xgb = r2_xgb = None
    if HAS_XGB:
        xgb = Pipeline([
            ("prep", make_preprocessor(features)),
            ("xgb", XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=1.0,
                colsample_bytree=1.0,
                eval_metric="logloss",
                n_jobs=1,
            )),
        ])
        xgb.fit(X_train, y_train_cls)
        y_pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test_cls, y_pred_xgb)
        precision_xgb = precision_score(y_test_cls, y_pred_xgb, average="macro", zero_division=0)
        recall_xgb = recall_score(y_test_cls, y_pred_xgb, average="macro", zero_division=0)
        f1_xgb = f1_score(y_test_cls, y_pred_xgb, average="macro", zero_division=0)
        # proxy regression metrics
        xgb_reg_pred = np.where(y_pred_xgb == 2, y_train_reg.quantile(0.9),
                        np.where(y_pred_xgb == 1, y_train_reg.quantile(0.6),
                                 y_train_reg.quantile(0.3)))
        rmse_xgb = np.sqrt(mean_squared_error(y_test_reg, xgb_reg_pred))
        mae_xgb = mean_absolute_error(y_test_reg, xgb_reg_pred)
        r2_xgb = r2_score(y_test_reg, xgb_reg_pred)

    metrics = {
        # GBC classification metrics (suffix _gbc)
        "acc_gbc": accuracy_score(y_test_cls, y_pred_gbc),
        "precision_macro_gbc": precision_score(y_test_cls, y_pred_gbc, average="macro", zero_division=0),
        "recall_macro_gbc": recall_score(y_test_cls, y_pred_gbc, average="macro", zero_division=0),
        "f1_macro_gbc": f1_score(y_test_cls, y_pred_gbc, average="macro", zero_division=0),
        # GBC regression metrics (suffix _gbc)
        "rmse_gbc": np.sqrt(mean_squared_error(y_test_reg, gbc_reg_pred)),
        "mae_gbc": mean_absolute_error(y_test_reg, gbc_reg_pred),
        "r2_gbc": r2_score(y_test_reg, gbc_reg_pred),
        # SVC classification metrics (suffix _svc)
        "acc_svc": accuracy_score(y_test_cls, y_pred_svc),
        "precision_macro_svc": precision_score(y_test_cls, y_pred_svc, average="macro", zero_division=0),
        "recall_macro_svc": recall_score(y_test_cls, y_pred_svc, average="macro", zero_division=0),
        "f1_macro_svc": f1_score(y_test_cls, y_pred_svc, average="macro", zero_division=0),
        # SVC regression metrics (suffix _svc)
        "rmse_svc": np.sqrt(mean_squared_error(y_test_reg, svc_reg_pred)),
        "mae_svc": mean_absolute_error(y_test_reg, svc_reg_pred),
        "r2_svc": r2_score(y_test_reg, svc_reg_pred),
    }
    if HAS_XGB and acc_xgb is not None:
        metrics.update({
            "acc_xgb": acc_xgb,
            "precision_macro_xgb": precision_xgb,
            "recall_macro_xgb": recall_xgb,
            "f1_macro_xgb": f1_xgb,
            "rmse_xgb": rmse_xgb,
            "mae_xgb": mae_xgb,
            "r2_xgb": r2_xgb,
        })
    return metrics


def ablation_loop(df: pd.DataFrame, features: List[str], baseline: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for remove_feat in features:
        # Do not modify original list; create a copy per iteration
        active_feats = [f for f in features if f != remove_feat]

        print("\n==============================================")
        print(f"Removed feature: {remove_feat}")
        print(f"Active features: {active_feats}")
        print("==============================================\n")

        m = train_baseline(df, active_feats)
        rows.append({
            "Removed Feature": remove_feat,
            # Report SVC metrics
            "acc_svc": m["acc_svc"],
            "precision_macro_svc": m["precision_macro_svc"],
            "recall_macro_svc": m["recall_macro_svc"],
            "f1_macro_svc": m["f1_macro_svc"],
            "rmse_svc": m["rmse_svc"],
            "mae_svc": m["mae_svc"],
            "r2_svc": m["r2_svc"],
            # Also include GBC metrics
            "acc_gbc": m["acc_gbc"],
            "precision_macro_gbc": m["precision_macro_gbc"],
            "recall_macro_gbc": m["recall_macro_gbc"],
            "f1_macro_gbc": m["f1_macro_gbc"],
            "rmse_gbc": m["rmse_gbc"],
            "mae_gbc": m["mae_gbc"],
            "r2_gbc": m["r2_gbc"],
            # Optional XGB metrics
            **({
                "acc_xgb": m.get("acc_xgb"),
                "precision_macro_xgb": m.get("precision_macro_xgb"),
                "recall_macro_xgb": m.get("recall_macro_xgb"),
                "f1_macro_xgb": m.get("f1_macro_xgb"),
                "rmse_xgb": m.get("rmse_xgb"),
                "mae_xgb": m.get("mae_xgb"),
                "r2_xgb": m.get("r2_xgb"),
            } if HAS_XGB else {}),
            # Deltas vs baseline (SVC)
            "accuracy_drop_svc": baseline["acc_svc"] - m["acc_svc"],
            "f1_drop_svc": baseline["f1_macro_svc"] - m["f1_macro_svc"],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Single-feature ablation for GBC (classification) and SVR (regression)")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--quality-format", action="store_true", help="Adapt quality_data.csv-style columns to expected ones")
    parser.add_argument("--out", type=str, default="ablation_results.csv", help="Path to save results CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.quality_format:
        df = ensure_quality_format(df)

    # Verify required columns exist
    check_columns(df, ORIG_FEATURES)

    print("Baseline training with all features:")
    baseline = train_baseline(df, ORIG_FEATURES)

    print("\nBaseline metrics (SVM):")
    svm_metrics = {k.replace('_svc', ''): v for k, v in baseline.items() if k.endswith('_svc')}
    for k, v in svm_metrics.items():
        print(f" - {k}: {v:.4f}")

    print("\nBaseline metrics (GBC):")
    gbc_metrics = {k.replace('_gbc', ''): v for k, v in baseline.items() if k.endswith('_gbc')}
    for k, v in gbc_metrics.items():
        print(f" - {k}: {v:.4f}")

    print("\nBaseline metrics (Combined):")
    combined_metrics = {}
    for metric in ['acc', 'precision_macro', 'recall_macro', 'f1_macro', 'rmse', 'mae', 'r2']:
        svm_val = baseline[f"{metric}_svc"]
        gbc_val = baseline[f"{metric}_gbc"]
        combined_metrics[metric] = (svm_val + gbc_val) / 2.0
        print(f" - {metric}: {combined_metrics[metric]:.4f}")

    print("\nRunning single-feature ablation...")
    results_df = ablation_loop(df, ORIG_FEATURES, baseline)

    print("\nAblation results (classification metrics):")
    print(results_df)

    # Create separate DataFrames for SVM, GBC, XGB (if available), and Combined
    svm_cols = ['Removed Feature', 'acc_svc', 'precision_macro_svc', 'recall_macro_svc', 'f1_macro_svc', 'rmse_svc', 'mae_svc', 'r2_svc', 'accuracy_drop_svc', 'f1_drop_svc']
    gbc_cols = ['Removed Feature', 'acc_gbc', 'precision_macro_gbc', 'recall_macro_gbc', 'f1_macro_gbc', 'rmse_gbc', 'mae_gbc', 'r2_gbc']
    xgb_cols = ['Removed Feature', 'acc_xgb', 'precision_macro_xgb', 'recall_macro_xgb', 'f1_macro_xgb', 'rmse_xgb', 'mae_xgb', 'r2_xgb'] if HAS_XGB and 'acc_xgb' in results_df.columns else None
    
    svm_results = results_df[svm_cols].copy()
    svm_results.columns = ['Removed Feature', 'acc', 'precision_macro', 'recall_macro', 'f1_macro', 'rmse', 'mae', 'r2', 'accuracy_drop', 'f1_drop']
    
    gbc_results = results_df[gbc_cols].copy()
    gbc_results.columns = ['Removed Feature', 'acc', 'precision_macro', 'recall_macro', 'f1_macro', 'rmse', 'mae', 'r2']
    if xgb_cols:
        xgb_results = results_df[xgb_cols].copy()
        xgb_results.columns = ['Removed Feature', 'acc', 'precision_macro', 'recall_macro', 'f1_macro', 'rmse', 'mae', 'r2']
    
    # Combined results
    combined_results = results_df[['Removed Feature']].copy()
    for metric in ['acc', 'precision_macro', 'recall_macro', 'f1_macro', 'rmse', 'mae', 'r2']:
        combined_results[metric] = (results_df[f"{metric}_svc"] + results_df[f"{metric}_gbc"]) / 2.0
    combined_results['accuracy_drop'] = (results_df['accuracy_drop_svc'] + 
                                       (baseline['acc_gbc'] - results_df['acc_gbc'])) / 2.0
    combined_results['f1_drop'] = (results_df['f1_drop_svc'] + 
                                 (baseline['f1_macro_gbc'] - results_df['f1_macro_gbc'])) / 2.0

    # Combined of three (if XGB available)
    combined3_results = None
    if xgb_cols:
        combined3_results = results_df[['Removed Feature']].copy()
        for metric in ['acc', 'precision_macro', 'recall_macro', 'f1_macro', 'rmse', 'mae', 'r2']:
            combined3_results[metric] = (
                results_df[f"{metric}_svc"].fillna(0) +
                results_df[f"{metric}_gbc"].fillna(0) +
                results_df[f"{metric}_xgb"].fillna(0)
            ) / 3.0
        # drops based on SVM/GBC/XGB baselines where available
        acc_drop_xgb = (baseline.get('acc_xgb', np.nan) - results_df.get('acc_xgb', np.nan))
        f1_drop_xgb = (baseline.get('f1_macro_xgb', np.nan) - results_df.get('f1_macro_xgb', np.nan))
        combined3_results['accuracy_drop'] = (
            results_df['accuracy_drop_svc'] + (baseline['acc_gbc'] - results_df['acc_gbc']) + acc_drop_xgb
        ) / 3.0
        combined3_results['f1_drop'] = (
            results_df['f1_drop_svc'] + (baseline['f1_macro_gbc'] - results_df['f1_macro_gbc']) + f1_drop_xgb
        ) / 3.0

    # Display separate results
    print("\n" + "="*60)
    print("SVM ABLATION RESULTS:")
    print("="*60)
    print(svm_results.to_string(index=False))
    
    print("\n" + "="*60)
    print("GBC ABLATION RESULTS:")
    print("="*60)
    print(gbc_results.to_string(index=False))
    
    if xgb_cols:
        print("\n" + "="*60)
        print("XGB ABLATION RESULTS:")
        print("="*60)
        print(xgb_results.to_string(index=False))
    
    print("\n" + "="*60)
    print("COMBINED ABLATION RESULTS:")
    print("="*60)
    print(combined_results.to_string(index=False))
    
    if combined3_results is not None:
        print("\n" + "="*60)
        print("COMBINED (SVM+GBC+XGB) ABLATION RESULTS:")
        print("="*60)
        print(combined3_results.to_string(index=False))

    # Save separate CSVs
    base_name = args.out.replace('.csv', '')
    svm_results.to_csv(f"{base_name}_svm.csv", index=False)
    gbc_results.to_csv(f"{base_name}_gbc.csv", index=False)
    combined_results.to_csv(f"{base_name}_combined.csv", index=False)
    if xgb_cols:
        xgb_results.to_csv(f"{base_name}_xgb.csv", index=False)
    if combined3_results is not None:
        combined3_results.to_csv(f"{base_name}_combined3.csv", index=False)
    
    print(f"\nSaved SVM ablation to: {os.path.abspath(f'{base_name}_svm.csv')}")
    print(f"Saved GBC ablation to: {os.path.abspath(f'{base_name}_gbc.csv')}")
    print(f"Saved Combined ablation to: {os.path.abspath(f'{base_name}_combined.csv')}")
    if xgb_cols:
        print(f"Saved XGB ablation to: {os.path.abspath(f'{base_name}_xgb.csv')}")
    if combined3_results is not None:
        print(f"Saved Combined-3 ablation to: {os.path.abspath(f'{base_name}_combined3.csv')}")


if __name__ == "__main__":
    main()

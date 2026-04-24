"""Run the IoT VQI + SVM/GBC/XGB + ablation pipeline on closed/closed.csv,
reusing unified_models but writing ALL outputs into the closed/ folder.

This does NOT modify unified_outputs or the original unified pipeline.
"""

import os
import pandas as pd

import unified_models as um


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "closed.csv")

    # Redirect all unified_models outputs to the closed/ folder
    out_dir = base_dir  # use closed/ as OUT_DIR
    um.OUT_DIR = out_dir
    um.DIR_SVM = os.path.join(out_dir, "svm"); os.makedirs(um.DIR_SVM, exist_ok=True)
    um.DIR_GBC = os.path.join(out_dir, "gbc"); os.makedirs(um.DIR_GBC, exist_ok=True)
    um.DIR_COMB = os.path.join(out_dir, "combined"); os.makedirs(um.DIR_COMB, exist_ok=True)
    um.DIR_XGB = os.path.join(out_dir, "xgb"); os.makedirs(um.DIR_XGB, exist_ok=True)

    um.SVM_MODEL_PATH = os.path.join(out_dir, "svm_cls_model.joblib")
    um.GBC_MODEL_PATH = os.path.join(out_dir, "gbc_fresh_model.joblib")
    um.XGB_MODEL_PATH = os.path.join(out_dir, "xgb_fresh_model.joblib")
    um.SCALER_PATH = os.path.join(out_dir, "scaler.joblib")
    um.LABEL_ENCODER_PATH = os.path.join(out_dir, "label_encoder.joblib")
    um.METRICS_JSON = os.path.join(out_dir, "metrics.json")
    um.ABLATION_CSV = os.path.join(out_dir, "ablation_results.csv")
    um.ABLATION_SVM_CSV = os.path.join(out_dir, "ablation_svm.csv")
    um.ABLATION_GBC_CSV = os.path.join(out_dir, "ablation_gbc.csv")
    um.ABLATION_COMBINED_CSV = os.path.join(out_dir, "ablation_combined.csv")

    # Load closed.csv (IoT-style: VOC, Temperature, Humidity)
    df = pd.read_csv(csv_path)

    if set(["VOC", "Temperature", "Humidity"]).issubset(df.columns):
        # Auto thresholds from this dataset, same as unified IoT pipeline
        voc_min = float(df["VOC"].min()); voc_max = float(df["VOC"].max())
        t_min = float(df["Temperature"].min()); t_max = float(df["Temperature"].max()); t_opt = float(df["Temperature"].median())
        h_min = float(df["Humidity"].min()); h_max = float(df["Humidity"].max()); h_opt = float(df["Humidity"].median())
        metrics = um.train_models_new_iot(df, voc_min, voc_max, t_min, t_opt, t_max, h_min, h_opt, h_max)
        print("\nClosed pipeline complete.")
        print(f"Outputs folder: {out_dir}")
        print(f"Ablation CSV:   {um.ABLATION_CSV}")
        print(f"Metrics JSON:   {um.METRICS_JSON}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        raise KeyError("closed.csv must contain VOC, Temperature, Humidity columns for the IoT pipeline.")


if __name__ == "__main__":
    main()

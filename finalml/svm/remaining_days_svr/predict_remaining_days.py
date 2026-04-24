import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from prepare_dataset import remaining_days_from_voc_and_edges


def predict_one_from_model(voc: float, model) -> int:
    X = pd.DataFrame({"VOC": [float(voc)]})
    pred = int(model.predict(X)[0])
    return pred


def predict_one_from_rule(voc: float, edges: np.ndarray, total_days: float) -> float:
    s = pd.Series([voc], name="VOC")
    pred = float(remaining_days_from_voc_and_edges(s, edges, int(total_days)).iloc[0])
    pred = float(np.clip(pred, 0.0, total_days))
    return pred


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\Alekhya\Desktop\4-2 project\svm (3)\svm (2)\svm\remaining_days_svr\outputs\logreg_model.joblib",
    )
    parser.add_argument(
        "--voc-rule",
        type=str,
        default=r"C:\Users\Alekhya\Desktop\4-2 project\svm (3)\svm (2)\svm\remaining_days_svr\outputs\voc_rule_edges.json",
    )
    parser.add_argument("--total-days", type=float, default=10.0)
    args = parser.parse_args(argv)

    model = None
    if os.path.exists(args.model):
        model = joblib.load(args.model)

    print("Enter sensor values to predict Remaining Days")
    voc = float(input("VOC: ").strip())
    _temperature = float(input("Temperature: ").strip())
    _humidity = float(input("Humidity: ").strip())

    if model is not None:
        pred_int = int(np.clip(predict_one_from_model(voc=voc, model=model), 0, int(args.total_days)))
        print(f"Predicted Remaining Days: {pred_int}")
        return 0

    # Fallback to the rule if the model file doesn't exist
    if not os.path.exists(args.voc_rule):
        raise FileNotFoundError(f"Model not found: {args.model} and VOC rule file not found: {args.voc_rule}")

    with open(args.voc_rule, "r", encoding="utf-8") as f:
        rule = json.load(f)
    edges = np.asarray(rule.get("edges", []), dtype=float)
    if edges.size < 3:
        raise ValueError("VOC rule edges are missing/invalid in voc_rule_edges.json")

    pred = predict_one_from_rule(voc=voc, edges=edges, total_days=args.total_days)
    print(f"Predicted Remaining Days: {pred:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
prepare_dataset.py

Purpose: Generate ground-truth labels for shelf life prediction using rule-based VOC thresholds.
This creates the training dataset with VOC and Remaining_Days columns.
Rules are ONLY used here for label generation - NOT used in the final ML pipeline.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

# Required columns in input CSV
REQUIRED_FEATURES = ["VOC", "Temperature", "Humidity"]
DEFAULT_TOTAL_SPOILAGE_DAYS = 7


def compute_voc_rule_edges(voc: pd.Series, total_spoilage_days: int) -> np.ndarray:
    """
    Compute VOC threshold edges for rule-based labeling.
    These thresholds map VOC values to remaining days (0-10).
    
    Rule mapping:
    VOC < 0.02 -> 10 days
    VOC < 0.03 -> 9 days
    VOC < 0.04 -> 8 days
    VOC < 0.05 -> 7 days
    VOC < 0.06 -> 6 days
    VOC < 0.07 -> 6 days
    VOC < 0.08 -> 5 days
    VOC < 0.09 -> 4 days
    VOC < 0.1 -> 3 days
    VOC < 0.12 -> 2 days
    VOC < 0.13 -> 2 days
    VOC < 0.14 -> 1 days
    VOC < 0.15 -> 0 days
    VOC >= 0.15 -> 0 days
    """
    edges = np.array([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, float('inf')])
    return edges


def remaining_days_from_voc_and_edges(voc: pd.Series, edges: np.ndarray, total_spoilage_days: int) -> pd.Series:
    """
    Apply rule-based VOC thresholds to generate Remaining_Days labels.
    This creates the ground-truth labels for training the ML model.
    
    Args:
        voc: VOC values
        edges: Threshold array for mapping
        total_spoilage_days: Total spoilage period (not used in current implementation)
    
    Returns:
        Series with remaining days (0-10)
    """
    voc_numeric = pd.to_numeric(voc, errors="coerce")
    remaining = pd.Series(np.full(len(voc), np.nan), index=voc.index)
    
    # Apply granular thresholds based on domain knowledge
    remaining[voc_numeric < 0.02] = 10.0
    remaining[(voc_numeric >= 0.02) & (voc_numeric < 0.03)] = 9.0
    remaining[(voc_numeric >= 0.03) & (voc_numeric < 0.04)] = 8.0
    remaining[(voc_numeric >= 0.04) & (voc_numeric < 0.05)] = 7.0
    remaining[(voc_numeric >= 0.05) & (voc_numeric < 0.06)] = 6.0
    remaining[(voc_numeric >= 0.06) & (voc_numeric < 0.07)] = 6.0
    remaining[(voc_numeric >= 0.07) & (voc_numeric < 0.08)] = 5.0
    remaining[(voc_numeric >= 0.08) & (voc_numeric < 0.09)] = 4.0
    remaining[(voc_numeric >= 0.09) & (voc_numeric < 0.1)] = 3.0
    remaining[(voc_numeric >= 0.1) & (voc_numeric < 0.12)] = 2.0
    remaining[(voc_numeric >= 0.12) & (voc_numeric < 0.13)] = 2.0
    remaining[(voc_numeric >= 0.13) & (voc_numeric < 0.14)] = 1.0
    remaining[(voc_numeric >= 0.14) & (voc_numeric < 0.15)] = 0.0
    remaining[voc_numeric >= 0.15] = 0.0   
    return remaining


def prepare_dataset(
    csv_path: str,
    total_spoilage_days: int = DEFAULT_TOTAL_SPOILAGE_DAYS,
    edges: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Prepare dataset with VOC and rule-based Remaining_Days labels.
    
    Args:
        csv_path: Path to input CSV with sensor data
        total_spoilage_days: Total spoilage period (for future use)
        edges: Optional pre-computed threshold edges
    
    Returns:
        DataFrame with VOC, Temperature, Humidity, and Remaining_Days columns
    """
    if not os.path.exists(csv_path): 
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df2 = df.copy()

    # Generate rule-based labels for training
    if edges is None:
        edges = compute_voc_rule_edges(df2["VOC"], total_spoilage_days)

    df2["Remaining_Days"] = remaining_days_from_voc_and_edges(df2["VOC"], edges, total_spoilage_days)

    return df2


def main(argv: Optional[list] = None) -> int:
    """
    Command-line interface for dataset preparation.
    
    Usage:
        python prepare_dataset.py --csv input.csv --out prepared_dataset.csv
    """
    parser = argparse.ArgumentParser(description="Prepare dataset with rule-based Remaining_Days labels")
    parser.add_argument("--csv", required=True, type=str, help="Input CSV with sensor data")
    parser.add_argument("--out", required=True, type=str, help="Output CSV with prepared dataset")
    parser.add_argument("--total-days", default=DEFAULT_TOTAL_SPOILAGE_DAYS, type=int, help="Total spoilage days")
    args = parser.parse_args(argv)

    df = prepare_dataset(args.csv, total_spoilage_days=args.total_days)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Dataset prepared: {args.out}")
    print(f"Shape: {df.shape}")
    print(f"Remaining_Days distribution: {df['Remaining_Days'].value_counts().sort_index().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

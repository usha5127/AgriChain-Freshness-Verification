import os
import matplotlib.pyplot as plt
import numpy as np

# ===== Raw metrics from your message =====

metrics = {
    "SVM": {
        "acc": 0.9770,
        "precision_macro": 0.9771,
        "recall_macro": 0.9771,
        "f1_macro": 0.9771,
    },
    "GBC": {
        "acc": 0.9770,
        "precision_macro": 0.9774,
        "recall_macro": 0.9770,
        "f1_macro": 0.9771,
    },
    "XGB": {
        "acc": 0.9847,
        "precision_macro": 0.9847,
        "recall_macro": 0.9848,
        "f1_macro": 0.9847,
    },
    "SVM+GBC": {
        "acc": 0.9770,
        "precision_macro": 0.9771,
        "recall_macro": 0.9771,
        "f1_macro": 0.9771,
    },
    "SVM+XGB": {
        "acc": 0.9770,
        "precision_macro": 0.9771,
        "recall_macro": 0.9771,
        "f1_macro": 0.9771,
    },
    "GBC+XGB": {
        "acc": 0.9770,
        "precision_macro": 0.9774,
        "recall_macro": 0.9770,
        "f1_macro": 0.9771,
    },
    "SVM+GBC+XGB": {
        "acc": 0.9847,
        "precision_macro": 0.9847,
        "recall_macro": 0.9848,
        "f1_macro": 0.9847,
    },
}

# Output folder
OUTPUT_DIR = "model_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed colors for metrics (for clear legend)
COLORS = {
    "Accuracy": "#1f77b4",          # blue
    "Precision (macro)": "#ff7f0e", # orange
    "Recall (macro)": "#2ca02c",    # green
    "F1-Score (macro)": "#d62728",  # red
}


def plot_group(model_list, title, filename):
    """
    Plot Accuracy, Precision, Recall, F1 for a given list of models.
    Legend (color explanation) is outside the bar area (below).
    """
    names = model_list
    acc = [metrics[m]["acc"] for m in names]
    prec = [metrics[m]["precision_macro"] for m in names]
    rec = [metrics[m]["recall_macro"] for m in names]
    f1 = [metrics[m]["f1_macro"] for m in names]

    x = np.arange(len(names))

    # Width smaller + extra spacing between metric bars
    width = 0.16
    gap = 0.05  # extra gap between metric bars inside a group

    # Positions: center bars around each x with small gaps
    offsets = np.array([
        -(1.5 * width + gap),
        -(0.5 * width + gap / 3),
        +(0.5 * width + gap / 3),
        +(1.5 * width + gap),
    ])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x + offsets[0], acc,  width, label="Accuracy",          color=COLORS["Accuracy"])
    ax.bar(x + offsets[1], prec, width, label="Precision (macro)", color=COLORS["Precision (macro)"])
    ax.bar(x + offsets[2], rec,  width, label="Recall (macro)",    color=COLORS["Recall (macro)"])
    ax.bar(x + offsets[3], f1,   width, label="F1-Score (macro)",  color=COLORS["F1-Score (macro)"])

    ax.set_ylim(0.95, 1.00)
    ax.set_ylabel("Score")
    ax.set_xlabel("Models")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Legend outside (below the chart) – not overlapping bars
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
    )

    fig.tight_layout()

    # Save to file
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    # 1) SVM vs GBC
    plot_group(
        ["SVM", "GBC"],
        "Performance Comparison: SVM vs GBC",
        "svm_vs_gbc.png",
    )

    # 2) GBC vs XGB
    plot_group(
        ["GBC", "XGB"],
        "Performance Comparison: GBC vs XGBoost",
        "gbc_vs_xgb.png",
    )

    # 3) SVM vs XGB
    plot_group(
        ["SVM", "XGB"],
        "Performance Comparison: SVM vs XGBoost",
        "svm_vs_xgb.png",
    )

    # 4) All base + combinations
    plot_group(
        ["SVM", "GBC", "XGB", "SVM+GBC", "SVM+XGB", "GBC+XGB", "SVM+GBC+XGB"],
        "Performance Comparison: All Models and Combinations",
        "all_models_and_combinations.png",
    )

    # If you still want to quickly preview the last figure, uncomment:
    # plt.show()


if __name__ == "__main__":
    main()
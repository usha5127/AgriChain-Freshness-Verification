import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from your table
metrics_data = {
    "SVM": {
        "Accuracy": 0.9770,
        "Precision (macro)": 0.9771,
        "Recall (macro)": 0.9771,
        "F1-Score (macro)": 0.9771,
    },
    "GBC": {
        "Accuracy": 0.9770,
        "Precision (macro)": 0.9774,
        "Recall (macro)": 0.9770,
        "F1-Score (macro)": 0.9771,
    },
    "XGBoost": {
        "Accuracy": 0.9767,
        "Precision (macro)": 0.9767,
        "Recall (macro)": 0.9768,
        "F1-Score (macro)": 0.9767,
    },
    "SVM+GBC+XGB": {
        "Accuracy": 0.9847,
        "Precision (macro)": 0.9847,
        "Recall (macro)": 0.9848,
        "F1-Score (macro)": 0.9847,
    }
}

# Colors for metrics
COLORS = {
    "Accuracy": "#1f77b4",          # blue
    "Precision (macro)": "#ff7f0e", # orange
    "Recall (macro)": "#2ca02c",    # green
    "F1-Score (macro)": "#d62728",  # red
}

def plot_model_comparison():
    """
    Create a bar chart comparing SVM, GBC, and XGBoost models (excluding SVM+GBC+XGB)
    """
    models = list(metrics_data.keys())
    models_to_plot = [model for model in models if model != "SVM+GBC+XGB"]
    metrics = list(metrics_data[models[0]].keys())
    
    # Prepare data for plotting
    x = np.arange(len(models_to_plot))
    
    # Reduced bar width for better visibility
    width = 0.15
    gap = 0.03  # extra gap between metric bars inside a group
    
    # Positions: center bars around each x with small gaps
    offsets = np.array([
        -(1.5 * width + gap),
        -(0.5 * width + gap / 3),
        +(0.5 * width + gap / 3),
        +(1.5 * width + gap),
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [metrics_data[model][metric] for model in models_to_plot]
        bars = ax.bar(x + offsets[i], values, width, label=metric, color=COLORS[metric])
        
        # Add value labels on top of each bar
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                   f'{value:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set axis properties
    ax.set_ylim(0.95, 1.00)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.set_title('Performance Comparison: SVM vs GBC vs XGBoost', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_to_plot, rotation=0, fontsize=11)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend outside the plot area at the bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=10,
        title='Metrics:',
        title_fontsize=11
    )
    
    # Adjust layout to prevent legend cutoff
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    # Save the plot
    plt.savefig('model_comparison_svm_gbc_xgb_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'model_comparison_svm_gbc_xgb_combined.png'")

def plot_individual_comparisons():
    """
    Create separate plots for individual model comparisons
    """
    models = list(metrics_data.keys())
    metrics = list(metrics_data[models[0]].keys())
    
    # SVM vs GBC
    models_svm_gbc = ["SVM", "GBC"]
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    
    x1 = np.arange(len(models_svm_gbc))
    width1 = 0.2
    gap1 = 0.02
    offsets1 = np.array([-0.11, -0.03, 0.03, 0.11])
    
    for i, metric in enumerate(metrics):
        values = [metrics_data[model][metric] for model in models_svm_gbc]
        bars = ax1.bar(x1 + offsets1[i], values, width1, label=metric, color=COLORS[metric])
        
        # Add value labels on top of each bar
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylim(0.974, 1.00)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Models')
    ax1.set_title('SVM vs GBC Performance Comparison')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models_svm_gbc)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig1.legend(
        handles1,
        labels1,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=10,
        title='Metrics:',
        title_fontsize=11
    )
    fig1.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig('svm_vs_gbc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # GBC vs XGBoost
    models_gbc_xgb = ["GBC", "XGBoost"]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    
    x2 = np.arange(len(models_gbc_xgb))
    for i, metric in enumerate(metrics):
        values = [metrics_data[model][metric] for model in models_gbc_xgb]
        bars = ax2.bar(x2 + offsets1[i], values, width1, label=metric, color=COLORS[metric])
        
        # Add value labels on top of each bar
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_ylim(0.974, 1.00)
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Models')
    ax2.set_title('GBC vs XGBoost Performance Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models_gbc_xgb)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig2.legend(
        handles2,
        labels2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=10,
        title='Metrics:',
        title_fontsize=11
    )
    fig2.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig('gbc_vs_xgb_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SVM vs XGBoost
    models_svm_xgb = ["SVM", "XGBoost"]
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    
    x3 = np.arange(len(models_svm_xgb))
    for i, metric in enumerate(metrics):
        values = [metrics_data[model][metric] for model in models_svm_xgb]
        bars = ax3.bar(x3 + offsets1[i], values, width1, label=metric, color=COLORS[metric])
        
        # Add value labels on top of each bar
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_ylim(0.974, 1.00)
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Models')
    ax3.set_title('SVM vs XGBoost Performance Comparison')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(models_svm_xgb)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    handles3, labels3 = ax3.get_legend_handles_labels()
    fig3.legend(
        handles3,
        labels3,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=10,
        title='Metrics:',
        title_fontsize=11
    )
    fig3.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig('svm_vs_xgb_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Individual comparison plots saved:")
    print("- svm_vs_gbc_comparison.png")
    print("- gbc_vs_xgb_comparison.png") 
    print("- svm_vs_xgb_comparison.png")

if __name__ == "__main__":
    # Create the main comparison plot
    plot_model_comparison()
    
    # Create individual comparison plots
    plot_individual_comparisons()

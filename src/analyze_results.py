"""
analyze_results.py
------------------
Post-training evaluation script for multi-class fraud detection for drivers license, passports, and ID Cards

Generates:
  - Classification report (precision/recall/F1 per class)
  - Macro/Micro/Weighted F1 scores
  - ROC Curves (One-vs-Rest)
  - Precision-Recall Curves (One-vs-Rest)
  - Summary metrics table saved to CSV
  - Confusion matrix (optional aggregated view)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from itertools import cycle
import pandas as pd

# ---------------- CONFIGURATION ----------------
RESULTS_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/results/resnet50/driverslicense"
CLASS_NAMES = ['genuine', 'copy-move', 'face-morph', 'face-replace', 'combined']
NUM_CLASSES = len(CLASS_NAMES)
os.makedirs(os.path.join(RESULTS_DIR, "analysis"), exist_ok=True)
SAVE_DIR = os.path.join(RESULTS_DIR, "analysis")
# ------------------------------------------------


def load_all_folds(results_dir):
    """Loads y_true, y_pred, and y_probs from all saved folds."""
    y_true_all, y_pred_all, y_probs_all = [], [], []
    for file in os.listdir(results_dir):
        if file.startswith("preds_fold_") and file.endswith(".npz"):
            data = np.load(os.path.join(results_dir, file))
            y_true_all.append(data["y_true"])
            y_pred_all.append(data["y_pred"])
            y_probs_all.append(data["y_probs"])
    if not y_true_all:
        raise FileNotFoundError("No preds_fold_*.npz files found in results directory.")
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(y_probs_all)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Aggregated Confusion Matrix (All Folds)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_all_folds.png"))
    plt.close()


def plot_roc_curves(y_true, y_probs):
    plt.figure(figsize=(8, 6))
    colors = cycle(['b', 'g', 'r', 'c', 'm'])
    for i, color in zip(range(NUM_CLASSES), colors):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "roc_curves_multiclass.png"))
    plt.close()


def plot_precision_recall_curves(y_true, y_probs):
    plt.figure(figsize=(8, 6))
    colors = cycle(['b', 'g', 'r', 'c', 'm'])
    for i, color in zip(range(NUM_CLASSES), colors):
        precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        ap = average_precision_score(y_true == i, y_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2, label=f"{CLASS_NAMES[i]} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "precision_recall_multiclass.png"))
    plt.close()


def compute_and_save_metrics(y_true, y_pred, y_probs):
    """Computes macro/micro/weighted F1, per-class metrics, and saves as CSV."""
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(SAVE_DIR, "classification_report.csv"), index=True)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    summary = pd.DataFrame({
        "Metric": ["Macro F1", "Micro F1", "Weighted F1"],
        "Value": [macro_f1, micro_f1, weighted_f1]
    })
    summary.to_csv(os.path.join(SAVE_DIR, "f1_summary.csv"), index=False)

    print("\n=== Macro/Micro/Weighted F1 Scores ===")
    print(summary)
    print("\n=== Classification Report ===")
    print(report_df)


def main():
    print("Loading predictions from all folds...")
    y_true, y_pred, y_probs = load_all_folds(RESULTS_DIR)
    print(f"Loaded {len(y_true)} samples from all folds.")

    print("Computing and saving metrics...")
    compute_and_save_metrics(y_true, y_pred, y_probs)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)

    print("Plotting ROC curves...")
    plot_roc_curves(y_true, y_probs)

    print("Plotting Precision-Recall curves...")
    plot_precision_recall_curves(y_true, y_probs)

    print(f"\nAll analysis results saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()


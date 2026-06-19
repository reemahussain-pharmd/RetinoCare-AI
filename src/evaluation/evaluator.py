"""
Model Evaluation — confusion matrix, classification report, ROC-AUC, model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras


CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]
COLORS      = ["#4CAF50", "#FFC107", "#F44336"]


# ── Predictions ────────────────────────────────────────────────────────────────
def get_predictions(model, X_test: np.ndarray, batch_size: int = 32) -> tuple:
    probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ── Confusion Matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",  "d"),
        (axes[1], cm_norm, "Normalized",  ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual",    fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = Path(save_dir) / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {path}")
    return cm


# ── Classification Report ──────────────────────────────────────────────────────
def get_classification_report(y_true, y_pred, model_name: str) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(f"\nClassification Report — {model_name}")
    print(df.to_string())
    return df


# ── ROC-AUC ────────────────────────────────────────────────────────────────────
def plot_roc_curves(y_true, y_probs, model_name: str, save_dir: str = "outputs") -> float:
    Path(save_dir).mkdir(exist_ok=True)
    n_classes = y_probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        fpr_dict[i], tpr_dict[i], auc_dict[i] = fpr, tpr, auc(fpr, tpr)

    macro_auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (color, name) in enumerate(zip(COLORS, CLASS_NAMES)):
        ax.plot(fpr_dict[i], tpr_dict[i], color=color, linewidth=2,
                label=f"{name} (AUC = {auc_dict[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}\nMacro-AUC = {macro_auc:.4f}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = Path(save_dir) / f"{model_name}_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved: {path}")
    return macro_auc


# ── Precision-Recall ───────────────────────────────────────────────────────────
def plot_precision_recall(y_true, y_probs, model_name: str, save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    n_classes = y_probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (color, name) in enumerate(zip(COLORS, CLASS_NAMES)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {model_name}", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = Path(save_dir) / f"{model_name}_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Full Evaluation ────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    save_dir: str = "outputs",
) -> dict:
    preds, probs = get_predictions(model, X_test)
    cm           = plot_confusion_matrix(y_test, preds, model_name, save_dir)
    report_df    = get_classification_report(y_test, preds, model_name)
    macro_auc    = plot_roc_curves(y_test, probs, model_name, save_dir)
    plot_precision_recall(y_test, probs, model_name, save_dir)

    report = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
    return {
        "model_name": model_name,
        "accuracy":   report["accuracy"],
        "precision":  report["macro avg"]["precision"],
        "recall":     report["macro avg"]["recall"],
        "f1":         report["macro avg"]["f1-score"],
        "auc":        macro_auc,
        "confusion_matrix": cm.tolist(),
    }


# ── Model Comparison Table ─────────────────────────────────────────────────────
def build_comparison_table(results: dict, save_dir: str = "outputs") -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(metrics.get("accuracy",  0), 4),
            "Precision": round(metrics.get("precision", 0), 4),
            "Recall":    round(metrics.get("recall",    0), 4),
            "F1 Score":  round(metrics.get("f1",        0), 4),
            "AUC":       round(metrics.get("auc",       0), 4),
        })

    df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    # Highlight best model
    best_idx = df["AUC"].idxmax()
    print(f"\n{'='*60}")
    print("MODEL COMPARISON TABLE")
    print(df.to_string(index=False))
    print(f"\nBest Model: {df.loc[best_idx, 'Model']} (AUC={df.loc[best_idx,'AUC']:.4f})")
    print(f"{'='*60}")

    # Save table
    Path(save_dir).mkdir(exist_ok=True)
    df.to_csv(Path(save_dir) / "model_comparison.csv", index=False)

    # Visual bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    width = 0.15
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(df["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df


# ── Error Analysis ─────────────────────────────────────────────────────────────
def error_analysis(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_dir: str = "outputs",
) -> dict:
    """Identify misclassified images and visualize hard cases."""
    errors = np.where(y_test != y_pred)[0]
    confident_errors = errors[y_probs[errors].max(axis=1) > 0.8]

    fp = {}; fn = {}
    for cls_idx in range(5):
        fp[CLASS_NAMES[cls_idx]] = np.where((y_pred == cls_idx) & (y_test != cls_idx))[0]
        fn[CLASS_NAMES[cls_idx]] = np.where((y_pred != cls_idx) & (y_test == cls_idx))[0]

    print(f"\nError Analysis:")
    print(f"  Total misclassified  : {len(errors)} / {len(y_test)} ({100*len(errors)/len(y_test):.1f}%)")
    print(f"  High-confidence wrong: {len(confident_errors)}")

    if len(errors) > 0:
        n_show = min(8, len(errors))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Misclassified Samples", fontsize=13, fontweight="bold")
        for ax, idx in zip(axes.flat, errors[:n_show]):
            disp_img = X_test[idx]
            disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min() + 1e-8)
            ax.imshow(np.clip(disp_img, 0, 1))
            conf = y_probs[idx].max()
            ax.set_title(
                f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]} ({conf:.2f})",
                fontsize=8
            )
            ax.axis("off")
        for ax in axes.flat[n_show:]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "error_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "total_errors": int(len(errors)),
        "confident_errors": int(len(confident_errors)),
        "false_positives": {k: len(v) for k, v in fp.items()},
        "false_negatives": {k: len(v) for k, v in fn.items()},
    }

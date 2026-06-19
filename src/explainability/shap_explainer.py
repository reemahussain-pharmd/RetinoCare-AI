"""
SHAP-based image explainability for retinopathy model predictions.
Uses DeepExplainer for per-pixel importance attribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import shap
import tensorflow as tf
from tensorflow import keras


CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


class SHAPExplainer:
    """SHAP DeepExplainer wrapper for Keras image classifiers."""

    def __init__(self, model: keras.Model, background: np.ndarray):
        """
        Args:
            model:      Trained Keras model.
            background: Small sample of training images for background (50-200 images).
        """
        self.model      = model
        self.explainer  = shap.DeepExplainer(model, background)

    def explain(self, images: np.ndarray) -> np.ndarray:
        """Compute SHAP values. Returns array of shape [n_classes, n_images, H, W, C]."""
        shap_values = self.explainer.shap_values(images)
        return shap_values

    def visualize_single(
        self,
        image: np.ndarray,
        pred_class: int,
        shap_values: np.ndarray = None,
        save_path: str = None,
    ) -> None:
        """Visualize SHAP values for a single image and predicted class."""
        if shap_values is None:
            shap_values = self.explain(np.expand_dims(image, 0))

        # SHAP values per class for this image: [n_classes, 1, H, W, C]
        class_shap = shap_values[pred_class][0]
        importance = class_shap.mean(axis=-1)  # average over channels

        # Normalize for display
        disp = image.copy()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"SHAP Explanation — Predicted: {CLASS_NAMES[pred_class]}\n"
            "Blue = reduces prediction confidence | Red = increases confidence",
            fontsize=12, fontweight="bold",
        )

        axes[0].imshow(np.clip(disp, 0, 1))
        axes[0].set_title("Retinal Image"); axes[0].axis("off")

        vmax = np.abs(importance).max()
        im = axes[1].imshow(importance, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("SHAP Attribution Map"); axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        overlay = np.clip(disp, 0, 1).copy()
        norm_imp = importance / (vmax + 1e-8)
        red_mask  = np.clip(norm_imp,  0, 1)
        blue_mask = np.clip(-norm_imp, 0, 1)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + red_mask  * 0.5, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + blue_mask * 0.5, 0, 1)

        axes[2].imshow(overlay)
        axes[2].set_title("SHAP Overlay on Retina"); axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_multiclass(
        self,
        image: np.ndarray,
        shap_values: np.ndarray = None,
        save_path: str = None,
    ) -> None:
        """Show SHAP maps for all 5 classes for one image."""
        if shap_values is None:
            shap_values = self.explain(np.expand_dims(image, 0))

        disp = image.copy()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("SHAP Values — All Retinopathy Classes", fontsize=14, fontweight="bold")

        axes[0, 0].imshow(np.clip(disp, 0, 1))
        axes[0, 0].set_title("Original Image"); axes[0, 0].axis("off")

        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for cls_idx, (r, c) in enumerate(positions):
            imp = shap_values[cls_idx][0].mean(axis=-1)
            vmax = np.abs(imp).max()
            axes[r, c].imshow(imp, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[r, c].set_title(CLASS_NAMES[cls_idx]); axes[r, c].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def feature_importance_summary(
        self,
        images: np.ndarray,
        n_samples: int = 50,
        save_path: str = None,
    ) -> None:
        """Aggregate pixel importance across samples using beeswarm-style summary."""
        subset = images[:n_samples]
        shap_values = self.explain(subset)

        # Use class 0 importance as example
        flat_shap = shap_values[0].reshape(n_samples, -1)
        flat_imgs = subset.reshape(n_samples, -1)

        # Top 50 pixels by mean |SHAP|
        mean_abs   = np.abs(flat_shap).mean(axis=0)
        top_pixels = np.argsort(mean_abs)[-50:][::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(50), mean_abs[top_pixels], color="#2196F3")
        ax.set_yticks(range(50))
        ax.set_yticklabels([f"Pixel-{p}" for p in top_pixels], fontsize=7)
        ax.set_xlabel("Mean |SHAP| Value")
        ax.set_title("Top-50 Important Pixels (No DR Class)", fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

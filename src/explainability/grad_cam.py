"""
Grad-CAM and Saliency Map implementation for retinopathy explainability.
Generates clinician-friendly visual explanations of model predictions.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]
SEVERITY_COLORS = {
    "No/Mild DR":             "#4CAF50",
    "Moderate DR":            "#FFC107",
    "Severe/Proliferative DR":"#F44336",
}


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: keras.Model, last_conv_layer: str = None):
        self.model = model
        self.layer_name = last_conv_layer or self._find_last_conv(model)

    @staticmethod
    def _find_last_conv(model: keras.Model) -> str:
        # Search top-level layers first (including sub-models like MobileNetV2 base)
        for layer in reversed(model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                return layer.name
            # If this layer is itself a model (e.g. MobileNetV2 base), search inside it
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                        return layer.name  # return the sub-model's name, not sub-layer
        raise ValueError("No Conv2D layer found in model.")

    def compute_heatmap(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        """Return normalized Grad-CAM heatmap for img_array [H,W,C] or [1,H,W,C]."""
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, 0)

        layer = self.model.get_layer(self.layer_name)
        # For nested sub-models (e.g. MobileNetV2 called as a layer), .output
        # refers to the sub-model's own input graph; get_output_at(0) returns
        # the tensor in the OUTER model's graph (connected to self.model.inputs).
        try:
            layer_out = layer.get_output_at(0)
        except AttributeError:
            layer_out = layer.output
        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[layer_out, self.model.output],
        )

        with tf.GradientTape() as tape:
            inputs     = tf.cast(img_array, tf.float32)
            conv_outs, preds = grad_model(inputs)
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads  = tape.gradient(loss, conv_outs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_outs[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    def overlay_heatmap(
        self,
        img_array: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on original image."""
        # Denormalize for display
        disp = img_array.copy()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
        disp = (disp * 255).astype(np.uint8)

        heat_resized = cv2.resize(heatmap, (disp.shape[1], disp.shape[0]))
        heat_uint8   = (heat_resized * 255).astype(np.uint8)
        heat_colored = cv2.applyColorMap(heat_uint8, colormap)
        heat_rgb     = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(disp, 1 - alpha, heat_rgb, alpha, 0)

    def explain(
        self,
        img_array: np.ndarray,
        class_idx: int = None,
        class_name: str = None,
        save_path: str = None,
        show: bool = True,
    ) -> np.ndarray:
        """Full Grad-CAM explanation with clinical annotation."""
        heatmap  = self.compute_heatmap(img_array, class_idx)
        overlaid = self.overlay_heatmap(img_array, heatmap)

        disp = img_array.copy()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Grad-CAM Explanation — {class_name or 'Predicted Class'}\n"
            "Highlighted regions are the retinal areas influencing the AI prediction.",
            fontsize=12, fontweight="bold",
        )

        axes[0].imshow(np.clip(disp, 0, 1))
        axes[0].set_title("Original Retinal Image")
        axes[0].axis("off")

        im = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap\n(Red = High Importance)")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(overlaid)
        axes[2].set_title("Overlay on Retina\n(Clinical View)")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return overlaid


# ── Saliency Maps ──────────────────────────────────────────────────────────────
class SaliencyMap:
    """Vanilla gradient saliency map."""

    def __init__(self, model: keras.Model):
        self.model = model

    def compute(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, 0)

        img_tensor = tf.Variable(tf.cast(img_array, tf.float32))
        with tf.GradientTape() as tape:
            preds = self.model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads    = tape.gradient(loss, img_tensor)[0].numpy()
        saliency = np.max(np.abs(grads), axis=-1)
        if saliency.max() > 0:
            saliency /= saliency.max()
        return saliency

    def visualize(
        self,
        img_array: np.ndarray,
        class_name: str = "",
        save_path: str = None,
    ) -> None:
        saliency = self.compute(img_array)
        disp = img_array.copy()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Saliency Map — {class_name}", fontsize=12, fontweight="bold")

        axes[0].imshow(np.clip(disp, 0, 1))
        axes[0].set_title("Original"); axes[0].axis("off")

        axes[1].imshow(saliency, cmap="hot")
        axes[1].set_title("Attention Regions"); axes[1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ── Batch XAI Report ───────────────────────────────────────────────────────────
def generate_xai_batch(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_samples: int = 10,
    save_dir: str = "outputs/xai",
):
    """Generate Grad-CAM and saliency for n_samples per class."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    gcam  = GradCAM(model)
    smap  = SaliencyMap(model)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        indices = np.where(preds == cls_idx)[0][:n_samples]
        for i, idx in enumerate(indices):
            tag = f"{cls_name.replace(' ', '_')}_sample{i}"
            gcam.explain(
                images[idx], class_idx=cls_idx, class_name=cls_name,
                save_path=f"{save_dir}/{tag}_gradcam.png", show=False,
            )
            smap.visualize(
                images[idx], class_name=cls_name,
                save_path=f"{save_dir}/{tag}_saliency.png",
            )
    print(f"XAI outputs saved to {save_dir}/")

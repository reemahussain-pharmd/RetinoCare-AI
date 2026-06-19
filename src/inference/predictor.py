"""
Inference engine — load model, preprocess one image, return structured prediction.
Used by both the Streamlit app and the REST API.
"""

import numpy as np
from pathlib import Path
from tensorflow import keras

from src.preprocessing.image_preprocessor import preprocess_single, load_image, resize_image
from src.explainability.grad_cam import GradCAM


CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]

SEVERITY_INFO = {
    "No/Mild DR": {
        "risk":  "Low",
        "color": "#4CAF50",
        "recommendation": (
            "No or mild retinopathy detected. Continue routine annual screening. "
            "Maintain good glycemic control and blood pressure management."
        ),
    },
    "Moderate DR": {
        "risk":  "Moderate",
        "color": "#FFC107",
        "recommendation": (
            "Moderate retinopathy present. Refer to ophthalmologist within 3-6 months. "
            "Intensified systemic risk factor management is strongly advised."
        ),
    },
    "Severe/Proliferative DR": {
        "risk":  "High / Critical",
        "color": "#F44336",
        "recommendation": (
            "Severe or proliferative retinopathy — URGENT ophthalmology referral required. "
            "High risk of vision loss. Treatment (laser/anti-VEGF/vitrectomy) may be needed immediately."
        ),
    },
}

DISCLAIMER = (
    "IMPORTANT DISCLAIMER: This AI system is an assistive diagnostic tool and is NOT a replacement "
    "for professional ophthalmological evaluation. All predictions must be reviewed and confirmed by a "
    "qualified ophthalmologist before clinical decisions are made. This tool has limitations and may "
    "produce incorrect results. Clinical judgement must always take precedence."
)


# ── Predictor Class ────────────────────────────────────────────────────────────
class RetinopathyPredictor:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model      = keras.models.load_model(model_path)
        self._grad_cam  = None  # lazy init — avoids crash on nested-model architectures
        print(f"Model loaded from {model_path}")

    @property
    def grad_cam(self):
        if self._grad_cam is None:
            self._grad_cam = GradCAM(self.model)
        return self._grad_cam

    def predict_from_path(self, image_path: str) -> dict:
        """Predict retinopathy class from an image file path."""
        img_array = preprocess_single(image_path, apply_clahe=True)
        return self._predict(img_array, image_path)

    def predict_from_array(self, img_array: np.ndarray) -> dict:
        """Predict from a preprocessed numpy array [H,W,3]."""
        return self._predict(img_array)

    def _predict(self, img_array: np.ndarray, source: str = "uploaded") -> dict:
        inp   = np.expand_dims(img_array, 0)
        probs = self.model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(probs))
        name  = CLASS_NAMES[idx]
        conf  = float(probs[idx])

        info = SEVERITY_INFO[name]

        return {
            "predicted_class":  name,
            "class_index":      idx,
            "confidence":       round(conf * 100, 2),
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "risk_level":       info["risk"],
            "risk_color":       info["color"],
            "clinical_recommendation": info["recommendation"],
            "disclaimer":       DISCLAIMER,
            "image_array":      img_array,
        }

    def predict_with_gradcam(self, image_path: str, save_path: str = None) -> dict:
        """Predict and generate Grad-CAM visualization."""
        result    = self.predict_from_path(image_path)
        img_array = result["image_array"]
        cls_idx   = result["class_index"]

        overlaid  = self.grad_cam.explain(
            img_array, class_idx=cls_idx,
            class_name=result["predicted_class"],
            save_path=save_path, show=False,
        )
        result["gradcam_overlay"] = overlaid
        return result

    def test_time_augmentation(self, image_path: str, n_augments: int = 10) -> dict:
        """TTA: average predictions over n_augments random augmentations."""
        import albumentations as A
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])

        base_img = load_image(image_path)
        base_img = resize_image(base_img)

        all_probs = []
        for _ in range(n_augments):
            aug_img = aug(image=base_img)["image"].astype(np.float32) / 255.0
            inp     = np.expand_dims(aug_img, 0)
            probs   = self.model.predict(inp, verbose=0)[0]
            all_probs.append(probs)

        mean_probs = np.mean(all_probs, axis=0)
        idx  = int(np.argmax(mean_probs))
        name = CLASS_NAMES[idx]
        info = SEVERITY_INFO[name]

        return {
            "predicted_class":  name,
            "class_index":      idx,
            "confidence":       round(float(mean_probs[idx]) * 100, 2),
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(mean_probs[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "risk_level":       info["risk"],
            "risk_color":       info["color"],
            "clinical_recommendation": info["recommendation"],
            "tta_runs":         n_augments,
            "disclaimer":       DISCLAIMER,
        }

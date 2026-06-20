"""
RetinoCare AI — Inference Engine v2.0
Predict retinopathy class, compute reliability score, and generate structured
clinical results for downstream use in the Streamlit app and PDF reports.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from tensorflow import keras

from src.preprocessing.image_preprocessor import preprocess_single, load_image, resize_image
from src.explainability.grad_cam import GradCAM


MODEL_VERSION  = "MobileNetV2-v1.0"
CLASS_NAMES    = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]

SEVERITY_INFO = {
    "No/Mild DR": {
        "risk":        "Low",
        "color":       "#27AE60",
        "urgency":     "Routine",
        "follow_up":   "Annual screening",
        "icd10":       "E11.319",
        "recommendation": (
            "No or minimal retinopathy detected. Continue routine annual fundus screening. "
            "Maintain good glycemic control (HbA1c < 7.0%) and blood pressure management "
            "(< 130/80 mmHg)."
        ),
        "clinical_summary": (
            "The AI model detected minimal or no signs of diabetic retinopathy. "
            "The retinal vasculature appears within normal limits for this diabetic patient. "
            "Continued preventive care and regular monitoring are recommended."
        ),
    },
    "Moderate DR": {
        "risk":        "Moderate",
        "color":       "#F39C12",
        "urgency":     "Non-Urgent Referral",
        "follow_up":   "3–6 months",
        "icd10":       "E11.339",
        "recommendation": (
            "Moderate non-proliferative diabetic retinopathy (NPDR) identified. "
            "Ophthalmology referral within 3–6 months is strongly recommended. "
            "Intensified systemic risk factor management is advised."
        ),
        "clinical_summary": (
            "Moderate NPDR features detected, consistent with microaneurysms, dot-blot "
            "haemorrhages, or hard exudates. These findings indicate progression of "
            "microvascular disease. Timely specialist evaluation is essential to prevent "
            "sight-threatening progression."
        ),
    },
    "Severe/Proliferative DR": {
        "risk":        "High / Critical",
        "color":       "#E74C3C",
        "urgency":     "URGENT Referral",
        "follow_up":   "Immediate (24–72 hours)",
        "icd10":       "E11.359",
        "recommendation": (
            "Severe or proliferative diabetic retinopathy (PDR) detected — URGENT "
            "ophthalmology referral required within 24–72 hours. High risk of vision loss. "
            "Treatment options include panretinal photocoagulation, anti-VEGF injections, "
            "or vitreoretinal surgery."
        ),
        "clinical_summary": (
            "Severe NPDR or PDR features identified, which may include neovascularisation, "
            "vitreous haemorrhage, or tractional retinal detachment risk. This represents "
            "a sight-threatening emergency requiring immediate specialist consultation to "
            "preserve visual function."
        ),
    },
}

DISCLAIMER = (
    "IMPORTANT DISCLAIMER: This AI system is an assistive diagnostic tool and is NOT a "
    "replacement for professional ophthalmological evaluation. All predictions must be "
    "reviewed and confirmed by a qualified ophthalmologist before clinical decisions are "
    "made. Clinical judgement must always take precedence."
)


def _compute_reliability(probs: np.ndarray) -> dict:
    """Reliability score from entropy and prediction margin."""
    sorted_p     = np.sort(probs)[::-1]
    margin       = float(sorted_p[0] - sorted_p[1])
    entropy      = float(-np.sum(probs * np.log(probs + 1e-10)))
    norm_entropy = entropy / np.log(len(probs))

    score = round((margin * 0.65 + (1 - norm_entropy) * 0.35) * 100, 1)
    score = max(0.0, min(100.0, score))

    if score >= 72:   level, color = "High",   "#27AE60"
    elif score >= 48: level, color = "Medium", "#F39C12"
    else:             level, color = "Low",    "#E74C3C"

    return {
        "reliability_score": score,
        "reliability_level": level,
        "reliability_color": color,
        "entropy":           round(entropy, 3),
        "prediction_margin": round(margin * 100, 1),
    }


class RetinopathyPredictor:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model      = keras.models.load_model(model_path)
        self._grad_cam  = None
        print(f"[RetinopathyPredictor] Loaded: {model_path}")

    @property
    def grad_cam(self):
        if self._grad_cam is None:
            self._grad_cam = GradCAM(self.model)
        return self._grad_cam

    def predict_from_path(self, image_path: str) -> dict:
        img_array = preprocess_single(image_path, apply_clahe=True)
        return self._predict(img_array)

    def predict_from_array(self, img_array: np.ndarray) -> dict:
        return self._predict(img_array)

    def _predict(self, img_array: np.ndarray) -> dict:
        inp   = np.expand_dims(img_array, 0)
        probs = self.model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(probs))
        name  = CLASS_NAMES[idx]
        conf  = float(probs[idx])
        info  = SEVERITY_INFO[name]

        reliability = _compute_reliability(probs)

        conf_pct = round(conf * 100, 2)
        if conf_pct >= 80:   conf_level, conf_color = "High",   "#27AE60"
        elif conf_pct >= 60: conf_level, conf_color = "Medium", "#F39C12"
        else:                conf_level, conf_color = "Low",    "#E74C3C"

        top3 = sorted(
            [(CLASS_NAMES[i], round(float(probs[i]) * 100, 2)) for i in range(len(CLASS_NAMES))],
            key=lambda x: x[1], reverse=True,
        )

        return {
            # Core prediction
            "predicted_class":    name,
            "class_index":        idx,
            "confidence":         conf_pct,
            "confidence_level":   conf_level,
            "confidence_color":   conf_color,
            "all_probabilities":  {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                                   for i in range(len(CLASS_NAMES))},
            "raw_probabilities":  probs.tolist(),
            "top3_predictions":   top3,
            # Clinical info
            "risk_level":         info["risk"],
            "risk_color":         info["color"],
            "urgency":            info["urgency"],
            "follow_up":          info["follow_up"],
            "icd10":              info["icd10"],
            "clinical_recommendation": info["recommendation"],
            "clinical_summary":   info["clinical_summary"],
            # Reliability
            **reliability,
            # Metadata
            "model_version":      MODEL_VERSION,
            "timestamp":          datetime.now().isoformat(),
            "disclaimer":         DISCLAIMER,
            "image_array":        img_array,
        }

    def predict_with_gradcam(self, image_path: str, save_path: str = None) -> dict:
        result    = self.predict_from_path(image_path)
        img_array = result["image_array"]
        cls_idx   = result["class_index"]

        overlaid = self.grad_cam.explain(
            img_array, class_idx=cls_idx,
            class_name=result["predicted_class"],
            save_path=save_path, show=False,
        )
        result["gradcam_overlay"] = overlaid
        return result

    def test_time_augmentation(self, image_path: str, n_augments: int = 10) -> dict:
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
            all_probs.append(self.model.predict(inp, verbose=0)[0])

        mean_probs = np.mean(all_probs, axis=0)
        std_probs  = np.std(all_probs, axis=0)
        idx   = int(np.argmax(mean_probs))
        name  = CLASS_NAMES[idx]
        info  = SEVERITY_INFO[name]
        conf  = float(mean_probs[idx])
        reliability = _compute_reliability(mean_probs)

        conf_pct = round(conf * 100, 2)
        if conf_pct >= 80:   conf_level, conf_color = "High",   "#27AE60"
        elif conf_pct >= 60: conf_level, conf_color = "Medium", "#F39C12"
        else:                conf_level, conf_color = "Low",    "#E74C3C"

        return {
            "predicted_class":    name,
            "class_index":        idx,
            "confidence":         conf_pct,
            "confidence_level":   conf_level,
            "confidence_color":   conf_color,
            "all_probabilities":  {CLASS_NAMES[i]: round(float(mean_probs[i]) * 100, 2)
                                   for i in range(len(CLASS_NAMES))},
            "raw_probabilities":  mean_probs.tolist(),
            "top3_predictions":   sorted(
                [(CLASS_NAMES[i], round(float(mean_probs[i]) * 100, 2)) for i in range(3)],
                key=lambda x: x[1], reverse=True,
            ),
            "tta_std":            {CLASS_NAMES[i]: round(float(std_probs[i]) * 100, 3)
                                   for i in range(len(CLASS_NAMES))},
            "tta_runs":           n_augments,
            "risk_level":         info["risk"],
            "risk_color":         info["color"],
            "urgency":            info["urgency"],
            "follow_up":          info["follow_up"],
            "icd10":              info["icd10"],
            "clinical_recommendation": info["recommendation"],
            "clinical_summary":   info["clinical_summary"],
            **reliability,
            "model_version":      MODEL_VERSION,
            "timestamp":          datetime.now().isoformat(),
            "disclaimer":         DISCLAIMER,
        }

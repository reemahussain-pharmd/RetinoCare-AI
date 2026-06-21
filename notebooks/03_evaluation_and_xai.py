"""
Notebook 03 — Model Evaluation, Comparison, and Explainable AI
Full evaluation pipeline with Grad-CAM, SHAP, and saliency maps.
"""

# %% [markdown]
# # Model Evaluation & Explainability — RetinaIQ

# %%
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras

from src.evaluation.evaluator import (
    evaluate_model, build_comparison_table, error_analysis,
    plot_confusion_matrix, plot_roc_curves, get_predictions
)
from src.explainability.grad_cam import GradCAM, SaliencyMap, generate_xai_batch

# ── Load data & models ──────────────────────────────────────────────────────────
PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
XAI_DIR    = OUTPUT_DIR / "xai"; XAI_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

images = np.load(PROC_DIR / "images.npy")
labels = np.load(PROC_DIR / "labels.npy")

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    images, labels, test_size=0.15, stratify=labels, random_state=42
)
print(f"Test set: {X_test.shape}")

# %% [markdown]
# ## 1. Evaluate All Saved Models

# %%
model_names = ["cnn", "mobilenetv2", "resnet50",
               "efficientnetb0", "efficientnetb3", "densenet121", "inceptionv3"]

eval_results = {}
trained_models = {}

for name in model_names:
    ckpt = MODEL_DIR / f"{name}_best.h5"
    if not ckpt.exists():
        print(f"[SKIP] {name} — checkpoint not found")
        continue

    print(f"\nEvaluating {name}...")
    model = keras.models.load_model(str(ckpt))
    trained_models[name] = model

    metrics = evaluate_model(model, X_test, y_test, name, str(OUTPUT_DIR))
    eval_results[name] = metrics

# %% [markdown]
# ## 2. Model Comparison Table

# %%
comparison_df = build_comparison_table(eval_results, str(OUTPUT_DIR))
display(comparison_df)

# %% [markdown]
# ## 3. Best Model Selection & Justification

# %%
best_name = comparison_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"""
{'='*60}
BEST MODEL: {best_name}
{'='*60}
Accuracy  : {comparison_df.iloc[0]['Accuracy']:.4f}
Precision : {comparison_df.iloc[0]['Precision']:.4f}
Recall    : {comparison_df.iloc[0]['Recall']:.4f}
F1 Score  : {comparison_df.iloc[0]['F1 Score']:.4f}
AUC       : {comparison_df.iloc[0]['AUC']:.4f}
{'='*60}

BUSINESS JUSTIFICATION:
• Highest AUC indicates best discrimination across all severity classes
• F1 score balances precision/recall — important in medical screening
• Model selected for deployment; saved as models/best_model.h5
""")

best_model.save(str(MODEL_DIR / "best_model.h5"))

# %% [markdown]
# ## 4. Error Analysis

# %%
preds, probs = get_predictions(best_model, X_test)
errors = error_analysis(X_test, y_test, preds, probs, str(OUTPUT_DIR))

print(f"\nError Summary:")
print(f"  Total errors        : {errors['total_errors']}")
print(f"  Confident errors    : {errors['confident_errors']}")
print(f"\nFalse Positives per class:")
for cls, count in errors['false_positives'].items():
    print(f"  {cls}: {count}")
print(f"\nFalse Negatives per class (missed cases):")
for cls, count in errors['false_negatives'].items():
    print(f"  {cls}: {count}")

print("""
IMPROVEMENT RECOMMENDATIONS:
  1. Augment training data for minority classes (Severe, Proliferative)
  2. Apply test-time augmentation (TTA) to reduce prediction variance
  3. Consider ensemble of top-3 models for hard cases
  4. Fine-tune on domain-specific pre-trained retinal weights if available
""")

# %% [markdown]
# ## 5. Grad-CAM Explainability

# %%
print("Generating Grad-CAM visualizations...")
gcam = GradCAM(best_model)
smap = SaliencyMap(best_model)

# One sample per class
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_test == cls_idx)[0]
    if len(class_indices) == 0:
        continue

    idx       = class_indices[0]
    img_array = X_test[idx]

    # Grad-CAM
    gcam.explain(
        img_array, class_idx=cls_idx, class_name=cls_name,
        save_path=str(XAI_DIR / f"gradcam_{cls_name.replace(' ', '_')}.png"),
        show=False,
    )

    # Saliency
    smap.visualize(
        img_array, class_name=cls_name,
        save_path=str(XAI_DIR / f"saliency_{cls_name.replace(' ', '_')}.png"),
    )

print(f"XAI outputs saved to {XAI_DIR}/")

# %% [markdown]
# ## 6. SHAP Analysis

# %%
try:
    from src.explainability.shap_explainer import SHAPExplainer

    # Use 100 training images as background
    background_size = min(100, len(images))
    background = images[:background_size]

    shap_exp = SHAPExplainer(best_model, background)

    # Explain one test sample
    sample_img  = X_test[0:1]
    shap_values = shap_exp.explain(sample_img)

    pred_class = int(np.argmax(best_model.predict(sample_img, verbose=0)[0]))
    shap_exp.visualize_single(
        X_test[0], pred_class, shap_values,
        save_path=str(XAI_DIR / "shap_single_explanation.png"),
    )
    shap_exp.visualize_multiclass(
        X_test[0], shap_values,
        save_path=str(XAI_DIR / "shap_multiclass.png"),
    )
    shap_exp.feature_importance_summary(
        X_test[:50], save_path=str(XAI_DIR / "shap_summary.png")
    )
    print("SHAP analysis complete.")
except Exception as e:
    print(f"SHAP skipped: {e}")

# %% [markdown]
# ## 7. Healthcare AI Interpretation

# %%
print("""
╔══════════════════════════════════════════════════════════════╗
║           HEALTHCARE AI CLINICAL INTERPRETATION              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CLASS: No DR (Grade 0)                                      ║
║  → No retinal lesions detected. Annual screening adequate.   ║
║                                                              ║
║  CLASS: Mild DR (Grade 1)                                    ║
║  → Microaneurysms only. Optimize metabolic control.          ║
║    Follow-up in 6-12 months.                                 ║
║                                                              ║
║  CLASS: Moderate DR (Grade 2)                                ║
║  → Microaneurysms + haemorrhages. Hard exudates possible.    ║
║    Ophthalmology referral within 3-6 months.                 ║
║                                                              ║
║  CLASS: Severe DR (Grade 3)                                  ║
║  → Intraretinal microvascular abnormalities. Cotton-wool     ║
║    spots. Urgent referral within 1 month.                    ║
║                                                              ║
║  CLASS: Proliferative DR (Grade 4)                           ║
║  → Neovascularization, vitreous haemorrhage possible.        ║
║    IMMEDIATE ophthalmology referral. Treatment urgently      ║
║    required (PRP laser / anti-VEGF / vitrectomy).           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  ⚠ AI is an ASSISTIVE TOOL — NOT a replacement for          ║
║    ophthalmologist evaluation. All predictions require       ║
║    clinical validation before patient management decisions.  ║
╚══════════════════════════════════════════════════════════════╝
""")

print("\n✅ Evaluation & XAI complete. Launch app: streamlit run app/streamlit_app.py")

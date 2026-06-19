"""
RetinoCare AI — Main Pipeline Orchestrator
Runs EDA → Preprocessing → Training → Evaluation → Explainability in sequence.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras

# ── Project Paths ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_CSV   = ROOT / "data" / "raw" / "labels.csv"
IMAGE_DIR  = ROOT / "data" / "raw" / "images"
PROC_DIR   = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for d in [PROC_DIR, MODEL_DIR, OUTPUT_DIR, OUTPUT_DIR / "eda", OUTPUT_DIR / "xai"]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]


# ── CLI Arguments ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RetinoCare AI Pipeline")
    p.add_argument("--mode",    default="full",
                   choices=["full", "train", "evaluate", "predict", "eda"],
                   help="Pipeline mode")
    p.add_argument("--models",  nargs="+",
                   default=["cnn", "mobilenetv2", "resnet50",
                             "efficientnetb0", "efficientnetb3",
                             "densenet121", "inceptionv3"],
                   help="Models to train")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--image_path",  type=str,   default=None,
                   help="Path for single prediction")
    return p.parse_args()


# ── Step 1: EDA ────────────────────────────────────────────────────────────────
def run_eda():
    print("\n" + "="*60)
    print("STEP 1: Exploratory Data Analysis")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "01_EDA.py")], check=False)


# ── Step 2: Preprocess ─────────────────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame) -> tuple:
    print("\n" + "="*60)
    print("STEP 2: Image Preprocessing")
    print("="*60)
    from src.preprocessing.image_preprocessor import RetinopathyPreprocessor

    preprocessor = RetinopathyPreprocessor(str(IMAGE_DIR), apply_clahe=True)

    print("Processing images...")
    images, labels = preprocessor.process_dataframe(df)

    # Normalize to [0,1] range if not already
    if images.max() > 1.0:
        images = images / 255.0

    preprocessor.save_processed(images, labels, str(PROC_DIR))
    print(f"Preprocessed: {images.shape}, Labels: {labels.shape}")
    return images, labels


# ── Step 3: Train ──────────────────────────────────────────────────────────────
def run_training(
    X_train, y_train, X_val, y_val,
    model_names: list,
    epochs: int,
    batch_size: int,
) -> dict:
    print("\n" + "="*60)
    print("STEP 3: Model Training")
    print("="*60)
    from src.training.trainer import train_all_models

    results = train_all_models(
        X_train, y_train, X_val, y_val,
        model_names=model_names,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=str(MODEL_DIR),
    )
    return results


# ── Step 4: Evaluate ───────────────────────────────────────────────────────────
def run_evaluation(
    trained_results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    print("\n" + "="*60)
    print("STEP 4: Model Evaluation")
    print("="*60)
    from src.evaluation.evaluator import evaluate_model, build_comparison_table, error_analysis

    eval_results = {}
    for name, res in trained_results.items():
        print(f"\nEvaluating {name}...")
        model = res["model"]
        metrics = evaluate_model(model, X_test, y_test, name, str(OUTPUT_DIR))
        eval_results[name] = metrics

    # Comparison table
    comparison_df = build_comparison_table(eval_results, str(OUTPUT_DIR))

    # Find best model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model      = trained_results[best_model_name]["model"]
    print(f"\n🏆 Best Model: {best_model_name}")

    # Save best model
    best_path = str(MODEL_DIR / "best_model.h5")
    best_model.save(best_path)
    print(f"Best model saved: {best_path}")

    # Error analysis on best model
    from src.evaluation.evaluator import get_predictions
    preds, probs = get_predictions(best_model, X_test)
    error_analysis(X_test, y_test, preds, probs, str(OUTPUT_DIR))

    return eval_results, best_model


# ── Step 5: Explainability ─────────────────────────────────────────────────────
def run_explainability(
    best_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 5,
):
    print("\n" + "="*60)
    print("STEP 5: Explainability (Grad-CAM + Saliency)")
    print("="*60)
    from src.explainability.grad_cam import generate_xai_batch
    from src.evaluation.evaluator import get_predictions

    preds, probs = get_predictions(best_model, X_test)

    generate_xai_batch(
        best_model, X_test, y_test, preds, probs,
        n_samples=n_samples,
        save_dir=str(OUTPUT_DIR / "xai"),
    )


# ── Step 6: Single Prediction ──────────────────────────────────────────────────
def run_single_prediction(image_path: str):
    print(f"\nRunning prediction on: {image_path}")
    from src.inference.predictor import RetinopathyPredictor
    from src.inference.report_generator import generate_pdf_report

    model_path = str(MODEL_DIR / "best_model.h5")
    predictor  = RetinopathyPredictor(model_path)

    gcam_path = str(OUTPUT_DIR / "prediction_gradcam.png")
    result    = predictor.predict_with_gradcam(image_path, save_path=gcam_path)

    print(f"\n{'='*50}")
    print(f"Prediction     : {result['predicted_class']}")
    print(f"Confidence     : {result['confidence']:.2f}%")
    print(f"Risk Level     : {result['risk_level']}")
    print(f"Recommendation : {result['clinical_recommendation']}")
    print(f"{'='*50}")

    report_path = generate_pdf_report(
        result,
        original_image_path=image_path,
        gradcam_array=result.get("gradcam_overlay"),
        save_path=str(ROOT / "reports" / "prediction_report.pdf"),
    )
    print(f"Report: {report_path}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  RetinoCare AI — Production Pipeline")
    print("="*60)
    print(f"  Mode     : {args.mode}")
    print(f"  Models   : {args.models}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print("="*60 + "\n")

    # ── Single prediction shortcut ──────────────────────────────────────────
    if args.mode == "predict":
        if not args.image_path:
            print("Error: --image_path required for predict mode")
            sys.exit(1)
        run_single_prediction(args.image_path)
        return

    # ── EDA only ────────────────────────────────────────────────────────────
    if args.mode == "eda":
        run_eda()
        return

    # ── Load data ────────────────────────────────────────────────────────────
    if not DATA_CSV.exists():
        print(f"ERROR: CSV not found at {DATA_CSV}")
        print("Place your labels CSV in data/raw/labels.csv")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    print(f"Dataset loaded: {len(df)} samples, columns: {df.columns.tolist()}")

    filename_col = df.columns[0]
    label_col    = df.columns[1]

    # ── Load or preprocess ───────────────────────────────────────────────────
    proc_img_path = PROC_DIR / "images.npy"
    proc_lbl_path = PROC_DIR / "labels.npy"

    if proc_img_path.exists() and proc_lbl_path.exists():
        print("Loading preprocessed data from cache...")
        images = np.load(proc_img_path)
        labels = np.load(proc_lbl_path)
    else:
        images, labels = run_preprocessing(df)

    # ── Train/Val/Test Split ─────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.15, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42  # ~15% of total
    )

    print(f"\nData Split:")
    print(f"  Train : {X_train.shape[0]} ({100*len(X_train)/len(images):.0f}%)")
    print(f"  Val   : {X_val.shape[0]}   ({100*len(X_val)/len(images):.0f}%)")
    print(f"  Test  : {X_test.shape[0]}  ({100*len(X_test)/len(images):.0f}%)")

    # ── EDA ──────────────────────────────────────────────────────────────────
    if args.mode in ("full", "eda"):
        run_eda()

    # ── Training ─────────────────────────────────────────────────────────────
    if args.mode in ("full", "train"):
        trained = run_training(
            X_train, y_train, X_val, y_val,
            model_names=args.models,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        # Load saved models for evaluate-only mode
        trained = {}
        for name in args.models:
            ckpt = MODEL_DIR / f"{name}_best.h5"
            if ckpt.exists():
                trained[name] = {"model": keras.models.load_model(str(ckpt))}

    # ── Evaluation ───────────────────────────────────────────────────────────
    if args.mode in ("full", "train", "evaluate") and trained:
        eval_results, best_model = run_evaluation(trained, X_test, y_test)
        run_explainability(best_model, X_test, y_test)

    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print(f"   Outputs : {OUTPUT_DIR}")
    print(f"   Models  : {MODEL_DIR}")
    print(f"   Reports : {ROOT / 'reports'}")
    print("="*60)
    print("\nTo launch the Streamlit app:")
    print("  streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()

"""
Notebook 02 — Model Training & Hyperparameter Tuning
Trains all 7 models with proper callbacks and LR scheduling.
"""

# %% [markdown]
# # Model Training — Retinopathy Classification
# **RetinaIQ** | Deep Learning Pipeline

# %%
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

from src.training.model_builder import get_model, compile_model, model_summary_stats
from src.training.trainer import (
    train_model, plot_training_curves, learning_rate_finder
)

# ── Configuration ──────────────────────────────────────────────────────────────
PROC_DIR    = Path("data/processed")
MODEL_DIR   = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR  = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-4
NUM_CLASSES = 5

# %% [markdown]
# ## 1. Load Preprocessed Data

# %%
images = np.load(PROC_DIR / "images.npy")
labels = np.load(PROC_DIR / "labels.npy")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# %% [markdown]
# ## 2. Train/Val/Test Split

# %%
X_temp, X_test, y_temp, y_test = train_test_split(
    images, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 3. Model Architecture Summaries

# %%
model_names = ["cnn", "mobilenetv2", "resnet50",
               "efficientnetb0", "efficientnetb3", "densenet121", "inceptionv3"]

print(f"\n{'Model':<20} {'Total Params':>15} {'Trainable':>15}")
print("-" * 55)
for name in model_names:
    m    = get_model(name)
    info = model_summary_stats(m)
    print(f"{name:<20} {info['total']:>15,} {info['trainable']:>15,}")
    tf.keras.backend.clear_session()

# %% [markdown]
# ## 4. Learning Rate Finder (EfficientNetB0)

# %%
print("\nRunning LR finder for EfficientNetB0...")
lr_results = learning_rate_finder(
    "efficientnetb0", X_train, y_train,
    lr_range=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    batch_size=BATCH_SIZE,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(lr_results.keys()), list(lr_results.values()), "o-", color="#2196F3")
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)")
ax.set_ylabel("Validation Loss")
ax.set_title("Learning Rate Finder — EfficientNetB0")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lr_finder.png", dpi=150)
plt.show()

best_lr = min(lr_results, key=lr_results.get)
print(f"Best LR: {best_lr}")

# %% [markdown]
# ## 5. Train All Models

# %%
all_histories = {}
all_models    = {}

for model_name in model_names:
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")
    try:
        model, history = train_model(
            model_name,
            X_train, y_train, X_val, y_val,
            num_classes=NUM_CLASSES,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            checkpoint_dir=str(MODEL_DIR),
        )
        all_models[model_name]    = model
        all_histories[model_name] = history
        plot_training_curves(history, model_name, str(OUTPUT_DIR))
    except Exception as e:
        print(f"[SKIP] {model_name}: {e}")

# %% [markdown]
# ## 6. Hyperparameter Sensitivity Analysis

# %%
# Batch size comparison (quick 5-epoch test)
from src.training.trainer import build_tf_dataset
from tensorflow import keras

batch_sizes   = [16, 32, 64]
batch_results = {}

for bs in batch_sizes:
    m  = get_model("mobilenetv2")
    m  = compile_model(m, learning_rate=1e-4)
    yc = keras.utils.to_categorical(y_train, NUM_CLASSES)
    yv = keras.utils.to_categorical(y_val, NUM_CLASSES)

    ds_t = build_tf_dataset(X_train, yc, bs, augment=False)
    ds_v = build_tf_dataset(X_val,   yv, bs, augment=False, shuffle=False)

    h = m.fit(ds_t, validation_data=ds_v, epochs=5, verbose=0)
    batch_results[bs] = {
        "val_acc": max(h.history["val_accuracy"]),
        "val_loss": min(h.history["val_loss"]),
    }
    keras.backend.clear_session()

print("\nBatch Size Comparison (5 epochs):")
for bs, res in batch_results.items():
    print(f"  bs={bs:3d}: val_acc={res['val_acc']:.4f}, val_loss={res['val_loss']:.4f}")

# %% [markdown]
# ## 7. Dropout Sensitivity

# %%
dropout_vals = [0.2, 0.3, 0.5, 0.6]
dropout_results = {}

for dp in dropout_vals:
    m = get_model("mobilenetv2", dropout=dp)
    m = compile_model(m, learning_rate=1e-4)
    yc = keras.utils.to_categorical(y_train, NUM_CLASSES)
    yv = keras.utils.to_categorical(y_val,   NUM_CLASSES)

    ds_t = build_tf_dataset(X_train, yc, 32, augment=False)
    ds_v = build_tf_dataset(X_val,   yv, 32, augment=False, shuffle=False)

    h = m.fit(ds_t, validation_data=ds_v, epochs=5, verbose=0)
    dropout_results[dp] = max(h.history["val_accuracy"])
    keras.backend.clear_session()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(dropout_results.keys()), list(dropout_results.values()), "s-", color="#9C27B0")
ax.set_xlabel("Dropout Rate"); ax.set_ylabel("Val Accuracy")
ax.set_title("Dropout Rate Sensitivity"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dropout_sensitivity.png", dpi=150)
plt.show()

print("\nAll models trained. Proceed to notebook 03 for evaluation.")

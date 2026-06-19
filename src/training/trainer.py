"""
Training Pipeline with callbacks, LR scheduling, and hyperparameter search.
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from src.training.model_builder import get_model, compile_model, model_summary_stats


CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]


# ── Callbacks ──────────────────────────────────────────────────────────────────
def build_callbacks(model_name: str, checkpoint_dir: str = "models") -> list:
    ckpt_path = Path(checkpoint_dir) / f"{model_name}_best.h5"
    return [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10, restore_best_weights=True,
            mode="max", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5,
            min_lr=1e-7, verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"outputs/logs/{model_name}_{int(time.time())}",
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger(f"outputs/{model_name}_history.csv"),
    ]


# ── Data Generator ─────────────────────────────────────────────────────────────
def build_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:

    def _augment(img, lbl):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(len(images), seed=42)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Class Weights ──────────────────────────────────────────────────────────────
def compute_weights(labels_int: np.ndarray) -> dict:
    classes = np.unique(labels_int)
    weights = compute_class_weight("balanced", classes=classes, y=labels_int)
    return dict(zip(classes.tolist(), weights.tolist()))


# ── Single Model Training ──────────────────────────────────────────────────────
def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "models",
) -> tuple:
    """Train one model; returns (model, history_dict)."""

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = keras.utils.to_categorical(y_val,   num_classes)

    model = get_model(model_name, num_classes=num_classes)
    model = compile_model(model, learning_rate=learning_rate)

    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    stats = model_summary_stats(model)
    print(f"  Total params   : {stats['total']:,}")
    print(f"  Trainable      : {stats['trainable']:,}")
    print(f"{'='*60}")

    train_ds = build_tf_dataset(X_train, y_train_cat, batch_size, augment=True)
    val_ds   = build_tf_dataset(X_val,   y_val_cat,   batch_size, augment=False, shuffle=False)

    class_weights = compute_weights(y_train)

    callbacks = build_callbacks(model_name, checkpoint_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history
    Path("outputs").mkdir(exist_ok=True)
    hist_path = f"outputs/{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return model, history.history


# ── Training Curve Plot ────────────────────────────────────────────────────────
def plot_training_curves(history: dict, model_name: str, save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    metrics = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss",     "val_loss",     "Loss"),
        ("auc",      "val_auc",      "AUC"),
    ]

    for ax, (train_m, val_m, title) in zip(axes, metrics):
        if train_m in history:
            ax.plot(history[train_m], label=f"Train {title}", linewidth=2)
        if val_m in history:
            ax.plot(history[val_m],   label=f"Val {title}",   linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / f"{model_name}_training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {save_path}")


# ── Hyperparameter Search ──────────────────────────────────────────────────────
def learning_rate_finder(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lr_range: list = None,
    batch_size: int = 32,
) -> dict:
    """Grid search over learning rates; returns dict of lr -> val_loss."""
    if lr_range is None:
        lr_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    results = {}
    y_train_cat = keras.utils.to_categorical(y_train, 5)

    X_t, X_v, y_t, y_v = train_test_split(
        X_train, y_train_cat, test_size=0.2, stratify=y_train, random_state=42
    )

    for lr in lr_range:
        print(f"\nTrying LR = {lr}")
        model = get_model(model_name, num_classes=5)
        model = compile_model(model, learning_rate=lr)

        ds_t = build_tf_dataset(X_t, y_t, batch_size, augment=False)
        ds_v = build_tf_dataset(X_v, y_v, batch_size, augment=False, shuffle=False)

        h = model.fit(ds_t, validation_data=ds_v, epochs=5, verbose=0)
        results[lr] = min(h.history["val_loss"])
        keras.backend.clear_session()

    best_lr = min(results, key=results.get)
    print(f"\nBest LR: {best_lr} (val_loss={results[best_lr]:.4f})")
    return results


# ── Train All Models ───────────────────────────────────────────────────────────
def train_all_models(
    X_train, y_train, X_val, y_val,
    model_names: list = None,
    epochs: int = 50,
    batch_size: int = 32,
    checkpoint_dir: str = "models",
) -> dict:
    if model_names is None:
        model_names = ["cnn", "mobilenetv2", "resnet50", "efficientnetb0",
                       "efficientnetb3", "densenet121", "inceptionv3"]

    results = {}
    for name in model_names:
        try:
            model, history = train_model(
                name, X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size,
                checkpoint_dir=checkpoint_dir,
            )
            plot_training_curves(history, name)
            results[name] = {"model": model, "history": history}
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    return results

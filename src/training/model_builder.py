"""
Model Builder — CNN from scratch + Transfer Learning models.
Supports: CNN, MobileNetV2, ResNet50, EfficientNetB0/B3, DenseNet121, InceptionV3.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, EfficientNetB3,
    DenseNet121, InceptionV3
)
import numpy as np


NUM_CLASSES = 3
IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)


# ── Custom CNN (Baseline) ──────────────────────────────────────────────────────
def build_cnn_scratch(num_classes: int = NUM_CLASSES, input_shape: tuple = INPUT_SHAPE) -> Model:
    inputs = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="CNN_Scratch")


# ── Transfer Learning Factory ──────────────────────────────────────────────────
def _build_transfer_model(
    base_class,
    name: str,
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = INPUT_SHAPE,
    fine_tune_at: int = None,
    dropout: float = 0.5,
    weights: str = "imagenet",
) -> Model:
    base = base_class(weights=weights, include_top=False, input_shape=input_shape)
    base.trainable = False

    if fine_tune_at is not None:
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True

    inputs = keras.Input(shape=input_shape, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout * 0.6)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name=name)


def build_mobilenetv2(**kw) -> Model:
    return _build_transfer_model(MobileNetV2, "MobileNetV2", **kw)


def build_resnet50(**kw) -> Model:
    return _build_transfer_model(ResNet50, "ResNet50", **kw)


def build_efficientnetb0(**kw) -> Model:
    return _build_transfer_model(EfficientNetB0, "EfficientNetB0", **kw)


def build_efficientnetb3(**kw) -> Model:
    input_shape = kw.pop("input_shape", (300, 300, 3))
    return _build_transfer_model(EfficientNetB3, "EfficientNetB3", input_shape=input_shape, **kw)


def build_densenet121(**kw) -> Model:
    return _build_transfer_model(DenseNet121, "DenseNet121", **kw)


def build_inceptionv3(**kw) -> Model:
    input_shape = kw.pop("input_shape", (299, 299, 3))
    return _build_transfer_model(InceptionV3, "InceptionV3", input_shape=input_shape, **kw)


# ── Ensemble Model ─────────────────────────────────────────────────────────────
def build_ensemble(model_paths: list, num_classes: int = NUM_CLASSES) -> Model:
    """Soft-voting ensemble from saved model paths."""
    loaded = [keras.models.load_model(p) for p in model_paths]
    inputs = keras.Input(shape=INPUT_SHAPE, name="input")

    preds = [m(inputs) for m in loaded]
    avg   = layers.Average()(preds)

    return Model(inputs, avg, name="Ensemble")


# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "cnn":            build_cnn_scratch,
    "mobilenetv2":    build_mobilenetv2,
    "resnet50":       build_resnet50,
    "efficientnetb0": build_efficientnetb0,
    "efficientnetb3": build_efficientnetb3,
    "densenet121":    build_densenet121,
    "inceptionv3":    build_inceptionv3,
}


def get_model(name: str, **kwargs) -> Model:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def compile_model(
    model: Model,
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.1,
) -> Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc", multi_label=True),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def model_summary_stats(model: Model) -> dict:
    trainable     = sum(np.prod(v.shape) for v in model.trainable_weights)
    non_trainable = sum(np.prod(v.shape) for v in model.non_trainable_weights)
    return {
        "name":           model.name,
        "trainable":      int(trainable),
        "non_trainable":  int(non_trainable),
        "total":          int(trainable + non_trainable),
        "input_shape":    model.input_shape,
        "output_shape":   model.output_shape,
    }

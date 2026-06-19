"""
Image Preprocessing Pipeline for Retinopathy Detection
Handles resizing, normalization, enhancement, and augmentation.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]


# ── Core Preprocessing ─────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img: np.ndarray, size: tuple = IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """ImageNet-style normalization."""
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img


def reduce_noise(img: np.ndarray) -> np.ndarray:
    """Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def enhance_contrast_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE on each RGB channel for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def standardize_rgb(img: np.ndarray) -> np.ndarray:
    """Per-image mean-std standardization."""
    img = img.astype(np.float32)
    mean = img.mean(axis=(0, 1), keepdims=True)
    std  = img.std(axis=(0, 1), keepdims=True) + 1e-8
    return (img - mean) / std


def preprocess_single(path: str, apply_clahe: bool = True) -> np.ndarray:
    """Full preprocessing pipeline for a single image."""
    img = load_image(path)
    img = reduce_noise(img)
    if apply_clahe:
        img = enhance_contrast_clahe(img)
    img = resize_image(img)
    img = normalize_image(img)
    return img


# ── Augmentation Pipelines ─────────────────────────────────────────────────────
def get_train_augmentation() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.4),
        A.Affine(shear=(-10, 10), p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])


def get_val_augmentation() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])


# ── Batch Preprocessing ────────────────────────────────────────────────────────
class RetinopathyPreprocessor:
    """Preprocesses an entire dataset split from a DataFrame."""

    def __init__(self, image_dir: str, img_size: tuple = IMG_SIZE, apply_clahe: bool = True):
        self.image_dir = Path(image_dir)
        self.img_size  = img_size
        self.apply_clahe = apply_clahe

    def process_dataframe(self, df, filename_col: str = "image", label_col: str = "label"):
        images, labels = [], []
        failed = []

        for _, row in df.iterrows():
            path = self.image_dir / row[filename_col]
            try:
                img = preprocess_single(str(path), self.apply_clahe)
                images.append(img)
                labels.append(row[label_col])
            except Exception as e:
                failed.append((row[filename_col], str(e)))

        if failed:
            print(f"[WARNING] {len(failed)} images failed to load:")
            for f, err in failed[:5]:
                print(f"  {f}: {err}")

        return np.array(images, dtype=np.float32), np.array(labels)

    def save_processed(self, images: np.ndarray, labels: np.ndarray, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "images.npy", images)
        np.save(out / "labels.npy", labels)
        print(f"Saved {len(images)} processed images to {out}")


# ── Visualization ──────────────────────────────────────────────────────────────
def visualize_augmentation_pipeline(image_path: str, save_path: str = None):
    """Show original vs 6 augmented versions side by side."""
    img = load_image(image_path)
    img = resize_image(img)

    aug = get_train_augmentation()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Augmentation Pipeline — Retinal Fundus Image", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontweight="bold")
    axes[0, 0].axis("off")

    aug_names = ["Rotation", "H-Flip", "Brightness", "Zoom/Shift", "Shear", "Noise", "CLAHE"]
    for idx, name in enumerate(aug_names):
        r, c = divmod(idx + 1, 4)
        augmented = aug(image=img)["image"]
        # Denormalize for display
        disp = augmented * STD + MEAN
        disp = np.clip(disp, 0, 1)
        axes[r, c].imshow(disp)
        axes[r, c].set_title(name)
        axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Augmentation diagram saved to {save_path}")
    plt.show()


def visualize_preprocessing_steps(image_path: str, save_path: str = None):
    """Show each preprocessing step as a figure."""
    original = load_image(image_path)

    steps = {
        "1. Original":       original,
        "2. Noise Reduced":  reduce_noise(original),
        "3. CLAHE Enhanced": enhance_contrast_clahe(original),
        "4. Resized (224)":  resize_image(original),
    }

    norm = normalize_image(resize_image(original))
    disp_norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    steps["5. Normalized"] = (disp_norm * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, len(steps), figsize=(20, 5))
    fig.suptitle("Preprocessing Pipeline Steps", fontsize=14, fontweight="bold")

    for ax, (title, img) in zip(axes, steps.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

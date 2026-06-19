"""
Generates synthetic retinal-like fundus images for demo/testing.
Creates 500 images across 5 DR severity classes.
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE       = Path(__file__).parent.parent
IMAGE_DIR  = BASE / "data" / "raw" / "images"
CSV_PATH   = BASE / "data" / "raw" / "labels.csv"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES   = ["No_DR", "Mild_DR", "Moderate_DR", "Severe_DR", "Proliferative_DR"]
SAMPLES       = [200, 80, 120, 50, 50]   # intentional imbalance

def make_retina(cls_idx: int, idx: int, size: int = 512) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r      = size // 2 - 10

    # Dark background
    cv2.circle(img, (cx, cy), r, (20, 10, 10), -1)

    # Base retinal texture (gradient)
    for row in range(size):
        for band in range(0, size, 40):
            val = int(30 + 15 * np.sin(row / 30))
            img[row, band:band+20, 1] = np.clip(val, 0, 60)

    # Optic disc (bright yellow-white circle)
    disc_x = cx + size // 5
    cv2.circle(img, (disc_x, cy), size // 14,  (200, 200, 120), -1)
    cv2.circle(img, (disc_x, cy), size // 20,  (240, 240, 180), -1)

    # Blood vessels (branching lines)
    n_vessels = 6 + cls_idx * 2
    for v in range(n_vessels):
        angle = v * (360 / n_vessels)
        rad   = np.radians(angle)
        ex    = int(disc_x + r * 0.8 * np.cos(rad))
        ey    = int(cy      + r * 0.8 * np.sin(rad))
        thickness = max(1, 3 - cls_idx // 2)
        cv2.line(img, (disc_x, cy), (ex, ey), (180, 80, 80), thickness)

    # Macula (darker center-left area)
    mac_x = cx - size // 5
    cv2.circle(img, (mac_x, cy), size // 15, (15, 8, 8), -1)
    cv2.circle(img, (mac_x, cy), size // 25, (25, 12, 12), -1)

    # Severity-specific lesions
    if cls_idx >= 1:  # Mild: microaneurysms (tiny red dots)
        n_ma = 5 + cls_idx * 8
        for _ in range(n_ma):
            angle  = np.random.uniform(0, 2 * np.pi)
            dist   = np.random.uniform(0.1, 0.7) * r
            x = int(cx + dist * np.cos(angle))
            y = int(cy + dist * np.sin(angle))
            if 0 < x < size and 0 < y < size:
                cv2.circle(img, (x, y), np.random.randint(2, 5), (180, 20, 20), -1)

    if cls_idx >= 2:  # Moderate: hard exudates (yellow patches)
        n_ex = 10 + cls_idx * 5
        for _ in range(n_ex):
            angle = np.random.uniform(0, 2 * np.pi)
            dist  = np.random.uniform(0.15, 0.65) * r
            x = int(cx + dist * np.cos(angle))
            y = int(cy + dist * np.sin(angle))
            if 0 < x < size and 0 < y < size:
                w = np.random.randint(8, 20)
                h = np.random.randint(8, 20)
                cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (200, 190, 80), -1)

    if cls_idx >= 3:  # Severe: haemorrhages (dark blotches)
        n_hm = 15 + cls_idx * 6
        for _ in range(n_hm):
            angle = np.random.uniform(0, 2 * np.pi)
            dist  = np.random.uniform(0.1, 0.75) * r
            x = int(cx + dist * np.cos(angle))
            y = int(cy + dist * np.sin(angle))
            if 0 < x < size and 0 < y < size:
                w = np.random.randint(10, 30)
                h = np.random.randint(10, 30)
                cv2.ellipse(img, (x, y), (w, h), np.random.randint(0, 180),
                            0, 360, (100, 10, 10), -1)

    if cls_idx == 4:  # Proliferative: neovascularization (bright irregular vessels)
        for _ in range(8):
            angle  = np.random.uniform(0, 2 * np.pi)
            dist   = np.random.uniform(0.2, 0.6) * r
            sx = int(cx + dist * np.cos(angle))
            sy = int(cy + dist * np.sin(angle))
            for seg in range(6):
                ex = sx + np.random.randint(-30, 30)
                ey = sy + np.random.randint(-30, 30)
                ex = np.clip(ex, 0, size - 1)
                ey = np.clip(ey, 0, size - 1)
                cv2.line(img, (sx, sy), (ex, ey), (220, 120, 120), 2)
                sx, sy = ex, ey

    # Vignette mask
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask, (cx, cy), r, 1.0, -1)
    kernel = np.ones((31, 31), np.float32) / 961
    mask   = cv2.filter2D(mask, -1, kernel)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * mask).astype(np.uint8)

    # Add slight noise
    noise = np.random.randint(0, 8, (size, size, 3), dtype=np.uint8)
    img   = cv2.add(img, noise)

    return img


rows = []
total = 0
for cls_idx, (cls_name, n) in enumerate(zip(CLASS_NAMES, SAMPLES)):
    print(f"Generating {n} images for class {cls_idx} ({cls_name})...")
    for i in range(n):
        fname = f"{cls_name}_{i:04d}.png"
        img   = make_retina(cls_idx, i)
        cv2.imwrite(str(IMAGE_DIR / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({"image": fname, "label": cls_idx})
        total += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(CSV_PATH, index=False)
print(f"\n✅ Created {total} synthetic images → {IMAGE_DIR}")
print(f"✅ Labels CSV → {CSV_PATH}")
print(f"\nClass distribution:\n{df['label'].value_counts().sort_index()}")

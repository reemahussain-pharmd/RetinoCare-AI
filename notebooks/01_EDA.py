"""
Notebook 01 — Exploratory Data Analysis for Retinopathy Dataset
Run as a script or convert to .ipynb with: jupytext --to notebook 01_EDA.py
"""

# %% [markdown]
# # Retinopathy Dataset — Exploratory Data Analysis
# **RetinoCare AI** | Healthcare AI Portfolio Project

# %%
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path
from PIL import Image
from collections import Counter

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_CSV   = "data/raw/labels.csv"
IMAGE_DIR  = "data/raw/images"
OUTPUT_DIR = "outputs/eda"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

CLASS_NAMES   = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
CLASS_COLORS  = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]

# %% [markdown]
# ## 1. Load Dataset

# %%
df = pd.read_csv(DATA_CSV)
print(f"Dataset shape   : {df.shape}")
print(f"Columns         : {df.columns.tolist()}")
print("\nFirst 5 rows:")
df.head()

# %% [markdown]
# ## 2. Dataset Dimensions & Basic Stats

# %%
print(f"\n{'='*50}")
print("DATASET OVERVIEW")
print(f"{'='*50}")
print(f"Total samples     : {len(df):,}")
print(f"Columns           : {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe(include='all')}")

# %% [markdown]
# ## 3. Missing Value Analysis

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(f"\nMissing Values:\n{missing_df}")

fig, ax = plt.subplots(figsize=(8, 4))
if missing.sum() > 0:
    missing[missing > 0].plot(kind="bar", ax=ax, color="#FF5722")
    ax.set_title("Missing Values per Column")
else:
    ax.text(0.5, 0.5, "No Missing Values Found!", ha="center", va="center",
            fontsize=16, color="#4CAF50", transform=ax.transAxes)
    ax.set_title("Missing Values Analysis")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Duplicate Analysis

# %%
dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After deduplication: {len(df):,} rows")

# %% [markdown]
# ## 5. Class Distribution & Imbalance Analysis

# %%
label_col  = df.columns[1]   # second column = label
counts     = df[label_col].value_counts().sort_index()
class_map  = {i: n for i, n in enumerate(CLASS_NAMES)}
counts.index = [class_map.get(i, str(i)) for i in counts.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Class Distribution — Retinopathy Severity", fontsize=14, fontweight="bold")

# Bar chart
axes[0].bar(counts.index, counts.values, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, f"{v:,}\n({100*v/counts.sum():.1f}%)",
                 ha="center", fontsize=9)
axes[0].set_title("Absolute Counts")
axes[0].set_ylabel("Number of Images")
axes[0].tick_params(axis="x", rotation=30)

# Pie chart
axes[1].pie(counts.values, labels=counts.index, colors=CLASS_COLORS,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Proportion by Class")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.show()

# Imbalance ratio
imbalance_ratio = counts.max() / counts.min()
print(f"\nClass Imbalance Ratio (max/min): {imbalance_ratio:.2f}x")
print("Recommended strategy: class_weight='balanced' or oversampling of minority classes")

# %% [markdown]
# ## 6. Image Dimension Analysis

# %%
def get_image_stats(img_dir: str, filenames: list, n_sample: int = 200) -> pd.DataFrame:
    sample = np.random.choice(filenames, min(n_sample, len(filenames)), replace=False)
    stats = []
    for fn in sample:
        path = Path(img_dir) / fn
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                h, w, c = img.shape
                stats.append({"filename": fn, "height": h, "width": w, "channels": c,
                               "aspect_ratio": round(w/h, 3)})
    return pd.DataFrame(stats)

filename_col = df.columns[0]
img_stats = get_image_stats(IMAGE_DIR, df[filename_col].tolist())
print(f"\nImage Dimension Statistics (sample of {len(img_stats)}):")
print(img_stats[["height", "width", "aspect_ratio"]].describe())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Image Dimension Analysis", fontsize=13, fontweight="bold")

axes[0].hist(img_stats["height"], bins=20, color="#2196F3", edgecolor="black")
axes[0].set_title("Height Distribution"); axes[0].set_xlabel("Pixels")

axes[1].hist(img_stats["width"], bins=20, color="#4CAF50", edgecolor="black")
axes[1].set_title("Width Distribution"); axes[1].set_xlabel("Pixels")

axes[2].hist(img_stats["aspect_ratio"], bins=20, color="#FFC107", edgecolor="black")
axes[2].set_title("Aspect Ratio"); axes[2].set_xlabel("W/H")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/image_dimensions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Pixel Intensity Analysis

# %%
def sample_pixel_intensities(img_dir: str, filenames: list, n: int = 50) -> dict:
    sample = np.random.choice(filenames, min(n, len(filenames)), replace=False)
    r_vals, g_vals, b_vals = [], [], []
    for fn in sample:
        path = Path(img_dir) / fn
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                r_vals.extend(img_rgb[:, :, 0].flatten()[::100].tolist())
                g_vals.extend(img_rgb[:, :, 1].flatten()[::100].tolist())
                b_vals.extend(img_rgb[:, :, 2].flatten()[::100].tolist())
    return {"R": np.array(r_vals), "G": np.array(g_vals), "B": np.array(b_vals)}

pixel_data = sample_pixel_intensities(IMAGE_DIR, df[filename_col].tolist())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Pixel Intensity Distribution (RGB Channels)", fontsize=13, fontweight="bold")
colors = {"R": "#F44336", "G": "#4CAF50", "B": "#2196F3"}
for ax, (ch, vals) in zip(axes, pixel_data.items()):
    ax.hist(vals, bins=50, color=colors[ch], alpha=0.7, edgecolor="none")
    ax.axvline(vals.mean(), color="black", linestyle="--", label=f"Mean={vals.mean():.1f}")
    ax.set_title(f"{ch} Channel"); ax.set_xlabel("Intensity (0-255)")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pixel_intensity.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Sample Images per Class

# %%
n_per_class = 4
fig = plt.figure(figsize=(20, len(CLASS_NAMES) * 4))
gs  = gridspec.GridSpec(len(CLASS_NAMES), n_per_class, figure=fig, hspace=0.4, wspace=0.1)
fig.suptitle("Sample Retinal Images by Severity Class", fontsize=16, fontweight="bold", y=1.01)

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_df = df[df[label_col] == cls_idx]
    samples = cls_df[filename_col].sample(min(n_per_class, len(cls_df)), random_state=42)

    for col_idx, fn in enumerate(samples):
        ax = fig.add_subplot(gs[cls_idx, col_idx])
        path = Path(IMAGE_DIR) / fn
        if path.exists():
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(cls_name, fontsize=11, fontweight="bold",
                          color=CLASS_COLORS[cls_idx], rotation=0,
                          labelpad=60, va="center")

plt.savefig(f"{OUTPUT_DIR}/sample_images_per_class.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Outlier Detection (file size)

# %%
def get_file_sizes(img_dir: str, filenames: list) -> list:
    sizes = []
    for fn in filenames:
        path = Path(img_dir) / fn
        if path.exists():
            sizes.append(path.stat().st_size / 1024)  # KB
    return sizes

file_sizes = get_file_sizes(IMAGE_DIR, df[filename_col].tolist())
if file_sizes:
    sizes_arr = np.array(file_sizes)
    q1, q3 = np.percentile(sizes_arr, [25, 75])
    iqr     = q3 - q1
    outliers = sizes_arr[(sizes_arr < q1 - 1.5*iqr) | (sizes_arr > q3 + 1.5*iqr)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(sizes_arr, bins=30, color="#9C27B0", edgecolor="black")
    axes[0].set_title("File Size Distribution (KB)")
    axes[1].boxplot(sizes_arr, vert=True)
    axes[1].set_title(f"Boxplot — {len(outliers)} outliers detected")
    plt.suptitle("Outlier Detection (File Size Proxy)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/outliers.png", dpi=150)
    plt.show()

# %% [markdown]
# ## 10. EDA Summary

# %%
print(f"""
{'='*60}
EDA SUMMARY — RetinoCare AI
{'='*60}
Total Samples    : {len(df):,}
Duplicate Rows   : {dupes}
Missing Values   : {missing.sum()}
Number of Classes: {len(CLASS_NAMES)}
Class Imbalance  : {imbalance_ratio:.2f}x
{'='*60}
RECOMMENDATIONS
  1. Apply class weighting / oversampling for imbalance
  2. Standardize image dimensions to 224x224
  3. Apply CLAHE for low-contrast images
  4. Use stratified train/val/test splits
{'='*60}
""")
print(f"All EDA outputs saved to: {OUTPUT_DIR}/")

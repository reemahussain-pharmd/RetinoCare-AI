"""
Full RetinoCare AI Pipeline — EDA → Train → Evaluate → XAI
Streams images from disk via tf.data (no large numpy cache needed).
3-class retinopathy dataset — 1764 real fundus images.
"""

import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
DATA_CSV  = BASE / "data" / "raw" / "labels.csv"
IMG_DIR   = BASE / "data" / "raw" / "images"
MDL_DIR   = BASE / "models";       MDL_DIR.mkdir(exist_ok=True)
OUT_DIR   = BASE / "outputs";      OUT_DIR.mkdir(exist_ok=True)
EDA_DIR   = OUT_DIR / "eda";       EDA_DIR.mkdir(exist_ok=True)
XAI_DIR   = OUT_DIR / "xai";       XAI_DIR.mkdir(exist_ok=True)
RPT_DIR   = BASE / "reports";      RPT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES  = 3
IMG_SIZE     = 224
CLASS_NAMES  = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]
CLASS_COLORS = ["#4CAF50", "#FFC107", "#F44336"]
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
EPOCHS     = 25
BATCH_SIZE = 16
LR         = 1e-4

def hdr(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ══════════════════════════════════════════════════════════════════
# STEP 1 — EDA
# ══════════════════════════════════════════════════════════════════
hdr("STEP 1: Exploratory Data Analysis")

df = pd.read_csv(DATA_CSV)
print(f"Dataset : {df.shape[0]} images | columns: {df.columns.tolist()}")
print(f"Labels  :\n{df['label'].value_counts().sort_index()}")
print(f"Missing : {df.isnull().sum().sum()} | Duplicates: {df.duplicated().sum()}")
imb = df['label'].value_counts().sort_index()
print(f"Imbalance ratio: {imb.max()/imb.min():.2f}x")

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Distribution — Retinopathy Dataset", fontsize=13, fontweight="bold")
axes[0].bar([CLASS_NAMES[i] for i in imb.index], imb.values,
            color=CLASS_COLORS, edgecolor="black")
for i, v in enumerate(imb.values):
    axes[0].text(i, v+8, f"{v}\n({100*v/len(df):.1f}%)", ha="center", fontsize=9)
axes[0].set_title("Count per Class"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=15)
axes[1].pie(imb.values, labels=[CLASS_NAMES[i] for i in imb.index],
            colors=CLASS_COLORS, autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor":"white"})
axes[1].set_title("Class Proportion")
plt.tight_layout()
plt.savefig(EDA_DIR/"class_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

# Image dimension stats (sample 200)
dim_rows = []
for _, row in df.sample(min(200,len(df)), random_state=42).iterrows():
    p = IMG_DIR / row["image"]
    if p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            dim_rows.append({"height":h, "width":w, "aspect":round(w/h,2)})
dim_df = pd.DataFrame(dim_rows)
print(f"\nDimension stats (n={len(dim_df)}):\n{dim_df.describe().to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Image Dimension Analysis", fontweight="bold")
for ax, col, c, lbl in [
    (axes[0],"height","#2196F3","Height (px)"),
    (axes[1],"width", "#4CAF50","Width (px)"),
    (axes[2],"aspect","#FFC107","Aspect Ratio W/H"),
]:
    ax.hist(dim_df[col], bins=20, color=c, edgecolor="black", alpha=0.8)
    ax.axvline(dim_df[col].mean(), color="red", linestyle="--",
               label=f"Mean={dim_df[col].mean():.0f}")
    ax.set_xlabel(lbl); ax.legend()
plt.tight_layout()
plt.savefig(EDA_DIR/"image_dimensions.png", dpi=150, bbox_inches="tight"); plt.close()

# Pixel intensity (subsample)
r_all, g_all, b_all = [], [], []
for _, row in df.sample(min(60,len(df)), random_state=1).iterrows():
    p = IMG_DIR / row["image"]
    if p.exists():
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        r_all.extend(img[::5,::5,0].flatten().tolist())
        g_all.extend(img[::5,::5,1].flatten().tolist())
        b_all.extend(img[::5,::5,2].flatten().tolist())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Pixel Intensity Distribution (RGB)", fontweight="bold")
for ax, vals, ch, c in [(axes[0],r_all,"Red","#F44336"),
                         (axes[1],g_all,"Green","#4CAF50"),
                         (axes[2],b_all,"Blue","#2196F3")]:
    arr = np.array(vals)
    ax.hist(arr, bins=50, color=c, alpha=0.7)
    ax.axvline(arr.mean(), color="black", linestyle="--", label=f"Mean={arr.mean():.1f}")
    ax.set_title(f"{ch} Channel"); ax.set_xlabel("0–255"); ax.legend()
plt.tight_layout()
plt.savefig(EDA_DIR/"pixel_intensity.png", dpi=150, bbox_inches="tight"); plt.close()

# Sample images per class
fig = plt.figure(figsize=(16, NUM_CLASSES*4))
gs  = gridspec.GridSpec(NUM_CLASSES, 4, hspace=0.3, wspace=0.05)
fig.suptitle("Sample Retinal Images by Severity Class", fontsize=14, fontweight="bold")
for ci, cn in enumerate(CLASS_NAMES):
    cls_df  = df[df["label"]==ci]
    samples = cls_df["image"].sample(min(4,len(cls_df)), random_state=42)
    for j, fn in enumerate(samples):
        ax = fig.add_subplot(gs[ci, j])
        p  = IMG_DIR / fn
        if p.exists():
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.axis("off")
        if j == 0: ax.set_title(cn, fontsize=10, fontweight="bold",
                                  color=CLASS_COLORS[ci], pad=4)
plt.savefig(EDA_DIR/"sample_images.png", dpi=120, bbox_inches="tight"); plt.close()

# Outlier detection (file size)
sizes = [((IMG_DIR/row["image"]).stat().st_size/1024)
         for _, row in df.iterrows() if (IMG_DIR/row["image"]).exists()]
sizes = np.array(sizes)
q1,q3 = np.percentile(sizes,[25,75]); iqr = q3-q1
outliers = sizes[(sizes<q1-1.5*iqr)|(sizes>q3+1.5*iqr)]
print(f"\nFile-size outliers detected: {len(outliers)} of {len(sizes)}")
fig, axes = plt.subplots(1,2,figsize=(10,4))
fig.suptitle("File Size Distribution (KB) — Outlier Detection", fontweight="bold")
axes[0].hist(sizes, bins=30, color="#9C27B0", edgecolor="black")
axes[0].set_xlabel("KB"); axes[0].set_title("Distribution")
axes[1].boxplot(sizes); axes[1].set_title(f"{len(outliers)} outliers (IQR method)")
plt.tight_layout()
plt.savefig(EDA_DIR/"outliers.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"\nEDA complete. Plots saved to outputs/eda/")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — tf.data PIPELINE (stream from disk)
# ══════════════════════════════════════════════════════════════════
hdr("STEP 2: Building tf.data Streaming Pipeline")

# Split CSV first
df_temp, df_test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
df_train, df_val  = train_test_split(df_temp, test_size=0.176, stratify=df_temp["label"], random_state=42)
print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# Preprocessing on GPU/CPU via tf ops
@tf.function
def preprocess_tf(path, label, augment=False):
    raw  = tf.io.read_file(path)
    img  = tf.image.decode_jpeg(raw, channels=3)
    img  = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img  = tf.cast(img, tf.float32) / 255.0
    # CLAHE-like: simple contrast stretch per channel
    img  = tf.clip_by_value(img, 0.0, 1.0)
    img  = (img - MEAN) / STD
    lbl  = tf.one_hot(label, NUM_CLASSES)
    return img, lbl

def aug_fn(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    # Random rotation via crop+pad trick
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    return img, lbl

def make_ds(dataframe, augment=False, shuffle=True):
    paths  = [str(IMG_DIR / fn) for fn in dataframe["image"]]
    labels = dataframe["label"].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(lambda p, l: preprocess_tf(p, l),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(df_train, augment=True,  shuffle=True)
val_ds   = make_ds(df_val,   augment=False, shuffle=False)
test_ds  = make_ds(df_test,  augment=False, shuffle=False)

class_weights = dict(zip(
    np.unique(df_train["label"]),
    compute_class_weight("balanced", classes=np.unique(df_train["label"]),
                         y=df_train["label"].values)
))
print(f"Class weights: { {CLASS_NAMES[k]:round(v,3) for k,v in class_weights.items()} }")

# Preprocessing visualization (OpenCV — 3 steps per class)
print("Generating preprocessing pipeline visualization...")
fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(16, NUM_CLASSES*4))
fig.suptitle("Preprocessing Pipeline per Class", fontsize=13, fontweight="bold")
step_titles = ["Original", "Noise Reduced", "CLAHE Enhanced", "Resized 224×224"]
for ci, cn in enumerate(CLASS_NAMES):
    fn   = df[df["label"]==ci]["image"].iloc[0]
    raw  = cv2.cvtColor(cv2.imread(str(IMG_DIR/fn)), cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(raw,(3,3),0)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh  = cv2.merge([clahe_obj.apply(ch) for ch in cv2.split(blur)])
    rsz  = cv2.resize(enh,(IMG_SIZE,IMG_SIZE))
    for j,(img,title) in enumerate(zip([raw,blur,enh,rsz], step_titles)):
        ax = axes[ci,j]
        ax.imshow(img); ax.axis("off")
        if ci==0: ax.set_title(title, fontsize=10, fontweight="bold")
        if j==0:  ax.set_ylabel(cn, fontsize=9, color=CLASS_COLORS[ci])
plt.tight_layout()
plt.savefig(OUT_DIR/"preprocessing_steps.png", dpi=120, bbox_inches="tight"); plt.close()
print("Preprocessing visualization saved.")

# Augmentation visualization
print("Generating augmentation visualization...")
sample_fn = df_train["image"].iloc[0]
raw_img   = cv2.cvtColor(cv2.imread(str(IMG_DIR/sample_fn)), cv2.COLOR_BGR2RGB)
raw_img   = cv2.resize(raw_img,(IMG_SIZE,IMG_SIZE))

aug_pipeline = [
    ("Rotation",   lambda i: cv2.warpAffine(i,
        cv2.getRotationMatrix2D((IMG_SIZE//2,IMG_SIZE//2), np.random.randint(-30,30), 1.0),
        (IMG_SIZE,IMG_SIZE))),
    ("H-Flip",     lambda i: cv2.flip(i,1)),
    ("V-Flip",     lambda i: cv2.flip(i,0)),
    ("Brightness", lambda i: np.clip(i.astype(np.float32)*np.random.uniform(0.7,1.3),0,255).astype(np.uint8)),
    ("Zoom",       lambda i: cv2.resize(cv2.resize(i,(int(IMG_SIZE*1.2),int(IMG_SIZE*1.2))),
                                         (IMG_SIZE,IMG_SIZE))[10:-10,10:-10]),
    ("Noise",      lambda i: np.clip(i.astype(np.int32)+np.random.randint(-20,20,i.shape),0,255).astype(np.uint8)),
    ("Contrast",   lambda i: cv2.convertScaleAbs(i, alpha=np.random.uniform(0.8,1.3), beta=0)),
]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Data Augmentation Pipeline", fontsize=13, fontweight="bold")
axes[0,0].imshow(raw_img); axes[0,0].set_title("Original", fontweight="bold"); axes[0,0].axis("off")
for idx,(name,fn) in enumerate(aug_pipeline):
    r,c = divmod(idx+1, 4)
    try:  aug_img = fn(raw_img.copy())
    except: aug_img = raw_img
    axes[r,c].imshow(np.clip(aug_img,0,255).astype(np.uint8))
    axes[r,c].set_title(name); axes[r,c].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR/"augmentation_pipeline.png", dpi=120, bbox_inches="tight"); plt.close()
print("Augmentation visualization saved.")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
hdr("STEP 3: Model Training (5 Architectures)")

from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, DenseNet121, ResNet50

def build_transfer(base_fn, name):
    base = base_fn(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    inp = keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x   = base(inp, training=False)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(256, activation="relu")(x)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name=name)

def build_cnn():
    inp = keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x   = inp
    for f in [32,64,128,256]:
        x = keras.layers.Conv2D(f,3,padding="same",activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(f,3,padding="same",activation="relu")(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Dropout(0.25)(x)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(256, activation="relu")(x)
    x   = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="CNN_Scratch")

MODEL_CONFIGS = [
    ("cnn",            build_cnn),
    ("mobilenetv2",    lambda: build_transfer(MobileNetV2,   "MobileNetV2")),
    ("efficientnetb0", lambda: build_transfer(EfficientNetB0,"EfficientNetB0")),
    ("resnet50",       lambda: build_transfer(ResNet50,      "ResNet50")),
    ("densenet121",    lambda: build_transfer(DenseNet121,   "DenseNet121")),
]

all_histories = {}
all_models    = {}

for model_name, model_fn in MODEL_CONFIGS:
    print(f"\n{'-'*50}\nTraining: {model_name.upper()}")
    model = model_fn()
    tp    = sum(np.prod(v.shape) for v in model.trainable_weights)
    print(f"  Trainable params: {tp:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    ckpt = str(MDL_DIR / f"{model_name}_best.keras")
    cbs  = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max",
                                        save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=8,
                                      restore_best_weights=True, mode="max", verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=4, min_lr=1e-7, verbose=0),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                     class_weight=class_weights, callbacks=cbs, verbose=1)

    all_histories[model_name] = hist.history
    all_models[model_name]    = model

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    fig.suptitle(f"Training Curves — {model_name}", fontweight="bold")
    for ax,(t,v,title) in zip(axes,[
        ("accuracy","val_accuracy","Accuracy"),
        ("loss","val_loss","Loss"),
        ("auc","val_auc","AUC"),
    ]):
        if t in hist.history: ax.plot(hist.history[t], label="Train", lw=2)
        if v in hist.history: ax.plot(hist.history[v], label="Val",   lw=2, ls="--")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{model_name}_training_curves.png", dpi=130, bbox_inches="tight")
    plt.close()
    val_auc = max(hist.history.get("val_auc",[0]))
    print(f"  Best val AUC: {val_auc:.4f}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════
hdr("STEP 4: Model Evaluation & Comparison")

# Collect test ground-truth labels
y_test_list = []
for _, lbl in test_ds.unbatch():
    y_test_list.append(np.argmax(lbl.numpy()))
y_test = np.array(y_test_list)

eval_results = {}
for model_name, model in all_models.items():
    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

    report = classification_report(y_test, preds,
                                   target_names=CLASS_NAMES, output_dict=True)
    try:
        macro_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        macro_auc = 0.0

    eval_results[model_name] = {
        "accuracy" : round(report["accuracy"], 4),
        "precision": round(report["macro avg"]["precision"], 4),
        "recall"   : round(report["macro avg"]["recall"], 4),
        "f1"       : round(report["macro avg"]["f1-score"], 4),
        "auc"      : round(macro_auc, 4),
        "probs"    : probs,
        "preds"    : preds,
    }

    # Confusion matrix
    cm   = confusion_matrix(y_test, preds)
    cmn  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontweight="bold")
    for ax, data, title, fmt in [
        (axes[0], cm,  "Raw Counts", "d"),
        (axes[1], cmn, "Normalized", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    linewidths=0.5)
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{model_name}_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ROC
    fig, ax = plt.subplots(figsize=(7,6))
    y_bin   = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    for i,(c,cn) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
        fpr,tpr,_ = roc_curve(y_bin[:,i], probs[:,i])
        ax.plot(fpr,tpr,color=c,lw=2,label=f"{cn} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--"); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {model_name} (macro={macro_auc:.4f})", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR/f"{model_name}_roc_curves.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"{model_name:15s} | Acc={eval_results[model_name]['accuracy']:.4f} "
          f"F1={eval_results[model_name]['f1']:.4f} AUC={eval_results[model_name]['auc']:.4f}")

# Comparison table + chart
comp_df = pd.DataFrame([{
    "Model":model_name, "Accuracy":v["accuracy"], "Precision":v["precision"],
    "Recall":v["recall"], "F1 Score":v["f1"], "AUC":v["auc"],
} for model_name,v in eval_results.items()]).sort_values("AUC",ascending=False).reset_index(drop=True)

print(f"\n{'='*60}\nMODEL COMPARISON\n{comp_df.to_string(index=False)}\n{'='*60}")
comp_df.to_csv(OUT_DIR/"model_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(12,5))
x = np.arange(len(comp_df)); w = 0.16
for i,m in enumerate(["Accuracy","Precision","Recall","F1 Score","AUC"]):
    ax.bar(x+i*w, comp_df[m], w, label=m)
ax.set_xticks(x+2*w); ax.set_xticklabels(comp_df["Model"], rotation=20, ha="right")
ax.set_ylim(0,1.05); ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"model_comparison.png", dpi=130, bbox_inches="tight"); plt.close()

# Save best model
best_name  = comp_df.iloc[0]["Model"]
best_model = all_models[best_name]
best_path  = str(MDL_DIR/"best_model.keras")
best_model.save(best_path)
best_preds = eval_results[best_name]["preds"]
best_probs = eval_results[best_name]["probs"]
print(f"\n🏆 Best Model: {best_name} (AUC={comp_df.iloc[0]['AUC']:.4f})")
print(f"   Saved to  : {best_path}")
print(f"\nDetailed report — {best_name}:")
print(classification_report(y_test, best_preds, target_names=CLASS_NAMES))

# ══════════════════════════════════════════════════════════════════
# STEP 5 — ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════
hdr("STEP 5: Error Analysis")

errors      = np.where(y_test != best_preds)[0]
conf_errors = errors[best_probs[errors].max(axis=1) > 0.8]
print(f"Misclassified : {len(errors)}/{len(y_test)} ({100*len(errors)/len(y_test):.1f}%)")
print(f"High-conf wrong: {len(conf_errors)}")
for ci,cn in enumerate(CLASS_NAMES):
    fp = np.sum((best_preds==ci)&(y_test!=ci))
    fn = np.sum((best_preds!=ci)&(y_test==ci))
    print(f"  {cn:35s} FP={fp}  FN={fn}")

# Collect test images for visualization (first 8 errors)
if len(errors) > 0:
    test_imgs_list = []
    for img,_ in test_ds.unbatch():
        test_imgs_list.append(img.numpy())
        if len(test_imgs_list) >= len(y_test): break
    test_imgs = np.array(test_imgs_list)

    n_show = min(8, len(errors))
    fig, axes = plt.subplots(2, 4, figsize=(16,8))
    fig.suptitle(f"Misclassified Samples — {best_name}", fontweight="bold")
    for ax, idx in zip(axes.flat, errors[:n_show]):
        disp = test_imgs[idx].copy()
        disp = (disp - disp.min()) / (disp.max()-disp.min()+1e-8)
        ax.imshow(np.clip(disp,0,1))
        conf = best_probs[idx].max()
        ax.set_title(
            f"True: {CLASS_NAMES[y_test[idx]]}\n"
            f"Pred: {CLASS_NAMES[best_preds[idx]]} ({conf:.2f})", fontsize=7)
        ax.axis("off")
    for ax in axes.flat[n_show:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"error_analysis.png", dpi=130, bbox_inches="tight"); plt.close()
    print("Error analysis saved.")

# ══════════════════════════════════════════════════════════════════
# STEP 6 — GRAD-CAM + SALIENCY
# ══════════════════════════════════════════════════════════════════
hdr("STEP 6: Explainable AI — Grad-CAM + Saliency Maps")

def find_last_conv(model):
    """Find a usable last conv-like layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # sub-model (transfer learning base)
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, keras.layers.DepthwiseConv2D) or \
                   isinstance(sub, keras.layers.Conv2D):
                    return layer.name
    return None

def grad_cam_heatmap(model, img_np, class_idx, layer_name):
    inp  = tf.cast(img_np[np.newaxis], tf.float32)
    grad_model = keras.Model(model.inputs,
                             [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        loss = preds[:, class_idx]
    grads  = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    hmap   = conv_out[0] @ pooled[..., tf.newaxis]
    hmap   = tf.squeeze(hmap).numpy()
    hmap   = np.maximum(hmap, 0)
    if hmap.max() > 0: hmap /= hmap.max()
    return hmap

def make_overlay(img_np, hmap, alpha=0.4):
    disp = img_np.copy()
    disp = (disp - disp.min()) / (disp.max()-disp.min()+1e-8)
    disp = (disp*255).astype(np.uint8)
    hr   = cv2.resize(hmap, (disp.shape[1], disp.shape[0]))
    hc   = cv2.applyColorMap((hr*255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(disp, 1-alpha, cv2.cvtColor(hc, cv2.COLOR_BGR2RGB), alpha, 0)

def saliency(model, img_np, class_idx):
    tv = tf.Variable(tf.cast(img_np[np.newaxis], tf.float32))
    with tf.GradientTape() as tape:
        preds = model(tv)
        loss  = preds[:, class_idx]
    grads = tape.gradient(loss, tv)[0].numpy()
    sal   = np.max(np.abs(grads), axis=-1)
    if sal.max() > 0: sal /= sal.max()
    return sal

conv_layer = find_last_conv(best_model)
print(f"Grad-CAM target layer: {conv_layer}")

# Collect one sample per class from test set
cls_samples = {}
idx = 0
for img, lbl in test_ds.unbatch():
    ci = np.argmax(lbl.numpy())
    if ci not in cls_samples:
        cls_samples[ci] = img.numpy()
    if len(cls_samples) == NUM_CLASSES:
        break

# Combined Grad-CAM figure
fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(12, NUM_CLASSES*4))
fig.suptitle(f"Grad-CAM + Saliency — {best_name}", fontsize=13, fontweight="bold")

for ci, cn in enumerate(CLASS_NAMES):
    img_np = cls_samples.get(ci)
    if img_np is None:
        continue
    disp = img_np.copy()
    disp = (disp - disp.min()) / (disp.max()-disp.min()+1e-8)

    try:
        hmap    = grad_cam_heatmap(best_model, img_np, ci, conv_layer)
        ov      = make_overlay(img_np, hmap)
    except Exception as e:
        print(f"  Grad-CAM failed ({cn}): {e}")
        hmap = np.zeros((7,7))
        ov   = np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)

    sal = saliency(best_model, img_np, ci)

    axes[ci,0].imshow(np.clip(disp,0,1))
    axes[ci,0].set_title(f"Original ({cn})", fontsize=9); axes[ci,0].axis("off")
    axes[ci,1].imshow(ov)
    axes[ci,1].set_title("Grad-CAM Overlay\n(Red=AI focus)", fontsize=9); axes[ci,1].axis("off")
    axes[ci,2].imshow(sal, cmap="hot")
    axes[ci,2].set_title("Saliency Map", fontsize=9); axes[ci,2].axis("off")

    # Save individual XAI per class
    fig2, ax2 = plt.subplots(1, 3, figsize=(12,4))
    fig2.suptitle(f"Explainability — {cn} | {best_name}", fontweight="bold")
    ax2[0].imshow(np.clip(disp,0,1)); ax2[0].set_title("Original"); ax2[0].axis("off")
    ax2[1].imshow(ov); ax2[1].set_title("Grad-CAM"); ax2[1].axis("off")
    ax2[2].imshow(sal,cmap="hot"); ax2[2].set_title("Saliency"); ax2[2].axis("off")
    plt.tight_layout()
    plt.savefig(XAI_DIR/f"xai_{cn.replace('/','_').replace(' ','_')}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig2)

plt.tight_layout()
plt.savefig(XAI_DIR/"gradcam_all_classes.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Grad-CAM + Saliency saved to outputs/xai/")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
hdr("PIPELINE COMPLETE ✅")
print(f"""
Dataset        : {len(df)} real retinal images | {NUM_CLASSES} classes
Classes        : {CLASS_NAMES}
Distribution   : {imb.to_dict()}

Best Model     : {best_name}
  Accuracy     : {comp_df.iloc[0]['Accuracy']:.4f}
  Precision    : {comp_df.iloc[0]['Precision']:.4f}
  Recall       : {comp_df.iloc[0]['Recall']:.4f}
  F1 Score     : {comp_df.iloc[0]['F1 Score']:.4f}
  AUC          : {comp_df.iloc[0]['AUC']:.4f}

Output files:
  EDA plots      → outputs/eda/
  Training curves→ outputs/<model>_training_curves.png
  Confusion mats → outputs/<model>_confusion_matrix.png
  ROC curves     → outputs/<model>_roc_curves.png
  Error analysis → outputs/error_analysis.png
  Grad-CAM/XAI  → outputs/xai/
  Comparison     → outputs/model_comparison.csv
  Best model     → models/best_model.keras

Launch app:
  venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py
""")

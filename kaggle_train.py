"""
Kaggle Training Notebook — Machine Fault Recognition
=====================================================
Paste each section below into a separate cell in a Kaggle Notebook.

Setup:
  1. Create a new Kaggle Notebook → enable GPU (Settings → Accelerator → GPU T4 x2)
  2. Add Dataset: click "+ Add Data" → search "alieldinalaa/nn-cmp27-dataset" → Add
  3. Upload your src/ folder as a SECOND dataset:
     - Go to kaggle.com/datasets → New Dataset → name it "machine-listener-src"
     - Upload your entire project folder (just src/ folder is enough)
     - Click Create
  4. Add that dataset too: "+ Add Data" → "Your Datasets" → "machine-listener-src"
  5. Paste each cell below and run!
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Setup & Install Dependencies                      ║
# ╚═══════════════════════════════════════════════════════════════╝

import subprocess, sys, os, shutil

# Install missing packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "timm", "onnxscript", "onnx", "onnxruntime", "librosa"])

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Copy src/ to working directory                    ║
# ╚═══════════════════════════════════════════════════════════════╝

# Find where your uploaded source code dataset is
# Kaggle mounts datasets at /kaggle/input/<dataset-name>/
INPUT_BASE = "/kaggle/input"

src_candidates = []
for ds in os.listdir(INPUT_BASE):
    ds_path = os.path.join(INPUT_BASE, ds)
    # Look for a 'src' folder inside
    if os.path.isdir(os.path.join(ds_path, "src")):
        src_candidates.append(os.path.join(ds_path, "src"))
    # Or maybe the dataset IS the src folder contents
    if os.path.isfile(os.path.join(ds_path, "preprocess.py")):
        src_candidates.append(ds_path)

if not src_candidates:
    print("WARNING: Could not auto-detect src/ folder.")
    print("Available datasets:", os.listdir(INPUT_BASE))
    print("\n>>> Manually set SRC_PATH below <<<")
    SRC_PATH = None
else:
    SRC_PATH = src_candidates[0]
    print(f"Found source code at: {SRC_PATH}")

# Copy to working directory so we can import it
WORK_DIR = "/kaggle/working"
os.chdir(WORK_DIR)

if SRC_PATH:
    dst = os.path.join(WORK_DIR, "src")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC_PATH, dst)
    print(f"Copied src/ → {dst}")
    print("Contents:", os.listdir(dst))

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Discover the dataset path                         ║
# ╚═══════════════════════════════════════════════════════════════╝

# Find the training data (Machine 1/2/3 folders)
DATA_PATH = None
for ds in os.listdir(INPUT_BASE):
    ds_path = os.path.join(INPUT_BASE, ds)
    # Check if this dataset contains "Machine 1", "Machine 2", etc.
    children = os.listdir(ds_path) if os.path.isdir(ds_path) else []
    if any("Machine" in c for c in children):
        DATA_PATH = ds_path
        break
    # Maybe nested one level deeper
    for child in children:
        child_path = os.path.join(ds_path, child)
        if os.path.isdir(child_path):
            grandchildren = os.listdir(child_path)
            if any("Machine" in g for g in grandchildren):
                DATA_PATH = child_path
                break
    if DATA_PATH:
        break

if DATA_PATH:
    print(f"Dataset found at: {DATA_PATH}")
    print("Contents:", sorted(os.listdir(DATA_PATH)))
else:
    print("ERROR: Could not find Machine 1/2/3 folders!")
    print("Available datasets and their contents:")
    for ds in os.listdir(INPUT_BASE):
        ds_path = os.path.join(INPUT_BASE, ds)
        if os.path.isdir(ds_path):
            print(f"  {ds}/: {os.listdir(ds_path)[:10]}")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Verify imports work                               ║
# ╚═══════════════════════════════════════════════════════════════╝

import sys
sys.path.insert(0, WORK_DIR)

from src.preprocess import preprocess_audio, TARGET_SR, TARGET_LENGTH
from src.features import extract_pcen_mel
from src.dataset import discover_samples, build_dataloaders, get_class_weights, LABEL_NAMES, NUM_CLASSES
from src.model import build_model, FocalLoss, export_to_onnx

# Quick sanity check
samples = discover_samples(DATA_PATH)
print(f"\nTotal samples found: {len(samples)}")
from collections import Counter
counts = Counter(label for _, label in samples)
for cls_id in sorted(counts):
    print(f"  Class {cls_id} ({LABEL_NAMES[cls_id]}): {counts[cls_id]} files")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Pre-compute features (optional but saves time)    ║
# ╚═══════════════════════════════════════════════════════════════╝

# This caches PCEN features to disk so training epochs are fast.
# Takes ~10-20 min on the full dataset but saves hours over many epochs.
# Skip this cell if you want to start training immediately.

CACHE_DIR = os.path.join(WORK_DIR, ".cache", "features")

from src.dataset import precompute_features
precompute_features(DATA_PATH, CACHE_DIR, noise_suppression=True)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Train!                                            ║
# ╚═══════════════════════════════════════════════════════════════╝

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from src.train import train

SAVE_DIR = os.path.join(WORK_DIR, "models")

results = train(
    data_root=DATA_PATH,
    backbone="efficientnet_b0",      # or "mobilenetv3_small" for faster
    pretrained=True,
    epochs=60,                        # increase to 80-100 for best accuracy
    batch_size=32,                    # GPU can handle larger batches
    lr=3e-4,
    weight_decay=1e-2,
    warmup_ratio=0.05,
    gamma=2.0,                        # Focal Loss focusing
    label_smoothing=0.1,
    drop_rate=0.3,
    drop_path_rate=0.2,
    max_grad_norm=1.0,
    patience=15,                      # early stopping
    val_ratio=0.15,
    test_ratio=0.15,
    num_workers=2,                    # Kaggle has 2+ CPU cores
    cache_dir=CACHE_DIR,              # use pre-computed features
    noise_suppression=True,
    use_amp=True,                     # fp16 on GPU = 2x speedup
    save_dir=SAVE_DIR,
    seed=42,
)

print("\nDone! Results:", results)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Download your trained model                       ║
# ╚═══════════════════════════════════════════════════════════════╝

# The trained files are saved at:
#   /kaggle/working/models/best_model.pt   ← PyTorch checkpoint
#   /kaggle/working/models/model.onnx      ← ONNX for inference
#   /kaggle/working/models/history.json    ← training curves
#
# To download: click the "Output" tab in the right panel of Kaggle,
# then click on each file to download it.
#
# After downloading, place them in your local project:
#   d:\CMP\NN\project\models\best_model.pt
#   d:\CMP\NN\project\models\model.onnx

print("Files saved to /kaggle/working/models/:")
for f in os.listdir(SAVE_DIR):
    fpath = os.path.join(SAVE_DIR, f)
    size = os.path.getsize(fpath) / 1e6
    print(f"  {f:25s} {size:8.2f} MB")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 8 (OPTIONAL) — Plot training curves                   ║
# ╚═══════════════════════════════════════════════════════════════╝

import json
import matplotlib.pyplot as plt

with open(os.path.join(SAVE_DIR, "history.json")) as f:
    history = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"], label="Val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(history["lr"])
axes[2].set_title("Learning Rate")
axes[2].set_xlabel("Epoch")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()
print("Training curves saved!")

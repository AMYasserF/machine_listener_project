"""
Local Training Script — Machine Fault Recognition
=====================================================
Setup Instructions for your laptop:
  1. Make sure your project folder looks like this:
     project/
     ├── data/                  <-- Put the unzipped dataset here (Machine 1, Machine 2, etc.)
     ├── src/                   <-- Your python modules (dataset.py, model.py, etc.)
     ├── models/                <-- Trained models will be saved here
     └── train_local.ipynb      <-- THIS notebook/script
  2. Install requirements: pip install torch torchvision torchaudio timm librosa onnx onnxscript onnxruntime scikit-learn matplotlib
  3. Run the cells below!
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Setup Paths & Verify Environment                    ║
# ╚═══════════════════════════════════════════════════════════════╝

import os
import sys

# 1. Define Local Paths (Assuming this script is in the project root)
PROJECT_ROOT = os.path.abspath(".")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "features")

# Ensure Python can find the 'src' folder
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Project Root: {PROJECT_ROOT}")
print(f"✅ Data Path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("❌ ERROR: 'data' folder not found! Please make sure your dataset is extracted in a folder named 'data' next to this script.")
elif not os.path.exists(os.path.join(PROJECT_ROOT, "src")):
    print("❌ ERROR: 'src' folder not found! Please make sure the 'src' folder is next to this script.")
else:
    print("✅ Project structure looks good!")

# Create save directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Verify imports & Dataset                            ║
# ╚═══════════════════════════════════════════════════════════════╝

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
# ║  CELL 3 — Pre-compute features (optional but saves time)      ║
# ╚═══════════════════════════════════════════════════════════════╝

# This caches PCEN features to disk so training epochs are fast.
# Takes some time initially but saves hours over many epochs.
from src.dataset import precompute_features
precompute_features(DATA_PATH, CACHE_DIR, noise_suppression=True)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Train!                                              ║
# ╚═══════════════════════════════════════════════════════════════╝

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: CUDA not available. Training on CPU will be slow.")

from src.train import train

# VERY IMPORTANT FOR WINDOWS LAPTOPS:
# PyTorch DataLoader multiprocessing on Windows requires the training 
# code to be inside an `if __name__ == '__main__':` block.
if __name__ == '__main__':
    results = train(
        data_root=DATA_PATH,
        backbone="efficientnet_b0",      # or "mobilenetv3_small" for faster laptop training
        pretrained=False,                # MUST be False — train from scratch!
        epochs=50,                       # good balance of accuracy vs overfitting
        batch_size=16,                   # Reduced from 32 to 16 for standard laptop GPUs (e.g., 4GB/6GB VRAM)
        lr=3e-4,
        weight_decay=1e-2,
        warmup_ratio=0.05,
        gamma=2.0,                       # Focal Loss focusing
        label_smoothing=0.1,
        drop_rate=0.3,
        drop_path_rate=0.2,
        max_grad_norm=1.0,
        patience=12,                     # early stopping
        val_ratio=0.15,
        test_ratio=0.20,                 # slightly larger test set for confidence
        num_workers=0,                   # MUST BE 0 ON WINDOWS LAPTOPS to avoid multiprocessing crash. Change to 2/4 if on Linux/Mac.
        cache_dir=CACHE_DIR,             # use pre-computed features
        noise_suppression=True,
        use_amp=torch.cuda.is_available(), # Only use AMP (fp16) if GPU is available
        save_dir=SAVE_DIR,
        seed=42,
        max_train_samples=None,          # set e.g. 500 to train on a small subset for quick laptop testing
    )

    print("\nDone! Results:", results)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Plot training curves                                ║
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
print(f"Training curves saved to {SAVE_DIR}!")
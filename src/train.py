"""
src/train.py — Training Loop with Cosine LR, Early Stopping & ONNX Export
===========================================================================
Full training pipeline for the Machine Fault Recognition system.

Usage
-----
::

    # Quick run (small dataset, no GPU)
    python -m src.train --data data/ --epochs 30 --batch-size 4 --workers 0

    # Full training (large dataset, GPU)
    python -m src.train --data data/ --epochs 100 --batch-size 32 \\
        --backbone efficientnet_b0 --cache .cache/features \\
        --lr 3e-4 --amp

    # Fast lightweight model
    python -m src.train --data data/ --backbone mobilenetv3_small \\
        --epochs 80 --batch-size 64 --amp

Features
--------
*  **AdamW** optimiser with weight decay
*  **Linear warm-up + cosine annealing** LR schedule
*  **Focal Loss** with inverse-frequency class weights & label smoothing
*  **Mixed-precision** training (``--amp``) for ~2× GPU speedup
*  **Gradient clipping** to prevent exploding gradients
*  **Early stopping** on validation loss with configurable patience
*  **Best-model checkpointing** (saves PyTorch ``.pt`` + ONNX ``.onnx``)
*  **Confusion matrix** & per-class metrics printed at the end
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tqdm import tqdm

from src.dataset import (
    discover_samples,
    build_dataloaders,
    get_class_weights,
    NUM_CLASSES,
    LABEL_NAMES,
)
from src.model import build_model, FocalLoss, export_to_onnx


# ──────────────────────────────────────────────────────────────
# Cosine LR schedule with linear warm-up
# ──────────────────────────────────────────────────────────────

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for *warmup_steps* then cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────
# Single-epoch routines
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    scaler: Optional[GradScaler],
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train for one full epoch.

    Returns
    -------
    metrics : dict
        ``{"loss": ..., "acc": ..., "lr": ...}``
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward ──
        if scaler is not None:
            with autocast(device_type=device.type):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()

        # ── Accumulate metrics ──
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "lr": scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate model on a data loader.

    Returns
    -------
    metrics : dict
        ``{"loss": ..., "acc": ..., "preds": [...], "targets": [...]}``
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    total = len(all_targets) or 1
    return {
        "loss": total_loss / total,
        "acc": accuracy_score(all_targets, all_preds),
        "preds": all_preds,
        "targets": all_targets,
    }


# ──────────────────────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────────────────────

def train(
    *,
    data_root: str = "data/",
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    warmup_ratio: float = 0.05,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
    max_grad_norm: float = 1.0,
    patience: int = 15,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    noise_suppression: bool = True,
    use_amp: bool = False,
    save_dir: str = "models/",
    seed: int = 42,
) -> Dict[str, object]:
    """End-to-end training procedure.

    Parameters
    ----------
    (see argparse below for full descriptions)

    Returns
    -------
    results : dict
        Final metrics, best epoch, paths to saved artefacts.
    """

    # ── Reproducibility ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Machine Fault Recognition — Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Backbone    : {backbone}")
    print(f"  Pretrained  : {pretrained}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {lr}")
    print(f"  Focal γ     : {gamma}")
    print(f"  Label smooth: {label_smoothing}")
    print(f"  AMP         : {use_amp}")
    print(f"  Patience    : {patience}")
    print(f"{'='*60}\n")

    # ── Data ──
    print("[1/5] Building data loaders ...")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        noise_suppression=noise_suppression,
        seed=seed,
    )
    print(f"  Train : {len(train_loader.dataset)} samples  ({len(train_loader)} batches)")
    print(f"  Val   : {len(val_loader.dataset)} samples  ({len(val_loader)} batches)")
    print(f"  Test  : {len(test_loader.dataset)} samples  ({len(test_loader)} batches)")

    # ── Class weights ──
    class_weights = get_class_weights(train_loader.dataset.samples).to(device)
    print(f"  Weights: {class_weights.cpu().tolist()}")

    # ── Model ──
    print(f"\n[2/5] Building model ({backbone}) ...")
    model = build_model(
        backbone=backbone,
        pretrained=pretrained,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Loss, Optimiser, Scheduler ──
    print(f"\n[3/5] Configuring optimiser & scheduler ...")
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = _build_scheduler(optimizer, total_steps, warmup_steps)
    print(f"  Total steps : {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    # ── Training loop ──
    print(f"\n[4/5] Training ...\n")
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_pt_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, max_grad_norm,
        )

        # ── Validate ──
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # ── Log ──
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(train_metrics["lr"])

        marker = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": val_metrics["acc"],
                "backbone": backbone,
            }, best_pt_path)
            marker = " ★ saved"
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"lr={train_metrics['lr']:.2e} │ "
            f"train_loss={train_metrics['loss']:.4f}  acc={train_metrics['acc']:.3f} │ "
            f"val_loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.3f} │ "
            f"{elapsed:.1f}s{marker}"
        )

        # ── Early stopping ──
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered (no improvement for {patience} epochs)")
            break

    # ── Load best model ──
    print(f"\n  Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f})")
    checkpoint = torch.load(best_pt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Final test evaluation ──
    print(f"\n[5/5] Evaluating on test set ...\n")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"  Test Loss     : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy : {test_metrics['acc']:.4f}")

    # Per-class report
    if test_metrics["targets"]:
        target_names = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
        report = classification_report(
            test_metrics["targets"],
            test_metrics["preds"],
            target_names=target_names,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
        print(f"\n{report}")

        cm = confusion_matrix(
            test_metrics["targets"],
            test_metrics["preds"],
            labels=list(range(NUM_CLASSES)),
        )
        print("  Confusion Matrix:")
        header = "       " + "  ".join(f"{i:>4}" for i in range(NUM_CLASSES))
        print(header)
        for i, row in enumerate(cm):
            print(f"    {i}: " + "  ".join(f"{v:>4}" for v in row))

    # ── ONNX export ──
    onnx_path = os.path.join(save_dir, "model.onnx")
    print(f"\n  Exporting to ONNX: {onnx_path}")
    export_to_onnx(model.cpu(), onnx_path)

    # ── Save training history ──
    history_path = os.path.join(save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "model_path": best_pt_path,
        "onnx_path": onnx_path,
        "total_params": total_params,
    }

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Test accuracy : {test_metrics['acc']:.4f}")
    print(f"  Model (.pt)   : {os.path.abspath(best_pt_path)}")
    print(f"  Model (.onnx) : {os.path.abspath(onnx_path)}")
    print(f"{'='*60}\n")

    return results


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Machine Fault Recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    g = p.add_argument_group("Data")
    g.add_argument("--data", type=str, default="data/",
                   help="Root directory with Machine {1,2,3}/machine_data/…")
    g.add_argument("--cache", type=str, default=None,
                   help="Cache dir for pre-computed features (.npy)")
    g.add_argument("--val-ratio", type=float, default=0.15)
    g.add_argument("--test-ratio", type=float, default=0.15)
    g.add_argument("--workers", type=int, default=0,
                   help="DataLoader num_workers (0 = main process)")
    g.add_argument("--no-denoise", action="store_true",
                   help="Disable MMSE-STSA noise suppression (faster)")

    # ── Model ──
    g = p.add_argument_group("Model")
    g.add_argument("--backbone", type=str, default="efficientnet_b0",
                   choices=["efficientnet_b0", "mobilenetv3_small", "mobilenetv3_large"])
    g.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    g.add_argument("--drop-rate", type=float, default=0.3)
    g.add_argument("--drop-path-rate", type=float, default=0.2)

    # ── Optimiser ──
    g = p.add_argument_group("Optimiser")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="Fraction of total steps used for LR warmup")
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--amp", action="store_true", help="Mixed-precision (fp16)")

    # ── Loss ──
    g = p.add_argument_group("Loss")
    g.add_argument("--gamma", type=float, default=2.0,
                   help="Focal Loss gamma (focusing exponent)")
    g.add_argument("--label-smoothing", type=float, default=0.1)

    # ── Training ──
    g = p.add_argument_group("Training")
    g.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (epochs)")
    g.add_argument("--save-dir", type=str, default="models/")
    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set UTF-8 for Windows console (emoji in ONNX logs)
    os.environ["PYTHONIOENCODING"] = "utf-8"

    train(
        data_root=args.data,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers,
        cache_dir=args.cache,
        noise_suppression=not args.no_denoise,
        use_amp=args.amp,
        save_dir=args.save_dir,
        seed=args.seed,
    )

"""
src/model.py — Model Architecture, Focal Loss & ONNX Export
=============================================================
Provides a factory for ultra-lightweight audio classifiers built on top of
``timm`` backbones (trained from scratch), plus a numerically stable Focal
Loss and an ONNX export utility.

**IMPORTANT**: All models are trained FROM SCRATCH — no pretrained weights
are used.  This is a hard requirement to avoid data leakage concerns and
ensure the model learns solely from the provided training data.

Available backbones
-------------------
+----------------------------+--------+-----------------------------------+
| Name (``backbone`` arg)    | Params |  Notes                            |
+============================+========+===================================+
| ``efficientnet_b0``        | 4.01 M | Best accuracy / efficiency trade  |
+----------------------------+--------+-----------------------------------+
| ``mobilenetv3_small_100``  | 1.52 M | Fastest inference (ONNX ≈ 3 ms)  |
+----------------------------+--------+-----------------------------------+
| ``mobilenetv3_large_100``  | 4.21 M | Middle ground                     |
+----------------------------+--------+-----------------------------------+

All backbones are automatically adapted for **single-channel** PCEN
spectrogram input ``(1, 128, 313)`` via ``timm``'s ``in_chans`` argument.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.dataset import NUM_CLASSES

# ──────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────

# Supported backbone names → timm model identifiers
_BACKBONE_REGISTRY = {
    "efficientnet_b0":        "efficientnet_b0",
    "mobilenetv3_small":      "mobilenetv3_small_100",
    "mobilenetv3_large":      "mobilenetv3_large_100",
}


def build_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Create a lightweight audio classifier from a ``timm`` backbone.

    **All models are trained from scratch** — pretrained weights are NOT
    allowed per project requirements.

    Parameters
    ----------
    backbone : str
        One of ``"efficientnet_b0"``, ``"mobilenetv3_small"``,
        ``"mobilenetv3_large"``.
    num_classes : int
        Number of output classes (default 6).
    pretrained : bool
        Must be ``False``.  Kept for API compatibility but will raise
        an error if set to ``True``.
    drop_rate : float
        Classifier dropout probability.
    drop_path_rate : float
        Stochastic depth / drop-path rate (regularisation for small
        datasets).

    Returns
    -------
    model : nn.Module
        Ready-to-train PyTorch model.  Input shape: ``(B, 1, 128, 313)``.
        Output shape: ``(B, num_classes)``  (raw logits, no softmax).
    """
    # ── HARD CONSTRAINT: no pretrained weights ──
    if pretrained:
        raise ValueError(
            "pretrained=True is NOT allowed. "
            "All models must be trained from scratch (no ImageNet weights). "
            "This is a project requirement to avoid data leakage."
        )

    timm_name = _BACKBONE_REGISTRY.get(backbone)
    if timm_name is None:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from: {list(_BACKBONE_REGISTRY.keys())}"
        )

    model = timm.create_model(
        timm_name,
        pretrained=False,          # ALWAYS from scratch
        in_chans=1,                # single-channel PCEN spectrogram
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    return model


# ──────────────────────────────────────────────────────────────
# Focal Loss  (Lin et al., ICCV 2017)
# ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class Focal Loss for addressing class imbalance.

    Focal Loss down-weights the contribution of easy (well-classified)
    examples and focuses training on hard misclassifications::

        FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    When γ = 0 this reduces to standard weighted cross-entropy.

    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weights, shape ``(C,)``.  Use the output of
        ``dataset.get_class_weights()`` for inverse-frequency weighting.
        If ``None``, all classes are weighted equally.
    gamma : float
        Focusing exponent.  Higher values increase the effect.
        Typical range: 1.0 – 3.0.  Default 2.0 (from the original paper).
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    label_smoothing : float
        Optional label smoothing ε ∈ [0, 1).  Softens hard targets to
        ``(1 − ε) · one_hot + ε / C``.  Helps prevent over-confident
        predictions on small datasets.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C)  raw model outputs (before softmax).
        targets : (B,)   integer class labels.

        Returns
        -------
        loss : scalar tensor.
        """
        C = logits.size(1)

        # Numerically stable log-softmax
        log_probs = F.log_softmax(logits, dim=1)      # (B, C)
        probs = log_probs.exp()                         # (B, C)

        # Gather the probability of the true class
        targets_oh = F.one_hot(targets, num_classes=C).float()  # (B, C)

        # Optional label smoothing
        if self.label_smoothing > 0:
            targets_oh = (
                (1.0 - self.label_smoothing) * targets_oh
                + self.label_smoothing / C
            )

        # Focal modulating factor: (1 - p_t)^γ
        p_t = (probs * targets_oh).sum(dim=1)           # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma        # (B,)

        # Cross-entropy per sample (using smoothed targets)
        ce = -(targets_oh * log_probs).sum(dim=1)       # (B,)

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]               # (B,)
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────────────────────

def export_to_onnx(
    model: nn.Module,
    save_path: str = "models/model.onnx",
    input_shape: tuple = (1, 1, 128, 313),
    opset_version: int = 18,
    validate: bool = True,
) -> str:
    """Export a trained PyTorch model to ONNX format for fast inference.

    Parameters
    ----------
    model : nn.Module
        Trained model (will be set to eval mode internally).
    save_path : str
        Output ``.onnx`` file path.
    input_shape : tuple
        ``(batch, channels, n_mels, n_frames)``.
    opset_version : int
        ONNX opset version (17 is widely supported).
    validate : bool
        If ``True``, reload the exported model with ``onnxruntime`` and
        verify outputs match PyTorch within tolerance.

    Returns
    -------
    save_path : str
        Absolute path to the saved ONNX file.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)

    # Workaround: torch.onnx on Windows may crash printing emoji with cp1252
    _prev_enc = os.environ.get("PYTHONIOENCODING")
    os.environ["PYTHONIOENCODING"] = "utf-8"

    try:
        torch.onnx.export(
            model,
            dummy,
            save_path,
            input_names=["spectrogram"],
            output_names=["logits"],
            dynamic_axes={
                "spectrogram": {0: "batch"},
                "logits":      {0: "batch"},
            },
            opset_version=opset_version,
        )
    finally:
        if _prev_enc is None:
            os.environ.pop("PYTHONIOENCODING", None)
        else:
            os.environ["PYTHONIOENCODING"] = _prev_enc

    abs_path = os.path.abspath(save_path)
    print(f"[ONNX] Exported → {abs_path}")

    # ── Optional validation ──
    if validate:
        import onnx
        import onnxruntime as ort

        # Structural check
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX] Model structure validated ✓")

        # Numerical check
        sess = ort.InferenceSession(save_path)
        np_input = dummy.cpu().numpy()

        with torch.no_grad():
            pt_out = model(dummy).cpu().numpy()

        ort_out = sess.run(None, {"spectrogram": np_input})[0]

        max_diff = np.abs(pt_out - ort_out).max()
        print(f"[ONNX] PyTorch vs ONNX max diff: {max_diff:.2e}")
        assert max_diff < 1e-4, f"ONNX validation failed! max_diff={max_diff}"
        print("[ONNX] Numerical validation passed ✓")

    return abs_path


# ──────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model smoke test")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=list(_BACKBONE_REGISTRY.keys()))
    parser.add_argument("--export", action="store_true",
                        help="Export to ONNX after test")
    args = parser.parse_args()

    print(f"Building model: {args.backbone}")
    model = build_model(args.backbone, pretrained=False)

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    print(f"  Size (float32):   {total * 4 / 1e6:>9.2f} MB")

    # Forward pass test
    x = torch.randn(2, 1, 128, 313)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {logits.shape}")

    # Focal Loss test
    targets = torch.tensor([0, 3])
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss = criterion(logits, targets)
    print(f"  Focal Loss: {loss.item():.4f}")

    # ONNX export
    if args.export:
        export_to_onnx(model, "models/model.onnx")

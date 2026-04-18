"""
src/dataset.py — PyTorch Dataset & DataLoader factories
========================================================
Handles two distinct use-cases:

1. **Training / Validation** (``MachineAudioDataset``)
   - Walks the on-disk folder hierarchy automatically::

         data/
         ├── Machine 1/machine_data/{Normal,Abnormal}/*.wav
         ├── Machine 2/machine_data/{Normal,Abnormal}/*.wav
         └── Machine 3/machine_data/{Normal,Abnormal}/*.wav

   - Maps (Machine, Condition) → integer label 0-5.
   - Optionally caches pre-computed PCEN features to ``.npy`` for fast
     subsequent epochs (MMSE-STSA is expensive).
   - Applies SpecAugment **online** during training (different mask every
     epoch → free data augmentation).

2. **Inference** (``InferenceDataset``)
   - Reads ``data/1.wav, 2.wav, …`` in strict numeric order.
   - No labels, no augmentation, no caching.
"""

from __future__ import annotations

import os
import re
import pathlib
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.preprocess import (
    TARGET_SR,
    TARGET_LENGTH,
    preprocess_audio,
    preprocess_waveform,
    load_audio,
)
from src.features import extract_pcen_mel, spec_augment

# ──────────────────────────────────────────────────────────────
# Label mapping  (Machine × Condition → int)
# ──────────────────────────────────────────────────────────────
LABEL_MAP: Dict[Tuple[str, str], int] = {
    ("Machine 1", "Normal"):   0,
    ("Machine 1", "Abnormal"): 1,
    ("Machine 2", "Normal"):   2,
    ("Machine 2", "Abnormal"): 3,
    ("Machine 3", "Normal"):   4,
    ("Machine 3", "Abnormal"): 5,
}

NUM_CLASSES: int = 6

LABEL_NAMES: List[str] = [
    "Machine 1 — Normal",
    "Machine 1 — Abnormal",
    "Machine 2 — Normal",
    "Machine 2 — Abnormal",
    "Machine 3 — Normal",
    "Machine 3 — Abnormal",
]


# ──────────────────────────────────────────────────────────────
# Discovery: walk the training directory tree
# ──────────────────────────────────────────────────────────────

def discover_samples(
    root: str | pathlib.Path,
) -> List[Tuple[str, int]]:
    """Scan *root* for ``.wav`` files and assign labels from folder names.

    Expected hierarchy::

        root/
        ├── Machine 1/machine_data/Normal/*.wav     → label 0
        ├── Machine 1/machine_data/Abnormal/*.wav   → label 1
        ├── Machine 2/machine_data/Normal/*.wav     → label 2
        ├── Machine 2/machine_data/Abnormal/*.wav   → label 3
        ├── Machine 3/machine_data/Normal/*.wav     → label 4
        └── Machine 3/machine_data/Abnormal/*.wav   → label 5

    Returns
    -------
    samples : list of (filepath, label)
        Sorted deterministically by filepath.
    """
    root = pathlib.Path(root)
    samples: List[Tuple[str, int]] = []

    for machine_dir in sorted(root.iterdir()):
        if not machine_dir.is_dir():
            continue
        machine_name = machine_dir.name  # "Machine 1", "Machine 2", …

        data_dir = machine_dir / "machine_data"
        if not data_dir.exists():
            # Try flat layout: Machine X/{Normal,Abnormal}
            data_dir = machine_dir

        for condition_dir in sorted(data_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name  # "Normal" or "Abnormal"

            key = (machine_name, condition)
            if key not in LABEL_MAP:
                continue

            label = LABEL_MAP[key]
            for wav in sorted(condition_dir.glob("*.wav")):
                samples.append((str(wav), label))

    if not samples:
        raise FileNotFoundError(
            f"No .wav files found under {root}. "
            f"Expected hierarchy: <root>/<Machine N>/machine_data/{{Normal,Abnormal}}/*.wav"
        )

    return samples


# ──────────────────────────────────────────────────────────────
# Training / Validation Dataset
# ──────────────────────────────────────────────────────────────

class MachineAudioDataset(Dataset):
    """PyTorch Dataset for machine-fault audio classification.

    Parameters
    ----------
    samples : list of (filepath, label)
        Output of :func:`discover_samples`.
    augment : bool
        If ``True``, apply SpecAugment to the PCEN spectrogram on every
        ``__getitem__`` call (training mode).
    cache_dir : str or None
        Directory to cache pre-computed PCEN ``.npy`` files.  If ``None``,
        features are computed on-the-fly every epoch.
    noise_suppression : bool
        Whether to run MMSE-STSA during preprocessing.  Disable for faster
        experimentation / ablation studies.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        *,
        augment: bool = False,
        cache_dir: Optional[str] = None,
        noise_suppression: bool = True,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.augment = augment
        self.cache_dir = cache_dir
        self.noise_suppression = noise_suppression

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, idx: int) -> Optional[str]:
        if self.cache_dir is None:
            return None
        filepath, _ = self.samples[idx]
        # Deterministic cache key: hash of absolute path
        key = pathlib.Path(filepath).stem + f"_{idx}"
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _extract_features(self, idx: int) -> np.ndarray:
        """Load, preprocess, and extract PCEN features — with optional caching."""
        cache_path = self._cache_path(idx)

        # ── Try cache first ──
        if cache_path is not None and os.path.isfile(cache_path):
            return np.load(cache_path)

        # ── Compute from scratch ──
        filepath, _ = self.samples[idx]
        waveform = preprocess_audio(
            filepath,
            noise_suppression=self.noise_suppression,
        )
        pcen = extract_pcen_mel(waveform)  # (1, n_mels, n_frames)

        # ── Save to cache ──
        if cache_path is not None:
            np.save(cache_path, pcen)

        return pcen

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pcen = self._extract_features(idx)       # (1, 128, 313)

        if self.augment:
            pcen = spec_augment(pcen)             # in-place copy inside

        tensor = torch.from_numpy(pcen)           # float32 tensor
        _, label = self.samples[idx]
        return tensor, label


# ──────────────────────────────────────────────────────────────
# Inference Dataset  (test-time, no labels)
# ──────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    """Lightweight dataset for graded evaluation.

    Reads ``data_dir/1.wav, 2.wav, …, N.wav`` in strict ascending order.
    Returns the *raw waveform* (not features) because the inference timer
    must start **after** the file read but **include** preprocessing +
    feature extraction time.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing numbered ``.wav`` files.
    """

    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)

        # Discover and sort by numeric filename
        wav_files = list(self.data_dir.glob("*.wav"))
        wav_files.sort(key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        self.files = wav_files

        if not self.files:
            raise FileNotFoundError(
                f"No .wav files found in {data_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Return (raw_waveform, filename) — preprocessing happens in infer.py."""
        filepath = self.files[idx]
        y = load_audio(str(filepath), sr=TARGET_SR)
        return y, filepath.name


# ──────────────────────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    noise_suppression: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from the on-disk directory tree.

    Parameters
    ----------
    data_root : str
        Root directory containing ``Machine {{1,2,3}}/machine_data/…``.
    val_ratio, test_ratio : float
        Fraction of data reserved for validation and test.  Splitting is
        **stratified** to preserve class proportions.
    batch_size : int
        Mini-batch size (applies to train; val/test use the same).
    num_workers : int
        DataLoader worker processes (set to 0 on Windows if you get
        multiprocessing errors).
    cache_dir : str or None
        Where to store cached ``.npy`` features.  ``None`` = no caching.
    noise_suppression : bool
        Toggle MMSE-STSA in the preprocessing stage.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    all_samples = discover_samples(data_root)
    filepaths, labels = zip(*all_samples)
    filepaths = list(filepaths)
    labels = list(labels)

    # ── Stratified split: train+val vs test ──
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        filepaths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # ── Stratified split: train vs val ──
    relative_val = val_ratio / (1.0 - test_ratio)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=relative_val,
        stratify=train_val_labels,
        random_state=seed,
    )

    train_samples = list(zip(train_files, train_labels))
    val_samples   = list(zip(val_files, val_labels))
    test_samples  = list(zip(test_files, test_labels))

    # ── Datasets ──
    train_cache = os.path.join(cache_dir, "train") if cache_dir else None
    val_cache   = os.path.join(cache_dir, "val")   if cache_dir else None
    test_cache  = os.path.join(cache_dir, "test")  if cache_dir else None

    train_ds = MachineAudioDataset(
        train_samples,
        augment=True,
        cache_dir=train_cache,
        noise_suppression=noise_suppression,
    )
    val_ds = MachineAudioDataset(
        val_samples,
        augment=False,
        cache_dir=val_cache,
        noise_suppression=noise_suppression,
    )
    test_ds = MachineAudioDataset(
        test_samples,
        augment=False,
        cache_dir=test_cache,
        noise_suppression=noise_suppression,
    )

    # ── DataLoaders ──
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(samples: List[Tuple[str, int]]) -> torch.Tensor:
    """Compute inverse-frequency class weights for Focal Loss / weighted CE.

    Parameters
    ----------
    samples : list of (filepath, label)

    Returns
    -------
    weights : torch.Tensor, shape (NUM_CLASSES,)
        Normalised so they sum to ``NUM_CLASSES`` (mean = 1.0).
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, label in samples:
        counts[label] += 1

    # Inverse frequency, then normalise to mean=1
    weights = 1.0 / (counts + 1e-6)
    weights *= NUM_CLASSES / weights.sum()

    return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Pre-computation utility  (run once before training)
# ──────────────────────────────────────────────────────────────

def precompute_features(
    data_root: str,
    cache_dir: str,
    noise_suppression: bool = True,
) -> None:
    """Walk the dataset and cache all PCEN features to disk.

    Run this **once** before training to avoid recomputing MMSE-STSA +
    PCEN on every epoch::

        python -m src.dataset --precompute --data data/ --cache .cache/features
    """
    from tqdm import tqdm

    samples = discover_samples(data_root)
    os.makedirs(cache_dir, exist_ok=True)

    ds = MachineAudioDataset(
        samples,
        augment=False,
        cache_dir=cache_dir,
        noise_suppression=noise_suppression,
    )

    print(f"Pre-computing {len(ds)} samples → {cache_dir}")
    for i in tqdm(range(len(ds)), desc="Caching features"):
        _ = ds[i]  # triggers compute + cache write

    print("Done.")


# ──────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("--data", type=str, default="data/",
                        help="Root data directory")
    parser.add_argument("--cache", type=str, default=None,
                        help="Cache dir for pre-computed features")
    parser.add_argument("--precompute", action="store_true",
                        help="Pre-compute and cache all PCEN features")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Disable MMSE-STSA noise suppression")
    args = parser.parse_args()

    if args.precompute:
        precompute_features(
            args.data,
            args.cache or ".cache/features",
            noise_suppression=not args.no_denoise,
        )
    elif args.stats:
        samples = discover_samples(args.data)
        from collections import Counter
        counts = Counter(label for _, label in samples)
        print(f"\nDataset root : {args.data}")
        print(f"Total samples: {len(samples)}\n")
        for cls_id in sorted(counts):
            print(f"  Class {cls_id} ({LABEL_NAMES[cls_id]}): {counts[cls_id]} files")

        weights = get_class_weights(samples)
        print(f"\nClass weights: {weights.tolist()}")
    else:
        parser.print_help()

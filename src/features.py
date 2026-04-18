"""
src/features.py — PCEN Mel-Spectrogram Feature Extraction & SpecAugment
=======================================================================
Converts a preprocessed waveform into a Per-Channel Energy Normalised (PCEN)
Mel-Spectrogram suitable for feeding into a 2-D convolutional / transformer
model.

Why PCEN instead of log-Mel?
----------------------------
*  **Automatic gain control** — adapts per frequency channel, compensating
   for microphone-specific gain curves (critical when different machines use
   different sensors).
*  **Dynamic range compression** — controlled via a power-law exponent,
   avoids the fragile ``log(x + ε)`` clipping problem at low energies.
*  **Stationary-noise robustness** — the temporal smoothing inherently
   suppresses slowly-varying factory hum residuals that survive MMSE-STSA.

Reference
---------
Wang, Y. et al. *"Trainable Frontend for Robust and Far-Field Keyword
Spotting"*, ICASSP 2017.

Output tensor shape
-------------------
With the default parameters and a 5-second / 16 kHz input::

    n_mels  = 128
    n_frames = 1 + TARGET_LENGTH // HOP_LENGTH = 313

    → (1, 128, 313)   single-channel 2-D "image"
"""

from __future__ import annotations

import numpy as np
import librosa

from src.preprocess import (
    TARGET_SR,
    TARGET_LENGTH,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)

# ──────────────────────────────────────────────────────────────
# Mel-spectrogram & PCEN hyper-parameters
# ──────────────────────────────────────────────────────────────
N_MELS: int = 128
FMIN: float = 20.0       # lowest frequency (Hz) — below machine-relevant range
FMAX: float = 8000.0     # Nyquist for 16 kHz

# PCEN parameters  (librosa defaults are already well-tuned for keyword
# spotting; we tweak slightly for industrial audio)
PCEN_GAIN: float = 0.98          # α  — AGC strength
PCEN_BIAS: float = 2.0           # δ  — stabilises near-zero energy bins
PCEN_POWER: float = 0.5          # r  — power-law compression exponent
PCEN_TIME_CONSTANT: float = 0.06 # τ  — IIR smoothing time-constant (seconds)
PCEN_EPS: float = 1e-6           # ε  — numerical floor

# SpecAugment defaults (Park et al., Interspeech 2019)
SPEC_TIME_MASK_PARAM: int = 30   # max consecutive time-frames to mask
SPEC_FREQ_MASK_PARAM: int = 15   # max consecutive mel-bands to mask
SPEC_NUM_TIME_MASKS: int = 2     # how many independent time masks
SPEC_NUM_FREQ_MASKS: int = 2     # how many independent frequency masks


# ──────────────────────────────────────────────────────────────
# Core feature extraction
# ──────────────────────────────────────────────────────────────

def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Compute a **linear-scale** Mel spectrogram (no log, no PCEN yet).

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Preprocessed waveform (float32, mono).

    Returns
    -------
    S : np.ndarray, shape (n_mels, n_frames), dtype float32
        Non-negative Mel-scaled energy spectrogram.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,        # energy (magnitude²)
    )
    return S.astype(np.float32)


def apply_pcen(
    S: np.ndarray,
    sr: int = TARGET_SR,
    hop_length: int = HOP_LENGTH,
    gain: float = PCEN_GAIN,
    bias: float = PCEN_BIAS,
    power: float = PCEN_POWER,
    time_constant: float = PCEN_TIME_CONSTANT,
    eps: float = PCEN_EPS,
) -> np.ndarray:
    """Apply Per-Channel Energy Normalisation to a **linear** Mel spectrogram.

    PCEN replaces the conventional ``log(S + ε)`` with a learned /
    parametric pipeline of three operations:

    1. **Temporal integration** — IIR low-pass per channel gives the
       smoothed filter-bank energy ``M(t,f)``.
    2. **Adaptive gain control** — ``S / (M + ε)^α`` normalises the
       instantaneous energy by the smoothed background.
    3. **Dynamic range compression** — ``(·)^r`` with ``0 < r ≤ 1``
       compresses the output (``r = 1`` ≈ no compression).

    The combined formula is::

        PCEN(t, f) = ( S(t,f) / (M(t,f) + ε)^α + δ )^r  −  δ^r

    Parameters
    ----------
    S : np.ndarray, shape (n_mels, n_frames)
        **Linear-scale** Mel spectrogram (output of :func:`compute_mel_spectrogram`).

    Returns
    -------
    P : np.ndarray, shape (n_mels, n_frames), dtype float32
        PCEN-normalised spectrogram.
    """
    P = librosa.pcen(
        S * (2 ** 31),          # librosa.pcen expects integer-scaled input;
                                # rescale to ~int32 range for stable AGC
        sr=sr,
        hop_length=hop_length,
        gain=gain,
        bias=bias,
        power=power,
        time_constant=time_constant,
        eps=eps,
    )
    return P.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Combined extraction
# ──────────────────────────────────────────────────────────────

def extract_pcen_mel(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """One-call convenience: waveform → PCEN Mel-Spectrogram.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)

    Returns
    -------
    pcen : np.ndarray, shape (1, n_mels, n_frames), dtype float32
        Single-channel 2-D feature map ready for a CNN / Transformer.
        The leading dimension is the channel axis.
    """
    S = compute_mel_spectrogram(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    P = apply_pcen(S, sr=sr, hop_length=hop_length)

    # Add channel dimension → (1, n_mels, n_frames)
    return P[np.newaxis, :, :]


# ──────────────────────────────────────────────────────────────
# SpecAugment  (Spectrogram Augmentation)
# ──────────────────────────────────────────────────────────────

def spec_augment(
    spec: np.ndarray,
    *,
    time_mask_param: int = SPEC_TIME_MASK_PARAM,
    freq_mask_param: int = SPEC_FREQ_MASK_PARAM,
    num_time_masks: int = SPEC_NUM_TIME_MASKS,
    num_freq_masks: int = SPEC_NUM_FREQ_MASKS,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply SpecAugment (Park et al., 2019) to a spectrogram **in-place**.

    Two augmentation policies are applied independently:

    1. **Frequency masking** — zero-out up to ``freq_mask_param`` consecutive
       Mel bands.  Simulates narrowband signal dropout / microphone resonance.
    2. **Time masking** — zero-out up to ``time_mask_param`` consecutive time
       frames.  Simulates short intermittent occlusions.

    Parameters
    ----------
    spec : np.ndarray, shape (…, n_mels, n_frames)
        Spectrogram (PCEN or log-Mel).  The last two axes are treated as
        (frequency, time).  A leading channel dimension is allowed.
    time_mask_param, freq_mask_param : int
        Maximum mask width for each axis.
    num_time_masks, num_freq_masks : int
        Number of independent masks to apply on each axis.
    fill_value : float
        Value used for masked regions (0.0 = standard SpecAugment).

    Returns
    -------
    spec : np.ndarray
        Augmented spectrogram (same shape, **modified in-place**).

    Notes
    -----
    This function uses ``np.random`` so it respects ``np.random.seed()`` for
    reproducibility.  In training, call this *after* feature extraction and
    *before* converting to a ``torch.Tensor``.
    """
    spec = spec.copy()  # avoid mutating cached data

    n_mels = spec.shape[-2]
    n_frames = spec.shape[-1]

    rng = np.random.default_rng()

    # ── Frequency masks ──────────────────────────────────────
    for _ in range(num_freq_masks):
        f = rng.integers(0, min(freq_mask_param, n_mels) + 1)
        if f == 0:
            continue
        f0 = rng.integers(0, n_mels - f + 1)
        spec[..., f0 : f0 + f, :] = fill_value

    # ── Time masks ───────────────────────────────────────────
    for _ in range(num_time_masks):
        t = rng.integers(0, min(time_mask_param, n_frames) + 1)
        if t == 0:
            continue
        t0 = rng.integers(0, n_frames - t + 1)
        spec[..., :, t0 : t0 + t] = fill_value

    return spec


# ──────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.preprocess import preprocess_audio

    if len(sys.argv) < 2:
        print("Usage: python -m src.features <path_to_wav>")
        sys.exit(1)

    path = sys.argv[1]

    # Full pipeline: preprocess → extract features
    waveform = preprocess_audio(path)
    pcen = extract_pcen_mel(waveform)

    print(f"Input     : {path}")
    print(f"Waveform  : shape={waveform.shape}  dtype={waveform.dtype}")
    print(f"PCEN Mel  : shape={pcen.shape}  dtype={pcen.dtype}")
    print(f"            min={pcen.min():.4f}  max={pcen.max():.4f}")
    print(f"            mean={pcen.mean():.4f}  std={pcen.std():.4f}")

    # Demo SpecAugment
    pcen_aug = spec_augment(pcen)
    zeros_before = np.sum(pcen == 0.0)
    zeros_after  = np.sum(pcen_aug == 0.0)
    print(f"\nSpecAugment demo:")
    print(f"  zeros before = {zeros_before}")
    print(f"  zeros after  = {zeros_after}  (+{zeros_after - zeros_before} masked)")

"""
src/preprocess.py — Audio Preprocessing Pipeline
=================================================
Implements the full preprocessing chain for the Machine Fault Recognition
system.  Each stage is independently callable and tested.

Pipeline
--------
1. Load & resample  →  16 kHz mono
2. Silence removal  →  RMS-Energy + ZCR dual thresholding
3. Noise suppression →  MMSE-STSA (Ephraim & Malah, 1984) with
                        decision-directed a-priori SNR estimation
4. Z-Score normalisation  →  zero-mean, unit-variance waveform
5. Pad / truncate   →  fixed 5-second window (center-crop / symmetric pad)
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.special import i0e, i1e         # Exponentially scaled Bessel functions

# ──────────────────────────────────────────────────────────────
# Global constants (single source of truth for the whole repo)
# ──────────────────────────────────────────────────────────────
TARGET_SR: int = 16_000
TARGET_DURATION: float = 5.0                         # seconds
TARGET_LENGTH: int = int(TARGET_SR * TARGET_DURATION) # 80 000 samples

# STFT parameters (shared with features.py)
N_FFT: int = 1024
HOP_LENGTH: int = 256
WIN_LENGTH: int = 1024

# Silence-removal parameters
_SILENCE_FRAME_LEN: int = 2048
_SILENCE_HOP: int = 512
_RMS_THRESH: float = 0.01
_ZCR_THRESH: float = 0.15

# MMSE-STSA parameters
_NOISE_FRAMES: int = 10      # initial frames assumed to be noise-only
_ALPHA_DD: float = 0.98      # decision-directed smoothing factor
_GAIN_FLOOR: float = 0.1     # minimum spectral gain (prevents musical noise)

# ──────────────────────────────────────────────────────────────
# 1.  Load audio
# ──────────────────────────────────────────────────────────────

def load_audio(filepath: str, sr: int = TARGET_SR) -> np.ndarray:
    """Load a WAV file, convert to mono and resample to *sr* Hz.

    Returns
    -------
    y : np.ndarray, shape (n_samples,)
        Mono waveform as float32.
    """
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    # Clean any NaNs or Infs early (factory sensors sometimes glitch)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 2.  Silence removal (RMS + ZCR dual gate)
# ──────────────────────────────────────────────────────────────

def remove_silence(
    y: np.ndarray,
    sr: int = TARGET_SR,
    rms_thresh: float = _RMS_THRESH,
    zcr_thresh: float = _ZCR_THRESH,
    frame_length: int = _SILENCE_FRAME_LEN,
    hop_length: int = _SILENCE_HOP,
) -> np.ndarray:
    """Trim leading / trailing silence using a dual RMS-energy + ZCR gate.

    A frame is considered *active* when its RMS exceeds ``rms_thresh``
    **and** its ZCR is below ``zcr_thresh`` (high ZCR + low energy ≈ noise /
    silence, low ZCR + high energy ≈ machine signal).

    Parameters
    ----------
    y : np.ndarray
        Input waveform.
    sr : int
        Sample rate (unused but kept for API symmetry).
    rms_thresh, zcr_thresh : float
        Activation thresholds.
    frame_length, hop_length : int
        Analysis-window parameters for the frame-level features.

    Returns
    -------
    y_trimmed : np.ndarray
        Trimmed waveform (never empty — returns original if all-silent).
    """
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]

    active = (rms > rms_thresh) & (zcr < zcr_thresh)

    if not np.any(active):
        return y  # safety: never return an empty signal

    idx = np.where(active)[0]
    start = idx[0] * hop_length
    end = min((idx[-1] + 1) * hop_length + frame_length, len(y))
    return y[start:end]


# ──────────────────────────────────────────────────────────────
# 3.  MMSE-STSA noise suppression
# ──────────────────────────────────────────────────────────────

def _estimate_noise_psd(
    Y_power: np.ndarray,
    n_noise_frames: int = _NOISE_FRAMES,
) -> np.ndarray:
    """Return noise PSD estimated from the first *n_noise_frames* of the STFT.

    Assumption: the first ~60 ms of a factory recording is mostly ambient
    machine hum (no transient fault event yet).

    Parameters
    ----------
    Y_power : np.ndarray, shape (n_freq, n_frames)
        Squared magnitude of the noisy STFT.

    Returns
    -------
    noise_psd : np.ndarray, shape (n_freq, 1)
    """
    n = min(n_noise_frames, Y_power.shape[1])
    return np.mean(Y_power[:, :n], axis=1, keepdims=True).clip(min=1e-10)


def mmse_stsa(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    noise_frames: int = _NOISE_FRAMES,
    alpha: float = _ALPHA_DD,
    gain_floor: float = _GAIN_FLOOR,
) -> np.ndarray:
    """Apply the Ephraim–Malah MMSE-STSA estimator to suppress stationary noise.

    This implements the classic MMSE Short-Time Spectral Amplitude estimator
    with the **decision-directed** approach for a-priori SNR estimation
    (Ephraim & Malah, IEEE TASSP 1984).

    Gain function
    ~~~~~~~~~~~~~
    For each TF-bin the clean spectral amplitude is estimated as::

        Â(k,t) = G(v) · |Y(k,t)|

    where the MMSE-optimal gain is

        G(v) = (√π / 2) · √v / γ · exp(−v/2) · [(1+v)·I₀(v/2) + v·I₁(v/2)]

    ``v = ξ·γ / (1+ξ)``  with a-posteriori SNR ``γ`` and a-priori SNR ``ξ``
    computed via the decision-directed method.

    Parameters
    ----------
    y : np.ndarray
        Input noisy waveform.
    sr, n_fft, hop_length, win_length : int
        STFT parameters.
    noise_frames : int
        Number of leading frames used to estimate the noise floor.
    alpha : float  ∈ (0, 1)
        Smoothing coefficient for the decision-directed SNR update.
        Higher → more temporal smoothing (less musical noise, slower adaptation).
    gain_floor : float  ∈ (0, 1)
        Minimum spectral gain.  Prevents aggressive nulling that causes
        "musical noise" artefacts.

    Returns
    -------
    x_hat : np.ndarray
        Enhanced (denoised) waveform, same length as *y*.
    """
    # ---------- forward STFT ----------
    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)
    Y_power = Y_mag ** 2

    n_freq, n_frames = Y.shape

    # ---------- noise PSD ----------
    lambda_n = np.broadcast_to(
        _estimate_noise_psd(Y_power, noise_frames), (n_freq, n_frames)
    ).copy()                        # writable copy

    # ---------- frame-wise MMSE gain ----------
    X_mag = np.empty_like(Y_mag)
    xi_prev = np.ones(n_freq, dtype=np.float64)

    SQRT_PI_2 = np.sqrt(np.pi) / 2.0

    for t in range(n_frames):
        # a-posteriori SNR
        gamma = (Y_power[:, t] / (lambda_n[:, t] + 1e-10)).clip(min=1e-5)

        # a-priori SNR (decision-directed)
        if t == 0:
            xi = np.maximum(gamma - 1.0, 1e-5)
        else:
            xi_ml = np.maximum(gamma - 1.0, 0.0)
            xi = alpha * (X_mag[:, t - 1] ** 2 / (lambda_n[:, t] + 1e-10)) \
                 + (1.0 - alpha) * xi_ml
            xi = np.maximum(xi, 1e-5)

        # v parameter
        v = (xi / (1.0 + xi)) * gamma
        v = np.minimum(v, 100.0)    # Cap v for numerical safety
        v_half = v / 2.0

        # MMSE-STSA gain G(v) using exponentially scaled Bessel functions for stability.
        # Original: G = (sqrt(pi)/2) * (sqrt(v)/gamma) * exp(-v/2) * [(1+v)I0(v/2) + vI1(v/2)]
        # Using i0e(x) = i0(x)*exp(-x), the exp terms cancel out.
        G = SQRT_PI_2 * np.sqrt(v) / (gamma + 1e-10) \
            * ((1.0 + v) * i0e(v_half) + v * i1e(v_half))

        G = G.clip(min=gain_floor, max=1.0)

        X_mag[:, t] = G * Y_mag[:, t]
        xi_prev = xi

    # ---------- inverse STFT ----------
    X_hat = X_mag * np.exp(1j * Y_phase)
    x_hat = librosa.istft(X_hat, hop_length=hop_length, win_length=win_length, length=len(y))

    return x_hat.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 4.  Z-Score normalisation
# ──────────────────────────────────────────────────────────────

def normalize_zscore(y: np.ndarray) -> np.ndarray:
    """Standardise waveform to zero mean and unit variance.

    If the signal is near-silent (std ≈ 0) the mean is subtracted but no
    division is performed to avoid amplifying quantisation noise.
    """
    mean = np.mean(y)
    std = np.std(y)
    if std < 1e-8:
        return (y - mean).astype(np.float32)
    return ((y - mean) / std).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 5.  Pad / truncate to fixed length
# ──────────────────────────────────────────────────────────────

def pad_or_truncate(y: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """Force waveform to exactly *target_length* samples.

    * **Too long** → centre-crop (keeps the most representative middle portion).
    * **Too short** → symmetric zero-pad.
    """
    n = len(y)
    if n > target_length:
        start = (n - target_length) // 2
        return y[start : start + target_length]
    elif n < target_length:
        pad_total = target_length - n
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        return np.pad(y, (pad_l, pad_r), mode="constant", constant_values=0.0)
    return y


# ──────────────────────────────────────────────────────────────
# Full pipeline (convenience entry-point)
# ──────────────────────────────────────────────────────────────

def preprocess_audio(
    filepath: str,
    sr: int = TARGET_SR,
    *,
    silence_removal: bool = True,
    noise_suppression: bool = True,
    normalise: bool = True,
    target_length: int = TARGET_LENGTH,
) -> np.ndarray:
    """End-to-end preprocessing: load → clean → normalise → pad.

    Parameters
    ----------
    filepath : str
        Path to the ``.wav`` file.
    sr : int
        Target sample rate.
    silence_removal, noise_suppression, normalise : bool
        Toggle individual stages (useful for ablation / faster inference).
    target_length : int
        Final waveform length in samples.

    Returns
    -------
    y : np.ndarray, shape (target_length,), dtype float32
        Cleaned, normalised, fixed-length waveform ready for feature extraction.
    """
    y = load_audio(filepath, sr=sr)

    if silence_removal:
        y = remove_silence(y, sr=sr)

    if noise_suppression:
        y = mmse_stsa(y, sr=sr)

    if normalise:
        y = normalize_zscore(y)

    y = pad_or_truncate(y, target_length=target_length)

    return y


def preprocess_waveform(
    y: np.ndarray,
    sr: int = TARGET_SR,
    *,
    silence_removal: bool = True,
    noise_suppression: bool = True,
    normalise: bool = True,
    target_length: int = TARGET_LENGTH,
) -> np.ndarray:
    """Same as :func:`preprocess_audio` but accepts a raw waveform instead of
    a file path.  Used when the audio has already been loaded (e.g. during
    inference where the timer must start *after* the I/O read).
    """
    if silence_removal:
        y = remove_silence(y, sr=sr)

    if noise_suppression:
        y = mmse_stsa(y, sr=sr)

    if normalise:
        y = normalize_zscore(y)

    y = pad_or_truncate(y, target_length=target_length)

    return y


# ──────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocess <path_to_wav>")
        sys.exit(1)

    path = sys.argv[1]
    waveform = preprocess_audio(path)
    print(f"Input  : {path}")
    print(f"Output : shape={waveform.shape}  dtype={waveform.dtype}")
    print(f"         mean={waveform.mean():.6f}  std={waveform.std():.6f}")
    print(f"         min={waveform.min():.4f}  max={waveform.max():.4f}")

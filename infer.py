"""
infer.py — Strict Graded Inference Script
==========================================
This script is the entry-point used for evaluation.  It produces exactly
two output files (``results.txt`` and ``time.txt``) as specified by the
grading rubric.

Timing contract
---------------
*  The **wall-clock timer** starts **after** the audio file is read from
   disk (I/O is excluded).
*  The timer stops **after** the ONNX model outputs the integer prediction.
*  Timing covers:  preprocessing (silence removal → MMSE-STSA →
   normalisation → pad/truncate)  +  feature extraction (PCEN Mel)  +
   ONNX Runtime inference  +  argmax.

Performance notes
-----------------
*  **No ``torch`` import** — avoids the ~3 s cold-start penalty.
*  **``onnxruntime``** with full graph optimisation and tuned thread count.
*  **No ``print()``** inside the inference loop — terminal I/O is measurably
   slow and would pollute the recorded time.

Usage
-----
::

    python infer.py                      # defaults: data/ → models/model.onnx
    python infer.py data/                # explicit data directory
    python infer.py data/ models/model.onnx  # explicit model path
"""

from __future__ import annotations

import os
import re
import sys
import time
import pathlib

import numpy as np
import onnxruntime as ort

from src.preprocess import preprocess_waveform, load_audio, TARGET_SR
from src.features import extract_pcen_mel


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "data/"
DEFAULT_MODEL_PATH = "models/model.onnx"


def _create_ort_session(model_path: str) -> ort.InferenceSession:
    """Create an optimised ONNX Runtime session for CPU inference.

    Graph-level optimisations and thread tuning are applied to squeeze
    every last millisecond out of the forward pass.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3          # suppress ORT info/warning chatter

    providers = ["CPUExecutionProvider"]
    # Prefer GPU if available (CUDA or DirectML)
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    elif "DmlExecutionProvider" in available:
        providers.insert(0, "DmlExecutionProvider")

    session = ort.InferenceSession(model_path, opts, providers=providers)
    return session


def _discover_wav_files(data_dir: str) -> list[pathlib.Path]:
    """Find all .wav files in *data_dir* and sort numerically.

    Handles filenames like ``1.wav``, ``2.wav``, …, ``100.wav``.
    Falls back to lexicographic sort if names are non-numeric.
    """
    root = pathlib.Path(data_dir)
    wavs = list(root.glob("*.wav"))

    def _numeric_key(p: pathlib.Path) -> int:
        digits = re.sub(r"\D", "", p.stem)
        return int(digits) if digits else 0

    wavs.sort(key=_numeric_key)
    return wavs


# ──────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────

def run_inference(
    data_dir: str = DEFAULT_DATA_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    results_path: str = "results.txt",
    time_path: str = "time.txt",
) -> None:
    """Run the full inference pipeline and write output files.

    Parameters
    ----------
    data_dir : str
        Directory containing ``1.wav, 2.wav, …, N.wav``.
    model_path : str
        Path to the ONNX model file.
    results_path : str
        Output file for predicted class integers (one per line).
    time_path : str
        Output file for per-sample inference times in seconds,
        rounded to 3 decimal places (one per line).
    """
    # ── Discover test files ──────────────────────────────────
    wav_files = _discover_wav_files(data_dir)
    n_files = len(wav_files)

    # ── Load ONNX model (once, outside the loop) ────────────
    session = _create_ort_session(model_path)
    input_name = session.get_inputs()[0].name      # "spectrogram"

    # ── Pre-allocate output buffers ──────────────────────────
    predictions: list[int] = []
    timings: list[float] = []

    # ── Inference loop ───────────────────────────────────────
    # ⚠  NO print() calls inside this loop — terminal I/O is slow
    for wav_path in wav_files:
        # ── 1. Read audio from disk (OUTSIDE the timer) ─────
        raw_waveform = load_audio(str(wav_path), sr=TARGET_SR)

        # ── 2. START timer (after I/O) ──────────────────────
        t_start = time.perf_counter()

        # ── 3. Preprocess: silence → denoise → normalise → pad
        clean = preprocess_waveform(raw_waveform)

        # ── 4. Feature extraction: PCEN Mel spectrogram
        pcen = extract_pcen_mel(clean)     # (1, 128, 313) float32

        # ── 5. Add batch dimension for ONNX
        pcen_batch = pcen[np.newaxis, :]   # (1, 1, 128, 313)

        # ── 6. ONNX Runtime forward pass
        logits = session.run(None, {input_name: pcen_batch})[0]  # (1, 6)

        # ── 7. Prediction (argmax)
        pred = int(np.argmax(logits, axis=1)[0])

        # ── 8. STOP timer ──────────────────────────────────
        t_end = time.perf_counter()

        predictions.append(pred)
        timings.append(round(t_end - t_start, 3))

    # ── Write output files ───────────────────────────────────
    with open(results_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    with open(time_path, "w") as f:
        for t in timings:
            f.write(f"{t:.3f}\n")


# ──────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse positional arguments (no argparse to keep import overhead minimal)
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    if not os.path.isfile(model_path):
        print(f"ERROR: ONNX model not found at '{model_path}'", file=sys.stderr)
        print(f"  Run training first:  python -m src.train --data data/", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found at '{data_dir}'", file=sys.stderr)
        sys.exit(1)

    # Run inference
    run_inference(data_dir=data_dir, model_path=model_path)

    # Post-loop summary (safe to print here — outside timed region)
    with open("results.txt") as f:
        results = f.read().strip().split("\n")
    with open("time.txt") as f:
        times = [float(x) for x in f.read().strip().split("\n")]

    print(f"Inference complete: {len(results)} files processed")
    print(f"  Total time : {sum(times):.3f} s")
    print(f"  Avg time   : {sum(times)/len(times):.3f} s/file")
    print(f"  Output     : results.txt, time.txt")

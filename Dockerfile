# ============================================================
#  Dockerfile — Machine Fault Recognition System
# ============================================================
#  Reproduces the exact runtime environment for inference.
#
#  Build:   docker build -t machine_listener .
#  Run:     docker run --rm -v /path/to/test/data:/app/data machine_listener
#  Custom:  docker run --rm -v /path/to/data:/app/data machine_listener python infer.py data/
# ============================================================

# ── Stage 1: Base image ─────────────────────────────────────
FROM python:3.12-slim AS base

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# System dependencies for audio processing
#   libsndfile1  — backend for soundfile / librosa
#   ffmpeg       — fallback audio decoder (various codecs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python dependencies ────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application code ───────────────────────────────
COPY src/ src/
COPY infer.py .
COPY models/ models/

# ── Default entrypoint ──────────────────────────────────────
#  Expects test .wav files to be mounted at /app/data/
#  Produces results.txt and time.txt in /app/
CMD ["python", "infer.py", "data/"]

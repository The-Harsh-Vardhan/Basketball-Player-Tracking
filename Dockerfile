# ============================================================
# Basketball Player Tracking — YOLO11 + ByteTrack
# Multi-stage Docker build for GPU-accelerated inference
# ============================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================
# Dependencies layer (cached unless requirements change)
# ============================================================
FROM base AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Application layer
# ============================================================
FROM deps AS app

WORKDIR /app

# Copy project source
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/

# Create directories for data and outputs
RUN mkdir -p data outputs

# Default environment variables
ENV WANDB_MODE=offline
ENV YOLO_VERBOSE=false

# Expose volume mount points
VOLUME ["/app/data", "/app/outputs", "/app/runs"]

# Default entrypoint: run inference
ENTRYPOINT ["python", "scripts/run_inference.py"]

# Default command (can be overridden)
CMD ["--help"]

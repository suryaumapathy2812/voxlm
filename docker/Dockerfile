# =============================================================================
# VoxLM Training Docker Image
# =============================================================================
# Multi-stage build for efficient training container.
#
# Build:
#   docker build -t voxlm:latest -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
#       voxlm:latest configs/voxlm-2b.yaml
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# -----------------------------------------------------------------------------
# Stage 2: Dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install -r requirements.txt

# Install additional training dependencies
RUN pip install \
    pyyaml \
    openai-whisper \
    soundfile

# -----------------------------------------------------------------------------
# Stage 3: Final image
# -----------------------------------------------------------------------------
FROM dependencies AS final

WORKDIR /app

# Copy project files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY docker/entrypoint.sh ./entrypoint.sh

# Make scripts executable
RUN chmod +x ./scripts/*.sh ./entrypoint.sh 2>/dev/null || true

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/checkpoints

# Set environment variables
ENV VOXLM_DATA_DIR=/app/data
ENV VOXLM_OUTPUT_DIR=/app/output
ENV VOXLM_CHECKPOINT_DIR=/app/checkpoints

# Default config
ENV VOXLM_CONFIG=configs/voxlm-2b.yaml

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["configs/voxlm-2b.yaml"]

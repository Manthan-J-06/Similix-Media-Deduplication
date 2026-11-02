FROM python:3.11-slim

# System environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    SIMILIX_DATA_DIR=/data \
    ARGUS_DATA_DIR=/data \
    SIMILIX_TRASH_DIR=/data/trash \
    ARGUS_TRASH_DIR=/data/trash \
    SIMILIX_LIBRARY_DIR=/data/library \
    ARGUS_LIBRARY_DIR=/data/library \
    SIMILIX_USERS_DIR=/data/users \
    ARGUS_USERS_DIR=/data/users \
    SIMILIX_ACCOUNTS_DB=/data/auth.db \
    ARGUS_ACCOUNTS_DB=/data/auth.db \
    TMPDIR=/tmp \
    HF_HOME=/home/similix/.cache/huggingface \
    TORCH_HOME=/home/similix/.cache/torch

# Install lightweight runtime dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and application directory with strict permissions
RUN useradd -m -u 1000 -s /bin/bash similix && \
    mkdir -p /data/users /data/trash /data/library /home/similix/.similix /home/similix/.argus /home/similix/.cache/huggingface /home/similix/.cache/torch /app && \
    ln -s /home/similix /home/argus && \
    chown -R similix:similix /data /home/similix /app

WORKDIR /app

# Upgrade pip and install CPU-only PyTorch wheels to avoid multi-GB CUDA overhead
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1 \
    torchvision==0.29.1

# Install pinned dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files and install package in editable mode
COPY . .
RUN pip install --no-cache-dir -e . && \
    chown -R similix:similix /app

# Pre-download open_clip model weights & tokenizer at build time for instant, offline startup
USER similix
RUN python3 -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); open_clip.get_tokenizer('ViT-B-32')"

# Expose web port (default 8000, compatible with Hugging Face Spaces $PORT 7860)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request, os; port=os.environ.get('PORT', 8000); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz')"

# Production execution: 1 web worker behind reverse proxy with forwarded headers
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]

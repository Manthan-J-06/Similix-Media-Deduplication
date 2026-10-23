FROM python:3.11-slim

# System environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ARGUS_USERS_DIR=/data/users \
    ARGUS_AUTH_DB=/data/auth.db \
    HF_HOME=/home/argus/.cache/huggingface \
    TORCH_HOME=/home/argus/.cache/torch

# Install lightweight runtime dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and application directory with strict permissions
RUN useradd -m -u 1000 -s /bin/bash argus && \
    mkdir -p /data/users /home/argus/.cache/huggingface /home/argus/.cache/torch /app && \
    chown -R argus:argus /data /home/argus /app

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
    chown -R argus:argus /app

# Pre-download open_clip model weights & tokenizer at build time for instant, offline startup
USER argus
RUN python3 -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); open_clip.get_tokenizer('ViT-B-32')"

# Expose web port (default 8000, compatible with Hugging Face Spaces $PORT 7860)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request, os; port=os.environ.get('PORT', 8000); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz')"

# Production execution: 1 web worker behind reverse proxy with forwarded headers
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]

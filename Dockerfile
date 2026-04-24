# LTX-Video 2.3 RunPod Serverless Worker
# Use plain NVIDIA CUDA base so we control PyTorch version
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git ffmpeg ca-certificates build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.5.1 with CUDA 12.4 (accepts string-form type annotations in torch.library)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Base Python deps
# Pin transformers to a version compatible with diffusers 0.30.3 (needs FLAX_WEIGHTS_NAME symbol)
RUN pip install runpod==1.7.9 boto3 requests \
    "huggingface_hub[hf_transfer]>=0.24.0,<0.27.0" hf_transfer \
    "transformers>=4.41.0,<4.46.0" "accelerate>=1.0.0,<1.4.0" \
    safetensors sentencepiece protobuf einops "imageio[ffmpeg]"

# Pin diffusers 0.30.3 (pre flash_attn_3 auto-registration)
RUN pip install "diffusers==0.30.3"

# LTX-Video (without deps to avoid pulling bad torch)
RUN pip install --no-deps git+https://github.com/Lightricks/LTX-Video.git

# Belt and braces: make sure flash_attn is NOT present and cannot re-install
RUN pip uninstall -y flash-attn flash_attn flash-attn-3 flash_attn_3 || true
RUN pip list | grep -i flash || echo "OK: no flash_attn packages installed"

WORKDIR /app
COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]

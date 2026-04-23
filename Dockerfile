# LTX-Video 2.3 RunPod Serverless Worker
# Requirements: Python >= 3.12, CUDA > 12.7, PyTorch ~2.7

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python/pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python3.12 -m pip install --upgrade pip

# PyTorch 2.7 with CUDA 12.8
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Core Python dependencies
RUN pip install \
    runpod==1.7.9 \
    boto3 \
    requests \
    huggingface_hub[hf_transfer] \
    transformers>=4.49.0 \
    accelerate \
    diffusers \
    imageio[ffmpeg] \
    einops \
    safetensors \
    sentencepiece \
    protobuf

# Install ltx-video from source (LTX-Video 2.3 pipeline)
RUN pip install git+https://github.com/Lightricks/LTX-Video.git

WORKDIR /app

COPY handler.py .
COPY download_models.py .

CMD ["python", "-u", "handler.py"]
# LTX-Video 2.3 RunPod Serverless Worker
# Requirements: Python >= 3.12, CUDA > 12.7, PyTorch ~2.7

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git wget curl ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# PyTorch 2.7 with CUDA 12.8
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Core Python deps
RUN pip install \
    runpod==1.7.9 \
    boto3 \
    requests \
    huggingface_hub \
    transformers>=4.49.0 \
    accelerate \
    diffusers \
    imageio[ffmpeg] \
    einops \
    safetensors

# Install ltx-video from source
RUN pip install git+https://github.com/Lightricks/LTX-Video.git

WORKDIR /app

COPY handler.py .
COPY download_models.py .

CMD ["python", "-u", "handler.py"]
# ── Base image: CUDA 12.8 + cuDNN + Ubuntu 24.04 ────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
        python3.12-dev \
            python3-pip \
                python3.12-venv \
                    git \
                        wget \
                            curl \
                                ffmpeg \
                                    libgl1 \
                                        libglib2.0-0 \
                                            && rm -rf /var/lib/apt/lists/*

                                            # Make python3.12 the default python/pip
                                            RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
                                                update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
                                                    python -m pip install --upgrade pip

                                                    # ── PyTorch 2.7 + CUDA 12.8 ─────────────────────────────────────────────────
                                                    RUN pip install --no-cache-dir \
                                                        torch==2.7.0 torchvision torchaudio \
                                                            --index-url https://download.pytorch.org/whl/cu128

                                                            # ── HuggingFace ecosystem ────────────────────────────────────────────────────
                                                            RUN pip install --no-cache-dir \
                                                                "transformers>=4.51.0" \
                                                                    "diffusers>=0.32.0" \
                                                                        "accelerate>=1.4.0" \
                                                                            "huggingface_hub[hf_transfer]>=0.27.0" \
                                                                                hf_transfer \
                                                                                    safetensors

                                                                                    # ── LTX-Video core 

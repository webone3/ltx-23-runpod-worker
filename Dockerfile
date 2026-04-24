# LTX-2.3 (22B dev) RunPod Serverless Worker — full two-stage pipeline
# Uses CUDA 12.9 / PyTorch 2.8+ / xformers per Lightricks/LTX-2 requirements.
# Note: cu129 wheels start at torch 2.8.0 — LTX-2's pyproject says ~=2.7 but 2.8 is
# the earliest cu129 build available and is API-compatible.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Stage 1: bootstrap essentials (curl + software-properties-common) so we can add deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python 3.11 (via deadsnakes PPA) + system media/build deps used by ltx-pipelines
# OpenImageIO provides 'openimageio' python bindings (dep of ltx-pipelines).
# libav* are needed so PyAV builds/loads cleanly for MP4 encoding.
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        git ffmpeg build-essential pkg-config \
        libopenimageio-dev python3-openimageio \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.8.x with CUDA 12.9 (earliest cu129 wheel; satisfies ltx-core API needs)
RUN pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# xformers for torch 2.8 / cu129 (attention optimizer for LTX-2.3 22B on H100 NVL)
RUN pip install --index-url https://download.pytorch.org/whl/cu129 xformers

# RunPod worker + R2 upload + media deps
RUN pip install \
        runpod==1.7.9 boto3 requests \
        "huggingface_hub[hf_transfer]>=0.27.0" hf_transfer \
        av tqdm pillow imageio imageio-ffmpeg

# Install Lightricks LTX-2 monorepo packages (ltx-core + ltx-pipelines) from GitHub.
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && pip install /opt/LTX-2/packages/ltx-core \
    && pip install /opt/LTX-2/packages/ltx-pipelines

# Build-time sanity check that the API we target actually imports
RUN python -c "from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline; from ltx_core.components.guiders import MultiModalGuiderParams; from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number; from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP; from ltx_pipelines.utils.constants import LTX_2_3_PARAMS; from ltx_pipelines.utils.media_io import encode_video; print('[build-check] LTX-2 imports OK')"

# Defensive: ensure flash_attn is NOT installed (we use xformers exclusively)
RUN pip uninstall -y flash-attn flash_attn flash-attn-3 flash_attn_3 || true
RUN pip list | grep -i flash || echo "OK: no flash_attn packages installed"

WORKDIR /app
COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]

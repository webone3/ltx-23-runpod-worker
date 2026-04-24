# LTX-Video 2.3 RunPod Serverless Worker
FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Prevent diffusers from trying to use FlashAttention 3 (avoids torch.library schema error)
ENV DIFFUSERS_DISABLE_FLASH_ATTN_3=1
ENV TRANSFORMERS_USE_FLASH_ATTN_3=0

# Install base dependencies
RUN pip install --no-cache-dir runpod==1.7.9 boto3 requests "huggingface_hub[hf_transfer]>=0.27.0" hf_transfer "transformers>=4.49.0" "accelerate>=1.4.0" safetensors sentencepiece protobuf einops "imageio[ffmpeg]"

# Install diffusers version compatible with LTX pipeline
RUN pip install --no-cache-dir "diffusers==0.32.2"

# Install LTX-Video without its dependency constraints
RUN pip install --no-cache-dir --no-deps git+https://github.com/Lightricks/LTX-Video.git

# Explicitly remove flash_attn packages if present (they cause schema registration errors)
RUN pip uninstall -y flash-attn flash_attn flash-attn-3 flash_attn_3 || true

WORKDIR /app
COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]

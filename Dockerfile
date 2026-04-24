# LTX-Video 2.3 RunPod Serverless Worker
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install base dependencies
RUN pip install --no-cache-dir runpod==1.7.9 boto3 requests "huggingface_hub[hf_transfer]>=0.27.0" hf_transfer "transformers>=4.49.0" "accelerate>=1.4.0" safetensors sentencepiece protobuf einops "imageio[ffmpeg]"

# Pin diffusers to 0.30.3 - older version that doesn't try to register flash_attn_3
RUN pip install --no-cache-dir "diffusers==0.30.3"

# Install LTX-Video without dependency overrides
RUN pip install --no-cache-dir --no-deps git+https://github.com/Lightricks/LTX-Video.git

# Remove any flash_attn packages that may have been pulled in
RUN pip uninstall -y flash-attn flash_attn flash-attn-3 flash_attn_3 2>/dev/null || true

WORKDIR /app
COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]

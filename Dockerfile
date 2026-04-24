# LTX-Video 2.3 RunPod Serverless Worker
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install base dependencies
RUN pip install --no-cache-dir runpod==1.7.9 boto3 requests "huggingface_hub[hf_transfer]>=0.27.0" hf_transfer "transformers>=4.49.0" "accelerate>=1.4.0" safetensors sentencepiece protobuf einops "imageio[ffmpeg]"

# Install LTX-Video package (without its dependency overrides)
RUN pip install --no-cache-dir --no-deps git+https://github.com/Lightricks/LTX-Video.git

# Pin compatible versions AFTER LTX-Video install - force reinstall
RUN pip install --no-cache-dir --force-reinstall --no-deps "diffusers==0.31.0" "pydantic==1.10.13"

WORKDIR /app
COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]

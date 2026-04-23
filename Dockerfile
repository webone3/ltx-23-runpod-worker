# LTX-Video 2.3 RunPod Serverless Worker
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv python3.12-distutils git wget curl ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir runpod==1.7.9 boto3 requests "huggingface_hub[hf_transfer]>=0.27.0" hf_transfer "transformers>=4.49.0" "accelerate>=1.4.0" "diffusers>=0.32.0" safetensors sentencepiece protobuf einops "imageio[ffmpeg]"

RUN pip install --no-cache-dir git+https://github.com/Lightricks/LTX-Video.git

WORKDIR /app

COPY handler.py .
COPY download_models.py .

CMD ["python3", "-u", "handler.py"]
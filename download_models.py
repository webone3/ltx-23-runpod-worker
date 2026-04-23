#!/usr/bin/env python3
"""
download_models.py - First-boot model downloader for LTX-Video 2.3 worker.
Downloads all required models from HuggingFace to the RunPod network volume.
Run this once before starting the worker, or at container startup.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

HF_TOKEN = os.environ["HF_TOKEN"]

VOLUME = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
LTX_DIR   = VOLUME / "models" / "ltx-2.3"
GEMMA_DIR = VOLUME / "models" / "gemma"

LTX_REPO   = "Lightricks/LTX-2.3"
GEMMA_REPO = "google/gemma-3-12b-it"

LTX_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
]


def already_downloaded(path: Path, min_bytes: int = 1_000_000) -> bool:
    return path.exists() and path.stat().st_size > min_bytes


def download_ltx():
    LTX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[downloader] LTX-2.3 target dir: {LTX_DIR}")

    for filename in LTX_FILES:
        dest = LTX_DIR / filename
        if already_downloaded(dest):
            print(f"[downloader] Already present: {filename} ({dest.stat().st_size / 1e9:.1f} GB)")
            continue
        print(f"[downloader] Downloading {filename}...")
        hf_hub_download(
            repo_id=LTX_REPO,
            filename=filename,
            local_dir=str(LTX_DIR),
            token=HF_TOKEN,
        )
        print(f"[downloader] Done: {filename}")


def download_gemma():
    GEMMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[downloader] Gemma target dir: {GEMMA_DIR}")

    # Check if already downloaded by looking for config.json
    config_path = GEMMA_DIR / "config.json"
    if already_downloaded(config_path, min_bytes=100):
        print(f"[downloader] Gemma already present at {GEMMA_DIR}")
        return

    print("[downloader] Downloading google/gemma-3-12b-it (full snapshot)...")
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=str(GEMMA_DIR),
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print("[downloader] Gemma download complete.")


if __name__ == "__main__":
    print("[downloader] === Starting model download ===")
    print(f"[downloader] Volume path: {VOLUME}")

    download_ltx()
    download_gemma()

    print("[downloader] === All models ready ===")

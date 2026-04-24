#!/usr/bin/env python3
"""
download_models.py - First-boot model downloader for the LTX-2.3 (22B dev) worker.

Downloads, onto the RunPod network volume:
  * LTX-2.3 22B dev checkpoint
  * LTX-2.3 22B distilled LoRA (required by TI2VidTwoStagesPipeline stage 2)
  * LTX-2.3 spatial upscaler x2 v1.1 (required by stage 2)
  * Gemma-3-12B-it QAT q4_0 unquantized (text encoder used by LTX-2)

Also cleans up the earlier 'google/gemma-3-12b-it' download that an older version of
this worker placed at VOLUME/models/gemma/, because (a) LTX-2 requires a different
Gemma variant and (b) the volume is space-constrained.
"""
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN") or ""

VOLUME = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))

# LTX-2.3 model assets
LTX_DIR = VOLUME / "models" / "ltx-2.3"
LTX_REPO = "Lightricks/LTX-2.3"
LTX_FILES = [
    "ltx-2.3-22b-dev.safetensors",                     # ~44 GB, main 22B dev checkpoint
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",  # distilled LoRA for stage 2
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",     # spatial upsampler for stage 2
]

# Gemma text encoder used by LTX-2 (QAT q4_0 unquantized, per LTX-2 README)
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_DIR = VOLUME / "models" / "gemma-qat-unquantized"

# Old (incorrect) Gemma download to clean up, if present
OLD_GEMMA_DIR = VOLUME / "models" / "gemma"


def _already_downloaded(path: Path, min_bytes: int = 1_000_000) -> bool:
    return path.exists() and path.stat().st_size > min_bytes


def _cleanup_old_gemma() -> None:
    if OLD_GEMMA_DIR.exists():
        print(f"[download] Removing old Gemma directory: {OLD_GEMMA_DIR}", flush=True)
        shutil.rmtree(OLD_GEMMA_DIR, ignore_errors=True)


def download_ltx() -> None:
    LTX_DIR.mkdir(parents=True, exist_ok=True)
    for filename in LTX_FILES:
        target = LTX_DIR / filename
        if _already_downloaded(target):
            print(f"[download] Already present: {target}", flush=True)
            continue
        print(f"[download] Fetching {LTX_REPO}/{filename} -> {target}", flush=True)
        hf_hub_download(
            repo_id=LTX_REPO,
            filename=filename,
            local_dir=str(LTX_DIR),
            local_dir_use_symlinks=False,
            token=HF_TOKEN or None,
        )


def download_gemma() -> None:
    marker = GEMMA_DIR / "config.json"
    if marker.exists():
        print(f"[download] Gemma already present at {GEMMA_DIR}", flush=True)
        return
    GEMMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching snapshot of {GEMMA_REPO} -> {GEMMA_DIR}", flush=True)
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=str(GEMMA_DIR),
        local_dir_use_symlinks=False,
        token=HF_TOKEN or None,
    )


def main() -> None:
    print(f"[download] Volume root: {VOLUME}", flush=True)
    _cleanup_old_gemma()
    download_ltx()
    download_gemma()
    print("[download] All LTX-2.3 assets ready.", flush=True)


if __name__ == "__main__":
    main()

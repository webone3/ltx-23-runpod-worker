"""
handler.py — LTX-2.3 (22B dev) RunPod Serverless worker.

Uses the Lightricks LTX-2 'TI2VidTwoStagesPipeline' (two-stage text/image-to-video,
stage 1 @ half-res with CFG, stage 2 @ 2x spatial upsample refined with the distilled
LoRA) for production-quality output at the configured resolution.

Model assets are expected on the RunPod network volume at the paths produced by
download_models.py. If environment variables are set they override those defaults.
"""
import json
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import boto3
import requests
import runpod

# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------
VOLUME = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
LTX_DIR = VOLUME / "models" / "ltx-2.3"
GEMMA_DIR = VOLUME / "models" / "gemma-qat-unquantized"

LTX_CHECKPOINT = os.environ.get("LTX_CHECKPOINT") or str(LTX_DIR / "ltx-2.3-22b-dev.safetensors")
DISTILLED_LORA = os.environ.get("DISTILLED_LORA") or str(LTX_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
SPATIAL_UPSCALER = os.environ.get("SPATIAL_UPSCALER") or str(LTX_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_ROOT = os.environ.get("GEMMA_ROOT") or str(GEMMA_DIR)

R2_ENDPOINT = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE = os.environ["R2_PUBLIC_URL"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")

# ---------------------------------------------------------------------------
# First-boot model download (idempotent)
# ---------------------------------------------------------------------------
def _ensure_models() -> None:
    try:
        import download_models  # local module next to handler.py
        download_models.main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[worker] download_models failed: {e}", flush=True)
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Pipeline lifecycle (lazy singleton)
# ---------------------------------------------------------------------------
_pipeline = None
_pipeline_ctx = None


def _build_pipeline():
    import torch
    from ltx_core.loader import (
        LTXV_LORA_COMFY_RENAMING_MAP,
        LoraPathStrengthAndSDOps,
    )
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.types import OffloadMode

    print("[worker] Constructing TI2VidTwoStagesPipeline (LTX-2.3 22B dev)...", flush=True)

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            LTX_CHECKPOINT_LORA := DISTILLED_LORA,
            1.0,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    ]

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=LTX_CHECKPOINT,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=SPATIAL_UPSCALER,
        gemma_root=GEMMA_ROOT,
        loras=[],
        device=torch.device("cuda"),
        quantization=None,
        torch_compile=False,
        # SUBMODELS offload lets us fit the 22B dev model + 12B text encoder on 94GB H100 NVL
        # by moving submodels to CPU when not in use.
        offload_mode=OffloadMode.SUBMODELS,
    )
    print("[worker] Pipeline ready.", flush=True)
    return pipeline


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    _ensure_models()
    _pipeline = _build_pipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# R2 upload helpers
# ---------------------------------------------------------------------------
def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )


def upload_to_r2(local_path: str, key: str, content_type: str = "video/mp4") -> str:
    _s3_client().upload_file(
        local_path,
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"{R2_PUBLIC_BASE.rstrip('/')}/{key}"


def send_webhook(payload: dict) -> None:
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"[worker] webhook send failed: {e}", flush=True)


def download_file(url: str, suffix: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _run_generation(pipeline, inp: dict, job_id: str) -> dict:
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.constants import LTX_2_3_PARAMS
    from ltx_pipelines.utils.media_io import encode_video

    # ---- defaults sourced from LTX-2.3 reference params ----
    params = LTX_2_3_PARAMS

    prompt = inp.get("prompt", "")
    if not prompt:
        raise ValueError("Missing required input 'prompt'")
    negative_prompt = inp.get(
        "negative_prompt",
        "worst quality, inconsistent motion, blurry, jittery, distorted",
    )

    # Target output size — must be divisible by 32. Stage 1 runs at half these dims.
    height = int(inp.get("height", 1024))
    width = int(inp.get("width", 1536))
    num_frames = int(inp.get("num_frames", params.num_frames))
    frame_rate = float(inp.get("fps", params.frame_rate))
    num_inference_steps = int(inp.get("num_inference_steps", params.num_inference_steps))
    seed = int(inp.get("seed", params.seed))

    # Optional image conditioning (single image url)
    images = None
    temp_image_path = None
    if inp.get("image_url"):
        from PIL import Image

        temp_image_path = download_file(inp["image_url"], ".png")
        images = [Image.open(temp_image_path).convert("RGB")]

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    video_guider = MultiModalGuiderParams(
        cfg_scale=float(inp.get("cfg_scale", params.video_guider_params.cfg_scale)),
        stg_scale=float(inp.get("stg_scale", params.video_guider_params.stg_scale)),
        rescale_scale=float(inp.get("rescale_scale", params.video_guider_params.rescale_scale)),
        modality_scale=float(inp.get("a2v_scale", params.video_guider_params.modality_scale)),
        skip_step=int(inp.get("skip_step", params.video_guider_params.skip_step)),
        stg_blocks=list(inp.get("stg_blocks", params.video_guider_params.stg_blocks)),
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=float(inp.get("audio_cfg_scale", params.audio_guider_params.cfg_scale)),
        stg_scale=float(inp.get("audio_stg_scale", params.audio_guider_params.stg_scale)),
        rescale_scale=float(inp.get("audio_rescale_scale", params.audio_guider_params.rescale_scale)),
        modality_scale=float(inp.get("v2a_scale", params.audio_guider_params.modality_scale)),
        skip_step=int(inp.get("audio_skip_step", params.audio_guider_params.skip_step)),
        stg_blocks=list(inp.get("audio_stg_blocks", params.audio_guider_params.stg_blocks)),
    )

    print(
        f"[worker] Running pipeline: {width}x{height}, {num_frames}f @ {frame_rate}fps, "
        f"steps={num_inference_steps}, seed={seed}",
        flush=True,
    )
    t0 = time.time()
    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        images=images,
        tiling_config=tiling_config,
        max_batch_size=1,
    )
    gen_time = time.time() - t0
    print(f"[worker] Generation done in {gen_time:.1f}s", flush=True)

    # ---- encode to MP4 ----
    out_path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
    encode_video(
        video=video,
        fps=int(round(frame_rate)),
        audio=audio,
        output_path=out_path,
        video_chunks_number=video_chunks_number,
    )

    key = f"videos/{job_id}.mp4"
    url = upload_to_r2(out_path, key, "video/mp4")

    # ---- cleanup ----
    try:
        os.remove(out_path)
    except OSError:
        pass
    if temp_image_path:
        try:
            os.remove(temp_image_path)
        except OSError:
            pass
    torch.cuda.empty_cache()

    return {
        "job_id": job_id,
        "status": "success",
        "video_url": url,
        "generation_time_sec": round(gen_time, 2),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": frame_rate,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# RunPod handler entrypoint
# ---------------------------------------------------------------------------
def handler(event):
    job_id = (event or {}).get("id") or str(uuid.uuid4())
    inp = (event or {}).get("input") or {}
    mode = inp.get("mode", "text_to_video")
    print(f"[worker] job={job_id} mode={mode}", flush=True)

    try:
        pipeline = get_pipeline()
        if mode in ("text_to_video", "image_to_video", "ti2v"):
            result = _run_generation(pipeline, inp, job_id)
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Supported: text_to_video, image_to_video.",
            )
        send_webhook(result)
        return result
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[worker] ERROR: {tb}", flush=True)
        err = {
            "job_id": job_id,
            "status": "failed",
            "mode": mode,
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
        }
        send_webhook(err)
        return err


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

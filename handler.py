import os
import json
import time
import uuid
import boto3
import requests
import runpod
from pathlib import Path

LTX_CHECKPOINT   = os.environ["LTX_CHECKPOINT"]
DISTILLED_LORA   = os.environ["DISTILLED_LORA"]
SPATIAL_UPSCALER = os.environ["SPATIAL_UPSCALER"]
GEMMA_ROOT       = os.environ["GEMMA_ROOT"]

R2_ENDPOINT    = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY  = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY  = os.environ["R2_SECRET_KEY"]
R2_BUCKET      = os.environ["R2_BUCKET"]
R2_PUBLIC_BASE = os.environ["R2_PUBLIC_BASE"]

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    print("[worker] Loading LTX-Video 2.3 pipeline...")
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    import torch
    _pipeline = LTXVideoPipeline.from_pretrained(
        ltx_checkpoint=LTX_CHECKPOINT,
        distilled_lora_checkpoint=DISTILLED_LORA,
        spatial_upscaler_checkpoint=SPATIAL_UPSCALER,
        gemma_model_root=GEMMA_ROOT,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    _pipeline.enable_model_cpu_offload()
    print("[worker] Pipeline ready.")
    return _pipeline


def upload_to_r2(local_path, key):
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    s3.upload_file(local_path, R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"{R2_PUBLIC_BASE.rstrip('/')}/{key}"


def send_webhook(payload):
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[worker] Webhook error: {e}")


def handler(job):
    job_id = job.get("id", str(uuid.uuid4()))
    inp    = job.get("input", {})
    prompt     = inp.get("prompt", "")
    negative   = inp.get("negative_prompt", "")
    width      = int(inp.get("width",  768))
    height     = int(inp.get("height", 512))
    num_frames = int(inp.get("num_frames", 121))
    fps        = int(inp.get("fps", 24))
    guidance   = float(inp.get("guidance_scale", 3.5))
    steps      = int(inp.get("num_inference_steps", 40))
    seed       = inp.get("seed", None)
    user_id    = inp.get("user_id", "unknown")
    project_id = inp.get("project_id", "unknown")
    import torch
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    t0 = time.time()
    try:
        pipe = get_pipeline()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=fps,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        frames = result.frames[0]
        import imageio
        out_path = f"/tmp/{job_id}.mp4"
        imageio.mimwrite(out_path, frames, fps=fps, quality=8)
        elapsed = round(time.time() - t0, 1)
        key = f"videos/{user_id}/{project_id}/{job_id}.mp4"
        url = upload_to_r2(out_path, key)
        os.remove(out_path)
        payload = {
            "job_id": job_id, "status": "completed",
            "video_url": url, "duration_s": elapsed,
            "user_id": user_id, "project_id": project_id,
        }
        send_webhook(payload)
        return payload
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[worker] ERROR: {tb}")
        payload = {
            "job_id": job_id, "status": "failed",
            "error": str(exc), "user_id": user_id, "project_id": project_id,
        }
        send_webhook(payload)
        return payload


runpod.serverless.start({"handler": handler})

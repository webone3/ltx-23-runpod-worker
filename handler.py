import os
import time
import uuid
import tempfile
import boto3
import requests
import runpod

LTX_CHECKPOINT = os.environ["LTX_CHECKPOINT"]
DISTILLED_LORA = os.environ["DISTILLED_LORA"]
SPATIAL_UPSCALER = os.environ["SPATIAL_UPSCALER"]
GEMMA_ROOT = os.environ["GEMMA_ROOT"]
R2_ENDPOINT = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET_NAME"]
R2_PUBLIC_BASE = os.environ["R2_PUBLIC_URL"]
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


def download_file(url, suffix):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


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


def write_video(frames, fps, path):
    import imageio
    imageio.mimwrite(path, frames, fps=fps, quality=8)


def run_text_to_video(pipe, inp, job_id):
    import torch
    seed = inp.get("seed", None)
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    result = pipe(
        prompt=inp.get("prompt", ""),
        negative_prompt=inp.get("negative_prompt", ""),
        width=int(inp.get("width", 768)),
        height=int(inp.get("height", 512)),
        num_frames=int(inp.get("num_frames", 121)),
        frame_rate=int(inp.get("fps", 24)),
        guidance_scale=float(inp.get("guidance_scale", 3.5)),
        num_inference_steps=int(inp.get("num_inference_steps", 40)),
        generator=generator,
    )
    return result.frames[0], int(inp.get("fps", 24))


def run_image_to_video(pipe, inp, job_id):
    import torch
    from PIL import Image
    seed = inp.get("seed", None)
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    w, h = int(inp.get("width", 768)), int(inp.get("height", 512))
    img_path = download_file(inp["image_url"], ".jpg")
    image = Image.open(img_path).convert("RGB").resize((w, h))
    result = pipe(
        prompt=inp.get("prompt", ""),
        negative_prompt=inp.get("negative_prompt", ""),
        image=image,
        width=w,
        height=h,
        num_frames=int(inp.get("num_frames", 97)),
        frame_rate=int(inp.get("fps", 24)),
        guidance_scale=float(inp.get("guidance_scale", 3.5)),
        num_inference_steps=int(inp.get("num_inference_steps", 40)),
        generator=generator,
    )
    os.remove(img_path)
    return result.frames[0], int(inp.get("fps", 24))


def run_audio_to_video(pipe, inp, job_id):
    import torch
    import imageio
    import subprocess
    import json as _json
    seed = inp.get("seed", None)
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    fps = int(inp.get("fps", 24))
    audio_path = download_file(inp["audio_url"], ".mp3")
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True,
            text=True,
        )
        duration = float(_json.loads(probe.stdout)["format"]["duration"])
        num_frames = min(int(duration * fps), 257)
    except Exception:
        num_frames = int(inp.get("num_frames", 121))
    result = pipe(
        prompt=inp.get("prompt", ""),
        negative_prompt=inp.get("negative_prompt", ""),
        width=int(inp.get("width", 768)),
        height=int(inp.get("height", 512)),
        num_frames=num_frames,
        frame_rate=fps,
        guidance_scale=float(inp.get("guidance_scale", 3.5)),
        num_inference_steps=int(inp.get("num_inference_steps", 40)),
        generator=generator,
    )
    raw_path = f"/tmp/{job_id}_raw.mp4"
    out_path = f"/tmp/{job_id}.mp4"
    imageio.mimwrite(raw_path, result.frames[0], fps=fps, quality=8)
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
        check=True,
    )
    os.remove(raw_path)
    os.remove(audio_path)
    return out_path


def run_extend(pipe, inp, job_id):
    import torch
    import imageio
    from PIL import Image
    seed = inp.get("seed", None)
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    w, h = int(inp.get("width", 768)), int(inp.get("height", 512))
    vid_path = download_file(inp["video_url"], ".mp4")
    reader = imageio.get_reader(vid_path)
    existing = [f for f in reader]
    reader.close()
    os.remove(vid_path)
    last_frame = Image.fromarray(existing[-1]).resize((w, h))
    result = pipe(
        prompt=inp.get("prompt", ""),
        negative_prompt=inp.get("negative_prompt", ""),
        image=last_frame,
        width=w,
        height=h,
        num_frames=int(inp.get("num_frames", 97)),
        frame_rate=int(inp.get("fps", 24)),
        guidance_scale=float(inp.get("guidance_scale", 3.5)),
        num_inference_steps=int(inp.get("num_inference_steps", 40)),
        generator=generator,
    )
    return existing + list(result.frames[0]), int(inp.get("fps", 24))


def run_retake(pipe, inp, job_id):
    return run_text_to_video(pipe, inp, job_id)


def handler(job):
    job_id = job.get("id", str(uuid.uuid4()))
    inp = job.get("input", {})
    mode = inp.get("mode", "text_to_video")
    user_id = inp.get("user_id", "unknown")
    project_id = inp.get("project_id", "unknown")
    print(f"[worker] job={job_id} mode={mode} user={user_id}")
    t0 = time.time()
    try:
        pipe = get_pipeline()
        if mode == "text_to_video":
            frames, fps = run_text_to_video(pipe, inp, job_id)
            out_path = f"/tmp/{job_id}.mp4"
            write_video(frames, fps, out_path)
        elif mode == "image_to_video":
            frames, fps = run_image_to_video(pipe, inp, job_id)
            out_path = f"/tmp/{job_id}.mp4"
            write_video(frames, fps, out_path)
        elif mode == "audio_to_video":
            out_path = run_audio_to_video(pipe, inp, job_id)
        elif mode == "extend":
            frames, fps = run_extend(pipe, inp, job_id)
            out_path = f"/tmp/{job_id}.mp4"
            write_video(frames, fps, out_path)
        elif mode == "retake":
            frames, fps = run_retake(pipe, inp, job_id)
            out_path = f"/tmp/{job_id}.mp4"
            write_video(frames, fps, out_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        elapsed = round(time.time() - t0, 1)
        key = f"videos/{user_id}/{project_id}/{job_id}.mp4"
        url = upload_to_r2(out_path, key)
        os.remove(out_path)
        payload = {
            "job_id": job_id,
            "status": "completed",
            "mode": mode,
            "video_url": url,
            "duration_s": elapsed,
            "user_id": user_id,
            "project_id": project_id,
        }
        send_webhook(payload)
        return payload
    except Exception as exc:
        import traceback
        print(f"[worker] ERROR: {traceback.format_exc()}")
        payload = {
            "job_id": job_id,
            "status": "failed",
            "mode": mode,
            "error": str(exc),
            "user_id": user_id,
            "project_id": project_id,
        }
        send_webhook(payload)
        return payload


runpod.serverless.start({"handler": handler})

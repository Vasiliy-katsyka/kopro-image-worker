"""Kopro video worker: LTX-2.3 distilled (ComfyUI). Two stages like the official ComfyUI template:
half resolution 8 steps -> x2 latent upscale -> 3 refine steps, video and audio generated together.
Input:  {prompt, width?, height?, seconds?, seed?, image?: base64 first frame} -> {video: base64 MP4, seconds, width, height}
        {warmup: true} -> loads every model with a tiny render."""
import base64, glob, io, os, subprocess, sys, time, uuid
import requests, runpod
from PIL import Image, ImageOps

COMFY = "http://127.0.0.1:8188"
OUT_DIR = "/comfyui/output"
IN_DIR = "/comfyui/input"
CK = "ltx-2.3-22b-distilled-fp8.safetensors"
TE = "gemma_3_12B_it_fp8_scaled.safetensors"
UPS = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
FPS = 24
NEG = "pc game, console game, video game, cartoon, childish, ugly, blurry, distorted, watermark, text overlay"

EXTRA = os.environ.get("COMFY_ARGS", "").split()
proc = subprocess.Popen([sys.executable, "/comfyui/main.py", "--listen", "127.0.0.1", "--port", "8188",
                         "--disable-auto-launch", "--dont-print-server", *EXTRA], cwd="/comfyui")
for _ in range(1200):
    try:
        if requests.get(f"{COMFY}/system_stats", timeout=2).ok:
            break
    except Exception:
        pass
    if proc.poll() is not None:
        raise SystemExit("ComfyUI exited during startup")
    time.sleep(0.5)
print("ComfyUI is up", flush=True)


def workflow(prompt, w, h, frames, seed, ref=None):
    wf = {
        "ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CK}},
        "te": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": TE, "ckpt_name": CK, "device": "default"}},
        "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": CK}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": NEG}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": float(FPS)}},
        "lat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w // 2, "height": h // 2, "length": frames, "batch_size": 1}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["avae", 0], "frames_number": frames, "frame_rate": FPS, "batch_size": 1}},
        "av1": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["lat", 0], "audio_latent": ["alat", 0]}},
        "g1": {"class_type": "CFGGuider", "inputs": {"model": ["ck", 0], "positive": ["cond", 0], "negative": ["cond", 1], "cfg": 1.0}},
        "n1": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "ks1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "sg1": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "s1": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["n1", 0], "guider": ["g1", 0], "sampler": ["ks1", 0], "sigmas": ["sg1", 0], "latent_image": ["av1", 0]}},
        "sep1": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s1", 0]}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["cond", 0], "negative": ["cond", 1], "latent": ["sep1", 0]}},
        "upm": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPS}},
        "up": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["sep1", 0], "upscale_model": ["upm", 0], "vae": ["ck", 2]}},
        "av2": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["up", 0], "audio_latent": ["sep1", 1]}},
        "g2": {"class_type": "CFGGuider", "inputs": {"model": ["ck", 0], "positive": ["crop", 0], "negative": ["crop", 1], "cfg": 1.0}},
        "n2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "ks2": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
        "sg2": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}},
        "s2": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["n2", 0], "guider": ["g2", 0], "sampler": ["ks2", 0], "sigmas": ["sg2", 0], "latent_image": ["av2", 0]}},
        "sep2": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s2", 0]}},
        "dec": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["sep2", 0], "vae": ["ck", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4}},
        "adec": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["sep2", 1], "audio_vae": ["avae", 0]}},
        "vid": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "audio": ["adec", 0], "fps": float(FPS)}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["vid", 0], "filename_prefix": "kopro", "format": "mp4", "format.codec": "h264",
                                                   # crf 23 keeps a 5 s 720p clip ~2-4 MB: the base64 result must fit RunPod's output limit
                                                   "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 23.0}},
    }
    if ref:  # image-to-video: the picture is pinned as frame 0 in both stages (template strengths 0.7 / 1.0)
        wf["img"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["pre"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["img", 0], "img_compression": 18}}
        wf["i2v1"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["ck", 2], "image": ["pre", 0], "latent": ["lat", 0], "strength": 0.7, "bypass": False}}
        wf["i2v2"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["ck", 2], "image": ["pre", 0], "latent": ["up", 0], "strength": 1.0, "bypass": False}}
        wf["av1"]["inputs"]["video_latent"] = ["i2v1", 0]
        wf["av2"]["inputs"]["video_latent"] = ["i2v2", 0]
    return wf


def run(wf):
    before = set(glob.glob(os.path.join(OUT_DIR, "**", "*.mp4"), recursive=True))
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"ComfyUI rejected workflow: {r.text[:800]}")
    pid = r.json()["prompt_id"]
    while True:
        if proc.poll() is not None:
            raise RuntimeError("ComfyUI crashed")
        hist = requests.get(f"{COMFY}/history/{pid}", timeout=10).json().get(pid)
        if hist:
            st = hist.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {str(st.get('messages'))[-800:]}")
            if st.get("completed"):
                new = sorted(set(glob.glob(os.path.join(OUT_DIR, "**", "*.mp4"), recursive=True)) - before, key=os.path.getmtime)
                if not new:
                    raise RuntimeError("no video in output")
                with open(new[-1], "rb") as f:
                    data = f.read()
                for p in new:
                    os.remove(p)
                return data
        time.sleep(0.5)


def snap(v, default):
    v = int(v or default)
    return max(256, min(1920, v - v % 64))  # stage 1 runs at half size, which must stay a multiple of 32


def handler(job):
    inp = job.get("input") or {}
    t = time.time()
    if inp.get("warmup"):
        run(workflow("a calm sea", 256, 256, 9, 1))
        return {"warm": True, "seconds": round(time.time() - t, 2)}
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    w, h = snap(inp.get("width"), 1280), snap(inp.get("height"), 720)
    secs = max(2.0, min(10.0, float(inp.get("seconds") or 5)))
    frames = int(secs * FPS) // 8 * 8 + 1  # LTX wants 8n+1 frames
    ref = None
    if inp.get("image"):
        src = Image.open(io.BytesIO(base64.b64decode(inp["image"].split(",", 1)[-1]))).convert("RGB")
        src = ImageOps.fit(src, (w, h), Image.LANCZOS)  # centre-crop to the clip's frame
        ref = f"ref_{uuid.uuid4().hex}.png"
        src.save(os.path.join(IN_DIR, ref))
    try:
        data = run(workflow(prompt, w, h, frames, int(inp.get("seed") or 0), ref))
    finally:
        if ref:
            try:
                os.remove(os.path.join(IN_DIR, ref))
            except OSError:
                pass
    return {"video": base64.b64encode(data).decode(), "seconds": round(time.time() - t, 2), "width": w, "height": h,
            "frames": frames, "fps": FPS}


runpod.serverless.start({"handler": handler})

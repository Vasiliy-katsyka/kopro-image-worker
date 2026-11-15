"""Qwen-Image-2.1-Turbo worker (ComfyUI backend).
Input: {prompt, width?, height?, seed?} -> {image: base64 JPEG, seconds, width, height}
       {warmup: true} -> loads every model into VRAM with a tiny render, so the next real job is fast."""
import base64, io, os, subprocess, sys, time, uuid
import requests, runpod
from PIL import Image

COMFY = "http://127.0.0.1:8188"
OUT_DIR = "/comfyui/output"
SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"

proc = subprocess.Popen([sys.executable, "/comfyui/main.py", "--listen", "127.0.0.1", "--port", "8188",
                         "--disable-auto-launch", "--dont-print-server"], cwd="/comfyui")
for _ in range(600):
    try:
        if requests.get(f"{COMFY}/system_stats", timeout=2).ok:
            break
    except Exception:
        pass
    if proc.poll() is not None:
        raise SystemExit("ComfyUI exited during startup")
    time.sleep(0.5)
print("ComfyUI is up", flush=True)


def workflow(prompt, w, h, seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}},
        "5": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["5", 0], "positive": ["4", 0], "negative": ["4", 1], "cfg": 1.0}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "10": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["6", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "kopro"}},
    }


def run(prompt, w, h, seed):
    r = requests.post(f"{COMFY}/prompt", json={"prompt": workflow(prompt, w, h, seed), "client_id": uuid.uuid4().hex}, timeout=30)
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
            for out in hist.get("outputs", {}).values():
                for im in out.get("images", []):
                    path = os.path.join(OUT_DIR, im.get("subfolder", ""), im["filename"])
                    img = Image.open(path).convert("RGB")
                    os.remove(path)
                    return img
            if st.get("completed"):
                raise RuntimeError("no image in output")
        time.sleep(0.15)


def snap(v, default):
    v = int(v or default)
    return max(256, min(2048, v - v % 32))


def handler(job):
    inp = job.get("input") or {}
    t = time.time()
    if inp.get("warmup"):
        run("a plain white square", 256, 256, 1)
        return {"warm": True, "seconds": round(time.time() - t, 2)}
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    w, h = snap(inp.get("width"), 1024), snap(inp.get("height"), 1024)
    img = run(prompt, w, h, int(inp.get("seed") or 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return {"image": base64.b64encode(buf.getvalue()).decode(), "seconds": round(time.time() - t, 2), "width": w, "height": h}


runpod.serverless.start({"handler": handler})

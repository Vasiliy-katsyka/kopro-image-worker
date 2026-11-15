# Kopro image worker: Qwen-Image-2.1-Turbo (8 steps) via ComfyUI on RunPod Serverless.
# Comfy-Org 8-bit files baked in (~17 GB total vs 32.5 GB bf16) -> fits 24 GB GPUs, fast cold start.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
LABEL org.opencontainers.image.source=https://github.com/Vasiliy-katsyka/kopro-image-worker
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b v0.45.0 https://github.com/Comfy-Org/ComfyUI /comfyui \
 && pip install --no-cache-dir -r /comfyui/requirements.txt runpod==1.7.13 requests
ARG HF=https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main
RUN curl -fL --retry 5 -o /comfyui/models/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors $HF/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors
RUN curl -fL --retry 5 -o /comfyui/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors $HF/text_encoders/qwen3vl_8b_int8_convrot.safetensors
RUN curl -fL --retry 5 -o /comfyui/models/vae/qwen_image_2.1_vae_bf16.safetensors $HF/vae/qwen_image_2.1_vae_bf16.safetensors
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]

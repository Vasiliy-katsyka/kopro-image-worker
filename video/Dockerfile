# Kopro video worker: LTX-2.3 22B distilled (fp8) via ComfyUI on RunPod Serverless.
# Text or image -> ~5 s 720p clip WITH generated audio, two stages (half-res 8 steps, x2 latent upscale, 3 steps).
# Weights are baked as <10 GB parts (GHCR layer limit), merged at load time by split_loader.py. Needs a 48 GB GPU.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
LABEL org.opencontainers.image.source=https://github.com/Vasiliy-katsyka/kopro-image-worker
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b v0.45.0 https://github.com/Comfy-Org/ComfyUI /comfyui \
 && pip install --no-cache-dir -r /comfyui/requirements.txt runpod==1.7.13 requests \
 && mkdir -p /models /comfyui/models/latent_upscale_models
COPY fetch_part.py /app/fetch_part.py
ARG CK=https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors
ARG TE=https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors
RUN python /app/fetch_part.py $CK /models/ck0.safetensors 0 4
RUN python /app/fetch_part.py $CK /models/ck1.safetensors 1 4
RUN python /app/fetch_part.py $CK /models/ck2.safetensors 2 4
RUN python /app/fetch_part.py $CK /models/ck3.safetensors 3 4
RUN python /app/fetch_part.py $TE /models/te0.safetensors 0 2
RUN python /app/fetch_part.py $TE /models/te1.safetensors 1 2
RUN curl -fL --retry 5 -o /comfyui/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
      https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
 && echo '{"parts": ["/models/ck0.safetensors", "/models/ck1.safetensors", "/models/ck2.safetensors", "/models/ck3.safetensors"]}' > /comfyui/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors \
 && echo '{"parts": ["/models/te0.safetensors", "/models/te1.safetensors"]}' > /comfyui/models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors
COPY split_loader.py /comfyui/custom_nodes/split_loader.py
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]

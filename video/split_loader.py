"""ComfyUI custom node: a tiny '<name>.safetensors' file containing {"parts": [...]} loads as the merge of those
parts (see fetch_part.py). Lets >10 GB checkpoints live in several Docker layers without a copy at boot."""
import json, os
import comfy.utils

_orig = comfy.utils.load_torch_file

def load_torch_file(ckpt, *args, **kwargs):
    try:
        if os.path.getsize(ckpt) < 65536:
            with open(ckpt, "rb") as f:
                head = f.read()
            if head.startswith(b'{"parts"'):
                parts = json.loads(head)["parts"]
                want_meta = kwargs.get("return_metadata", False)
                sd, meta = {}, None
                for p in parts:
                    r = _orig(p, *args, **kwargs)
                    if want_meta:
                        r, m = r
                        meta = meta or m
                    sd.update(r)
                return (sd, meta) if want_meta else sd
    except OSError:
        pass
    return _orig(ckpt, *args, **kwargs)

comfy.utils.load_torch_file = load_torch_file
NODE_CLASS_MAPPINGS = {}

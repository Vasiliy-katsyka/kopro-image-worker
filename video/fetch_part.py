"""Downloads part i of n of a remote .safetensors as a standalone valid .safetensors (HTTP range requests).
GHCR caps image layers at 10 GB, so big checkpoints are baked as several parts, one Docker layer each;
split_loader.py (ComfyUI custom node) merges them back at load time. Part 0 carries the original metadata.
Usage: fetch_part.py URL OUT_PATH I N"""
import json, struct, sys, time, urllib.request

url, out, i, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def get(a, b):
    for attempt in range(8):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=60)
        except Exception as e:
            print("retry", attempt, e, flush=True); time.sleep(5 * (attempt + 1))
    raise SystemExit("download failed")

hn = struct.unpack("<Q", get(0, 7).read())[0]
hdr = json.loads(get(8, 7 + hn).read())
meta = hdr.pop("__metadata__", None)
names = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
total = hdr[names[-1]]["data_offsets"][1]
# contiguous groups of roughly equal bytes, in file order
groups, cur, acc = [[] for _ in range(n)], 0, 0
for k in names:
    sz = hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0]
    if acc >= total * (cur + 1) / n and cur < n - 1:
        cur += 1
    groups[cur].append(k); acc += sz
mine = groups[i]
start, end = hdr[mine[0]]["data_offsets"][0], hdr[mine[-1]]["data_offsets"][1]
new = {k: dict(hdr[k], data_offsets=[hdr[k]["data_offsets"][0] - start, hdr[k]["data_offsets"][1] - start]) for k in mine}
if i == 0 and meta:
    new["__metadata__"] = meta
hb = json.dumps(new, separators=(",", ":")).encode()
hb += b" " * (-len(hb) % 8)
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb)
    pos = start
    while pos < end:  # 512 MB chunks, each retried on its own
        b = min(end, pos + (512 << 20))
        r = get(8 + hn + pos, 8 + hn + b - 1)
        while True:
            c = r.read(8 << 20)
            if not c:
                break
            f.write(c); pos += len(c)
        if pos < b:
            raise SystemExit(f"short read at {pos}")
print(f"part {i}/{n}: {len(mine)} tensors, {(end - start) / 1e9:.2f} GB -> {out}", flush=True)

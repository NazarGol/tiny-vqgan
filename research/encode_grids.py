"""Encode COCO val2017 + CelebA-HQ to 32x32 VQGAN token grids (512 px centre crops) for tiny-decoder training data.
Writes web/research/data/tokens32_{coco,celeba}.npy (uint16 [N,1024]). Usage: .venv-export/bin/python web/research/encode_grids.py
"""
import glob, io, os, sys, time
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); EXPORT = os.path.join(HERE, "..", "export")
sys.path.insert(0, EXPORT); sys.path.insert(0, os.path.join(EXPORT, "taming-transformers"))
import export_decoder as E
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
SIDE, B = 512, 8
dev = "mps" if torch.backends.mps.is_available() else "cpu"
cfg, sd = E.load_cfg_and_sd(); enc = E.build_encoder(cfg, sd).to(dev)
def center_crop(im, side):
    w, h = im.size; s = side / min(w, h)
    im = im.resize((max(side, round(w * s)), max(side, round(h * s))), Image.LANCZOS)
    l, t = (im.width - side) // 2, (im.height - side) // 2
    return im.crop((l, t, l + side, t + side))
def coco():
    for p in sorted(glob.glob(os.path.join(EXPORT, "data", "val2017", "*.jpg"))):
        try: yield Image.open(p).convert("RGB")
        except Exception: continue
def celeba(maxn=1500):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(EXPORT, "data", "celeba_val.parquet"))
    col = next(c for c in t.column_names if "image" in c.lower())
    for i, row in enumerate(t.column(col).to_pylist()):
        if i >= maxn: break
        b = row["bytes"] if isinstance(row, dict) else row
        yield Image.open(io.BytesIO(b)).convert("RGB")
@torch.no_grad()
def run(name, gen):
    path = os.path.join(OUT, f"tokens32_{name}.npy")
    if os.path.exists(path): print("exists", path); return
    rows, batch, t0, n = [], [], time.time(), 0
    def flush():
        x = torch.from_numpy(np.stack(batch)).to(dev)
        rows.append(enc(x).reshape(len(batch), -1).cpu().numpy().astype(np.uint16)); batch.clear()
    for im in gen:
        a = np.asarray(center_crop(im, SIDE), dtype=np.float32).transpose(2, 0, 1) / 255.0
        batch.append(a); n += 1
        if len(batch) == B: flush()
        if n % 500 == 0: print(f"{name}: {n} images, {time.time()-t0:.0f}s", flush=True)
    if batch: flush()
    arr = np.concatenate(rows); np.save(path, arr); print("wrote", path, arr.shape, f"{time.time()-t0:.0f}s")
if __name__ == "__main__":
    run("celeba", celeba()); run("coco", coco())

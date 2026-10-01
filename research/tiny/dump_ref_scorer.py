"""Reference for the browser scorer test: PyTorch TokenScorer (fp16-rounded weights) embeddings + scores vs two text targets.
Usage: python dump_ref_scorer.py CKPT VARIANT OUT_DIR"""
import base64, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
ck, var, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
sd = torch.load(ck, map_location="cpu")["variants"][var]["ema"]
c0 = sd["emb.weight"].shape[1]; widths = tuple(sd[f"convs.{i}.0.bias"].shape[0] for i in range(4))
m = C.TokenScorer(c0=c0, widths=widths); m.load_state_dict(sd); m.eval()
with torch.no_grad():
    for p in m.parameters(): p.copy_(p.half().float())
C.export_scorer(m, os.path.join(out, f"tiny_scorer_{var}.bin"), os.path.join(out, f"tiny_scorer_{var}.json"), meta={"variant": var})
g16, g32 = C.load_grids(os.path.join(C.HERE, "..", "data")); s = C.GridSampler(g16, g32, seed=9)
rng = np.random.default_rng(3); targets = rng.normal(size=(2, 512)).astype(np.float32); targets /= np.linalg.norm(targets, axis=1, keepdims=True)
cases = []
with torch.no_grad():
    for S, n in ((8, 5), (16, 3), (12, 4), (6, 2), (20, 2)):
        grids = np.stack([s.sample(S) for _ in range(n)]); e = m(torch.from_numpy(grids)).numpy()
        cases.append({"S": S, "n": n, "tokens": grids.reshape(-1).tolist(), "emb": base64.b64encode(e.astype(np.float32).tobytes()).decode(), "scores": (e @ targets.T).tolist()})
json.dump({"variant": var, "targets": base64.b64encode(targets.tobytes()).decode(), "cases": cases}, open(os.path.join(out, f"ref_scorer_{var}.json"), "w"))
print("wrote scorer ref", widths, c0)

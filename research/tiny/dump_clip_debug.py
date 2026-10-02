"""Per-op checkpoint statistics of the clipvision IR (torch executor) on clip_img_0.png, for localising shader bugs."""
import base64, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common as C, ir_torch as IR
d = sys.argv[1]; man, tensor = IR.load(d)
u8 = np.asarray(Image.open(os.path.join(C.HERE, "test", "clip_img_0.png")).convert("RGB")).astype(np.float32) / 255
x = torch.from_numpy(u8).permute(2, 0, 1)[None]; x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
# re-run the executor but keep every op output
T = {man["ops"][0]["in"]: F.pad(x, (0, 0, 0, 0, 0, 1))}; outs = []
with torch.no_grad():
    for i, op in enumerate(man["ops"]):
        sub = dict(man); sub["ops"] = [op]
        y = IR.run(sub, tensor, None) if False else None
        # evaluate op by op using the executor's internals: simplest is to call run() on a prefix (cheap enough: 118 ops, small net)
    for i in range(len(man["ops"])):
        sub = dict(man); sub["ops"] = man["ops"][:i + 1]
        y = IR.run(sub, tensor, x)
        a = y.numpy(); flat = a.reshape(-1)
        idx = np.linspace(0, flat.size - 1, 16).astype(int)
        outs.append({"i": i, "op": man["ops"][i]["op"], "out": man["ops"][i]["out"], "shape": list(a.shape), "mean": float(flat.mean()), "std": float(flat.std()), "absmax": float(np.abs(flat).max()), "samples": [float(v) for v in flat[idx]], "sample_idx": [int(v) for v in idx]})
json.dump(outs, open(os.path.join(C.HERE, "test", "ref_clip_debug.json"), "w")); print("wrote", len(outs), "checkpoints")

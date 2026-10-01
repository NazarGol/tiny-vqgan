"""Reference outputs for the browser runtime test: run a TinyDec (fp16-rounded weights, fp32 activations) on a few grids.
Usage: python dump_ref.py CKPT VARIANT OUT_DIR   -> OUT_DIR/tiny_decoder_{V}.bin/.json + ref_{V}.json"""
import base64, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
ck, var, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
st = torch.load(ck, map_location="cpu")
sd = st["variants"][var]["ema"]
widths = tuple(sd[k].shape[0] for k in sd if k.endswith(".0.bias") and "stages" in k)   # stage in-convs, in order
m = C.TinyDec(widths=widths, blocks=tuple(sum(1 for k in sd if k.startswith(f"stages.{i}.") and k.endswith(".c1.bias")) for i in range(len(widths))))
m.load_state_dict(sd); m.eval()
with torch.no_grad():
    for p in m.parameters(): p.copy_(p.half().float())     # what the browser sees
man = C.export_tinydec(m, os.path.join(out, f"tiny_decoder_{var}.bin"), os.path.join(out, f"tiny_decoder_{var}.json"), meta={"variant": var})
g16, g32 = C.load_grids(os.path.join(C.HERE, "..", "data")); s = C.GridSampler(g16, g32, seed=5)
cases = []
acts = []
def hook(mod, inp, outp): acts.append(float(outp.abs().max()))
hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, torch.nn.Conv2d)]
with torch.no_grad():
    for name, S in (("stroke_blank", 8), ("photo16", 16), ("stroke_canvas", 12), ("mosaic", 11)):
        g = s.sample(S, name); y = m(torch.from_numpy(g)[None])[0].clamp(0, 1)
        cases.append({"name": name, "h": S, "w": S, "tokens": g.reshape(-1).tolist(), "expected": base64.b64encode(y.numpy().astype(np.float32).tobytes()).decode()})
json.dump({"variant": var, "max_abs_activation": max(acts), "cases": cases}, open(os.path.join(out, f"ref_{var}.json"), "w"))
print("wrote", out, "max activation", max(acts), "layers", len(man["layers"]), "widths", widths)

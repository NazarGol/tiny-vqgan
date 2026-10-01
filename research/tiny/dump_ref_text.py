"""Reference for the JS text-encoder test: export a TinyText (fp16-rounded) and embed a few prompts. Usage: python dump_ref_text.py CKPT VARIANT OUT_DIR"""
import base64, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from tokenizers import Tokenizer
ck, var, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
sd = torch.load(ck, map_location="cpu")["variants"][var]["ema"]
e, w = sd["tok.weight"].shape[1], sd["pos"].shape[1]; L = sum(1 for k in sd if k.endswith(".qkv.weight")); h = 4
m = C.TinyText(emb_dim=e, width=w, layers=L, heads=h); m.load_state_dict(sd); m.eval()
with torch.no_grad():
    for p in m.parameters(): p.copy_(p.half().float())
C.export_text(m, os.path.join(out, f"tiny_text_{var}.bin"), os.path.join(out, f"tiny_text_{var}.json"), meta={"variant": var})
tk = Tokenizer.from_file(os.path.join(C.WEB, "models", "mobileclip_s0", "tokenizer.json"))
def enc(t):
    ids = tk.encode(t).ids[:77]; ids = ids if ids[-1] == 49407 else ids[:76] + [49407]; return ids + [0] * (77 - len(ids))
texts = ["a red forest", "the sea at night", "deadline on Friday", "I miss my grandmother's kitchen and the smell of bread in the morning, when everything was still simple"]
ids = torch.tensor([enc(t) for t in texts])
with torch.no_grad(): emb = m(ids).numpy()
json.dump({"variant": var, "cases": [{"text": t, "ids": ids[i].tolist(), "emb": base64.b64encode(emb[i].astype(np.float32).tobytes()).decode()} for i, t in enumerate(texts)]}, open(os.path.join(out, f"ref_text_{var}.json"), "w"))
print("wrote text ref", e, w, L)

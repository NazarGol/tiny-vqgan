"""Reference executor for the clipvision op list (export_clip_vision.py), unpacking weights with the same formulas the shaders use.
Usage: python ir_torch.py DIR [--int8]   -> cosine vs onnxruntime on decoded images"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common as C

def load(dirpath, int8=False):
    man = json.load(open(os.path.join(dirpath, "clip_vision_i8.json" if int8 else "clip_vision.json")))
    blob = open(os.path.join(dirpath, "clip_vision_i8.bin" if int8 else "clip_vision.bin"), "rb").read()
    def tensor(name):
        e = man["tensors"][name]
        if e["dtype"] == "f16": return np.frombuffer(blob, np.float16, e["len"], e["offset"]).astype(np.float32)
        q = np.frombuffer(blob, np.int8, e["len"], e["offset"]).astype(np.float32); sc = np.frombuffer(blob, np.float32, e["nscales"], e["scales"])
        if e.get("scales_per_channel"):   # packed order (go, gi, tap, j, i): element index -> output channel 4*go + i
            go = e["cout"] // 4; per_go = e["len"] // go; idx = np.arange(e["len"]); o = (idx // per_go) * 4 + (idx % 4); return q * sc[o]
        return q * sc
    return man, tensor

def dense_weight(flat, cout, cin, k):
    go, gi = cout // 4, cin // 4; p = flat.reshape(go, gi, k * k, 4, 4)                # [go, gi, tap, j, i]
    return torch.from_numpy(np.ascontiguousarray(p.transpose(0, 4, 1, 3, 2).reshape(cout, cin, k, k)))

def dw_weight(flat, cout, k):
    p = flat.reshape(cout // 4, k * k, 4)                                               # [g, tap, 4]
    return torch.from_numpy(np.ascontiguousarray(p.transpose(0, 2, 1).reshape(cout, 1, k, k)))

def gelu(x): return 0.5 * x * (1 + torch.erf(x / 1.4142135))
def act(x, a): return gelu(x) if a == "gelu" else F.relu(x) if a == "relu" else x

@torch.no_grad()
def run(man, tensor, img):   # img [B,3,256,256] float
    B = img.shape[0]; T = {man["ops"][0]["in"]: F.pad(img, (0, 0, 0, 0, 0, 1))}      # 4th input channel = 0
    for op in man["ops"]:
        kind = op["op"]
        if kind == "conv":
            w = dense_weight(tensor(op["w"]), op["cout"], op["cin"], op["k"]); b = torch.from_numpy(tensor(op["b"]))
            x = T[op["in"]]; x = F.pad(x, (0, 0, 0, 0, 0, op["cin"] - x.shape[1])) if x.shape[1] < op["cin"] else x
            T[op["out"]] = act(F.conv2d(x, w, b, stride=op["stride"], padding=op["pad"]), op["act"])
        elif kind == "dwconv":
            w = dw_weight(tensor(op["w"]), op["cout"], op["k"]); b = torch.from_numpy(tensor(op["b"])); x = T[op["in"]][:, :op["cin"]]
            T[op["out"]] = act(F.conv2d(x, w[:op["cin"] * op["m"]], b[:op["cin"] * op["m"]], stride=op["stride"], padding=op["pad"], groups=op["cin"]), op["act"])
            if T[op["out"]].shape[1] < op["cout"]: T[op["out"]] = F.pad(T[op["out"]], (0, 0, 0, 0, 0, op["cout"] - T[op["out"]].shape[1]))
        elif kind == "se":
            x = T[op["in"]]; v = x.mean((2, 3), keepdim=True)
            w1 = dense_weight(tensor(op["fc1w"]), op["r"], op["c"], 1); b1 = torch.from_numpy(tensor(op["fc1b"])); w2 = dense_weight(tensor(op["fc2w"]), op["c"], op["r"], 1); b2 = torch.from_numpy(tensor(op["fc2b"]))
            h = F.relu(F.conv2d(v, w1, b1)); s = torch.sigmoid(F.conv2d(h, w2, b2)); T[op["out"]] = act(x * s, op["act"])
        elif kind == "scale_add":
            s = torch.from_numpy(tensor(op["s"])); T[op["out"]] = T[op["a"]] + s[None, :, None, None] * T[op["b"]]
        elif kind == "gelu": T[op["out"]] = gelu(T[op["in"]])
        elif kind == "attn":
            x = T[op["in"]]; Bn, Cc, H, Wd = x.shape; qw = dense_weight(tensor(op["qkvw"]), 3 * Cc, Cc, 1); qb = torch.from_numpy(tensor(op["qkvb"]))
            qkv = F.conv2d(x, qw, qb).reshape(Bn, 3, op["heads"], Cc // op["heads"], H * Wd)   # [B, 3, h, d, N]
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            att = torch.softmax((q.transpose(-1, -2) @ k) * op["scale"], -1)              # [B, h, N, N]
            o = (att @ v.transpose(-1, -2)).transpose(-1, -2).reshape(Bn, Cc, H, Wd)           # [B, C, H, W]
            pw = dense_weight(tensor(op["projw"]), Cc, Cc, 1); pb = torch.from_numpy(tensor(op["projb"])); T[op["out"]] = F.conv2d(o, pw, pb)
        elif kind == "gmean": T[op["out"]] = T[op["in"]].mean((2, 3), keepdim=True)
        elif kind == "head":
            w = dense_weight(tensor(op["w"]), op["cout"], op["cin"], 1); T[op["out"]] = F.conv2d(T[op["in"]], w).flatten(1)
        else: raise ValueError(kind)
    return T[man["ops"][-1]["out"]]

if __name__ == "__main__":
    import onnxruntime as ort
    d = sys.argv[1]; int8 = "--int8" in sys.argv
    man, tensor = load(d, int8)
    sess = ort.InferenceSession(os.path.join(C.WEB, "models", "mobileclip_s0", "onnx", "vision_model.onnx"), providers=["CPUExecutionProvider"])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); teacher, _ = C.load_teacher(dev); g16, g32 = C.load_grids(os.path.join(C.HERE, "..", "data")); s = C.GridSampler(g16, g32, seed=3)
    toks = torch.from_numpy(np.stack([s.sample(16) for _ in range(6)]))
    with torch.no_grad(): img = teacher(toks.to(dev)).float().clamp(0, 1).cpu()
    ref = sess.run(None, {"pixel_values": img.numpy()})[0]; ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    out = run(man, tensor, img).numpy(); out /= np.linalg.norm(out, axis=1, keepdims=True)
    cos = (ref * out).sum(1); print(f"{'int8' if int8 else 'fp16'} IR vs onnxruntime: cosine per image {np.round(cos, 5)}  min {cos.min():.5f}")

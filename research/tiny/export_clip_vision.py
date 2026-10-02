"""MobileCLIP-S0 image tower (Xenova ONNX) -> a compact op list + packed weights for the WebGL2 runtime (lib/clipvision.js).
Ops: conv (dense, k 1/3, cin padded to 4), dwconv (1 input channel per group, m outputs per channel), se, scale_add (LayerScale +
residual), attn (BN folded into qkv), gmean, head. Weights fp16 (default) or int8 with per-output-channel scales (--int8).
Usage: python export_clip_vision.py [--int8] [--onnx path] [--out dir]"""
import argparse, json, os, sys
import numpy as np, onnx
from onnx import numpy_helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
ap = argparse.ArgumentParser(); ap.add_argument("--onnx", default=os.path.join(C.WEB, "models", "mobileclip_s0", "onnx", "vision_model.onnx")); ap.add_argument("--out", default=os.path.join(C.WEB, "models", "tiny")); ap.add_argument("--int8", action="store_true")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
m = onnx.load(args.onnx); g = m.graph
W = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value": consts[n.output[0]] = numpy_helper.to_array(a.t)
nodes = [n for n in g.node if n.op_type != "Constant"]
SKIP = {"Shape", "Gather", "Unsqueeze", "Concat", "Reshape", "Transpose", "Slice", "Squeeze", "Split", "Pad"}
attrs = lambda n: {a.name: a for a in n.attribute}

arrays, meta_entries, off = [], {}, 0
def add(name, arr, scales=None, qscale_full=None):
    """scales: compact per-output-channel vector stored after the int8 data; qscale_full: the per-element scale used to quantise."""
    global off
    a = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
    if args.int8 and scales is not None:
        q = np.clip(np.round(a / qscale_full), -127, 127).astype(np.int8); arrays.append(("i8", q)); entry = {"offset": off, "len": int(q.size), "dtype": "i8"}
        pad = (-int(q.size)) % 4; arrays.append(("i8", np.zeros(pad, np.int8)))   # keep the fp32 scales 4-byte aligned
        arrays.append(("f32", scales.astype(np.float32))); entry["scales"] = off + int(q.size) + pad; entry["nscales"] = int(scales.size); off += int(q.size) + pad + int(scales.size) * 4
    else:
        h = a.astype(np.float16); arrays.append(("f16", h)); entry = {"offset": off, "len": int(h.size), "dtype": "f16"}; off += int(h.size) * 2
    meta_entries[name] = entry; return entry

def pad4(x):  # pad a channel count to a multiple of 4
    return (x + 3) // 4 * 4

def pack_dense(w, b, name, quant=True):
    """w [cout, cin, k, k] -> (go, gi, tap, j, i) with cin padded to 4; int8 scale per output channel (repeated in packed order)."""
    cout, cin, k, _ = w.shape; cin4 = pad4(cin); cout4 = pad4(cout)
    w4 = np.zeros((cout4, cin4, k, k), np.float32); w4[:cout, :cin] = w; b4 = np.zeros(cout4, np.float32); b4[:cout] = b
    go, gi = cout4 // 4, cin4 // 4
    p = w4.reshape(go, 4, gi, 4, k * k).transpose(0, 2, 4, 3, 1)             # [go, gi, tap, j, i]
    if args.int8 and quant and w.size >= 64 * 192:   # int8 only for the big 1x1 ConvFFN layers; small/sensitive layers stay fp16
        s = np.maximum(np.abs(w4).reshape(cout4, -1).max(1), 1e-8) / 127.0     # per output channel
        full = np.broadcast_to(s.reshape(go, 1, 1, 1, 4), p.shape).reshape(-1)
        e = add(name + ".w", p, scales=s, qscale_full=full); e["scales_per_channel"] = True; e["cout"] = int(cout4); e["gi"] = int(gi); e["taps"] = int(k * k)
    else: add(name + ".w", p)
    add(name + ".b", b4)
    return cout4, cin4

def pack_dw(w, b, name):
    """w [cout, 1, k, k], out channel c uses input channel c // m. Layout (group, tap, 4): texel (x = tap, y = g) = channels 4g..4g+3."""
    cout, _, k, _ = w.shape; cout4 = pad4(cout)
    w4 = np.zeros((cout4, k * k), np.float32); w4[:cout] = w.reshape(cout, -1); b4 = np.zeros(cout4, np.float32); b4[:cout] = b
    p = w4.reshape(cout4 // 4, 4, k * k).transpose(0, 2, 1)                    # [g, tap, 4]: texture row g, column tap, RGBA = 4 channels
    add(name + ".w", p); add(name + ".b", b4); return cout4

ops, i = [], 0
act_target = None   # the op whose output the next GELU applies to
def is_gelu(j):
    return j + 4 < len(nodes) and [n.op_type for n in nodes[j:j + 5]] == ["Div", "Erf", "Add", "Mul", "Mul"]
def conv_attrs(n):
    a = attrs(n); return tuple(a["kernel_shape"].ints), (tuple(a["strides"].ints) if "strides" in a else (1, 1)), (a["group"].i if "group" in a else 1), (tuple(a["pads"].ints) if "pads" in a else (0, 0, 0, 0))
while i < len(nodes):
    n = nodes[i]
    if n.op_type in SKIP: i += 1; continue
    if n.op_type == "Conv":
        k, st, grp, pads = conv_attrs(n); w, b = W[n.input[1]], W[n.input[2]]; cout, cin_g = w.shape[0], w.shape[1]
        name = f"op{len(ops)}"
        if grp == 1:
            cout4, cin4 = pack_dense(w, b, name); ops.append({"op": "conv", "in": n.input[0], "out": n.output[0], "k": k[0], "stride": st[0], "pad": pads[0], "cin": int(cin4), "cout": int(cout4), "w": name + ".w", "b": name + ".b", "act": "none"})
        else:
            assert cin_g == 1 and cout % grp == 0; m_ = cout // grp
            cout4 = pack_dw(w, b, name); ops.append({"op": "dwconv", "in": n.input[0], "out": n.output[0], "k": k[0], "stride": st[0], "pad": pads[0], "cin": int(grp), "cout": int(cout4), "m": int(m_), "w": name + ".w", "b": name + ".b", "act": "none"})
        act_target = ops[-1]; i += 1; continue
    if is_gelu(i):
        src = n.input[0]
        if act_target is not None and act_target["out"] == src and act_target["act"] == "none": act_target["act"] = "gelu"; act_target["out"] = nodes[i + 4].output[0]
        else: ops.append({"op": "gelu", "in": src, "out": nodes[i + 4].output[0]})
        act_target = None; i += 5; continue
    if n.op_type == "Mul" and n.input[0] in W and W[n.input[0]].ndim == 3:   # LayerScale then residual Add
        s = W[n.input[0]].reshape(-1); nxt = nodes[i + 1]; assert nxt.op_type == "Add" and n.output[0] in nxt.input
        resid = [x for x in nxt.input if x != n.output[0]][0]; name = f"op{len(ops)}"
        s4 = np.zeros(pad4(s.size), np.float32); s4[:s.size] = s; add(name + ".s", s4)
        ops.append({"op": "scale_add", "a": resid, "b": n.input[1], "s": name + ".s", "out": nxt.output[0], "c": int(pad4(s.size))}); act_target = None; i += 2; continue
    if (n.op_type == "ReduceMean" and attrs(n).get("keepdims", None) is not None and attrs(n)["keepdims"].i == 1) or n.op_type == "AveragePool":   # SE
        src = n.input[0] if n.op_type == "ReduceMean" else nodes[i - 1].input[0]   # AveragePool follows a zero Pad of the same tensor
        j = i + 1; seq = []
        while len(seq) < 4:
            if nodes[j].op_type in SKIP: j += 1; continue
            seq.append(nodes[j]); j += 1
        fc1, relu, fc2, sig = seq; assert fc1.op_type == "Conv" and relu.op_type == "Relu" and fc2.op_type == "Conv" and sig.op_type == "Sigmoid", [x.op_type for x in seq]
        while nodes[j].op_type in SKIP: j += 1
        mul = nodes[j]; assert mul.op_type == "Mul" and src in mul.input; j += 1
        name = f"op{len(ops)}"; w1, b1, w2, b2 = W[fc1.input[1]], W[fc1.input[2]], W[fc2.input[1]], W[fc2.input[2]]
        r4, c4 = pack_dense(w1, b1, name + ".fc1", quant=False); c4b, r4b = pack_dense(w2, b2, name + ".fc2", quant=False)
        op = {"op": "se", "in": src, "out": mul.output[0], "c": int(c4), "r": int(r4), "fc1w": name + ".fc1.w", "fc1b": name + ".fc1.b", "fc2w": name + ".fc2.w", "fc2b": name + ".fc2.b", "act": "none"}
        ops.append(op); act_target = op; i = j; continue
    if n.op_type == "BatchNormalization":   # attention block: BN folded into qkv
        a = attrs(n); eps = [x.f for x in n.attribute if x.name == "epsilon"][0]
        gamma, beta, mean, var = (W[x] for x in n.input[1:5]); s = gamma / np.sqrt(var + eps); t = beta - mean * s
        j = i + 1; mm = []
        while len(mm) < 2:
            if nodes[j].op_type == "MatMul" and nodes[j].input[1] in W: mm.append(nodes[j])
            j += 1
        qkv, proj = mm; qw = W[qkv.input[1]]; pw = W[proj.input[1]]              # [in, out]
        scale_node = next(x for x in nodes[i:j] if x.op_type == "Mul" and x.input[1] in consts and consts[x.input[1]].size == 1); scale = float(consts[scale_node.input[1]])
        proj_add = nodes[j]; assert proj_add.op_type == "Add" and proj.output[0] in proj_add.input; pb = W[[x for x in proj_add.input if x in W][0]]
        qw_f = (s[:, None] * qw); qb = t @ qw                                        # fold BN
        name = f"op{len(ops)}"
        pack_dense(qw_f.T.reshape(qw.shape[1], qw.shape[0], 1, 1), qb, name + ".qkv", quant=False); pack_dense(pw.T.reshape(pw.shape[1], pw.shape[0], 1, 1), pb, name + ".proj", quant=False)
        # then LayerScale + residual
        k2 = j + 1
        while not (nodes[k2].op_type == "Mul" and nodes[k2].input[0] in W and W[nodes[k2].input[0]].ndim == 3): k2 += 1
        ls = nodes[k2]; addn = nodes[k2 + 1]; assert addn.op_type == "Add"; resid = [x for x in addn.input if x != ls.output[0]][0]
        sv = W[ls.input[0]].reshape(-1); add(name + ".s", sv)
        ops.append({"op": "attn", "in": n.input[0], "out": ls.input[1], "c": int(qw.shape[0]), "heads": 16, "scale": scale, "qkvw": name + ".qkv.w", "qkvb": name + ".qkv.b", "projw": name + ".proj.w", "projb": name + ".proj.b"})
        ops.append({"op": "scale_add", "a": resid, "b": ls.input[1], "s": name + ".s", "out": addn.output[0], "c": int(sv.size)}); act_target = None; i = k2 + 2; continue
    if n.op_type == "ReduceMean":   # global pool (keepdims=0) then the head MatMul
        j = i + 1
        while nodes[j].op_type != "MatMul": j += 1
        hw = W[nodes[j].input[1]]; name = f"op{len(ops)}"; cout4, cin4 = pack_dense(hw.T.reshape(hw.shape[1], hw.shape[0], 1, 1), np.zeros(hw.shape[1], np.float32), name, quant=False)
        ops.append({"op": "gmean", "in": n.input[0], "out": n.output[0], "c": int(pad4(hw.shape[0]))}); ops.append({"op": "head", "in": n.output[0], "out": nodes[j].output[0], "cin": int(cin4), "cout": int(cout4), "w": name + ".w"}); i = j + 1; continue
    raise RuntimeError(f"unhandled node {i} {n.op_type} {n.name}")
# write
blob = bytearray()
for kind, a in arrays:
    blob += a.tobytes()
bin_path = os.path.join(args.out, "clip_vision_i8.bin" if args.int8 else "clip_vision.bin"); open(bin_path, "wb").write(blob)
man = {"type": "clipvision", "input": {"w": 256, "h": 256}, "ops": ops, "tensors": meta_entries, "bytes": len(blob), "dtype": "i8" if args.int8 else "f16", "source": os.path.basename(args.onnx), "notice": "MobileCLIP-S0 image tower (Apple ML Research Model License, research use only)"}
json.dump(man, open(os.path.join(args.out, "clip_vision_i8.json" if args.int8 else "clip_vision.json"), "w"))
print(f"{len(ops)} ops, {len(blob)/2**20:.1f} MiB -> {bin_path}"); import collections; print(collections.Counter(o['op'] for o in ops))

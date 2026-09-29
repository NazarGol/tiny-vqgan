"""Spike 1: export the VQGAN (imagenet f16 16384) decoder to ONNX.

Model = codebook lookup (tokens -> z) + post_quant_conv + taming Decoder.
Input  tokens: int32 [B, H, W]   (H, W = token grid, 16x16 -> 256px)
Output image:  float32 [B, 3, 16H, 16W] in [0, 1]

Writes:
  ../models/decoder_fp32.onnx
  ../models/decoder_fp16.onnx        (weights fp16, io fp32/int32)
  ../spike1/test_tokens.json         (real tokens from encoding a test image, 16x16 and 32x32)
  ../spike1/out/torch_*.png          (PyTorch decodes of those tokens, for comparison)

Usage:
  .venv-export/bin/python web/export/export_decoder.py --test-image some.jpg
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "taming-transformers"))
from taming.modules.diffusionmodules.model import Decoder, Encoder  # noqa: E402


# The .ckpt is a PyTorch-Lightning checkpoint whose pickle references pytorch_lightning
# classes (callbacks etc.). Stub the package so torch.load works without installing Lightning.
import importlib.abc
import importlib.util
import types


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    ROOTS = ("pytorch_lightning", "lightning")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.ROOTS:
            return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        def _ga(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _Dummy
        m.__getattr__ = _ga
        return m

    def exec_module(self, module):
        pass



CKPT_DIR = os.path.join(HERE, "checkpoints")
MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))
SPIKE_DIR = os.path.normpath(os.path.join(HERE, "..", "spike1"))


class TokenDecoder(nn.Module):
    """tokens [B,H,W] int -> image [B,3,16H,16W] in [0,1]."""

    def __init__(self, ddconfig, n_embed, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.decoder = Decoder(**ddconfig)

    def forward(self, tokens):
        z = self.embedding(tokens).permute(0, 3, 1, 2)  # B,C,H,W
        x = self.decoder(self.post_quant_conv(z))
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


class TokenEncoder(nn.Module):
    """image [B,3,H,W] in [0,1] -> tokens [B,H/16,W/16] (nearest codebook entry)."""

    def __init__(self, ddconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.embedding = nn.Embedding(n_embed, embed_dim)

    def forward(self, img):
        z = self.quant_conv(self.encoder(img * 2.0 - 1.0))  # B,C,h,w
        z = z.permute(0, 2, 3, 1)  # B,h,w,C
        cb = self.embedding.weight  # N,C
        d = (z.pow(2).sum(-1, keepdim=True) + cb.pow(2).sum(1) - 2 * z @ cb.T)
        return d.argmin(-1)


def load_cfg_and_sd():
    cfg = yaml.safe_load(open(os.path.join(CKPT_DIR, "vqgan_imagenet_f16_16384.yaml")))["model"]["params"]
    finder = _StubFinder()
    sys.meta_path.append(finder)
    try:
        sd = torch.load(os.path.join(CKPT_DIR, "vqgan_imagenet_f16_16384.ckpt"), map_location="cpu", weights_only=False)["state_dict"]
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.split(".")[0] in _StubFinder.ROOTS]:
            del sys.modules[k]
    return cfg, sd


def build_decoder(cfg, sd):
    m = TokenDecoder(cfg["ddconfig"], cfg["n_embed"], cfg["embed_dim"])
    sub = {"embedding.weight": sd["quantize.embedding.weight"]}
    for k, v in sd.items():
        if k.startswith("decoder.") or k.startswith("post_quant_conv."):
            sub[k] = v
    missing, unexpected = m.load_state_dict(sub, strict=True), None
    return m.eval()


def build_encoder(cfg, sd):
    m = TokenEncoder(cfg["ddconfig"], cfg["n_embed"], cfg["embed_dim"])
    sub = {"embedding.weight": sd["quantize.embedding.weight"]}
    for k, v in sd.items():
        if k.startswith("encoder.") or k.startswith("quant_conv."):
            sub[k] = v
    m.load_state_dict(sub, strict=True)
    return m.eval()


def rescale_tail(dec, s):
    """Fold 1/s into the last stage so fp16 does not overflow on GPU.

    The residual stream after up[1].upsample.conv reaches ~1e5 (> fp16 max 65504) for real
    images. Every consumer of that stream is a GroupNorm (scale-invariant up to eps) or a
    residual add, so dividing the stream by s and also dividing each up[0] block's conv2
    output by s keeps x/s + h/s = (x + h)/s exactly; norm_out then removes the scale.
    """
    with torch.no_grad():
        d = dec.decoder
        d.up[1].upsample.conv.weight.div_(s)
        d.up[1].upsample.conv.bias.div_(s)
        for blk in d.up[0].block:
            assert blk.in_channels == blk.out_channels, "no shortcut conv expected in up[0]"
            blk.conv2.weight.div_(s)
            blk.conv2.bias.div_(s)
    return dec


def param_report(dec):
    total = sum(p.numel() for p in dec.parameters())
    print(f"[params] decoder+codebook total: {total/1e6:.1f} M  (fp32 {total*4/2**20:.0f} MiB, fp16 {total*2/2**20:.0f} MiB)")
    groups = {}
    for n, p in dec.named_parameters():
        key = n.split(".")[0]
        if key == "decoder":
            parts = n.split(".")
            key = "decoder." + (parts[1] if parts[1] != "up" else f"up[{parts[2]}]")
        groups[key] = groups.get(key, 0) + p.numel()
    for k, v in groups.items():
        print(f"    {k:24s} {v/1e6:6.2f} M")


def to_pil(x):
    return Image.fromarray((x[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8))


def export_onnx(dec, path, opset):
    tokens = torch.randint(0, 16384, (1, 16, 16), dtype=torch.int32)
    t0 = time.time()
    try:
        torch.onnx.export(
            dec, (tokens,), path, dynamo=False, opset_version=opset,
            input_names=["tokens"], output_names=["image"],
            dynamic_axes={"tokens": {0: "batch", 1: "h", 2: "w"},
                          "image": {0: "batch", 2: "height", 3: "width"}},
            do_constant_folding=True,
        )
        how = "torchscript exporter"
    except Exception as e:  # legacy exporter gone or failed -> dynamo exporter
        print(f"[export] legacy exporter failed ({type(e).__name__}: {str(e)[:200]}), trying dynamo=True")
        from torch.export import Dim
        prog = torch.onnx.export(
            dec, (tokens,), dynamo=True, opset_version=opset,
            input_names=["tokens"], output_names=["image"],
            dynamic_shapes={"tokens": {0: Dim("batch"), 1: Dim("h"), 2: Dim("w")}},
        )
        prog.optimize()
        prog.save(path)
        how = "dynamo exporter"
    print(f"[export] {how}, opset {opset}, {time.time()-t0:.1f}s -> {path} ({os.path.getsize(path)/2**20:.1f} MiB)")


def convert_fp16(src, dst):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(src)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, dst)
    print(f"[fp16] -> {dst} ({os.path.getsize(dst)/2**20:.1f} MiB)")


def op_histogram(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    hist = {}
    for n in m.graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    print(f"[ops] {os.path.basename(path)}: " + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))


def verify(path, tokens_np, ref_img, tag):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    t0 = time.time()
    out = sess.run(None, {"tokens": tokens_np})[0]
    dt = time.time() - t0
    diff = np.abs(out - ref_img.numpy())
    print(f"[verify:{tag}] {tokens_np.shape[1]}x{tokens_np.shape[2]} tokens -> {out.shape[2]}x{out.shape[3]} px, "
          f"ORT-CPU {dt*1000:.0f} ms, max|diff| {diff.max():.4f}, mean|diff| {diff.mean():.5f} (0..1 scale)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-image", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--rescale-tail", type=float, default=16.0, help="1/s scale folded into the last stage (0 = off)")
    args = ap.parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(SPIKE_DIR, "out"), exist_ok=True)

    cfg, sd = load_cfg_and_sd()
    dec, enc = build_decoder(cfg, sd), build_encoder(cfg, sd)
    param_report(dec)

    # real tokens from the encoder, for 256px (16x16) and 512px (32x32)
    img = Image.open(args.test_image).convert("RGB")
    tokens_json, refs = {}, {}
    with torch.no_grad():
        for side in (256, 512):
            w, h = img.size
            s = side / min(w, h)
            im = img.resize((max(side, round(w * s)), max(side, round(h * s))), Image.LANCZOS)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side))
            x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
            toks = enc(x)
            t0 = time.time()
            ref = dec(toks.to(torch.int32))
            print(f"[torch-cpu] decode {side//16}x{side//16} in {(time.time()-t0)*1000:.0f} ms")
            n = side // 16
            tokens_json[str(n)] = toks[0].tolist()
            refs[n] = (toks.to(torch.int32).numpy(), ref)
            im.save(os.path.join(SPIKE_DIR, "out", f"input_{side}.png"))
            to_pil(ref).save(os.path.join(SPIKE_DIR, "out", f"torch_{side}.png"))
            print(f"[tokens] {n}x{n}: {len(set(toks.flatten().tolist()))} unique of {n*n}")
    with open(os.path.join(SPIKE_DIR, "test_tokens.json"), "w") as f:
        json.dump(tokens_json, f)

    if args.rescale_tail:
        dec_ref = dec
        dec = rescale_tail(build_decoder(cfg, sd), args.rescale_tail)
        with torch.no_grad():
            for n, (tk, ref) in refs.items():
                out = dec(torch.from_numpy(tk))
                print(f"[rescale s={args.rescale_tail:g}] {n}x{n}: max|diff| vs unscaled {float((out - ref).abs().max()):.2e}")
    p32 = os.path.join(MODELS_DIR, "decoder_fp32.onnx")
    p16 = os.path.join(MODELS_DIR, "decoder_fp16.onnx")
    export_onnx(dec, p32, args.opset)
    op_histogram(p32)
    convert_fp16(p32, p16)

    for n, (tk, ref) in refs.items():
        out = verify(p32, tk, ref, "fp32")
        out16 = verify(p16, tk, ref, "fp16")
        Image.fromarray((np.clip(out16[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)).save(
            os.path.join(SPIKE_DIR, "out", f"ort_fp16_{n*16}.png"))


if __name__ == "__main__":
    main()

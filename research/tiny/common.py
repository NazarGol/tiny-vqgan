"""Shared code for the tiny VQGAN engine research: teacher loading, on-the-fly token-grid data, student nets, export.

Everything the browser runs later is restricted to: embedding lookup, 3x3 conv (zero pad), ReLU, nearest 2x upsample,
residual add, 1x1 conv, average pooling, so each module here maps to one WebGL2 fragment-shader pass.
"""
import json, math, os, sys, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.normpath(os.path.join(HERE, "..", ".."))
EXPORT = os.path.join(WEB, "export")
N_EMBED, BLANK = 16384, 6328


# ----------------------------------------------------------------------------- teacher
def load_teacher(device):
    sys.path.insert(0, EXPORT); sys.path.insert(0, os.path.join(EXPORT, "taming-transformers"))
    import export_decoder as E
    cfg, sd = E.load_cfg_and_sd()
    dec = E.build_decoder(cfg, sd).to(device).eval()
    for p in dec.parameters(): p.requires_grad_(False)
    return dec, dec.embedding.weight.detach().float().cpu()   # [16384, 256] codebook


# ----------------------------------------------------------------------------- data
class GridSampler:
    """Token grids like the app produces: bank photos / paintings (crops), random, mosaics, strokes on a canvas, mutations."""
    def __init__(self, grids16, grids32, seed=0, blank=BLANK, n_embed=N_EMBED):
        self.g16 = grids16.astype(np.int64).reshape(-1, 16, 16)
        self.g32 = grids32.astype(np.int64).reshape(-1, 32, 32) if grids32 is not None and len(grids32) else None
        self.rng = np.random.default_rng(seed); self.blank = blank; self.n = n_embed
        self.modes = [("crop32", 0.25), ("photo16", 0.20), ("random", 0.05), ("mosaic", 0.12), ("stroke_blank", 0.18), ("stroke_canvas", 0.20)]
        if self.g32 is None: self.modes = [(m, w) for m, w in self.modes if m != "crop32"]
        self.mode_p = np.array([w for _, w in self.modes]); self.mode_p /= self.mode_p.sum()

    # --- sources
    def source(self, S):
        """Any S x S grid taken from a real photo/painting grid (crop or nearest-neighbour stretch)."""
        r = self.rng
        if self.g32 is not None and (S > 16 or r.random() < 0.5):
            g = self.g32[r.integers(len(self.g32))]
        else:
            g = self.g16[r.integers(len(self.g16))]
        return self.fit(g, S)

    def fit(self, g, S):
        r = self.rng; H = g.shape[0]
        if H >= S:
            y, x = r.integers(H - S + 1), r.integers(H - S + 1); return g[y:y + S, x:x + S].copy()
        idx = (np.arange(S) * H // S); return g[np.ix_(idx, idx)].copy()

    def random(self, S): return self.rng.integers(0, self.n, (S, S))

    def mosaic(self, S, patch=None):
        r = self.rng; patch = patch or int(r.choice([2, 3, 4, 4, 6]))
        srcs = [self.source(S) for _ in range(int(r.integers(2, 5)))]
        out = np.empty((S, S), np.int64)
        for by in range(0, S, patch):
            for bx in range(0, S, patch):
                src = srcs[r.integers(len(srcs))]
                out[by:by + patch, bx:bx + patch] = src[by:by + patch, bx:bx + patch]
        # a few cells from anywhere (palette-like proposals)
        k = int(r.integers(0, max(1, S * S // 8)))
        if k: out.flat[r.integers(0, S * S, k)] = r.integers(0, self.n, k)
        return out

    def blob(self, S):
        """Irregular mask like the app's lasso: an ellipse perturbed by low-frequency noise."""
        r = self.rng
        cy, cx = r.uniform(0.3, 0.7, 2) * S; ry, rx = r.uniform(0.2, 0.5, 2) * S
        yy, xx = np.mgrid[0:S, 0:S]
        d = ((yy + 0.5 - cy) / ry) ** 2 + ((xx + 0.5 - cx) / rx) ** 2
        noise = r.normal(0, 0.35, (3, 3)); noise = np.kron(noise, np.ones((S, S)))[:S, :S] if S >= 3 else 0
        m = d + (noise if np.ndim(noise) else 0) < 1.0
        if m.sum() < 2: m[int(cy), int(cx)] = True
        return m

    def stroke(self, S, on_blank):
        r = self.rng
        if on_blank:
            bg = np.full((S, S), self.blank if r.random() < 0.7 else int(r.integers(self.n)), np.int64)
        else:
            bg = self.source(S) if r.random() < 0.7 else self.mosaic(S)
        m = self.blob(S)
        fill = self.mosaic(S) if r.random() < 0.5 else self.source(S)
        out = bg.copy(); out[m] = fill[m]
        # edge band grown from the surrounding canvas, like the app's seeds
        grow = r.uniform(0.3, 0.9)
        ys, xs = np.nonzero(m)
        for y, x in zip(ys, xs):
            nb = [(y + dy, x + dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= y + dy < S and 0 <= x + dx < S and not m[y + dy, x + dx]]
            if nb and r.random() < grow:
                ny, nx = nb[r.integers(len(nb))]; out[y, x] = bg[ny, nx]
        return out

    def mutate(self, g, frac=None):
        """Search-style mutations on a fraction of the cells: random token, neighbour copy, block move, swap, source patch."""
        r = self.rng; S = g.shape[0]; g = g.copy()
        n = max(1, int(round(S * S * (frac if frac is not None else r.uniform(0.02, 0.35)))))
        src = self.source(S) if r.random() < 0.6 else None
        for _ in range(n):
            y, x = r.integers(S), r.integers(S); u = r.random()
            if u < 0.30: g[y, x] = r.integers(self.n)
            elif u < 0.55:
                dy, dx = [(0, 1), (0, -1), (1, 0), (-1, 0)][r.integers(4)]
                g[y, x] = g[min(S - 1, max(0, y + dy)), min(S - 1, max(0, x + dx))]
            elif u < 0.70:
                k = int(r.integers(2, 4)); sy, sx = r.integers(S), r.integers(S)
                h, w = min(k, S - y, S - sy), min(k, S - x, S - sx)
                g[y:y + h, x:x + w] = g[sy:sy + h, sx:sx + w].copy()
            elif u < 0.80:
                y2, x2 = r.integers(S), r.integers(S); g[y, x], g[y2, x2] = g[y2, x2], g[y, x]
            elif src is not None:
                k = int(r.integers(1, 4)); h, w = min(k, S - y), min(k, S - x)
                g[y:y + h, x:x + w] = src[y:y + h, x:x + w]
        return g

    def sample(self, S, mode=None):
        r = self.rng
        mode = mode or self.modes[r.choice(len(self.modes), p=self.mode_p)][0]
        if mode == "crop32": g = self.fit(self.g32[r.integers(len(self.g32))], S)
        elif mode == "photo16": g = self.fit(self.g16[r.integers(len(self.g16))], S)
        elif mode == "random": g = self.random(S)
        elif mode == "mosaic": g = self.mosaic(S)
        elif mode == "stroke_blank": g = self.stroke(S, True)
        elif mode == "stroke_canvas": g = self.stroke(S, False)
        else: raise ValueError(mode)
        if mode != "random" and r.random() < 0.5: g = self.mutate(g)
        return g

    def batch(self, B, S):
        return torch.from_numpy(np.stack([self.sample(S) for _ in range(B)]))

    @staticmethod
    def pick_size(rng, sizes=((8, 0.10), (10, 0.10), (12, 0.15), (16, 0.50), (20, 0.10), (24, 0.05))):
        p = np.array([w for _, w in sizes]); return int(sizes[rng.choice(len(sizes), p=p / p.sum())][0])


def load_grids(data_dir):
    """16x16 grids: bank_photos (6500) + paintings (1500) + bank (2000, overlaps); 32x32: COCO + CelebA at 512 px."""
    g16, g32 = [], []
    for name in ("bank_photos_tokens_16.u16", "bank_tokens_16.u16"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p): g16.append(np.fromfile(p, np.uint16).reshape(-1, 256))
    p = os.path.join(data_dir, "paintings_tokens16.npy")
    if os.path.exists(p): g16.append(np.load(p))
    for name in ("tokens32_coco.npy", "tokens32_celeba.npy"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p): g32.append(np.load(p))
    g16 = np.concatenate(g16) if g16 else np.zeros((0, 256), np.uint16)
    g32 = np.concatenate(g32) if g32 else None
    return g16, g32


# ----------------------------------------------------------------------------- student decoder
class Block(nn.Module):
    """conv-relu-conv + skip, relu after the sum (TAESD-like, two convs so the high-res stages stay cheap)."""
    def __init__(self, c):
        super().__init__(); self.c1 = nn.Conv2d(c, c, 3, padding=1); self.c2 = nn.Conv2d(c, c, 3, padding=1)
    def forward(self, x): return F.relu(x + self.c2(F.relu(self.c1(x))))


class TinyDec(nn.Module):
    """tokens [B,H,W] -> image [B,3,16H,16W] in 0..1 (raw output, clamp at use). widths: channels at 1/16, 1/8, 1/4, 1/2, 1 res."""
    def __init__(self, widths=(64, 64, 64, 32, 16), blocks=(2, 2, 2, 1, 1), codebook=None, n_embed=N_EMBED):
        super().__init__()
        self.widths, self.blocks = list(widths), list(blocks)
        self.emb = nn.Embedding(n_embed, widths[0])
        if codebook is not None:   # init from the PCA of the real codebook so tokens start out meaningfully placed
            c = codebook - codebook.mean(0, keepdim=True); u, s, v = torch.pca_lowrank(c, q=widths[0], center=False)
            proj = c @ v[:, :widths[0]]; proj = proj / proj.std()
            with torch.no_grad(): self.emb.weight.copy_(proj)
        self.stages = nn.ModuleList()
        cin = widths[0]
        for i, (c, nb) in enumerate(zip(widths, blocks)):
            layers = [nn.Conv2d(cin, c, 3, padding=1)] + [Block(c) for _ in range(nb)]
            self.stages.append(nn.Sequential(*layers)); cin = c
        self.out = nn.Conv2d(cin, 3, 3, padding=1)

    def forward(self, tokens):
        x = self.emb(tokens).permute(0, 3, 1, 2)
        for i, st in enumerate(self.stages):
            if i > 0: x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = st[0](x); x = F.relu(x)
            for b in st[1:]: x = b(x)
        return self.out(x)

    def n_params(self): return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------- student scorer (phase 2)
class TokenScorer(nn.Module):
    """tokens [B,H,W] (any H,W) -> unit CLIP-space embedding [B,512] of the decoded crop resized to 256 px.
    The embedding grid is resized to 16x16 (as the app resizes the crop) and read by a small conv net."""
    def __init__(self, c0=64, widths=(96, 128, 192, 256), out_dim=512, n_embed=N_EMBED, codebook=None):
        super().__init__()
        self.c0, self.widths = c0, list(widths)
        self.emb = nn.Embedding(n_embed, c0)
        if codebook is not None:
            c = codebook - codebook.mean(0, keepdim=True); u, s, v = torch.pca_lowrank(c, q=c0, center=False)
            proj = c @ v[:, :c0]; proj = proj / proj.std()
            with torch.no_grad(): self.emb.weight.copy_(proj)
        layers, cin = [], c0
        for i, c in enumerate(widths):   # 16 -> 16 -> 8 -> 4 -> 2
            layers.append(nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, stride=1 if i == 0 else 2), nn.ReLU()))
            cin = c
        self.convs = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.Linear(cin * 4, 512), nn.ReLU(), nn.Linear(512, out_dim))

    def forward(self, tokens):
        x = self.emb(tokens).permute(0, 3, 1, 2)
        if x.shape[-1] != 16 or x.shape[-2] != 16: x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
        for c in self.convs: x = c(x)
        return F.normalize(self.head(x.flatten(1)), dim=-1)   # 2x2 x widths[-1] after three stride-2 convs

    def n_params(self): return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------- export
def pack_conv(w, b):
    """Conv weight [cout,cin,k,k] -> for each out group, in group, tap: a column-major 4x4 (GLSL mat4: m[j][i] = w[i][j])."""
    cout, cin, k, _ = w.shape; go, gi = cout // 4, cin // 4
    assert cout % 4 == 0 and cin % 4 == 0, (cout, cin)
    w = w.reshape(go, 4, gi, 4, k * k)                 # [go, i, gi, j, tap]
    w = w.permute(0, 2, 4, 3, 1).contiguous()          # [go, gi, tap, j, i]  -> flat index ((go*gi+gi)*taps+tap)*16 + j*4 + i
    return w.reshape(-1), b.reshape(go, 4).reshape(-1)


def export_tinydec(model, path_bin, path_json, meta=None):
    """Binary: little-endian fp16 arrays back to back; JSON: the layer list with offsets (in elements)."""
    model = model.eval().cpu(); arrays, layers, off = [], [], 0
    def add(name, t):
        nonlocal off; t = t.detach().float().reshape(-1).numpy().astype(np.float16); arrays.append(t); o = off; off += t.size; return o, t.size
    c0 = model.widths[0]
    o, n = add("emb", model.emb.weight)   # token-major: token t -> c0 floats
    emb = {"offset": o, "len": n, "n": model.emb.weight.shape[0], "c": c0}
    def conv_entry(name, conv, cin, cout, up, relu, residual=None):
        w, b = pack_conv(conv.weight.detach().float(), conv.bias.detach().float())
        ow, nw = add(name + ".w", torch.from_numpy(w.numpy() if hasattr(w, 'numpy') else w)); ob, nb = add(name + ".b", b)
        layers.append({"name": name, "cin": cin, "cout": cout, "k": conv.kernel_size[0], "up": up, "relu": relu, "residual": residual, "w": ow, "b": ob})
    cin = c0
    for i, st in enumerate(model.stages):
        c = model.widths[i]
        conv_entry(f"s{i}.in", st[0], cin, c, up=i > 0, relu=True); cin = c
        for j, blk in enumerate(st[1:]):
            conv_entry(f"s{i}.b{j}.c1", blk.c1, c, c, up=False, relu=True)
            conv_entry(f"s{i}.b{j}.c2", blk.c2, c, c, up=False, relu=True, residual="input")   # relu(x + conv(h)): residual = block input
    # the final 3-channel conv is padded to 4 output channels
    w = model.out.weight.detach().float(); b = model.out.bias.detach().float()
    w4 = torch.zeros(4, w.shape[1], 3, 3); w4[:3] = w; b4 = torch.zeros(4); b4[:3] = b
    wp, bp = pack_conv(w4, b4); ow, nw = add("out.w", wp); ob, nb = add("out.b", bp)
    layers.append({"name": "out", "cin": cin, "cout": 4, "k": 3, "up": False, "relu": False, "residual": None, "w": ow, "b": ob})
    data = np.concatenate(arrays)
    data.tofile(path_bin)
    manifest = {"type": "tinydec", "dtype": "f16", "widths": model.widths, "blocks": model.blocks, "emb": emb, "layers": layers, "elements": int(data.size), **(meta or {})}
    json.dump(manifest, open(path_json, "w"))
    return manifest


def export_scorer(model, path_bin, path_json, meta=None):
    model = model.eval().cpu(); arrays, layers, off = [], [], 0
    def add(t):
        nonlocal off; t = t.detach().float().reshape(-1).numpy().astype(np.float16); arrays.append(t); o = off; off += t.size; return o, t.size
    o, n = add(model.emb.weight); emb = {"offset": o, "len": n, "n": model.emb.weight.shape[0], "c": model.c0}
    cin = model.c0
    for i, seq in enumerate(model.convs):
        for j, conv in enumerate([seq[0], seq[2]]):
            w, b = pack_conv(conv.weight.detach().float(), conv.bias.detach().float())
            ow, _ = add(torch.from_numpy(w.numpy())); ob, _ = add(b)
            layers.append({"name": f"c{i}.{j}", "cin": conv.in_channels, "cout": conv.out_channels, "k": 3, "stride": conv.stride[0], "relu": True, "w": ow, "b": ob})
    fcs = []
    for lin in (model.head[0], model.head[2]):
        ow, _ = add(lin.weight.detach().float()); ob, _ = add(lin.bias.detach().float())
        fcs.append({"in": lin.in_features, "out": lin.out_features, "w": ow, "b": ob})
    data = np.concatenate(arrays); data.tofile(path_bin)
    manifest = {"type": "tokenscorer", "dtype": "f16", "c0": model.c0, "widths": model.widths, "emb": emb, "layers": layers, "fcs": fcs, "elements": int(data.size), **(meta or {})}
    json.dump(manifest, open(path_json, "w")); return manifest


def to_u8(img):  # [3,H,W] float 0..1 -> HWC uint8
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- student text encoder (phase 3)
class TextBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__(); self.ln1 = nn.LayerNorm(w); self.qkv = nn.Linear(w, 3 * w); self.proj = nn.Linear(w, w); self.ln2 = nn.LayerNorm(w)
        self.fc1 = nn.Linear(w, 4 * w); self.fc2 = nn.Linear(4 * w, w); self.heads = heads
    def forward(self, x, mask):
        B, T, W = x.shape; h = self.heads
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, h, W // h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(W // h) + mask
        x = x + self.proj((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, W))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class TinyText(nn.Module):
    """CLIP-style causal transformer: ids [B,77] (BOS … EOS, 0-padded) -> unit embedding [B,512], pooled at the EOS position (argmax id)."""
    def __init__(self, vocab=49408, ctx=77, emb_dim=32, width=256, layers=4, heads=4, out_dim=512):
        super().__init__(); self.cfg = dict(vocab=vocab, ctx=ctx, emb_dim=emb_dim, width=width, layers=layers, heads=heads, out_dim=out_dim)
        self.tok = nn.Embedding(vocab, emb_dim); self.proj_in = nn.Linear(emb_dim, width); self.pos = nn.Parameter(torch.zeros(ctx, width))
        self.blocks = nn.ModuleList([TextBlock(width, heads) for _ in range(layers)]); self.ln_f = nn.LayerNorm(width); self.out = nn.Linear(width, out_dim, bias=False)
        nn.init.normal_(self.pos, std=0.01); nn.init.normal_(self.tok.weight, std=0.02)
        self.register_buffer("mask", torch.full((ctx, ctx), float("-inf")).triu(1), persistent=False)
    def forward(self, ids):
        T = ids.shape[1]; x = self.proj_in(self.tok(ids)) + self.pos[:T]
        for b in self.blocks: x = b(x, self.mask[:T, :T])
        x = self.ln_f(x); pooled = x[torch.arange(x.shape[0]), ids.argmax(-1)]
        return F.normalize(self.out(pooled), dim=-1)
    def n_params(self): return sum(p.numel() for p in self.parameters())


def export_text(model, path_bin, path_json, meta=None):
    model = model.eval().cpu(); arrays, off, entries = [], 0, {}
    def add(name, t):
        nonlocal off; a = t.detach().float().reshape(-1).numpy().astype(np.float16); arrays.append(a); entries[name] = {"offset": off, "shape": list(t.shape)}; off += a.size
    add("tok", model.tok.weight); add("proj_in.w", model.proj_in.weight); add("proj_in.b", model.proj_in.bias); add("pos", model.pos)
    for i, b in enumerate(model.blocks):
        for n, t in (("ln1.g", b.ln1.weight), ("ln1.b", b.ln1.bias), ("qkv.w", b.qkv.weight), ("qkv.b", b.qkv.bias), ("proj.w", b.proj.weight), ("proj.b", b.proj.bias),
                     ("ln2.g", b.ln2.weight), ("ln2.b", b.ln2.bias), ("fc1.w", b.fc1.weight), ("fc1.b", b.fc1.bias), ("fc2.w", b.fc2.weight), ("fc2.b", b.fc2.bias)): add(f"b{i}.{n}", t)
    add("ln_f.g", model.ln_f.weight); add("ln_f.b", model.ln_f.bias); add("out.w", model.out.weight)
    data = np.concatenate(arrays); data.tofile(path_bin)
    json.dump({"type": "tinytext", "dtype": "f16", **model.cfg, "tensors": entries, "elements": int(data.size), **(meta or {})}, open(path_json, "w"))


# ----------------------------------------------------------------------------- one-pass starting model (phase 5)
class OnePass(nn.Module):
    """(context tokens [B,S,S], mask [B,S,S] bool, text [B,512]) -> per-cell 64-d vectors [B,c0,S,S]; token logits = h · table^T.
    Shader-friendly: table lookup, 3×3 convs with residual + a per-channel bias from the text (FiLM), ReLU, 1×1 out conv."""
    def __init__(self, c0=64, width=96, layers=6, n_embed=N_EMBED, text_dim=512, table=None):
        super().__init__(); self.cfg = dict(c0=c0, width=width, layers=layers, n_embed=n_embed, text_dim=text_dim)
        self.emb = nn.Embedding(n_embed, c0); self.out_table = nn.Embedding(n_embed, c0)
        if table is not None:
            with torch.no_grad(): self.emb.weight.copy_(table); self.out_table.weight.copy_(table)
        self.inp = nn.Conv2d(c0 + 1, width, 3, padding=1); self.convs = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(layers)])
        self.film = nn.Linear(text_dim, width * (layers + 1)); self.out = nn.Conv2d(width, c0, 1)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
    def forward(self, tokens, mask, text):
        B = tokens.shape[0]; W = self.cfg["width"]
        f = self.film(text).reshape(B, -1, W)                                        # [B, layers+1, W]
        x = torch.cat([self.emb(tokens).permute(0, 3, 1, 2), mask[:, None].float()], 1)
        x = F.relu(self.inp(x) + f[:, 0, :, None, None])
        for i, c in enumerate(self.convs): x = F.relu(x + c(x) + f[:, i + 1, :, None, None])
        return self.out(x)
    def logits(self, h): return torch.einsum("bchw,nc->bhwn", h.float(), self.out_table.weight.float())
    def predict(self, tokens, mask, text):   # argmax tokens on masked cells, context elsewhere
        h = self.forward(tokens, mask, text); best = self.logits(h).argmax(-1)
        return torch.where(mask, best, tokens)
    def n_params(self): return sum(p.numel() for p in self.parameters())


def export_onepass(model, path_bin, path_json, meta=None):
    model = model.eval().cpu(); arrays, off, entries = [], 0, {}
    def add(name, t):
        nonlocal off; a = t.detach().float().reshape(-1).numpy().astype(np.float16); arrays.append(a); entries[name] = {"offset": off, "shape": list(t.shape)}; off += a.size
    add("emb", model.emb.weight); add("out_table", model.out_table.weight)
    layers = []
    cin = model.cfg["c0"] + 4;   # cin = c0+1 is not a multiple of 4: pad to c0+4 (mask + 3 zero channels)
    w4 = torch.zeros(model.inp.out_channels, cin, 3, 3); w4[:, :model.cfg["c0"] + 1] = model.inp.weight.detach().float(); wp, bp = pack_conv(w4, model.inp.bias.detach().float())
    add("inp.w", torch.from_numpy(wp.numpy())); add("inp.b", bp)
    layers.append({"name": "inp", "cin": cin, "cout": model.inp.out_channels, "k": 3, "relu": True, "residual": None, "w": entries["inp.w"]["offset"], "b": entries["inp.b"]["offset"]})
    for i, c in enumerate(model.convs):
        wp, bp = pack_conv(c.weight.detach().float(), c.bias.detach().float()); add(f"c{i}.w", torch.from_numpy(wp.numpy())); add(f"c{i}.b", bp)
        layers.append({"name": f"c{i}", "cin": c.in_channels, "cout": c.out_channels, "k": 3, "relu": True, "residual": "input", "w": entries[f"c{i}.w"]["offset"], "b": entries[f"c{i}.b"]["offset"]})
    w1 = torch.zeros(model.out.out_channels, model.out.in_channels, 3, 3); w1[:, :, 1, 1] = model.out.weight.detach().float()[:, :, 0, 0]
    wp, bp = pack_conv(w1, model.out.bias.detach().float()); add("out.w", torch.from_numpy(wp.numpy())); add("out.b", bp)
    layers.append({"name": "head", "cin": model.out.in_channels, "cout": model.out.out_channels, "k": 3, "relu": False, "residual": None, "w": entries["out.w"]["offset"], "b": entries["out.b"]["offset"]})
    add("film.w", model.film.weight); add("film.b", model.film.bias)
    data = np.concatenate(arrays); data.tofile(path_bin)
    json.dump({"type": "onepass", "dtype": "f16", **model.cfg, "tensors": entries, "layers": layers, "elements": int(data.size), **(meta or {})}, open(path_json, "w"))

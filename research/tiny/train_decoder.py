"""Distil the VQGAN f16 decoder into tiny conv decoders (one or more variants trained on the same teacher batches).

Data: token grids generated on the fly (common.GridSampler); targets: the original decoder (fp16 on a second GPU when there is one).
Loss: L1 + lpips_w * LPIPS(vgg). Time-based cosine schedule, EMA weights exported to the browser format.
Usage: python train_decoder.py --data DIR --out DIR --hours 3 [--variants "A:64,64,64,32,16:2,2,2,1,1;B:64,64,48,24,12:2,2,2,1,1"]
"""
import argparse, json, os, sys, threading, queue, time, math, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--hours", type=float, default=3.0); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--variants", default="A:64,64,64,32,16:2,2,2,1,1;B:64,64,48,24,12:2,2,2,1,1")
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--lpips", type=float, default=1.0); ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--ckpt-every", type=float, default=900); ap.add_argument("--eval-every", type=float, default=1800)
ap.add_argument("--smoke", action="store_true", help="a few tiny steps on the local machine"); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--clip", default="", help="MobileCLIP vision ONNX for the CLIP-agreement metric (optional)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
sdev = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tdev = torch.device("cuda:1") if torch.cuda.device_count() > 1 else sdev
print("student on", sdev, "teacher on", tdev, flush=True)
if args.smoke: args.hours, args.batch, args.ckpt_every, args.eval_every = 0.02, 2, 1e9, 1e9

# ---- teacher + data
teacher, codebook = C.load_teacher(tdev)
g16, g32 = C.load_grids(args.data); print(f"grids: {len(g16)} x16, {0 if g32 is None else len(g32)} x32", flush=True)
sampler = C.GridSampler(g16, g32, seed=int(time.time()) % 100000)
amp = tdev.type == "cuda"

@torch.no_grad()
def teach(tokens):
    with torch.autocast(tdev.type, dtype=torch.float16, enabled=amp):
        return teacher(tokens.to(tdev)).float().clamp(0, 1)

q = queue.Queue(maxsize=6); stop = False
def producer():
    rng = np.random.default_rng(12345)
    while not stop:
        S = 8 if args.smoke else C.GridSampler.pick_size(rng)
        tok = sampler.batch(args.batch, S)
        tgt = teach(tok)
        if tdev != sdev: tgt = tgt.to(sdev, non_blocking=True)
        q.put((tok.to(sdev), tgt))
th = threading.Thread(target=producer, daemon=True); th.start()

# ---- students
variants = {}
for spec in args.variants.split(";"):
    name, w, b = spec.split(":"); widths = tuple(int(x) for x in w.split(",")); blocks = tuple(int(x) for x in b.split(","))
    m = C.TinyDec(widths, blocks, codebook=codebook).to(sdev)
    ema = copy.deepcopy(m).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    variants[name] = dict(model=m, ema=ema, opt=opt, widths=widths, blocks=blocks)
    print(f"variant {name}: widths {widths} blocks {blocks} params {m.n_params()/1e6:.2f} M -> fp16 {m.n_params()*2/2**20:.1f} MiB", flush=True)
import lpips
lp = lpips.LPIPS(net="vgg", verbose=False).to(sdev).eval()
for p in lp.parameters(): p.requires_grad_(False)
scaler = torch.amp.GradScaler(enabled=sdev.type == "cuda")

step, elapsed0 = 0, 0.0
ck = os.path.join(args.out, "ckpt.pt")
if os.path.exists(ck):
    st = torch.load(ck, map_location=sdev); step, elapsed0 = st["step"], st["elapsed"]
    for name, v in variants.items():
        if name in st["variants"]: v["model"].load_state_dict(st["variants"][name]["model"]); v["ema"].load_state_dict(st["variants"][name]["ema"]); v["opt"].load_state_dict(st["variants"][name]["opt"])
    print(f"resumed at step {step}, {elapsed0/3600:.2f} h", flush=True)

def save_ckpt():
    torch.save({"step": step, "elapsed": elapsed(), "variants": {n: {"model": v["model"].state_dict(), "ema": v["ema"].state_dict(), "opt": v["opt"].state_dict()} for n, v in variants.items()}}, ck + ".tmp"); os.replace(ck + ".tmp", ck)

# ---- eval set (fixed): 30 grids: modes x sizes, plus real strokes from the app when data/strokes.json exists
def eval_set():
    rs = C.GridSampler(g16, g32, seed=1234); out = []
    for mode in ("photo16", "crop32", "mosaic", "stroke_blank", "stroke_canvas", "random"):
        for S in (16, 16, 12, 10):
            if mode == "random" and S != 16: continue
            out.append((f"{mode}{S}", rs.sample(S, mode)))
    p = os.path.join(args.data, "strokes.json")
    if os.path.exists(p):
        import base64
        for i, s in enumerate(json.load(open(p))[:12]):
            u = np.frombuffer(base64.b64decode(s["tokens"]), np.uint16).astype(np.int64).reshape(s["crop"]["h"], s["crop"]["w"])
            out.append((f"stroke{i}:{s['text'][:18]}", u))
    return out
EVAL = eval_set()
clip_sess = None
if args.clip and os.path.exists(args.clip):
    import onnxruntime as ort
    clip_sess = ort.InferenceSession(args.clip, providers=["CPUExecutionProvider"])
def clip_embed(img):   # [3,H,W] 0..1 -> unit 512
    x = F.interpolate(img[None], size=(256, 256), mode="bilinear", align_corners=False, antialias=True).cpu().numpy().astype(np.float32)
    e = clip_sess.run(None, {"pixel_values": x})[0][0]; return e / (np.linalg.norm(e) + 1e-8)

@torch.no_grad()
def evaluate(final=False):
    from PIL import Image, ImageDraw
    res = {n: dict(l1=[], lpips=[], psnr=[], clipcos=[]) for n in variants}
    tiles = []
    for label, g in EVAL:
        tok = torch.from_numpy(g)[None]; tgt = teach(tok).to(sdev)
        row = [C.to_u8(tgt[0])]; ce = clip_embed(tgt[0]) if clip_sess else None
        for n, v in variants.items():
            with torch.autocast(sdev.type, dtype=torch.float16, enabled=sdev.type == "cuda"):
                pred = v["ema"](tok.to(sdev)).float().clamp(0, 1)
            res[n]["l1"].append((pred - tgt).abs().mean().item()); res[n]["psnr"].append(-10 * math.log10(((pred - tgt) ** 2).mean().item() + 1e-10))
            res[n]["lpips"].append(lp(pred * 2 - 1, tgt * 2 - 1).mean().item())
            if ce is not None: res[n]["clipcos"].append(float(np.dot(ce, clip_embed(pred[0]))))
            row.append(C.to_u8(pred[0]))
        tiles.append((label, row))
    summary = {n: {k: float(np.mean(v)) for k, v in r.items() if v} for n, r in res.items()}
    print(f"[eval step {step}] " + " | ".join(f"{n}: L1 {s['l1']:.4f} LPIPS {s['lpips']:.4f} PSNR {s['psnr']:.1f}" + (f" CLIPcos {s['clipcos']:.4f}" if 'clipcos' in s else "") for n, s in summary.items()), flush=True)
    # contact sheet: each row = original | variants, tiles scaled to 160 px
    T = 160; cols = 1 + len(variants); W = cols * T + 8; H = len(tiles) * (T + 14)
    sheet = Image.new("RGB", (W, H), (40, 40, 40)); d = ImageDraw.Draw(sheet)
    for r, (label, row) in enumerate(tiles):
        y = r * (T + 14); d.text((4, y + 1), label, fill=(230, 230, 230))
        for c, im in enumerate(row): sheet.paste(Image.fromarray(im).resize((T, T), Image.LANCZOS), (c * T + 4, y + 12))
    d.text((4 + T, 1), "  ".join(["original"] + [f"tiny {n}" for n in variants]), fill=(255, 200, 255))
    sheet.save(os.path.join(args.out, "eval_sheet.png"))
    if final:   # full-resolution pairs for the first 8 grids
        for r, (label, row) in enumerate(tiles[:8] + [t for t in tiles if t[0].startswith("stroke")][:6]):
            pair = Image.new("RGB", (sum(im.shape[1] for im in row) + 4 * (len(row) - 1), row[0].shape[0]), (40, 40, 40)); x = 0
            for im in row: pair.paste(Image.fromarray(im), (x, 0)); x += im.shape[1] + 4
            pair.save(os.path.join(args.out, f"pair_{r:02d}_{label.split(':')[0]}.png"))
    json.dump({"step": step, "elapsed_h": elapsed() / 3600, "metrics": summary}, open(os.path.join(args.out, "eval.json"), "w"), indent=1)
    return summary

# ---- train
t_start = time.time(); elapsed = lambda: elapsed0 + time.time() - t_start
budget = args.hours * 3600; warm = 200
last_ck, last_ev, last_log = time.time(), time.time(), time.time(); seen = 0; acc = {n: dict(l1=0.0, lp=0.0, k=0) for n in variants}
while elapsed() < budget and not (args.max_steps and step >= args.max_steps):
    tok, tgt = q.get()
    frac = min(1.0, elapsed() / budget); lr = args.lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05) * min(1.0, (step + 1) / warm)
    for n, v in variants.items():
        for g in v["opt"].param_groups: g["lr"] = lr
        with torch.autocast(sdev.type, dtype=torch.float16, enabled=sdev.type == "cuda"):
            pred = v["model"](tok)
            l1 = (pred.float() - tgt).abs().mean()
            lpv = lp(pred.float().clamp(0, 1) * 2 - 1, tgt * 2 - 1).mean() if args.lpips > 0 else torch.zeros((), device=sdev)
            loss = l1 + args.lpips * lpv
        v["opt"].zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(v["opt"]); torch.nn.utils.clip_grad_norm_(v["model"].parameters(), 1.0); scaler.step(v["opt"])
        with torch.no_grad():
            d = args.ema if step > warm else 0.0
            for pe, pm in zip(v["ema"].parameters(), v["model"].parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        acc[n]["l1"] += l1.item(); acc[n]["lp"] += lpv.item(); acc[n]["k"] += 1
    scaler.update(); step += 1; seen += tok.shape[0]
    if time.time() - last_log > 60 or args.smoke:
        dt = time.time() - last_log; print(f"step {step} {elapsed()/3600:.2f}h lr {lr:.2e} S={tok.shape[1]} " + " | ".join(f"{n}: L1 {a['l1']/max(1,a['k']):.4f} LPIPS {a['lp']/max(1,a['k']):.4f}" for n, a in acc.items()) + f"  {seen/dt:.1f} img/s q={q.qsize()}", flush=True)
        last_log = time.time(); seen = 0; acc = {n: dict(l1=0.0, lp=0.0, k=0) for n in variants}
    if time.time() - last_ck > args.ckpt_every: save_ckpt(); last_ck = time.time()
    if time.time() - last_ev > args.eval_every: evaluate(); last_ev = time.time()
stop = True
save_ckpt(); summary = evaluate(final=True)
for n, v in variants.items():
    man = C.export_tinydec(copy.deepcopy(v["ema"]).cpu(), os.path.join(args.out, f"tiny_decoder_{n}.bin"), os.path.join(args.out, f"tiny_decoder_{n}.json"), meta={"variant": n, "step": step, "metrics": summary[n]})
    print(f"exported {n}: {os.path.getsize(os.path.join(args.out, f'tiny_decoder_{n}.bin'))/2**20:.2f} MiB, {len(man['layers'])} conv layers", flush=True)
print("done", flush=True)

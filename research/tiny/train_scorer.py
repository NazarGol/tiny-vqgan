"""Token-space CLIP scorer: tokens -> MobileCLIP-S0 image embedding of the decoded crop (resized to 256 px), distilled from
teacher decoder + the real (ONNX) MobileCLIP vision tower. Variants trained on the same batches.
Usage: python train_scorer.py --data DIR --out DIR --hours 2.5 --clip-vision vision_model.onnx --clip-text text_model.onnx --tokenizer tokenizer.json
"""
import argparse, json, os, sys, threading, queue, time, math, copy, base64
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--hours", type=float, default=2.5); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--variants", default="S:64:64,96,128,192;M:64:96,128,192,256")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--pairs", type=float, default=0.4, help="fraction of each batch that are (grid, small mutation) pairs")
ap.add_argument("--pair-w", type=float, default=10.0); ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--ckpt-every", type=float, default=900); ap.add_argument("--eval-every", type=float, default=1800)
ap.add_argument("--clip-vision", required=True); ap.add_argument("--clip-text", default=""); ap.add_argument("--tokenizer", default="")
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--extra-npz", default="", help="one-pass data npz: its search results/contexts are mixed into the batches (hard negatives for a second round)")
ap.add_argument("--extra-frac", type=float, default=0.3)
ap.add_argument("--workers", type=int, default=3, help="token-grid producer processes")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
sdev = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tdev = torch.device("cuda:1") if torch.cuda.device_count() > 1 else sdev
if args.smoke: args.hours, args.batch, args.ckpt_every, args.eval_every = 0.02, 4, 1e9, 1e9
print("student on", sdev, "teacher on", tdev, flush=True)

import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = max(2, (os.cpu_count() or 4) - 1)
providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
vis = ort.InferenceSession(args.clip_vision, so, providers=providers); print("CLIP vision providers:", vis.get_providers(), flush=True)
def clip_images(img):   # torch [B,3,H,W] 0..1 (any device) -> unit embeddings np [B,512]
    x = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False, antialias=True).float().cpu().numpy()
    e = vis.run(None, {"pixel_values": np.ascontiguousarray(x, dtype=np.float32)})[0]
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

teacher, codebook = C.load_teacher(tdev)
g16, g32 = C.load_grids(args.data); print(f"grids: {len(g16)} x16, {0 if g32 is None else len(g32)} x32", flush=True)
sampler = C.GridSampler(g16, g32, seed=int(time.time()) % 100000)
amp = tdev.type == "cuda"
SIZES = ((4, 0.05), (6, 0.10), (8, 0.15), (10, 0.15), (12, 0.20), (14, 0.10), (16, 0.15), (20, 0.10))

def teach(tokens): return C.teach_safe(teacher, tokens, tdev, amp)

EXTRA = {}
if args.extra_npz and os.path.exists(args.extra_npz):
    D = np.load(args.extra_npz, allow_pickle=True)
    for k in D.files:
        if k.startswith("result_") or k.startswith("ctx_"): EXTRA.setdefault(int(k.split("_")[1]), []).append(D[k].astype(np.int64))
    EXTRA = {S: np.concatenate(v) for S, v in EXTRA.items()}; print("extra grids:", {S: len(v) for S, v in EXTRA.items()}, flush=True)
def make_batch(B, S, pairs):
    """Returns tokens [B,S,S] where the first 2*npairs rows are pairs (a, mutate(a, small))."""
    npairs = int(B * pairs) // 2; rows = []
    ex = EXTRA.get(S)
    def extra_or_sample():
        if ex is not None and np.random.random() < args.extra_frac: return ex[np.random.randint(len(ex))].copy()
        return sampler.sample(S)
    for _ in range(npairs):
        a = extra_or_sample(); rows += [a, sampler.mutate(a, frac=float(np.random.uniform(0.01, 0.12)))]
    while len(rows) < B: rows.append(extra_or_sample())
    return torch.from_numpy(np.stack(rows)), npairs

import multiprocessing as mp
def grid_worker(seed, out_q, B, pairs, smoke, data_dir, extra_npz, extra_frac):
    np.random.seed(seed); rng = np.random.default_rng(seed)
    global sampler, EXTRA, args
    g16_, g32_ = C.load_grids(data_dir); sampler = C.GridSampler(g16_, g32_, seed=seed)
    class A: pass
    args = A(); args.extra_frac = extra_frac
    EXTRA = {}
    if extra_npz and os.path.exists(extra_npz):
        D = np.load(extra_npz, allow_pickle=True)
        for k in D.files:
            if k.startswith("result_") or k.startswith("ctx_"): EXTRA.setdefault(int(k.split("_")[1]), []).append(D[k].astype(np.int64))
        EXTRA = {S: np.concatenate(v) for S, v in EXTRA.items()}
    while True:
        S = 8 if smoke else C.GridSampler.pick_size(rng, SIZES)
        tok, npairs = make_batch(B, S, pairs); out_q.put((tok.numpy(), npairs))
q = queue.Queue(maxsize=4); stop = False
ctx_mp = mp.get_context("fork"); grid_q = ctx_mp.Queue(maxsize=16)
workers = [ctx_mp.Process(target=grid_worker, args=(1000 + i, grid_q, args.batch, args.pairs, args.smoke, args.data, args.extra_npz, args.extra_frac), daemon=True) for i in range(args.workers)]
for w in workers: w.start()
def producer():
    while not stop:
        tok_np, npairs = grid_q.get(); tok = torch.from_numpy(tok_np)
        img = teach(tok); emb = torch.from_numpy(clip_images(img))
        if not np.isfinite(emb.numpy()).all(): print("non-finite CLIP embedding, batch dropped", flush=True); continue
        q.put((tok.to(sdev), emb.to(sdev), npairs))
th = threading.Thread(target=producer, daemon=True); th.start()

variants = {}
for spec in args.variants.split(";"):
    name, c0, w = spec.split(":"); widths = tuple(int(x) for x in w.split(","))
    m = C.TokenScorer(c0=int(c0), widths=widths, codebook=codebook).to(sdev); ema = copy.deepcopy(m).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    variants[name] = dict(model=m, ema=ema, opt=opt)
    print(f"variant {name}: c0 {c0} widths {widths} params {m.n_params()/1e6:.2f} M -> fp16 {m.n_params()*2/2**20:.1f} MiB", flush=True)
step, elapsed0 = 0, 0.0; ck = os.path.join(args.out, "ckpt.pt")
if os.path.exists(ck):
    st = torch.load(ck, map_location=sdev); step, elapsed0 = st["step"], st["elapsed"]
    for name, v in variants.items():
        if name in st["variants"]: v["model"].load_state_dict(st["variants"][name]["model"]); v["ema"].load_state_dict(st["variants"][name]["ema"]); v["opt"].load_state_dict(st["variants"][name]["opt"])
    print(f"resumed at step {step}", flush=True)
def save_ckpt():
    torch.save({"step": step, "elapsed": elapsed(), "variants": {n: {"model": v["model"].state_dict(), "ema": v["ema"].state_dict(), "opt": v["opt"].state_dict()} for n, v in variants.items()}}, ck + ".tmp"); os.replace(ck + ".tmp", ck)

# ---- eval: held-out grids + real strokes, 40 prompts, pairs
PROMPTS = ['a face', 'a red forest', 'the sea at night', 'a city street', 'green hills under a blue sky', 'a cat', 'deadline on Friday', "I miss my grandmother's kitchen", 'the smell of rain', 'a lighthouse in a storm', 'a bowl of oranges', 'snow on a mountain', 'a crowded market', 'a sleeping dog', 'fire', 'a glass of water on a table', 'we should talk about the budget', 'a yellow bicycle', 'an old library', "a child's drawing of a house", 'sunset over the plains', 'a portrait of a woman in blue', 'a horse in a field', 'mushrooms in the forest', 'the moon over the sea', 'a broken clock', 'a train station in winter', 'flowers in a vase', 'I feel tired today', 'a river through a canyon', 'a dark room with one candle', 'the first day of school', 'a bird on a wire', 'an abandoned factory', 'waves crashing on rocks', 'a wedding', 'a bridge in fog', 'coffee in the morning', 'a map of an imaginary island', 'the sound of a cello']
text_embs = None
if args.clip_text and args.tokenizer and os.path.exists(args.clip_text):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer); ts = ort.InferenceSession(args.clip_text, providers=["CPUExecutionProvider"])
    embs = []
    for p in PROMPTS:
        ids = tk.encode(p).ids[:77]; assert ids[0] == 49406, ids[:3]
        ids = ids + [0] * (77 - len(ids))   # like lib/clip_tokenizer.js: BOS ... EOS, zero padded (pad "!" = 0)
        e = ts.run(None, {"input_ids": np.array([ids], np.int64)})[0][0]; embs.append(e / (np.linalg.norm(e) + 1e-8))
    text_embs = np.stack(embs); print("text embeddings for", len(PROMPTS), "prompts", flush=True)

def eval_sets():
    rs = C.GridSampler(g16, g32, seed=4321); grids, pairs = [], []
    for S in (6, 8, 10, 12, 14, 16, 20):
        for _ in range(24): grids.append(rs.sample(S))
        for _ in range(12): a = rs.sample(S); pairs.append((a, rs.mutate(a, frac=float(rs.rng.uniform(0.01, 0.12)))))
    strokes = []
    p = os.path.join(args.data, "strokes.json")
    if os.path.exists(p):
        for s in json.load(open(p)):
            strokes.append(np.frombuffer(base64.b64decode(s["tokens"]), np.uint16).astype(np.int64).reshape(s["crop"]["h"], s["crop"]["w"]))
    return grids, pairs, strokes
EV_GRIDS, EV_PAIRS, EV_STROKES = eval_sets()

@torch.no_grad()
def embed_real(grids):
    out = []
    for g in grids: out.append(clip_images(teach(torch.from_numpy(g)[None]))[0])
    return np.stack(out)
@torch.no_grad()
def embed_pred(model, grids):
    out = []
    for g in grids:
        out.append(model(torch.from_numpy(g)[None].to(sdev)).float().cpu().numpy()[0])
    return np.stack(out)
REAL = {"grids": embed_real(EV_GRIDS), "a": embed_real([a for a, b in EV_PAIRS]), "b": embed_real([b for a, b in EV_PAIRS]), "strokes": embed_real(EV_STROKES) if EV_STROKES else None}

def evaluate():
    from scipy.stats import spearmanr, pearsonr
    summary = {}
    for n, v in variants.items():
        m = v["ema"]; r = {}
        P = embed_pred(m, EV_GRIDS); r["cos_grids"] = float((P * REAL["grids"]).sum(1).mean())
        if REAL["strokes"] is not None: r["cos_strokes"] = float((embed_pred(m, EV_STROKES) * REAL["strokes"]).sum(1).mean())
        if text_embs is not None:
            sr, sp = REAL["grids"] @ text_embs.T, P @ text_embs.T   # [grids, prompts]
            r["pearson"] = float(np.mean([pearsonr(sr[:, j], sp[:, j])[0] for j in range(len(PROMPTS))]))
            r["spearman"] = float(np.mean([spearmanr(sr[:, j], sp[:, j])[0] for j in range(len(PROMPTS))]))
            Pa, Pb = embed_pred(m, [a for a, b in EV_PAIRS]), embed_pred(m, [b for a, b in EV_PAIRS])
            dr, dp = (REAL["b"] - REAL["a"]) @ text_embs.T, (Pb - Pa) @ text_embs.T
            r["pair_sign_agree"] = float(np.mean(np.sign(dr) == np.sign(dp))); r["pair_delta_corr"] = float(pearsonr(dr.reshape(-1), dp.reshape(-1))[0])
            big = np.abs(dr) > 0.005; r["pair_sign_agree_big"] = float(np.mean((np.sign(dr) == np.sign(dp))[big])) if big.any() else None
        summary[n] = r
    print(f"[eval step {step}] " + json.dumps(summary), flush=True)
    json.dump({"step": step, "elapsed_h": elapsed() / 3600, "metrics": summary}, open(os.path.join(args.out, "eval.json"), "w"), indent=1)
    return summary

t_start = time.time(); elapsed = lambda: elapsed0 + time.time() - t_start
budget = elapsed0 + args.hours * 3600; warm = 200 if step == 0 else 1; last_ck, last_ev, last_log = time.time(), time.time(), time.time(); seen = 0
acc = {n: dict(cos=0.0, pair=0.0, k=0) for n in variants}
while elapsed() < budget and not (args.max_steps and step >= args.max_steps):
    tok, emb, npairs = q.get()
    frac = min(1.0, (elapsed() - elapsed0) / (args.hours * 3600)); lr = args.lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05) * min(1.0, (step + 1) / warm)
    stepped = False
    for n, v in variants.items():
        for g in v["opt"].param_groups: g["lr"] = lr
        pred = v["model"](tok).float()   # fp32: the scorer has no normalisation layers and overflowed under fp16 autocast
        cos = 1 - (pred * emb).sum(1).mean()
        if npairs:
            dp = pred[1:2 * npairs:2] - pred[0:2 * npairs:2]; dr = emb[1:2 * npairs:2] - emb[0:2 * npairs:2]
            pair = (((dp - dr) ** 2).sum(1) / ((dr ** 2).sum(1) + 1e-3)).mean()   # bounded when a pair barely differs
        else: pair = torch.zeros((), device=sdev)
        loss = cos + args.pair_w * pair
        if not torch.isfinite(loss): print(f"step {step}: non-finite loss for {n}, batch skipped", flush=True); continue
        v["opt"].zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(v["model"].parameters(), 1.0); v["opt"].step(); stepped = True
        with torch.no_grad():
            d = args.ema if step > warm else 0.0
            for pe, pm in zip(v["ema"].parameters(), v["model"].parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        acc[n]["cos"] += cos.item(); acc[n]["pair"] += pair.item(); acc[n]["k"] += 1
    step += 1; seen += tok.shape[0]
    if step == 30 and not any(a_k for a_k in ([acc[n]["k"] for n in acc] if isinstance(acc[next(iter(acc))], dict) else [acc[n][1] for n in acc])): raise SystemExit("no finite loss in the first 30 steps: aborting instead of burning the budget")
    if time.time() - last_log > 60 or args.smoke:
        dt = time.time() - last_log; print(f"step {step} {elapsed()/3600:.2f}h lr {lr:.2e} S={tok.shape[1]} " + " | ".join(f"{n}: 1-cos {a['cos']/max(1,a['k']):.4f} pair {a['pair']/max(1,a['k']):.3f}" for n, a in acc.items()) + f"  {seen/dt:.1f} img/s q={q.qsize()}", flush=True)
        last_log = time.time(); seen = 0; acc = {n: dict(cos=0.0, pair=0.0, k=0) for n in variants}
    if time.time() - last_ck > args.ckpt_every: save_ckpt(); last_ck = time.time()
    if time.time() - last_ev > args.eval_every: evaluate(); last_ev = time.time()
stop = True; save_ckpt(); summary = evaluate()
for n, v in variants.items():
    man = C.export_scorer(copy.deepcopy(v["ema"]).cpu(), os.path.join(args.out, f"tiny_scorer_{n}.bin"), os.path.join(args.out, f"tiny_scorer_{n}.json"), meta={"variant": n, "step": step, "metrics": summary[n]})
    print(f"exported {n}: {os.path.getsize(os.path.join(args.out, f'tiny_scorer_{n}.bin'))/2**20:.2f} MiB", flush=True)
print("done", flush=True)
import sys as _sys; _sys.stdout.flush(); os._exit(0)   # daemon producer threads/processes holding CUDA can hang the interpreter at shutdown and keep the Kaggle session alive

"""Distil the MobileCLIP-S0 text tower (ONNX) into a small causal transformer (TinyText), same 512-d space.
Texts: COCO captions (train, val held out), painting prompts (web/export/paintings/prompts.py), metaphors, synthetic notes,
random token-id sequences. Loss 1-cos. Usage: python train_text.py --data DIR --out DIR --clip-text text_model.onnx --tokenizer tokenizer.json [--captions annotations_dir]
"""
import argparse, json, os, re, sys, threading, queue, time, math, copy, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "export", "paintings"))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--captions", default="", help="dir with captions_train2017.json / captions_val2017.json")
ap.add_argument("--hours", type=float, default=1.5); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--variants", default="S:32,192,4,4;M:32,256,4,4", help="name:emb_dim,width,layers,heads")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--clip-text", required=True); ap.add_argument("--tokenizer", required=True)
ap.add_argument("--ckpt-every", type=float, default=900); ap.add_argument("--eval-every", type=float, default=1200)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--max-steps", type=int, default=0)
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if args.smoke: args.hours, args.batch, args.ckpt_every, args.eval_every = 0.02, 16, 1e9, 1e9
print("device", dev, flush=True)

# ---- tokenizer + teacher
from tokenizers import Tokenizer
import onnxruntime as ort
tk = Tokenizer.from_file(args.tokenizer)
BOS, EOS, CTX = 49406, 49407, 77
def encode(text):
    ids = tk.encode(text).ids[:CTX]
    if ids[-1] != EOS: ids = ids[:CTX - 1] + [EOS]
    return ids + [0] * (CTX - len(ids))
so = ort.SessionOptions(); so.intra_op_num_threads = max(2, (os.cpu_count() or 4) - 1)
providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
ts = ort.InferenceSession(args.clip_text, so, providers=providers); print("teacher providers", ts.get_providers(), flush=True)
def teach(ids):   # np [B,77] -> unit [B,512]
    e = ts.run(None, {"input_ids": ids.astype(np.int64)})[0]; return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

# ---- texts
import texts as TX
texts = TX.all_texts(args.captions); print({k: len(v) for k, v in texts.items()}, flush=True)
MIX = [("coco", 0.55), ("prompts", 0.15), ("metaphors", 0.05), ("notes", 0.10), ("ids", 0.10), ("crop", 0.05)]
MIX = [(k, w) for k, w in MIX if k in ("ids", "crop") or texts.get(k)]
mix_p = np.array([w for _, w in MIX]); mix_p /= mix_p.sum()

def sample_ids(nrng):
    kind = MIX[nrng.choice(len(MIX), p=mix_p)][0]
    if kind == "ids":
        n = int(nrng.integers(1, 6)); ids = [BOS] + [int(x) for x in nrng.integers(1, 49406, n)] + [EOS]; return ids + [0] * (CTX - len(ids))
    if kind == "crop":
        src = texts["coco"] or texts["prompts"] or texts["notes"]; words = src[nrng.integers(len(src))].split(); k = int(nrng.integers(1, max(2, len(words) + 1)))
        return encode(" ".join(words[:k]))
    src = texts[kind]; return encode(src[nrng.integers(len(src))])

q = queue.Queue(maxsize=8); stop = False
def producer():
    nrng = np.random.default_rng(int(time.time()) % 100000)
    while not stop:
        ids = np.array([sample_ids(nrng) for _ in range(args.batch)], np.int64); emb = teach(ids)
        q.put((torch.from_numpy(ids).to(dev), torch.from_numpy(emb).to(dev)))
threads = [threading.Thread(target=producer, daemon=True) for _ in range(2)]
for th in threads: th.start()

variants = {}
for spec in args.variants.split(";"):
    name, cfg = spec.split(":"); e, w, l, h = (int(x) for x in cfg.split(","))
    m = C.TinyText(emb_dim=e, width=w, layers=l, heads=h).to(dev); ema = copy.deepcopy(m).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    variants[name] = dict(model=m, ema=ema, opt=torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98)))
    print(f"variant {name}: emb {e} width {w} layers {l} heads {h}: {m.n_params()/1e6:.2f} M params -> fp16 {m.n_params()*2/2**20:.1f} MiB", flush=True)
step, elapsed0 = 0, 0.0; ck = os.path.join(args.out, "ckpt.pt")
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev); step, elapsed0 = st["step"], st["elapsed"]
    for n, v in variants.items():
        if n in st["variants"]: v["model"].load_state_dict(st["variants"][n]["model"]); v["ema"].load_state_dict(st["variants"][n]["ema"]); v["opt"].load_state_dict(st["variants"][n]["opt"])
    print("resumed at", step, flush=True)
def save_ckpt():
    torch.save({"step": step, "elapsed": elapsed(), "variants": {n: {"model": v["model"].state_dict(), "ema": v["ema"].state_dict(), "opt": v["opt"].state_dict()} for n, v in variants.items()}}, ck + ".tmp"); os.replace(ck + ".tmp", ck)

# ---- eval sets
ev_rng = np.random.default_rng(1)
EV = {"val": (texts["val"] or texts["coco"])[:2000] if (texts["val"] or texts["coco"]) else [], "prompts40": TX.PROMPTS40, "_unused": ['a face', 'a red forest', 'the sea at night', 'a city street', 'green hills under a blue sky', 'a cat', 'deadline on Friday', "I miss my grandmother's kitchen", 'the smell of rain', 'a lighthouse in a storm', 'a bowl of oranges', 'snow on a mountain', 'a crowded market', 'a sleeping dog', 'fire', 'a glass of water on a table', 'we should talk about the budget', 'a yellow bicycle', 'an old library', "a child's drawing of a house", 'sunset over the plains', 'a portrait of a woman in blue', 'a horse in a field', 'mushrooms in the forest', 'the moon over the sea', 'a broken clock', 'a train station in winter', 'flowers in a vase', 'I feel tired today', 'a river through a canyon', 'a dark room with one candle', 'the first day of school', 'a bird on a wire', 'an abandoned factory', 'waves crashing on rocks', 'a wedding', 'a bridge in fog', 'coffee in the morning', 'a map of an imaginary island', 'the sound of a cello'],
      "metaphors": texts["metaphors"], "notes": texts["notes"][:500]}
EV_IDS = {k: np.array([encode(t) for t in v], np.int64) for k, v in EV.items() if v}
EV_T = {k: np.concatenate([teach(ids[i:i + 256]) for i in range(0, len(ids), 256)]) for k, ids in EV_IDS.items()}
@torch.no_grad()
def evaluate():
    out = {}
    for n, v in variants.items():
        r = {}
        for k, ids in EV_IDS.items():
            P = np.concatenate([v["ema"](torch.from_numpy(ids[i:i + 256]).to(dev)).float().cpu().numpy() for i in range(0, len(ids), 256)])
            r[f"cos_{k}"] = float((P * EV_T[k]).sum(1).mean())
            if k == "val" and len(ids) >= 500:   # retrieval agreement: top-10 neighbours among the val captions
                St, Sp = EV_T[k] @ EV_T[k].T, P @ P.T; np.fill_diagonal(St, -1); np.fill_diagonal(Sp, -1)
                tt, tp = np.argsort(-St, 1)[:, :10], np.argsort(-Sp, 1)[:, :10]
                r["top10_overlap"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(tt, tp)]))
        out[n] = r
    print(f"[eval step {step}] " + json.dumps(out), flush=True)
    json.dump({"step": step, "elapsed_h": elapsed() / 3600, "metrics": out}, open(os.path.join(args.out, "eval.json"), "w"), indent=1); return out

t_start = time.time(); elapsed = lambda: elapsed0 + time.time() - t_start
budget = args.hours * 3600; warm = 300; last_ck, last_ev, last_log = time.time(), time.time(), time.time(); seen = 0; acc = {n: [0.0, 0] for n in variants}
while elapsed() < budget and not (args.max_steps and step >= args.max_steps):
    ids, emb = q.get()
    frac = min(1.0, elapsed() / budget); lr = args.lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05) * min(1.0, (step + 1) / warm)
    for n, v in variants.items():
        for g in v["opt"].param_groups: g["lr"] = lr
        pred = v["model"](ids); loss = 1 - (pred * emb).sum(1).mean()
        v["opt"].zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(v["model"].parameters(), 1.0); v["opt"].step()
        with torch.no_grad():
            d = args.ema if step > warm else 0.0
            for pe, pm in zip(v["ema"].parameters(), v["model"].parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        acc[n][0] += loss.item(); acc[n][1] += 1
    step += 1; seen += ids.shape[0]
    if time.time() - last_log > 60 or args.smoke:
        dt = time.time() - last_log; print(f"step {step} {elapsed()/3600:.2f}h lr {lr:.2e} " + " | ".join(f"{n}: 1-cos {a[0]/max(1,a[1]):.4f}" for n, a in acc.items()) + f"  {seen/dt:.0f} texts/s q={q.qsize()}", flush=True)
        last_log = time.time(); seen = 0; acc = {n: [0.0, 0] for n in variants}
    if time.time() - last_ck > args.ckpt_every: save_ckpt(); last_ck = time.time()
    if time.time() - last_ev > args.eval_every: evaluate(); last_ev = time.time()
stop = True; save_ckpt(); summary = evaluate()
for n, v in variants.items():
    C.export_text(copy.deepcopy(v["ema"]).cpu(), os.path.join(args.out, f"tiny_text_{n}.bin"), os.path.join(args.out, f"tiny_text_{n}.json"), meta={"variant": n, "step": step, "metrics": summary[n]})
    print(f"exported {n}: {os.path.getsize(os.path.join(args.out, f'tiny_text_{n}.bin'))/2**20:.2f} MiB", flush=True)
print("done", flush=True)

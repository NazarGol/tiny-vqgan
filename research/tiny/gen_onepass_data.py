"""Phase 5 data: run the token-space search over many (text, context, mask) and save (text_emb, context, mask, result, scores).
Usage: python gen_onepass_data.py --scorer-ckpt scorer/ckpt.pt --variant S --data DIR --clip-text text_model.onnx --tokenizer tokenizer.json --out onepass_data.npz --n 6000 [--hours 1.2]"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, texts as TX
from token_search import TokenSearch
ap = argparse.ArgumentParser()
ap.add_argument("--scorer-ckpt", required=True); ap.add_argument("--variant", default="S"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--clip-text", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--captions", default="")
ap.add_argument("--n", type=int, default=6000); ap.add_argument("--k", type=int, default=64); ap.add_argument("--generations", type=int, default=120); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--hours", type=float, default=1.2); ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if args.smoke: args.n, args.k, args.generations, args.batch = 8, 4, 5, 8
sd = torch.load(args.scorer_ckpt, map_location="cpu")["variants"][args.variant]["ema"]
scorer = C.TokenScorer(c0=sd["emb.weight"].shape[1], widths=tuple(sd[f"convs.{i}.0.bias"].shape[0] for i in range(4))); scorer.load_state_dict(sd); scorer.to(dev)
g16, g32 = C.load_grids(args.data); sampler = C.GridSampler(g16, g32, seed=11)
ts = TokenSearch(scorer, g16, dev, seed=11)
# texts -> teacher embeddings (ONNX)
import onnxruntime as ort
from tokenizers import Tokenizer
tk = Tokenizer.from_file(args.tokenizer); sess = ort.InferenceSession(args.clip_text, providers=[p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()])
def enc(t):
    ids = tk.encode(t).ids[:77]; ids = ids if ids[-1] == 49407 else ids[:76] + [49407]; return ids + [0] * (77 - len(ids))
pool = TX.all_texts(args.captions); rng = np.random.default_rng(11)
mix = [(k, w) for k, w in (("coco", 0.35), ("prompts", 0.25), ("notes", 0.25), ("metaphors", 0.05), ("prompts40", 0.10)) if (pool.get(k) if k != "prompts40" else True)]
mp = np.array([w for _, w in mix]); mp /= mp.sum()
def pick_text():
    k = mix[rng.choice(len(mix), p=mp)][0]; src = TX.PROMPTS40 if k == "prompts40" else pool[k]; return src[rng.integers(len(src))]
def embed_texts(ts_):
    ids = np.array([enc(t) for t in ts_], np.int64); e = sess.run(None, {"input_ids": ids})[0]; return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
out = dict(text=[], text_emb=[], ctx=[], mask=[], result=[], seed_score=[], final_score=[], S=[])
prev_results = []; t0 = time.time(); done = 0
while done < args.n and time.time() - t0 < args.hours * 3600:
    S = int(rng.choice([12, 16, 20], p=[0.35, 0.45, 0.20])); K = args.k; texts_ = [pick_text() for _ in range(K)]; T = torch.from_numpy(embed_texts(texts_))
    ctx = np.empty((K, S, S), np.int64); masks = np.zeros((K, S, S), bool)
    for k in range(K):
        u = rng.random()
        if u < 0.40: ctx[k] = C.BLANK if rng.random() < 0.85 else rng.integers(C.N_EMBED)
        elif u < 0.70 or not prev_results: ctx[k] = sampler.source(S)
        else: pr = prev_results[rng.integers(len(prev_results))]; ctx[k] = sampler.fit(pr, S)
        m = sampler.blob(S)
        while m.sum() < 0.15 * S * S: m = sampler.blob(S)
        masks[k] = m
    best, sc, hist = ts.run(T, ctx, masks, generations=args.generations, batch=args.batch)
    for k in range(K): out["text"].append(texts_[k]); out["text_emb"].append(T[k].numpy().astype(np.float16)); out["ctx"].append(ctx[k].astype(np.uint16)); out["mask"].append(masks[k]); out["result"].append(best[k].astype(np.uint16)); out["seed_score"].append(hist[0]); out["final_score"].append(sc[k]); out["S"].append(S)
    prev_results.extend(best[:8]); prev_results = prev_results[-200:]; done += K
    print(f"{done}/{args.n} S={S} seed {hist[0]:.4f} -> final {sc.mean():.4f}  {(time.time()-t0)/60:.1f} min", flush=True)
# ragged sizes: store per-size arrays
np.savez_compressed(args.out, text=np.array(out["text"]), text_emb=np.stack(out["text_emb"]), S=np.array(out["S"]), seed_score=np.array(out["seed_score"]), final_score=np.array(out["final_score"]),
                    **{f"{k}_{S}": np.stack([a for a, s in zip(out[k], out["S"]) if s == S]) for k in ("ctx", "mask", "result") for S in sorted(set(out["S"]))},
                    **{f"idx_{S}": np.array([i for i, s in enumerate(out["S"]) if s == S]) for S in sorted(set(out["S"]))})
print("wrote", args.out, done, "samples", flush=True)

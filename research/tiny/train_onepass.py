"""Phase 5: train the one-pass starting model on search data; compare search-from-one-pass vs search-from-mosaic at equal budget.
Usage: python train_onepass.py --npz onepass_data.npz --scorer-ckpt scorer/ckpt.pt --data DIR --out DIR [--hours 1.0]"""
import argparse, json, os, sys, time, math, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from token_search import TokenSearch
ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True); ap.add_argument("--scorer-ckpt", required=True); ap.add_argument("--variant", default="S"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--hours", type=float, default=1.0); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--width", type=int, default=96); ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--eval-gens", type=int, default=40); ap.add_argument("--smoke", action="store_true")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if args.smoke: args.hours, args.batch, args.eval_gens = 0.01, 4, 3
D = np.load(args.npz, allow_pickle=True); sizes = sorted(int(k.split("_")[1]) for k in D.files if k.startswith("ctx_"))
sets = {S: dict(ctx=D[f"ctx_{S}"].astype(np.int64), mask=D[f"mask_{S}"], res=D[f"result_{S}"].astype(np.int64), emb=D["text_emb"][D[f"idx_{S}"]].astype(np.float32), idx=D[f"idx_{S}"]) for S in sizes}
n_total = sum(len(v["idx"]) for v in sets.values()); print("samples", n_total, {S: len(v["idx"]) for S, v in sets.items()}, flush=True)
hold = {S: np.arange(len(v["idx"]))[-max(1, min(len(v["idx"]) // 2, len(v["idx"]) // 20)):] for S, v in sets.items()}; train_idx = {S: np.arange(len(v["idx"]))[:-len(hold[S])] for S, v in sets.items()}
sd = torch.load(args.scorer_ckpt, map_location="cpu")["variants"][args.variant]["ema"]
scorer = C.TokenScorer(c0=sd["emb.weight"].shape[1], widths=tuple(sd[f"convs.{i}.0.bias"].shape[0] for i in range(4))); scorer.load_state_dict(sd); scorer.to(dev).eval()
model = C.OnePass(width=args.width, layers=args.layers, table=sd["emb.weight"].float()).to(dev); ema = copy.deepcopy(model).eval()
for p in ema.parameters(): p.requires_grad_(False)
print(f"OnePass params {model.n_params()/1e6:.2f} M -> fp16 {model.n_params()*2/2**20:.1f} MiB", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01); rng = np.random.default_rng(0)
def batch():
    S = sizes[rng.choice(len(sizes), p=np.array([len(train_idx[s]) for s in sizes]) / sum(len(train_idx[s]) for s in sizes))]; v = sets[S]; i = rng.choice(train_idx[S], size=min(args.batch, len(train_idx[S])), replace=False)
    return (torch.from_numpy(v["ctx"][i]).to(dev), torch.from_numpy(v["mask"][i]).to(dev), torch.from_numpy(v["emb"][i]).to(dev), torch.from_numpy(v["res"][i]).to(dev))
t0 = time.time(); step = 0; budget = args.hours * 3600; acc = [0.0, 0.0, 0]
while time.time() - t0 < budget:
    ctx, mask, emb, res = batch(); frac = (time.time() - t0) / budget; lr = args.lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05) * min(1, (step + 1) / 200)
    for g in opt.param_groups: g["lr"] = lr
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        h = model(ctx, mask, emb)
    logits = model.logits(h)[mask]; target = res[mask]
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        d = 0.999 if step > 200 else 0.0
        for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    acc[0] += loss.item(); acc[1] += (logits.argmax(-1) == target).float().mean().item(); acc[2] += 1; step += 1
    if step % 100 == 0 or args.smoke: print(f"step {step} {(time.time()-t0)/60:.1f} min lr {lr:.2e} CE {acc[0]/acc[2]:.3f} acc {acc[1]/acc[2]:.3f}", flush=True); acc = [0.0, 0.0, 0]
# ---- eval: held-out CE/acc; seed scores; search from one-pass seed vs mosaic at equal budget
ts = TokenSearch(scorer, C.load_grids(args.data)[0], dev, seed=5); rep = {}
with torch.no_grad():
    for S in sizes:
        v = sets[S]; i = hold[S]; ctx, mask, emb, res = (torch.from_numpy(v["ctx"][i]).to(dev), torch.from_numpy(v["mask"][i]).to(dev), torch.from_numpy(v["emb"][i]).to(dev), torch.from_numpy(v["res"][i]).to(dev))
        h = ema(ctx, mask, emb); lg = ema.logits(h)[mask]; ce = F.cross_entropy(lg, res[mask]).item(); ac = (lg.argmax(-1) == res[mask]).float().mean().item()
        pred = ema.predict(ctx, mask, emb)
        def sc(grids): e = scorer(grids.to(dev)).float(); return (e * emb).sum(-1).cpu().numpy()
        s_pred, s_res, s_ctx = sc(pred), sc(res), sc(ctx)
        K = len(i); T = emb.cpu(); c_np, m_np = v["ctx"][i], v["mask"][i]
        b1, f1, h1 = ts.run(T, c_np, m_np, generations=args.eval_gens, batch=16, init=pred.cpu().numpy())
        b2, f2, h2 = ts.run(T, c_np, m_np, generations=args.eval_gens, batch=16)
        rep[S] = dict(n=K, ce=ce, acc=ac, score_onepass=float(s_pred.mean()), score_context=float(s_ctx.mean()), score_search_data=float(s_res.mean()), score_mosaic_seed=float(h2[0]), search_from_onepass=float(f1.mean()), search_from_mosaic=float(f2.mean()), wins=int((f1 > f2).sum()))
        print(f"[eval S={S}] " + json.dumps(rep[S]), flush=True)
json.dump({"step": step, "metrics": rep}, open(os.path.join(args.out, "eval.json"), "w"), indent=1)
C.export_onepass(copy.deepcopy(ema).cpu(), os.path.join(args.out, "onepass.bin"), os.path.join(args.out, "onepass.json"), meta={"metrics": rep})
torch.save({"model": ema.state_dict(), "cfg": ema.cfg}, os.path.join(args.out, "onepass_ema.pt"))
print("done, exported", os.path.getsize(os.path.join(args.out, "onepass.bin")) / 2 ** 20, "MiB", flush=True)

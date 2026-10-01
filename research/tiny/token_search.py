"""Token-space search in PyTorch (batched over many independent searches): the engine's hill-climb (mosaic seeds from retrieved
bank grids, bank patches, palette samples, edge growth, neighbour copies, block moves, swaps) scored by a TokenScorer.
Used on Kaggle to generate (text embedding, context, mask, result) data for the one-pass model and to evaluate scorers."""
import numpy as np, torch, torch.nn.functional as F


class TokenSearch:
    def __init__(self, scorer, bank16, device, n_embed=16384, blank=6328, seed=0):
        self.scorer = scorer.eval(); self.dev = device; self.n = n_embed; self.blank = blank; self.rng = np.random.default_rng(seed)
        self.bank = bank16.astype(np.int64).reshape(-1, 16, 16)
        with torch.no_grad():
            self.bank_emb = torch.cat([self._emb(torch.from_numpy(self.bank[i:i + 512])) for i in range(0, len(self.bank), 512)])       # [N,512]
            toks = torch.arange(n_embed).reshape(-1, 1, 1)
            self.pal_emb = torch.cat([self._emb(toks[i:i + 2048]) for i in range(0, n_embed, 2048)])                                       # [16384,512] single-token grids

    @torch.no_grad()
    def _emb(self, grids):
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
            return self.scorer(grids.to(self.dev)).float()

    def _fit(self, g, h, w):
        H, W = g.shape
        if H >= h and W >= w: oy, ox = (H - h) // 2, (W - w) // 2; return g[oy:oy + h, ox:ox + w]
        ys, xs = np.arange(h) * H // h, np.arange(w) * W // w; return g[np.ix_(ys, xs)]

    @torch.no_grad()
    def run(self, targets, contexts, masks, generations=120, batch=32, seeds=8, init=None, bank_top=24, sources=4, patch=4, grow_edge=0.8, mutation=0.08, anneal=0.003, bank_patch=0.30, temperature=0.03, top_k=512, log=None):
        """targets [K,512] unit (torch, any device); contexts [K,S,S] int64 numpy; masks [K,S,S] bool numpy. Returns best [K,S,S], scores [K], history."""
        K, S, _ = contexts.shape; r = self.rng; T = targets.to(self.dev).float()
        # retrieval + palette samplers per search
        bank_scores = (self.bank_emb @ T.T).T.cpu().numpy()                      # [K,N]
        pal_scores = (self.pal_emb @ T.T).T.cpu().numpy()                        # [K,16384]
        per = []
        for k in range(K):
            ys, xs = np.nonzero(masks[k]); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            rw, rh = x1 - x0, y1 - y0
            top = np.argsort(-bank_scores[k])[:bank_top]; chosen = r.choice(top, size=min(sources, len(top)), replace=False)
            retrieved = [self._fit(self.bank[i], rh, rw) for i in chosen]
            idx = np.argsort(-pal_scores[k])[:top_k]; p = np.exp((pal_scores[k][idx] - pal_scores[k][idx[0]]) / temperature); p /= p.sum()
            cells = list(zip(ys, xs)); inm = masks[k]
            nbrs = lambda y, x: [(y + dy, x + dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= y + dy < S and 0 <= x + dx < S]
            edge = [(y, x) for y, x in cells if any(not inm[ny, nx] for ny, nx in nbrs(y, x))]
            per.append(dict(region=(y0, x0, rh, rw), retrieved=retrieved, pal_idx=idx, pal_p=p, cells=cells, edge=edge, nbrs=nbrs))
        def sample_pal(k, n=1): return r.choice(per[k]["pal_idx"], size=n, p=per[k]["pal_p"])
        def mosaic(k):
            P = per[k]; y0, x0, rh, rw = P["region"]; cand = contexts[k].copy(); inm = masks[k]
            for by in range(0, rh, patch):
                for bx in range(0, rw, patch):
                    src = P["retrieved"][r.integers(len(P["retrieved"]))] if P["retrieved"] else None
                    for y in range(by, min(rh, by + patch)):
                        for x in range(bx, min(rw, bx + patch)):
                            gy, gx = y0 + y, x0 + x
                            if not inm[gy, gx]: continue
                            cand[gy, gx] = src[y, x] if src is not None and r.random() > 0.15 else sample_pal(k)[0]
            for (y, x) in P["edge"]:
                if r.random() < grow_edge:
                    nb = [(ny, nx) for ny, nx in P["nbrs"](y, x) if not inm[ny, nx]]
                    if nb: ny, nx = nb[r.integers(len(nb))]; cand[y, x] = contexts[k][ny, nx]
            return cand
        def mutate(k, cand, frac):
            P = per[k]; y0, x0, rh, rw = P["region"]; inm = masks[k]; n = max(1, int(round(len(P["cells"]) * frac))); cur = cand[y0:y0 + rh, x0:x0 + rw].copy()
            for _ in range(n):
                u = r.random(); y, x = P["cells"][r.integers(len(P["cells"]))]; cy, cx = y - y0, x - x0
                if P["retrieved"] and u < bank_patch:
                    src = P["retrieved"][r.integers(len(P["retrieved"]))]; s = 1 + r.integers(3); h, w = min(s, rh - cy), min(s, rw - cx)
                    blk = inm[y:y + h, x:x + w]; cand[y:y + h, x:x + w][blk] = src[cy:cy + h, cx:cx + w][blk]
                elif u < 0.50: cand[y, x] = sample_pal(k)[0]
                elif u < 0.62 and P["edge"]:
                    ey, ex = P["edge"][r.integers(len(P["edge"]))]; nb = [(ny, nx) for ny, nx in P["nbrs"](ey, ex) if not inm[ny, nx]]
                    if nb: ny, nx = nb[r.integers(len(nb))]; cand[ey, ex] = contexts[k][ny, nx]
                elif u < 0.82: nb = P["nbrs"](y, x); ny, nx = nb[r.integers(len(nb))]; cand[y, x] = cand[ny, nx]
                elif u < 0.95:
                    s = 2 + r.integers(2); sy, sx = r.integers(rh), r.integers(rw); h, w = min(s, rh - cy, rh - sy), min(s, rw - cx, rw - sx)
                    if h > 0 and w > 0: blk = inm[y:y + h, x:x + w]; cand[y:y + h, x:x + w][blk] = cur[sy:sy + h, sx:sx + w][blk]
                else: y2, x2 = P["cells"][r.integers(len(P["cells"]))]; cand[y, x], cand[y2, x2] = cand[y2, x2], cand[y, x]
            return cand
        def score(cands):   # [K,B,S,S] -> [K,B]
            B = cands.shape[1]; e = self._emb(torch.from_numpy(cands.reshape(K * B, S, S))).reshape(K, B, -1)
            return (e * T[:, None, :]).sum(-1).cpu().numpy()
        # seeds: mosaics, or (phase 5) a given starting grid plus light mutations of it
        if init is None: cands = np.stack([np.stack([mosaic(k) for _ in range(seeds)]) for k in range(K)])
        else: cands = np.stack([np.stack([init[k].copy()] + [mutate(k, init[k].copy(), 0.05) for _ in range(seeds - 1)]) for k in range(K)])
        sc = score(cands)
        bi = sc.argmax(1); best = cands[np.arange(K), bi]; best_sc = sc[np.arange(K), bi]; hist = [best_sc.mean()]
        for g in range(generations):
            prog = g / generations; rate = mutation * (1 - prog) + 0.01; temp = anneal * (1 - prog)
            cands = np.stack([np.stack([mutate(k, best[k].copy(), rate) for _ in range(batch)]) for k in range(K)]); sc = score(cands)
            bi = sc.argmax(1); cs = sc[np.arange(K), bi]
            acc = (cs > best_sc) | (r.random(K) < np.exp(np.clip((cs - best_sc) / max(temp, 1e-6), -50, 0)))
            best[acc] = cands[np.arange(K), bi][acc]; best_sc[acc] = cs[acc]; hist.append(best_sc.mean())
            if log and (g % 20 == 0 or g == generations - 1): log(f"gen {g}: mean score {best_sc.mean():.4f}")
        return best, best_sc, hist

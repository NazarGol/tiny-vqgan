// Token-space painter: the same hill-climb as lib/search.js (mosaic seeds, bank patches, palette, neighbour copies, block
// moves, swaps) but candidates are scored in batches by the token scorer (no decode, no CLIP image model), and only the
// current best is decoded by the tiny decoder for the live preview.
import { fitGrid } from '../lib/bank.js';

export function expandRegion(grid, r, margin) {
  const x = Math.max(0, r.x - margin), y = Math.max(0, r.y - margin);
  const x1 = Math.min(grid.w, r.x + r.w + margin), y1 = Math.min(grid.h, r.y + r.h + margin);
  return { x, y, w: x1 - x, h: y1 - y };
}
/** Square crop of side max(w,h)+2·margin around the region, shifted to stay inside the grid (the scorer scores squares). */
export function squareCrop(grid, r, margin) {
  const side = Math.min(Math.max(r.w, r.h) + 2 * margin, grid.w, grid.h);
  let x = r.x + ((r.w - side) >> 1), y = r.y + ((r.h - side) >> 1);
  x = Math.max(0, Math.min(grid.w - side, x)); y = Math.max(0, Math.min(grid.h - side, y));
  return { x, y, w: side, h: side };
}
export function readRegion(grid, r) {
  const out = new Int32Array(r.w * r.h);
  for (let yy = 0; yy < r.h; yy++) out.set(grid.tokens.subarray((r.y + yy) * grid.w + r.x, (r.y + yy) * grid.w + r.x + r.w), yy * r.w);
  return out;
}
export function writeRegion(grid, r, tokens) {
  for (let yy = 0; yy < r.h; yy++) grid.tokens.set(tokens.subarray(yy * r.w, yy * r.w + r.w), (r.y + yy) * grid.w + r.x);
}

export class TokenPainter {
  constructor({ scorer, decoder, clip = null, palette, bank = null }) { this.scorer = scorer; this.decoder = decoder; this.clip = clip; this.palette = palette; this.bank = bank; }

  /**
   * Paint the cells of `mask` ({x,y,w,h,cells}) in `grid` toward `target` (unit embedding).
   * Returns {score, steps, generations, accepted, elapsed, image: {rgba,w,h}, crop, tokens, changed}.
   */
  async paint({ grid, mask, target, seconds = 10, margin = 2, batch = 32, seeds = 8, bankTop = 24, sources = 4, patch = 4, growEdge = 0.8, mutation = 0.08, anneal = 0.003, bankPatch = 0.30,
                temperature = 0.03, topK = 512, blankToken = -1, parent = null, parentMix = 0.5, photo = null, photoMix = 0.6, onProgress, progressEvery = 300, signal,
                mode = 'token', clipBatch = 4, prefilterTop = 4, poolBudget = 32 * 2 ** 20 }) {
    if (mode !== 'token' && !this.clip) throw new Error('mode ' + mode + ' needs the CLIP image tower');
    if (mode === 'prefilter' && !this.scorer) mode = 'clip';
    const t0 = performance.now();
    const sampler = this.palette.sampler(this.palette.scores(target), { topK, temperature });
    const region = { x: mask.x, y: mask.y, w: mask.w, h: mask.h };
    const side = Math.max(region.w, region.h);
    const retrievedAll = this.bank ? this.bank.top(this.bank.scores(target), bankTop).map((i) => fitGrid(this.bank.grid(i, side), region.w, region.h)) : [];
    const retrieved = retrievedAll.slice().sort(() => Math.random() - 0.5).slice(0, sources);
    const photoGrid = photo && photo.tokens ? fitGrid(photo, region.w, region.h) : null;

    const crop = expandRegion(grid, region, margin);           // what is decoded and written back (like the app)
    const sq = squareCrop(grid, region, margin);               // what is scored
    const base = readRegion(grid, crop);
    const rx = region.x - crop.x, ry = region.y - crop.y;
    const inMask = new Uint8Array(crop.w * crop.h), cells = [];
    for (let y = 0; y < region.h; y++) for (let x = 0; x < region.w; x++) if (mask.cells[y * region.w + x]) { const c = (ry + y) * crop.w + rx + x; inMask[c] = 1; cells.push(c); }
    const nCells = cells.length; if (!nCells) throw new Error('empty mask');
    const neighbors = (c) => { const x = c % crop.w, y = (c - x) / crop.w, out = []; if (x > 0) out.push(c - 1); if (x < crop.w - 1) out.push(c + 1); if (y > 0) out.push(c - crop.w); if (y < crop.h - 1) out.push(c + crop.w); return out; };
    const usable = (c) => !inMask[c];
    const grown = (c) => !inMask[c] && base[c] !== blankToken;   // edge cells grow from painted canvas only: on blank canvas the stroke fills its whole shape
    const edge = cells.filter((c) => neighbors(c).some(grown));
    const rnd = (n) => (Math.random() * n) | 0;
    const parentAt = parent ? (c) => { const x = crop.x + c % crop.w, y = crop.y + (c - c % crop.w) / crop.w; const px = Math.min(parent.crop.w - 1, Math.max(0, x - parent.crop.x)), py = Math.min(parent.crop.h - 1, Math.max(0, y - parent.crop.y)); return parent.tokens[py * parent.crop.w + px]; } : null;

    // scoring: the square crop is read from the grid with the candidate's crop tokens overlaid
    const sqBase = readRegion(grid, sq), S = sq.w, cropToSq = new Int32Array(crop.w * crop.h).fill(-1);
    for (let y = 0; y < crop.h; y++) for (let x = 0; x < crop.w; x++) { const gx = crop.x + x - sq.x, gy = crop.y + y - sq.y; if (gx >= 0 && gy >= 0 && gx < S && gy < S) cropToSq[y * crop.w + x] = gy * S + gx; }
    const sqBatch = new Uint16Array(S * S * batch);
    const scoreBatch = (cands) => {
      for (let b = 0; b < cands.length; b++) {
        const o = b * S * S; sqBatch.set(sqBase, o);
        const cand = cands[b]; for (let c = 0; c < cand.length; c++) { const j = cropToSq[c]; if (j >= 0) sqBatch[o + j] = cand[c]; }
      }
      return this.scorer.score(cands.length === batch ? sqBatch : sqBatch.subarray(0, cands.length * S * S), S, cands.length);
    };
    const whileHidden = async () => { while (typeof document !== 'undefined' && document.visibilityState === 'hidden' && !(signal && signal.aborted)) await new Promise((r) => setTimeout(r, 250)); };
    const yieldUI = () => new Promise((r) => setTimeout(r, 0));

    const mosaic = () => {
      const cand = base.slice(), bw = Math.ceil(region.w / patch), bh = Math.ceil(region.h / patch);
      for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) {
        const src = retrieved.length ? retrieved[rnd(retrieved.length)] : null, fromParent = parentAt && Math.random() < parentMix, fromPhoto = photoGrid && Math.random() < photoMix;
        for (let y = by * patch; y < Math.min(region.h, (by + 1) * patch); y++) for (let x = bx * patch; x < Math.min(region.w, (bx + 1) * patch); x++) {
          const c = (ry + y) * crop.w + rx + x; if (!inMask[c]) continue;
          cand[c] = fromPhoto ? photoGrid[y * region.w + x] : fromParent ? parentAt(c) : src && Math.random() > 0.15 ? src[y * region.w + x] : sampler.sample();
        }
      }
      for (const c of edge) if (Math.random() < growEdge) { const nb = neighbors(c).filter(grown); if (nb.length) cand[c] = base[nb[rnd(nb.length)]]; }
      return cand;
    };
    const copyBlock = (cand, src, sx, sy, dx, dy, s) => {
      for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
        const gx = dx + x, gy = dy + y, hx = sx + x, hy = sy + y;
        if (gx < 0 || gy < 0 || gx >= region.w || gy >= region.h || hx < 0 || hy < 0 || hx >= region.w || hy >= region.h) continue;
        const c = (ry + gy) * crop.w + rx + gx; if (inMask[c]) cand[c] = src[hy * region.w + hx];
      }
    };
    const regionOf = (cand) => { const out = new Int32Array(region.w * region.h); for (let y = 0; y < region.h; y++) for (let x = 0; x < region.w; x++) out[y * region.w + x] = cand[(ry + y) * crop.w + rx + x]; return out; };
    const mutate = (cand, frac) => {
      const n = Math.max(1, Math.round(nCells * frac)), cur = regionOf(cand);
      for (let i = 0; i < n; i++) {
        const r = Math.random(), c = cells[rnd(nCells)], cx = c % crop.w - rx, cy = ((c - c % crop.w) / crop.w) - ry;
        if (retrieved.length && r < bankPatch) copyBlock(cand, retrieved[rnd(retrieved.length)], cx, cy, cx, cy, 1 + rnd(3));
        else if (parentAt && r < bankPatch + 0.12) cand[c] = parentAt(c);
        else if (photoGrid && r < bankPatch + 0.30) copyBlock(cand, photoGrid, cx, cy, cx, cy, 1 + rnd(3));
        else if (r < 0.50) cand[c] = sampler.sample();
        else if (r < 0.62 && edge.length) { const e = edge[rnd(edge.length)]; const nb = neighbors(e).filter(grown); if (nb.length) cand[e] = base[nb[rnd(nb.length)]]; }
        else if (r < 0.82) { const nb = neighbors(c); cand[c] = cand[nb[rnd(nb.length)]]; }
        else if (r < 0.95) copyBlock(cand, cur, rnd(region.w), rnd(region.h), cx, cy, 2 + rnd(2));
        else { const c2 = cells[rnd(nCells)]; const t = cand[c]; cand[c] = cand[c2]; cand[c2] = t; }
      }
    };
    const changedCells = (cand) => { const out = []; for (const c of cells) if (cand[c] !== base[c]) { const x = c % crop.w; out.push([crop.x + x, crop.y + (c - x) / crop.w]); } return out; };

    // real CLIP on the tiny decoder's output (rect crop, resized to 256 like the app's full path); tries = real CLIP evaluations
    let clipEvals = 0, tokenEvals = 0;
    const clipScore = (cand) => { const d = this.decoder.decodeToTexture(cand, crop.h, crop.w); this.clip.setInput(d.tex, d.w, d.h); clipEvals++; return this.clip.score()[0]; };
    const evaluate = (cands) => {   // -> Float32Array of scores for the candidates that were really evaluated, plus their indices
      if (mode === 'token') { tokenEvals += cands.length; return { idx: cands.map((_, i) => i), scores: scoreBatch(cands) }; }
      if (mode === 'clip') return { idx: cands.map((_, i) => i), scores: Float32Array.from(cands, clipScore) };
      const pre = scoreBatch(cands); tokenEvals += cands.length;   // prefilter: top few by the token scorer, real CLIP decides
      // slow GPUs (a real-CLIP evaluation over ~120 ms): judge 2 instead of 4 per generation, so the search still moves
      const top = this.clip.stats.lastMs > 120 ? Math.min(2, prefilterTop) : prefilterTop;
      const idx = Array.from(pre.keys()).sort((a, b) => pre[b] - pre[a]).slice(0, Math.min(top, cands.length));
      return { idx, scores: Float32Array.from(idx, (i) => clipScore(cands[i])) };
    };
    if (this.scorer) this.scorer.setTargets([target]);
    if (this.clip && mode !== 'token') this.clip.setTargets([target]);
    await whileHidden();
    let best = null, bestScore = -Infinity, steps = 0, generations = 0, accepted = 0;
    const nSeeds = mode === 'clip' ? Math.min(seeds, 6) : Math.min(seeds, batch);
    const seedCands = []; for (let k = 0; k < nSeeds; k++) seedCands.push(mosaic());
    const seedEval = evaluate(seedCands), seedScores = new Float32Array(seedCands.length).fill(-Infinity); seedEval.idx.forEach((i, j) => { seedScores[i] = seedEval.scores[j]; }); steps += seedEval.scores.length;
    for (let k = 0; k < seedCands.length; k++) if (seedScores[k] > bestScore) { bestScore = seedScores[k]; best = seedCands[k]; }
    const preview = (final = false) => {
      const img = this.decoder.decodeRGBA(best, crop.h, crop.w);
      const res = { score: bestScore, steps, generations, accepted, elapsed: elapsed(), image: img, crop, tokens: best, changed: changedCells(best), final, mode, clipEvals, tokenEvals };
      onProgress?.(res); return res;
    };
    let lastReport = 0, pausedMs = 0;
    const elapsed = () => (performance.now() - t0 - pausedMs) / 1000;
    preview();
    const genSize = mode === 'clip' ? clipBatch : batch, cands = new Array(genSize);
    while (best && elapsed() < seconds && !(signal && signal.aborted)) {
      const tp = performance.now(); await whileHidden(); pausedMs += performance.now() - tp;
      const progress = Math.min(1, elapsed() / seconds), rate = mutation * (1 - progress) + 0.01;
      for (let b = 0; b < genSize; b++) { const cand = best.slice(); mutate(cand, rate); cands[b] = cand; }
      const ev = evaluate(cands); steps += ev.scores.length; generations++;
      let bj = 0; for (let j = 1; j < ev.scores.length; j++) if (ev.scores[j] > ev.scores[bj]) bj = j;
      const bi = ev.idx[bj], sc = ev.scores[bj], temp = anneal * (1 - progress);
      if (sc > bestScore || Math.random() < Math.exp((sc - bestScore) / Math.max(temp, 1e-6))) { if (sc > bestScore) accepted++; best = cands[bi]; bestScore = sc; }
      if (performance.now() - lastReport > progressEvery) { lastReport = performance.now(); preview(); }
      this.decoder.nn.trim(poolBudget);   // keep the GPU working set bounded during the stroke (big crops on phones)
      await yieldUI();
    }
    if (!best) throw new Error('aborted before any candidate was scored');
    writeRegion(grid, crop, best);
    return preview(true);
  }
}

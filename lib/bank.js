// Token bank: real photos as VQGAN token grids (4x4, 6x6, 8x8, 16x16) with MobileCLIP embeddings (PCA-128).
// A prompt retrieves the closest grids; the painter uses them as seeds and as a source of patches.
import { fetchCached, fetchJsonCached } from './models.js';

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class Bank {
  static async load(base) {
    const meta = await fetchJsonCached(base + 'bank.json');
    const pca = f16ToF32(new Uint16Array(await fetchCached(base + 'bank_pca128.f16')));
    const basis = new Float32Array(await fetchCached(base + 'bank_basis.f32'));
    const meandot = new Float32Array(await fetchCached(base + 'bank_meandot.f32'));
    const tokens = {};
    for (const s of meta.sizes) tokens[s] = new Uint16Array(await fetchCached(base + `bank_tokens_${s}.u16`));
    return new Bank(meta, pca, basis, meandot, tokens);
  }

  constructor(meta, pca, basis, meandot, tokens) {
    this.n = meta.n; this.d = meta.pca_dim; this.e = meta.embed_dim; this.meanNorm2 = meta.mean_norm2; this.sizes = meta.sizes;
    this.pca = pca; this.mean = basis.subarray(0, this.e); this.comps = basis.subarray(this.e); this.meandot = meandot; this.tokens = tokens;
  }

  /** Cosine similarity of every bank image with the (unit) text embedding. */
  scores(textEmb) {
    const p = new Float32Array(this.d);
    let mt = 0;
    for (let k = 0; k < this.e; k++) {
      const v = textEmb[k] - this.mean[k]; mt += this.mean[k] * textEmb[k];
      if (v === 0) continue;
      const row = k * this.d;
      for (let j = 0; j < this.d; j++) p[j] += v * this.comps[row + j];
    }
    const c = mt - this.meanNorm2, out = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      let s = 0; const row = i * this.d;
      for (let j = 0; j < this.d; j++) s += this.pca[row + j] * p[j];
      out[i] = s + this.meandot[i] + c;
    }
    return out;
  }

  top(scores, k) {
    const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
    return idx.slice(0, k);
  }

  /** Token grid of image i at the bank size closest to (>=) the requested side, as {w, h, tokens}. */
  grid(i, side) {
    const s = this.sizes.find((x) => x >= side) || this.sizes[this.sizes.length - 1];
    return { w: s, h: s, tokens: this.tokens[s].subarray(i * s * s, (i + 1) * s * s) };
  }
}

/** Fit a bank grid to a w x h region: crop if the grid is bigger, nearest-neighbour stretch if smaller. Returns Int32Array(w*h). */
export function fitGrid(g, w, h, ox = null, oy = null) {
  const out = new Int32Array(w * h);
  if (g.w >= w && g.h >= h) {
    if (ox == null) ox = (g.w - w) >> 1;
    if (oy == null) oy = (g.h - h) >> 1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = g.tokens[(oy + y) * g.w + ox + x];
  } else {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = g.tokens[Math.floor(y * g.h / h) * g.w + Math.floor(x * g.w / w)];
  }
  return out;
}

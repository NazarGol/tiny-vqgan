// Token palette: per-codebook-token MobileCLIP embedding (PCA-128), used to rank/sample tokens for a text prompt.
import { fetchCached, fetchJsonCached } from './models.js';

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class Palette {
  static async load(base) {
    const meta = await fetchJsonCached(base + 'palette.json');
    const pca = f16ToF32(new Uint16Array(await fetchCached(base + 'palette_pca128.f16')));
    const basis = new Float32Array(await fetchCached(base + 'palette_basis.f32'));
    const rgb = new Uint8Array(await fetchCached(base + 'palette_rgb.u8'));
    return new Palette(meta, pca, basis, rgb);
  }

  constructor(meta, pca, basis, rgb) {
    this.n = meta.n; this.d = meta.pca_dim; this.e = meta.embed_dim;
    this.pca = pca; this.mean = basis.subarray(0, this.e); this.comps = basis.subarray(this.e); this.rgb = rgb;
    this.norms = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      let s = 0; for (let j = 0; j < this.d; j++) { const v = pca[i * this.d + j]; s += v * v; }
      this.norms[i] = Math.sqrt(s) + 1e-8;
    }
  }

  /** Project a 512-d (unit) text embedding into PCA space. */
  project(emb) {
    const p = new Float32Array(this.d);
    for (let k = 0; k < this.e; k++) {
      const v = emb[k] - this.mean[k];
      if (v === 0) continue;
      const row = k * this.d;
      for (let j = 0; j < this.d; j++) p[j] += v * this.comps[row + j];
    }
    return p;
  }

  /** Approximate cosine similarity of every token tile with the text embedding. Float32Array(n). */
  scores(textEmb) {
    const p = this.project(textEmb);
    let pn = 0; for (let j = 0; j < this.d; j++) pn += p[j] * p[j]; pn = Math.sqrt(pn) + 1e-8;
    const out = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      let s = 0; const row = i * this.d;
      for (let j = 0; j < this.d; j++) s += this.pca[row + j] * p[j];
      out[i] = s / (this.norms[i] * pn);
    }
    return out;
  }

  /** Softmax sampler over the top-K tokens. temperature in cosine units (0.02 = peaky, 0.1 = broad). */
  sampler(scores, { topK = 512, temperature = 0.03 } = {}) {
    const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, topK);
    const max = scores[idx[0]];
    const cum = new Float64Array(idx.length);
    let acc = 0;
    for (let i = 0; i < idx.length; i++) { acc += Math.exp((scores[idx[i]] - max) / temperature); cum[i] = acc; }
    return {
      top: idx,
      sample() {
        const r = Math.random() * acc;
        let lo = 0, hi = cum.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < r) lo = mid + 1; else hi = mid; }
        return idx[lo];
      },
    };
  }
}

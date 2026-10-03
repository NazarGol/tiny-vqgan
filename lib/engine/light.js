// The light engine behind the same facade as lib/engine/client.js (Engine): init / paint / embedLong / embedText / embedImages /
// decode / encode / release / terminate, plus the decoder / clip / painter facades room.js uses. Runs on the page thread
// (WebGL2 in a worker needs OffscreenCanvas, which the minimum phones' Safari lacks); the search yields between generations.
// Photos still need the VQGAN encoder (ORT) → delegated to the ORT engine worker, loaded only for that call.
import { Engine as LightCore, fetchLowMem } from '../../engine/engine.js';
import { fetchCached } from '../models.js';
import { embedLongText } from '../text.js';
import { writeRegion } from '../decoder.js';

const toCHW = (img) => { const n = img.w * img.h, data = new Float32Array(3 * n), p = img.rgba; for (let i = 0; i < n; i++) { data[i] = p[i * 4] / 255; data[n + i] = p[i * 4 + 1] / 255; data[2 * n + i] = p[i * 4 + 2] / 255; } return { data, w: img.w, h: img.h, rgba: img.rgba }; };
const PRE = new RegExp("'s|'t|'re|'ve|'m|'ll|'d|\\p{L}+|\\p{N}|[^\\s\\p{L}\\p{N}]+", 'gu');
const estTokenizer = { tokenize: (s) => { const out = []; for (const m of s.toLowerCase().matchAll(PRE)) { const n = Math.max(1, Math.ceil(m[0].length / 4)); for (let i = 0; i < n; i++) out.push(m[0]); } return out; } };

export class LightEngine {
  constructor({ mode = 'prefilter', variant = 'auto', scorer = 'S', text = 'M', clip = 'clip_vision', batch = 32 } = {}) {
    this.core = null; this.ready = false; this.broken = null; this.gpu = false; this.speed = null; this.light = true; this.mode = mode; this.cfg = { variant, scorer, text, clip, batch }; this.ortEngine = null; this.opts = null;
    const self = this;
    this.decoder = { decode: async (tokens, h, w) => { const r = toCHW(self.core.decode(tokens, h, w)); self.decoder.lastMs = self.core.decoder.stats.lastMs; self.decoder.times.push(self.decoder.lastMs); if (self.decoder.times.length > 50) self.decoder.times.shift(); return r; }, lastMs: 0, times: [], tiny: true };
    this.clip = { size: 256, tok: estTokenizer, embedText: (text) => self.embedText(text), embedImages: (chw, n = 1) => self.embedImages(chw, n), tiny: true };
    this.painter = { paint: (o) => self.paint(o), tiny: true };
  }
  start() {}
  /** Same shape as the ORT engine's init: {modelBase, modelFallback, bankName, lowMem, ...}; progress {stage, loaded, total, cached}. */
  async init(opts, onProgress) {
    this.opts = opts; const base = opts.modelBase, total = 38 * 2 ** 20, seen = {}, cachedSeen = {};
    const onP = (p) => { seen[p.url] = p.loaded; cachedSeen[p.url] = !!p.cached; const loaded = Object.values(seen).reduce((a, b) => a + b, 0); onProgress?.({ stage: 'download', loaded, total, cached: Object.values(cachedSeen).every(Boolean) }); };
    try {
      onProgress?.({ stage: 'engine' });
      const mirror = opts.modelFallback ? (u) => (u.startsWith(opts.modelBase) ? opts.modelFallback + u.slice(opts.modelBase.length) : null) : null;
      this.core = await LightCore.load({ base, onProgress: onP, bank: opts.bankName || 'bank', variant: this.cfg.variant, scorer: this.cfg.scorer, text: this.cfg.text, clip: this.cfg.clip, mirrorOf: mirror, fetchBuf: (u) => (new URLSearchParams(location.search).get('loader') === 'cached' ? fetchCached(u, { onProgress: onP }) : fetchLowMem(u, { onProgress: onP, mirror })) });
      this.core.decode(new Int32Array(256).fill(opts.blankToken | 0), 16, 16); this.speed = Math.round(this.core.decoder.stats.lastMs);
      this.gpu = true; this.ready = true; this.variant = this.core.decoder.variant + (this.core.decoder.hash ? '@' + this.core.decoder.hash.slice(0, 4) : '');
      return { gpu: true, speed: this.speed, ep: 'webgl2', light: true, variant: this.variant };
    } catch (e) { this.broken = e.message || 'light engine failed'; throw e; }
  }
  async paint({ grid, mask, target, parent = null, photo = null, signal = null, onProgress = null, ...opts }) {
    if (!this.core) throw new Error('engine not ready');
    const res = await this.core.paintStroke({ grid, mask, target, parent, photo, signal, mode: this.mode, batch: this.cfg.batch, ...opts, onPreview: onProgress ? (p) => onProgress({ ...p, image: toCHW(p.image), changed: [] }) : null });
    writeRegion(grid, res.crop, res.tokens);
    return { ...res, image: toCHW(res.image) };
  }
  async embedText(text) { return this.core.encodeText(text); }
  async embedLong(text) { const r = await embedLongText(this.clip, text); return { target: r.target, chunks: r.chunks.length, hardSplits: r.hardSplits }; }
  async embedImages(chw, n = 1) {
    const side = Math.round(Math.sqrt(chw.length / 3)), rgba = new Uint8ClampedArray(side * side * 4), plane = side * side;
    for (let i = 0; i < plane; i++) { rgba[i * 4] = chw[i] * 255; rgba[i * 4 + 1] = chw[plane + i] * 255; rgba[i * 4 + 2] = chw[2 * plane + i] * 255; rgba[i * 4 + 3] = 255; }
    return [this.core.embedImage(rgba, side, side)];
  }
  decode(tokens, h, w) { return this.decoder.decode(tokens, h, w); }
  /** photo → tokens needs the VQGAN encoder: the ORT engine worker does it (loaded for that call, released after). */
  async encode(chw, size, opts = null) {
    if (!this.ortEngine) { const { Engine } = await import('./client.js'); this.ortEngine = new Engine(); }
    return this.ortEngine.encode(chw, size, opts || this.opts);
  }
  async release() { if (this.core) { this.core.release(); this.core = null; } this.ready = false; if (this.ortEngine) { this.ortEngine.terminate(); this.ortEngine = null; } }
  terminate() { this.release(); }
  get stats() { return this.core ? this.core.stats : null; }
}

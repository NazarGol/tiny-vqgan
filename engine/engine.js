// The light painting engine: tiny decoder + token scorer as WebGL2 passes, palette + bank data, no ML runtime.
// load() -> encodeText() (worker, see text.js) -> paintStroke() -> decode(); release() frees everything.
import { GLNN } from '../lib/glnn.js';
import { TinyDecoderGL } from '../lib/tinydec.js';
import { TinyScorerGL } from '../lib/tinyscorer.js';
import { ClipVisionGL } from '../lib/clipvision.js';
import { Palette } from '../lib/palette.js';
import { Bank } from '../lib/bank.js';
import { fetchCached } from '../lib/models.js';

/** Low-memory model fetch for first visits: stream the response straight into Cache Storage (no JS copy), then read it back once.
 *  WebKit holds several copies of a body fetched into JS (a 22 MB file cost +77 MB plain, +134 MB with put(clone) + read). */
export async function fetchLowMem(url, { onProgress = null, cacheName = 'vqpaint-models-v1', mirror = null } = {}) {
  let cache = null; try { cache = await caches.open(cacheName); } catch (_) { cache = null; }
  let tryUrls = mirror ? [url, mirror(url)].filter(Boolean) : [url];
  if (tryUrls.length > 1 && /\/tiny\//.test(url)) tryUrls = [tryUrls[1], tryUrls[0]];   // the light engine's files live on GitHub Pages only: skip the Hugging Face round trip
  if (cache) {
    for (const u of tryUrls) { const hit = await cache.match(u); if (hit) { const buf = await hit.arrayBuffer(); onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, cached: true }); return buf; } }
    for (const u of tryUrls) {
      try { const resp = await fetch(u); if (!resp.ok) throw new Error(`fetch ${u}: ${resp.status}`); onProgress?.({ url, loaded: 0, total: +resp.headers.get('content-length') || 0, cached: false }); await cache.put(u, resp); }
      catch (e) { if (u === tryUrls[tryUrls.length - 1]) throw e; console.warn('model host failed, trying the fallback:', e && e.message); continue; }
      const hit = await cache.match(u); if (hit) { const buf = await hit.arrayBuffer(); onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, cached: false }); return buf; }
    }
  }
  return fetchCached(url, { onProgress, cacheName });
}
import { TokenPainter } from './search.js';

export class Engine {
  /**
   * base: URL of the model files (tiny/, palette/, bank/). variant: decoder variant letter; scorer: scorer variant letter.
   * fetchBuf(url) may be the app's cached fetch; onProgress({url, loaded, total}) per file.
   */
  static async load({ base, variant = 'A', scorer = 'S', text = 'S', clip = 'clip_vision', bank = 'bank', fetchBuf = null, onProgress = null, textEncoder = null, tokenizerUrl = null, mirrorOf = null } = {}) {
    const fb = fetchBuf || ((u) => fetchLowMem(u, { onProgress, mirror: mirrorOf }));
    // the text model files are fetched now (so they are cached and counted) but only parsed inside a worker, per note
    const textUrls = text ? { jsonUrl: base + `tiny/tiny_text_${text}.json`, binUrl: base + `tiny/tiny_text_${text}.bin`, tokenizerUrl: tokenizerUrl || base + 'mobileclip_s0/tokenizer.json', mirror: mirrorOf } : null;
    if (textUrls) { for (const u of [textUrls.tokenizerUrl, textUrls.jsonUrl, textUrls.binUrl]) await fb(u); }   // cached now; the worker reads them from Cache Storage (with the same mirror)
    const nn = new GLNN();
    // variant 'auto': start with A; if a 256 px decode takes more than ~60 ms on this GPU, switch to the lighter B
    const want = variant === 'auto' ? 'A' : variant;
    // one model at a time: each download buffer is released before the next one is held
    let dec = await TinyDecoderGL.load(base + `tiny/tiny_decoder_${want}.json`, base + `tiny/tiny_decoder_${want}.bin`, { fetchBuf: fb, nn });
    const sc = scorer ? await TinyScorerGL.load(base + `tiny/tiny_scorer_${scorer}.json`, base + `tiny/tiny_scorer_${scorer}.bin`, { fetchBuf: fb, nn }) : null;
    const cv = clip ? await ClipVisionGL.load(base + `tiny/${clip}.json`, base + `tiny/${clip}.bin`, { fetchBuf: fb, nn }) : null;   // the real MobileCLIP-S0 image tower (21.7 MB fp16)
    const palette = await Palette.load(base + 'palette/');
    const bk = bank ? await Bank.load(base + bank + '/').catch((e) => { console.warn('bank', e); return null; }) : null;
    if (variant === 'auto') {
      const probe = new Int32Array(256).fill(6328), ms = []; for (let i = 0; i < 4; i++) { dec.decodeRGBA(probe, 16, 16); ms.push(dec.stats.lastMs); }
      const med = ms.sort((a, b) => a - b)[2];
      if (med > 60) { try { const b = await TinyDecoderGL.load(base + 'tiny/tiny_decoder_B.json', base + 'tiny/tiny_decoder_B.bin', { fetchBuf: fb, nn }); dec.release(); dec = b; dec.variant = 'B'; } catch (e) { console.warn('variant B unavailable', e); } }
      dec.probeMs = med; dec.variant ||= 'A';
    }
    dec.m = { emb: dec.m.emb, widths: dec.m.widths }; if (cv) cv.m = { input: cv.m.input, ops: cv.m.ops.length };   // manifests are not needed after load
    return new Engine({ nn, decoder: dec, scorer: sc, clip: cv, palette, bank: bk, textEncoder, base, textUrls });
  }

  constructor({ nn, decoder, scorer, clip = null, palette, bank, textEncoder, base, textUrls = null }) {
    this.nn = nn; this.decoder = decoder; this.scorer = scorer; this.clip = clip; this.palette = palette; this.bank = bank; this.textEncoder = textEncoder; this.base = base; this.textUrls = textUrls; this.worker = null;
    this.painter = new TokenPainter({ scorer, decoder, clip, palette, bank });
    this.stats = { strokes: 0, lastTries: 0, lastSeconds: 0, decodeMs: () => decoder.stats.lastMs, renderer: nn.renderer, halfRT: nn.halfRT, weightBytes: decoder.weightBytes + (scorer ? scorer.weightBytes : 0) + (clip ? clip.weightBytes : 0) };
  }

  /** texts -> unit embeddings (512 each). One Worker holds the tokenizer + tiny text model (~25 MB) off the main thread; it is
   *  created on first use and kept (WebKit does not give a terminated worker's memory back, so one long-lived worker is cheaper
   *  than one per note); releaseText() terminates it. */
  async encodeTexts(texts) {
    if (this.textEncoder) return Promise.all(texts.map((t) => this.textEncoder(t)));
    if (!this.textUrls) throw new Error('no text encoder');
    const abs = (u) => new URL(u, location.href).href;
    if (!this.worker) {
      const w = new Worker(new URL('./text_worker.js', import.meta.url), { type: 'module' });
      const call = (msg) => new Promise((resolve, reject) => { const id = Math.random(); const h = (ev) => { if (ev.data.id !== id) return; w.removeEventListener('message', h); ev.data.error ? reject(new Error(ev.data.error)) : resolve(ev.data); }; w.addEventListener('message', h); w.onerror = (e) => reject(new Error(e.message)); w.postMessage({ id, ...msg }); });
      const m = this.textUrls.mirror, alt = (u) => (m ? m(u) : null);
      this.worker = { w, call, ready: call({ init: { tokenizerUrl: abs(this.textUrls.tokenizerUrl), jsonUrl: abs(this.textUrls.jsonUrl), binUrl: abs(this.textUrls.binUrl), alt: { tokenizerUrl: alt(this.textUrls.tokenizerUrl), jsonUrl: alt(this.textUrls.jsonUrl), binUrl: alt(this.textUrls.binUrl) } } }) };
    }
    try { await this.worker.ready; } catch (e) { this.releaseText(); throw e; }
    const r = await this.worker.call({ texts }); this.stats.lastTextMs = r.ms; return r.embeddings;
  }
  releaseText() { if (this.worker) { this.worker.w.terminate(); this.worker = null; } }
  async encodeText(text) { return (await this.encodeTexts([text]))[0]; }
  /** Object with the Clip.embedText signature, so lib/text.js embedLongText() works unchanged. */
  get clipLike() { return { embedText: (t) => this.encodeText(t) }; }

  /** Paint one stroke. mask {x,y,w,h,cells}; grid {w,h,tokens}; target unit embedding; onPreview({image:{rgba,w,h}, crop, tokens, score, steps, elapsed, final}).
   *  mode: 'clip' (real MobileCLIP on the tiny decoder's output, default), 'prefilter' (token scorer picks the top few of each batch, real CLIP decides), 'token' (token scorer only; not for production). */
  async paintStroke({ grid, mask, target, seconds = 10, onPreview = null, signal = null, mode = this.clip ? 'clip' : 'token', ...params }) {
    const t0 = performance.now();
    let res;
    try { res = await this.painter.paint({ grid, mask, target, seconds, onProgress: onPreview, signal, mode, ...params }); }
    finally { this.nn.trim(this.poolBudget || 24 * 2 ** 20); }   // keep GPU memory flat across strokes of different sizes
    this.stats.strokes++; this.stats.lastTries = res.steps; this.stats.lastSeconds = (performance.now() - t0) / 1000;
    return res;
  }

  /** tokens (Int32Array|Uint16Array h*w) -> {rgba, w, h} pixels. */
  decode(tokens, h, w) { return this.decoder.decodeRGBA(tokens, h, w); }

  /** Unit CLIP embedding of an RGBA8 image (Uint8ClampedArray w*h*4), e.g. a note's photo. */
  embedImage(rgba, w, h) {
    if (!this.clip) throw new Error('no CLIP image tower loaded');
    const gl = this.nn.gl, tex = this.nn.tex(gl.RGBA8, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba); this.clip.setInput(tex, w, h); const e = this.clip.embed(); gl.deleteTexture(tex); return e;
  }
  release() { this.releaseText(); this.decoder.release(); if (this.scorer) this.scorer.release(); if (this.clip) this.clip.release(); this.nn.destroy(); }
}

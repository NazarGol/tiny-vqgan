// The light painting engine: tiny decoder + token scorer as WebGL2 passes, palette + bank data, no ML runtime.
// load() -> encodeText() (worker, see text.js) -> paintStroke() -> decode(); release() frees everything.
import { GLNN } from '../lib/glnn.js';
import { TinyDecoderGL } from '../lib/tinydec.js';
import { TinyScorerGL } from '../lib/tinyscorer.js';
import { ClipVisionGL } from '../lib/clipvision.js';
import { Palette } from '../lib/palette.js';
import { Bank } from '../lib/bank.js';
import { fetchCached } from '../lib/models.js';
import { TokenPainter } from './search.js';

/** Model fetch that never hangs: tiny files from the mirror (GitHub Pages) first, per-attempt timeout with stall detection,
 *  3 attempts across the URLs, streaming byte progress, Cache Storage when it works (put after the download) and plain memory
 *  when it does not (private tabs, quota errors). `mirror` and `mirrorOf` are the same option (older callers used either name). */
export async function fetchLowMem(url, { onProgress = null, cacheName = 'vqpaint-models-v1', mirror = null, mirrorOf = null, timeoutMs = 45000, attempts = 3, log = null } = {}) {
  const mir = mirror || mirrorOf; const alt = mir ? mir(url) : null;
  let urls = alt ? [url, alt] : [url];
  if (alt && /\/tiny\//.test(url)) urls = [alt, url];               // the light engine's files live on Pages only
  const rlog = (e) => { try { (log || globalThis.__rlog)?.({ t: 'fetch', url: e.url || url, ...e }); } catch (_) {} };
  let cache = null;
  if (!(globalThis.__noCache || /[?&]nocache=1/.test(typeof location !== 'undefined' ? location.search || '' : ''))) { try { cache = await caches.open(cacheName); } catch (e) { cache = null; rlog({ cache: 'unavailable', error: String(e && e.message) }); } }
  if (cache) {
    for (const u of urls) {
      let hit = null; try { hit = await cache.match(u); } catch (_) {}
      if (hit) { const buf = await hit.arrayBuffer(); onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, cached: true }); rlog({ url: u, status: 'cache', bytes: buf.byteLength, ms: 0 }); return buf; }
    }
  }
  let lastErr = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const u = urls[Math.min(attempt, urls.length - 1)], t0 = performance.now(), ac = new AbortController();
    let stallTimer = null; const arm = () => { clearTimeout(stallTimer); stallTimer = setTimeout(() => ac.abort(), timeoutMs); };
    try {
      arm(); const resp = await fetch(u, { signal: ac.signal });
      if (!resp.ok) { rlog({ url: u, status: resp.status, ms: Math.round(performance.now() - t0) }); throw new Error(`fetch ${u}: ${resp.status}`); }
      const total = +resp.headers.get('content-length') || 0; let buf = total ? new Uint8Array(total) : null, loaded = 0; const chunks = buf ? null : [];
      const reader = resp.body.getReader();
      for (;;) { const { done, value } = await reader.read(); if (done) break; arm();
        if (buf) { if (loaded + value.length > buf.length) { const big = new Uint8Array(Math.max(buf.length * 2, loaded + value.length)); big.set(buf.subarray(0, loaded)); buf = big; } buf.set(value, loaded); } else chunks.push(value);
        loaded += value.length; onProgress?.({ url, loaded, total, cached: false }); }
      clearTimeout(stallTimer);
      if (!buf) { buf = new Uint8Array(loaded); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; } } else if (loaded !== buf.length) buf = buf.slice(0, loaded);
      rlog({ url: u, status: 200, bytes: loaded, ms: Math.round(performance.now() - t0), attempt });
      if (cache) { try { await cache.put(u, new Response(buf, { headers: { 'Content-Type': resp.headers.get('content-type') || 'application/octet-stream', 'Content-Length': String(loaded) } })); } catch (e) { rlog({ url: u, cache: 'put failed', error: String(e && e.message) }); } }
      return buf.buffer;
    } catch (e) { clearTimeout(stallTimer); lastErr = e; rlog({ url: u, error: String(e && e.message), ms: Math.round(performance.now() - t0), attempt }); }
  }
  throw new Error(`could not load ${url.split('/').slice(-2).join('/')}: ${lastErr && lastErr.message}`);
}

/** Fresh copy of tiny/manifest.json ({ file: hash }): Pages first like the other tiny files, 8 s per URL, never throws. A copy
 *  is kept in Cache Storage so a flaky network does not fall back to unversioned URLs (which would re-download everything). */
async function loadManifest(url, mirror, rlog) {
  const alt = mirror ? mirror(url) : null, urls = alt ? [alt, url] : [url], key = url + '?last=1';
  let cache = null; if (!globalThis.__noCache) { try { cache = await caches.open('vqpaint-models-v1'); } catch (_) {} }
  for (const u of urls) {
    const ac = new AbortController(), timer = setTimeout(() => ac.abort(), 8000);
    try {
      const r = await fetch(u, { cache: 'no-cache', signal: ac.signal }); clearTimeout(timer);
      if (r.ok) { const txt = await r.text(), m = JSON.parse(txt); if (cache) { try { await cache.put(key, new Response(txt, { headers: { 'Content-Type': 'application/json' } })); } catch (_) {} } rlog({ step: 'manifest', url: u, files: Object.keys(m).length }); return m; }
      rlog({ step: 'manifest', url: u, status: r.status });
    } catch (e) { clearTimeout(timer); rlog({ step: 'manifest', url: u, error: String(e && e.message) }); }
  }
  if (cache) { try { const hit = await cache.match(key); if (hit) { const m = await hit.json(); rlog({ step: 'manifest', url: 'cache', files: Object.keys(m).length }); return m; } } catch (_) {} }
  return {};
}

/** Delete cached tiny files whose ?v= hash differs from the manifest (or that have none); files the manifest does not list stay. */
async function evictStale(man, rlog) {
  if (!Object.keys(man).length || globalThis.__noCache) return;
  try {
    const cache = await caches.open('vqpaint-models-v1'); let n = 0;
    for (const req of await cache.keys()) {
      const u = req.url, i = u.indexOf('/tiny/'); if (i < 0) continue;
      const [name, q] = u.slice(i + 6).split('?'), h = man[name]; if (!h || q === 'v=' + h) continue;
      if (await cache.delete(req)) n++;
    }
    if (n) rlog({ step: 'evict', removed: n });
  } catch (_) {}
}

export class Engine {
  /**
   * base: URL of the model files (tiny/, palette/, bank/). variant: decoder variant letter; scorer: scorer variant letter.
   * fetchBuf(url) may be the app's cached fetch; onProgress({url, loaded, total}) per file.
   */
  static async load({ base, variant = 'A', scorer = 'S', text = 'S', clip = 'clip_vision', bank = 'bank', fetchBuf = null, onProgress = null, textEncoder = null, tokenizerUrl = null, mirrorOf = null } = {}) {
    const fb = fetchBuf || ((u) => fetchLowMem(u, { onProgress, mirror: mirrorOf }));
    const rlog = (e) => { try { globalThis.__rlog?.({ t: 'engine', ...e }); } catch (_) {} };
    // content hashes: tiny/manifest.json (small, fetched fresh) → every tiny file is requested as name?v=<hash>, so a new model
    // reaches every device by itself (the Cache Storage key changes) and stale copies are evicted after a successful load
    const man = await loadManifest(base + 'tiny/manifest.json', mirrorOf, rlog);
    const T = (name) => { const u = base + 'tiny/' + name, h = man[name]; return h ? `${u}?v=${h}` : u; };
    // the text model files are fetched now (so they are cached and counted) but only parsed inside a worker, per note
    const textUrls = text ? { jsonUrl: T(`tiny_text_${text}.json`), binUrl: T(`tiny_text_${text}.bin`), tokenizerUrl: tokenizerUrl || base + 'mobileclip_s0/tokenizer.json', mirror: mirrorOf } : null;
    if (textUrls) { for (const u of [textUrls.tokenizerUrl, textUrls.jsonUrl, textUrls.binUrl]) await fb(u); }   // cached now; the worker reads them from Cache Storage (with the same mirror)
    let nn; try { nn = new GLNN(); } catch (e) { rlog({ step: 'webgl2', error: String(e && e.message) }); throw new Error('WebGL2 is not available: ' + (e && e.message)); }
    rlog({ step: 'webgl2', renderer: nn.renderer, maxTex: nn.maxTex, maxUbo: nn.maxUbo, halfRT: nn.halfRT, extF: nn.extF, worker: typeof document === 'undefined' });
    // variant 'auto': start with A; if a 256 px decode takes more than ~60 ms on this GPU, switch to the lighter B
    const want = variant === 'auto' ? 'A' : variant;
    // one model at a time: each download buffer is released before the next one is held
    let dec = await TinyDecoderGL.load(T(`tiny_decoder_${want}.json`), T(`tiny_decoder_${want}.bin`), { fetchBuf: fb, nn });
    dec.hash = man[`tiny_decoder_${want}.bin`] || null; rlog({ step: 'decoder', variant: want, hash: dec.hash, ms: Math.round(performance.now()) });
    const sc = scorer ? await TinyScorerGL.load(T(`tiny_scorer_${scorer}.json`), T(`tiny_scorer_${scorer}.bin`), { fetchBuf: fb, nn }) : null;
    rlog({ step: 'scorer', variant: scorer || null, hash: scorer ? man[`tiny_scorer_${scorer}.bin`] || null : null });
    const cv = clip ? await ClipVisionGL.load(T(`${clip}.json`), T(`${clip}.bin`), { fetchBuf: fb, nn }) : null;   // the real MobileCLIP-S0 image tower (21.7 MB fp16)
    rlog({ step: 'clip', file: clip || null, hash: clip ? man[`${clip}.bin`] || null : null, programs: nn.programs.size });
    const palette = await Palette.load(base + 'palette/');
    const bk = bank ? await Bank.load(base + bank + '/').catch((e) => { console.warn('bank', e); return null; }) : null;
    rlog({ step: 'data', bank: !!bk });
    if (variant === 'auto') {
      const probe = new Int32Array(256).fill(6328), ms = []; for (let i = 0; i < 4; i++) { dec.decodeRGBA(probe, 16, 16); ms.push(dec.stats.lastMs); }
      const med = ms.sort((a, b) => a - b)[2];
      if (med > 60) { try { const b = await TinyDecoderGL.load(T('tiny_decoder_B.json'), T('tiny_decoder_B.bin'), { fetchBuf: fb, nn }); dec.release(); dec = b; dec.variant = 'B'; dec.hash = man['tiny_decoder_B.bin'] || null; } catch (e) { console.warn('variant B unavailable', e); } }
      dec.probeMs = med; dec.variant ||= 'A';
    }
    dec.m = { emb: dec.m.emb, widths: dec.m.widths }; if (cv) cv.m = { input: cv.m.input, ops: cv.m.ops.length };   // manifests are not needed after load
    evictStale(man, rlog);   // background: drop cached tiny files whose hash is no longer current
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

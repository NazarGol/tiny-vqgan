// The light painting engine: tiny decoder + token scorer as WebGL2 passes, palette + bank data, no ML runtime.
// load() -> encodeText() (worker, see text.js) -> paintStroke() -> decode(); release() frees everything.
import { GLNN } from '../lib/glnn.js';
import { TinyDecoderGL } from '../lib/tinydec.js';
import { TinyScorerGL } from '../lib/tinyscorer.js';
import { Palette } from '../lib/palette.js';
import { Bank } from '../lib/bank.js';
import { fetchCached } from '../lib/models.js';
import { TokenPainter } from './search.js';

export class Engine {
  /**
   * base: URL of the model files (tiny/, palette/, bank/). variant: decoder variant letter; scorer: scorer variant letter.
   * fetchBuf(url) may be the app's cached fetch; onProgress({url, loaded, total}) per file.
   */
  static async load({ base, variant = 'A', scorer = 'S', text = 'S', bank = 'bank', fetchBuf = null, onProgress = null, textEncoder = null, tokenizerUrl = null } = {}) {
    const fb = fetchBuf || ((u) => fetchCached(u, { onProgress }));
    // the text model files are fetched now (so they are cached and counted) but only parsed inside a worker, per note
    const textUrls = text ? { jsonUrl: base + `tiny/tiny_text_${text}.json`, binUrl: base + `tiny/tiny_text_${text}.bin`, tokenizerUrl: tokenizerUrl || base + 'mobileclip_s0/tokenizer.json' } : null;
    if (textUrls) await Promise.all([fb(textUrls.tokenizerUrl), fb(textUrls.jsonUrl), fb(textUrls.binUrl)]);   // cached; the worker parses them
    const nn = new GLNN();
    const [dec, sc, palette, bk] = await Promise.all([
      TinyDecoderGL.load(base + `tiny/tiny_decoder_${variant}.json`, base + `tiny/tiny_decoder_${variant}.bin`, { fetchBuf: fb, nn }),
      TinyScorerGL.load(base + `tiny/tiny_scorer_${scorer}.json`, base + `tiny/tiny_scorer_${scorer}.bin`, { fetchBuf: fb, nn }),
      Palette.load(base + 'palette/'),
      bank ? Bank.load(base + bank + '/').catch((e) => { console.warn('bank', e); return null; }) : null,
    ]);
    return new Engine({ nn, decoder: dec, scorer: sc, palette, bank: bk, textEncoder, base, textUrls });
  }

  constructor({ nn, decoder, scorer, palette, bank, textEncoder, base, textUrls = null }) {
    this.nn = nn; this.decoder = decoder; this.scorer = scorer; this.palette = palette; this.bank = bank; this.textEncoder = textEncoder; this.base = base; this.textUrls = textUrls; this.worker = null;
    this.painter = new TokenPainter({ scorer, decoder, palette, bank });
    this.stats = { strokes: 0, lastTries: 0, lastSeconds: 0, decodeMs: () => decoder.stats.lastMs, renderer: nn.renderer, halfRT: nn.halfRT, weightBytes: decoder.weightBytes + scorer.weightBytes };
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
      this.worker = { w, call, ready: call({ init: { tokenizerUrl: abs(this.textUrls.tokenizerUrl), jsonUrl: abs(this.textUrls.jsonUrl), binUrl: abs(this.textUrls.binUrl) } }) };
    }
    try { await this.worker.ready; } catch (e) { this.releaseText(); throw e; }
    const r = await this.worker.call({ texts }); this.stats.lastTextMs = r.ms; return r.embeddings;
  }
  releaseText() { if (this.worker) { this.worker.w.terminate(); this.worker = null; } }
  async encodeText(text) { return (await this.encodeTexts([text]))[0]; }
  /** Object with the Clip.embedText signature, so lib/text.js embedLongText() works unchanged. */
  get clipLike() { return { embedText: (t) => this.encodeText(t) }; }

  /** Paint one stroke. mask {x,y,w,h,cells}; grid {w,h,tokens}; target unit embedding; onPreview({image:{rgba,w,h}, crop, tokens, score, steps, elapsed, final}). */
  async paintStroke({ grid, mask, target, seconds = 10, onPreview = null, signal = null, ...params }) {
    const t0 = performance.now();
    let res;
    try { res = await this.painter.paint({ grid, mask, target, seconds, onProgress: onPreview, signal, ...params }); }
    finally { this.nn.trim(this.poolBudget || 24 * 2 ** 20); }   // keep GPU memory flat across strokes of different sizes
    this.stats.strokes++; this.stats.lastTries = res.steps; this.stats.lastSeconds = (performance.now() - t0) / 1000;
    return res;
  }

  /** tokens (Int32Array|Uint16Array h*w) -> {rgba, w, h} pixels. */
  decode(tokens, h, w) { return this.decoder.decodeRGBA(tokens, h, w); }

  release() { this.releaseText(); this.decoder.release(); this.scorer.release(); this.nn.destroy(); }
}

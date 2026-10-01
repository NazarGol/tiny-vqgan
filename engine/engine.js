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
  static async load({ base, variant = 'A', scorer = 'S', bank = 'bank', fetchBuf = null, onProgress = null, textEncoder = null } = {}) {
    const fb = fetchBuf || ((u) => fetchCached(u, { onProgress }));
    const nn = new GLNN();
    const [dec, sc, palette, bk] = await Promise.all([
      TinyDecoderGL.load(base + `tiny/tiny_decoder_${variant}.json`, base + `tiny/tiny_decoder_${variant}.bin`, { fetchBuf: fb, nn }),
      TinyScorerGL.load(base + `tiny/tiny_scorer_${scorer}.json`, base + `tiny/tiny_scorer_${scorer}.bin`, { fetchBuf: fb, nn }),
      Palette.load(base + 'palette/'),
      bank ? Bank.load(base + bank + '/').catch((e) => { console.warn('bank', e); return null; }) : null,
    ]);
    return new Engine({ nn, decoder: dec, scorer: sc, palette, bank: bk, textEncoder, base });
  }

  constructor({ nn, decoder, scorer, palette, bank, textEncoder, base }) {
    this.nn = nn; this.decoder = decoder; this.scorer = scorer; this.palette = palette; this.bank = bank; this.textEncoder = textEncoder; this.base = base;
    this.painter = new TokenPainter({ scorer, decoder, palette, bank });
    this.stats = { strokes: 0, lastTries: 0, lastSeconds: 0, decodeMs: () => decoder.stats.lastMs, renderer: nn.renderer, halfRT: nn.halfRT, weightBytes: decoder.weightBytes + scorer.weightBytes };
  }

  /** text -> unit embedding (512). Uses the injected text encoder (phase 3); callers may also pass targets straight to paintStroke. */
  async encodeText(text) { if (!this.textEncoder) throw new Error('no text encoder attached'); return this.textEncoder(text); }

  /** Paint one stroke. mask {x,y,w,h,cells}; grid {w,h,tokens}; target unit embedding; onPreview({image:{rgba,w,h}, crop, tokens, score, steps, elapsed, final}). */
  async paintStroke({ grid, mask, target, seconds = 10, onPreview = null, signal = null, ...params }) {
    const t0 = performance.now();
    const res = await this.painter.paint({ grid, mask, target, seconds, onProgress: onPreview, signal, ...params });
    this.stats.strokes++; this.stats.lastTries = res.steps; this.stats.lastSeconds = (performance.now() - t0) / 1000;
    return res;
  }

  /** tokens (Int32Array|Uint16Array h*w) -> {rgba, w, h} pixels. */
  decode(tokens, h, w) { return this.decoder.decodeRGBA(tokens, h, w); }

  release() { this.decoder.release(); this.scorer.release(); this.nn.destroy(); }
}

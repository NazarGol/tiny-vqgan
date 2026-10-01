// Tiny CLIP-space text encoder in plain JavaScript (typed arrays): ids (77, BOS … EOS, 0-padded) -> unit embedding (512).
// Weights: fp16 binary + JSON manifest from web/research/tiny/common.py export_text. Runs once per note, usually in a Worker.
import { f16ToF32 } from '../lib/glnn.js';

export class TinyTextJS {
  static async load(jsonUrl, binUrl, { fetchBuf = (u) => fetch(u).then((r) => r.arrayBuffer()) } = {}) {
    const [manifest, bin] = await Promise.all([fetchBuf(jsonUrl).then((b) => JSON.parse(new TextDecoder().decode(b))), fetchBuf(binUrl)]);
    return new TinyTextJS(manifest, new Uint16Array(bin));
  }
  constructor(m, f16) {
    // all weights stay fp16 (one copy, 6–9 MB); each tensor is converted to fp32 while it is used and freed right after
    this.m = m; this.f16 = f16; const T = m.tensors; this.t = (n) => { const t = T[n], len = t.shape.reduce((a, b) => a * b, 1); return f16ToF32(f16.subarray(t.offset, t.offset + len)); };
    this.W = m.width; this.H = m.heads; this.E = m.emb_dim; this.ctx = m.ctx; this.out_dim = m.out_dim; this.tokOff = T.tok.offset;
    this.weightBytes = f16.byteLength;
  }
  // y[T,O] = x[T,I] · W[O,I]^T + b
  _linear(x, Tn, I, w, b, O) { const y = new Float32Array(Tn * O); for (let t = 0; t < Tn; t++) { const xo = t * I; for (let o = 0; o < O; o++) { let s = b ? b[o] : 0; const wo = o * I; for (let i = 0; i < I; i++) s += x[xo + i] * w[wo + i]; y[t * O + o] = s; } } return y; }
  _ln(x, Tn, W, g, b) { const y = new Float32Array(Tn * W); for (let t = 0; t < Tn; t++) { let m = 0; for (let i = 0; i < W; i++) m += x[t * W + i]; m /= W; let v = 0; for (let i = 0; i < W; i++) { const d = x[t * W + i] - m; v += d * d; } const r = 1 / Math.sqrt(v / W + 1e-5); for (let i = 0; i < W; i++) y[t * W + i] = (x[t * W + i] - m) * r * g[i] + b[i]; } return y; }
  /** ids: Int32Array|number[] of length ≤ 77 (BOS … EOS, zero padded). Returns Float32Array(512), unit length. */
  encode(ids) {
    const W = this.W, H = this.H, E = this.E, hd = W / H;
    let Tn = ids.length; let eos = 0; for (let t = 0; t < ids.length; t++) if (ids[t] > ids[eos]) eos = t;   // CLIP pools at argmax(ids) = EOS
    Tn = Math.min(Tn, eos + 1);   // positions after EOS cannot influence it (causal), so skip them
    // token table (fp16 rows) -> proj_in + pos
    const x = new Float32Array(Tn * W), pinw = this.t('proj_in.w'), pinb = this.t('proj_in.b'), pos = this.t('pos');
    for (let t = 0; t < Tn; t++) {
      const r = f16ToF32(this.f16.subarray(this.tokOff + ids[t] * E, this.tokOff + ids[t] * E + E));
      for (let o = 0; o < W; o++) { let s = pinb[o]; const wo = o * E; for (let i = 0; i < E; i++) s += r[i] * pinw[wo + i]; x[t * W + o] = s + pos[t * W + o]; }
    }
    const att = new Float32Array(Tn);
    for (let bi = 0; bi < this.m.layers; bi++) {
      const g = (n) => this.t(`b${bi}.${n}`), B = { ln1g: g('ln1.g'), ln1b: g('ln1.b'), qkvw: g('qkv.w'), qkvb: g('qkv.b'), pw: g('proj.w'), pb: g('proj.b'), ln2g: g('ln2.g'), ln2b: g('ln2.b'), f1w: g('fc1.w'), f1b: g('fc1.b'), f2w: g('fc2.w'), f2b: g('fc2.b') };
      const h = this._ln(x, Tn, W, B.ln1g, B.ln1b), qkv = this._linear(h, Tn, W, B.qkvw, B.qkvb, 3 * W), ctx = new Float32Array(Tn * W), scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) for (let t = 0; t < Tn; t++) {
        let mx = -Infinity; const qo = t * 3 * W + hh * hd;
        for (let s = 0; s <= t; s++) { let d = 0; const ko = s * 3 * W + W + hh * hd; for (let i = 0; i < hd; i++) d += qkv[qo + i] * qkv[ko + i]; att[s] = d * scale; if (att[s] > mx) mx = att[s]; }
        let z = 0; for (let s = 0; s <= t; s++) { att[s] = Math.exp(att[s] - mx); z += att[s]; }
        for (let s = 0; s <= t; s++) { const p = att[s] / z, vo = s * 3 * W + 2 * W + hh * hd; for (let i = 0; i < hd; i++) ctx[t * W + hh * hd + i] += p * qkv[vo + i]; }
      }
      const o = this._linear(ctx, Tn, W, B.pw, B.pb, W); for (let i = 0; i < x.length; i++) x[i] += o[i];
      const h2 = this._ln(x, Tn, W, B.ln2g, B.ln2b), f = this._linear(h2, Tn, W, B.f1w, B.f1b, 4 * W);
      for (let i = 0; i < f.length; i++) { const v = f[i]; f[i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))); }
      const o2 = this._linear(f, Tn, 4 * W, B.f2w, B.f2b, W); for (let i = 0; i < x.length; i++) x[i] += o2[i];
    }
    const last = this._ln(x.subarray(eos * W, eos * W + W), 1, W, this.t('ln_f.g'), this.t('ln_f.b')), e = this._linear(last, 1, W, this.t('out.w'), null, this.out_dim);
    let n = 0; for (let i = 0; i < e.length; i++) n += e[i] * e[i]; n = Math.sqrt(n) + 1e-8; for (let i = 0; i < e.length; i++) e[i] /= n;
    return e;
  }
}

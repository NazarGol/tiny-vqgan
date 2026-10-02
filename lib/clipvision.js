// MobileCLIP-S0 image tower as WebGL2 passes (no ML runtime). Op list + packed weights from web/research/tiny/export_clip_vision.py.
// embed(texture) -> unit 512-d embedding; score(texture) -> cosine with the targets set by setTargets(). Apple ML Research Model
// License applies to these weights (research use only).
import { GLNN, VS, GELU, ACT, f16ToF32, f32ToF16 } from './glnn.js';

const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;`;
const RESIZE_FS = `${HEAD}
uniform sampler2D uSrc; uniform ivec2 uSrcSize; uniform int uOut;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); vec2 sc = vec2(uSrcSize) / float(uOut);
  vec2 f = clamp((vec2(p) + 0.5) * sc - 0.5, vec2(0.0), vec2(uSrcSize) - 1.0); ivec2 i0 = ivec2(f); ivec2 i1 = min(i0 + 1, uSrcSize - 1); vec2 w = f - vec2(i0);
  vec4 c = mix(mix(texelFetch(uSrc, i0, 0), texelFetch(uSrc, ivec2(i1.x, i0.y), 0), w.x), mix(texelFetch(uSrc, ivec2(i0.x, i1.y), 0), texelFetch(uSrc, i1, 0), w.x), w.y);
  o = vec4(c.rgb, 0.0);
}`;
const DW_FS = ({ K, stride, pad, M, act }) => `${HEAD}
uniform sampler2D uIn, uW, uB; uniform ivec2 uInSize; uniform int uInGpr; uniform ivec2 uOutSize; uniform int uOutGpr; uniform int uG;
out vec4 o; ${GELU}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int col = p.x / uOutSize.x, row = p.y / uOutSize.y; int g = row * uOutGpr + col;
  if (g >= uG) { o = vec4(0.0); return; }
  int x = p.x - col * uOutSize.x, y = p.y - row * uOutSize.y;
  vec4 acc = texelFetch(uB, ivec2(g, 0), 0);
  ${M === 1 ? 'int gi = g;' : 'int gi = g >> 1; int c0 = (g & 1) * 2;'}
  for (int t = 0; t < ${K * K}; t++) {
    int sx = x * ${stride} + (t % ${K}) - ${pad}, sy = y * ${stride} + (t / ${K}) - ${pad};
    if (sx < 0 || sy < 0 || sx >= uInSize.x || sy >= uInSize.y) continue;
    vec4 v = texelFetch(uIn, ivec2((gi % uInGpr) * uInSize.x + sx, (gi / uInGpr) * uInSize.y + sy), 0);
    ${M === 1 ? '' : 'v = vec4(v[c0], v[c0], v[c0 + 1], v[c0 + 1]);'}
    acc += texelFetch(uW, ivec2(t, g), 0) * v;
  }
  ${ACT[act]}
  o = acc;
}`;
const SCALE_ADD_FS = `${HEAD}
uniform sampler2D uA, uB, uS; uniform ivec2 uSize; uniform int uGpr;
out vec4 o;
void main() { ivec2 p = ivec2(gl_FragCoord.xy); int g = (p.y / uSize.y) * uGpr + p.x / uSize.x; o = texelFetch(uA, p, 0) + texelFetch(uS, ivec2(g, 0), 0) * texelFetch(uB, p, 0); }`;
const GMEAN_FS = `${HEAD}
uniform sampler2D uIn; uniform ivec2 uSize; uniform int uGpr;
out vec4 o;
void main() { int g = int(gl_FragCoord.x); vec4 acc = vec4(0.0); int bx = (g % uGpr) * uSize.x, by = (g / uGpr) * uSize.y;
  for (int y = 0; y < uSize.y; y++) for (int x = 0; x < uSize.x; x++) acc += texelFetch(uIn, ivec2(bx + x, by + y), 0);
  o = acc / float(uSize.x * uSize.y); }`;
const CHANMUL_FS = (act) => `${HEAD}
uniform sampler2D uIn, uS; uniform ivec2 uSize; uniform int uGpr;
out vec4 o; ${GELU}
void main() { ivec2 p = ivec2(gl_FragCoord.xy); int g = (p.y / uSize.y) * uGpr + p.x / uSize.x; vec4 acc = texelFetch(uIn, p, 0) * texelFetch(uS, ivec2(g, 0), 0); ${ACT[act]} o = acc; }`;
// attention over N = W*H tokens: scores[q, h, key] packed 4 keys per texel (x = key/4, y = q*heads + h)
const ATT_SCORES_FS = `${HEAD}
uniform sampler2D uQKV; uniform ivec2 uSize; uniform int uGpr; uniform int uHeads; uniform int uDG; uniform float uScale;
out vec4 o;
vec4 at(int g, int t) { return texelFetch(uQKV, ivec2((g % uGpr) * uSize.x + t % uSize.x, (g / uGpr) * uSize.y + t / uSize.x), 0); }
void main() { ivec2 p = ivec2(gl_FragCoord.xy); int kx = p.x, q = p.y / uHeads, h = p.y - q * uHeads; int C4 = uHeads * uDG;
  vec4 s = vec4(0.0);
  for (int d = 0; d < uDG; d++) { vec4 qv = at(h * uDG + d, q); int kg = C4 + h * uDG + d;
    s += vec4(dot(qv, at(kg, kx * 4)), dot(qv, at(kg, kx * 4 + 1)), dot(qv, at(kg, kx * 4 + 2)), dot(qv, at(kg, kx * 4 + 3))); }
  o = s * uScale; }`;
const ATT_APPLY_FS = `${HEAD}
uniform sampler2D uQKV, uS; uniform ivec2 uSize; uniform int uGpr; uniform int uOutGpr; uniform int uHeads; uniform int uDG; uniform int uN;
out vec4 o;
vec4 at(int g, int t) { return texelFetch(uQKV, ivec2((g % uGpr) * uSize.x + t % uSize.x, (g / uGpr) * uSize.y + t / uSize.x), 0); }
void main() { ivec2 p = ivec2(gl_FragCoord.xy); int col = p.x / uSize.x, row = p.y / uSize.y; int g = row * uOutGpr + col; int x = p.x - col * uSize.x, y = p.y - row * uSize.y; int q = y * uSize.x + x;
  int h = g / uDG; int vg = 2 * uHeads * uDG + g; int nk4 = uN / 4; float m = -1e30;
  for (int k4 = 0; k4 < nk4; k4++) { vec4 s = texelFetch(uS, ivec2(k4, q * uHeads + h), 0); m = max(m, max(max(s.x, s.y), max(s.z, s.w))); }
  float z = 0.0; vec4 acc = vec4(0.0);
  for (int k4 = 0; k4 < nk4; k4++) { vec4 e = exp(texelFetch(uS, ivec2(k4, q * uHeads + h), 0) - m); z += e.x + e.y + e.z + e.w;
    acc += e.x * at(vg, k4 * 4) + e.y * at(vg, k4 * 4 + 1) + e.z * at(vg, k4 * 4 + 2) + e.w * at(vg, k4 * 4 + 3); }
  o = acc / z; }`;
const PACK_FS = `${HEAD}
uniform sampler2D uIn; uniform int uG;
out vec4 o;
void main() { int i = int(gl_FragCoord.x); float n = 0.0; for (int g = 0; g < uG; g++) { vec4 v = texelFetch(uIn, ivec2(g, 0), 0); n += dot(v, v); }
  vec4 v = texelFetch(uIn, ivec2(i >> 2, 0), 0); float c = (i & 3) == 0 ? v.x : (i & 3) == 1 ? v.y : (i & 3) == 2 ? v.z : v.w;
  uint b = floatBitsToUint(c / sqrt(max(n, 1e-12))); o = vec4(uvec4(b & 255u, (b >> 8) & 255u, (b >> 16) & 255u, (b >> 24) & 255u)) / 255.0; }`;
const SCORE_FS = `${HEAD}
uniform sampler2D uIn, uT; uniform int uG;
out vec4 o;
void main() { int t = int(gl_FragCoord.x); float n = 0.0, d = 0.0; for (int g = 0; g < uG; g++) { vec4 v = texelFetch(uIn, ivec2(g, 0), 0); n += dot(v, v); d += dot(v, texelFetch(uT, ivec2(g, t), 0)); }
  uint b = floatBitsToUint(d / sqrt(max(n, 1e-12))); o = vec4(uvec4(b & 255u, (b >> 8) & 255u, (b >> 16) & 255u, (b >> 24) & 255u)) / 255.0; }`;

export class ClipVisionGL {
  static async load(jsonUrl, binUrl, { fetchBuf = (u) => fetch(u).then((r) => r.arrayBuffer()), nn = null } = {}) {
    const [manifest, bin] = await Promise.all([fetchBuf(jsonUrl).then((b) => JSON.parse(new TextDecoder().decode(b))), fetchBuf(binUrl)]);
    return new ClipVisionGL(manifest, bin, nn);
  }
  constructor(manifest, bin, nn = null) {
    this.m = manifest; this.sharedNN = !!nn; this.nn = nn || new GLNN(); const gl = this.gl = this.nn.gl; this.weightBytes = bin.byteLength;
    const u8 = new Uint8Array(bin), T = manifest.tensors;
    // tensor -> fp16 array (int8 is dequantised with its per-output-channel scales)
    this.f16 = (name) => {
      const e = T[name];
      if (e.dtype === 'f16') return new Uint16Array(bin, e.offset, e.len);
      const q = new Int8Array(bin, e.offset, e.len), sc = new Float32Array(bin, e.scales, e.nscales), out = new Uint16Array(e.len);
      const perGo = e.len / (e.cout / 4);
      for (let i = 0; i < e.len; i++) out[i] = f32ToF16(q[i] * sc[((i / perGo) | 0) * 4 + (i & 3)]);
      return out;
    };
    this.f32 = (name) => f16ToF32(this.f16(name));
    this.progs = { resize: this.nn.program('cv-resize', VS, RESIZE_FS), scaleAdd: this.nn.program('cv-scaleadd', VS, SCALE_ADD_FS), gmean: this.nn.program('cv-gmean', VS, GMEAN_FS), scores: this.nn.program('cv-attn-s', VS, ATT_SCORES_FS), apply: this.nn.program('cv-attn-a', VS, ATT_APPLY_FS), pack: this.nn.program('cv-pack', VS, PACK_FS), score: this.nn.program('cv-score', VS, SCORE_FS) };
    this.vecTex = (name, C) => this.nn.tex(gl.RGBA16F, C / 4, 1, gl.RGBA, gl.HALF_FLOAT, this.f16(name));
    this.layers = manifest.ops.map((op) => this._prepare(op));
    // reference counts: release an activation after its last consumer
    this.uses = {}; for (const op of manifest.ops) for (const k of ['in', 'a', 'b']) if (op[k]) this.uses[op[k]] = (this.uses[op[k]] || 0) + 1;
    this.inputName = manifest.ops[0].in; this.outputName = manifest.ops[manifest.ops.length - 1].out;
    this.inTex = null; this.targetTex = null; this.nTargets = 0; this.scoreTex = null; this.packTex = null; this.stats = { lastMs: 0, embeds: 0 };
  }
  _prepare(op) {
    const gl = this.gl, nn = this.nn;
    const dense = (wName, bName, cin, cout, k, stride, act) => nn.convLayer({ name: 'cv', cin, cout, k, stride, relu: false, act, w: 0, b: cout * cin * k * k }, this._packed(wName, bName, cin, cout, k));
    if (op.op === 'conv') return { op, layer: dense(op.w, op.b, op.cin, op.cout, op.k, op.stride, op.act) };
    if (op.op === 'head') return { op, layer: dense(op.w, null, op.cin, op.cout, 1, 1, 'none') };
    if (op.op === 'dwconv') { const G = op.cout / 4; return { op, w: nn.tex(gl.RGBA16F, op.k * op.k, G, gl.RGBA, gl.HALF_FLOAT, this.f16(op.w)), b: this.vecTex(op.b, op.cout), G, prog: nn.program(`cv-dw-${op.k}-${op.stride}-${op.m}-${op.act}`, VS, DW_FS({ K: op.k, stride: op.stride, pad: op.pad, M: op.m, act: op.act })) }; }
    if (op.op === 'se') return { op, fc1: dense(op.fc1w, op.fc1b, op.c, op.r, 1, 1, 'relu'), fc2: dense(op.fc2w, op.fc2b, op.r, op.c, 1, 1, 'sigmoid'), prog: nn.program(`cv-chanmul-${op.act}`, VS, CHANMUL_FS(op.act)) };
    if (op.op === 'scale_add') return { op, s: this.vecTex(op.s, op.c) };
    if (op.op === 'attn') return { op, qkv: dense(op.qkvw, op.qkvb, op.c, 3 * op.c, 1, 1, 'none'), proj: dense(op.projw, op.projb, op.c, op.c, 1, 1, 'none') };
    if (op.op === 'gmean' || op.op === 'gelu') return { op };
    throw new Error('unknown op ' + op.op);
  }
  /** convLayer() expects one fp16 array holding [weights..., bias...]: build it. */
  _packed(wName, bName, cin, cout, k) {
    const w = this.f16(wName), n = cout * cin * k * k, out = new Uint16Array(n + cout); out.set(w.subarray(0, n), 0);
    if (bName) out.set(this.f16(bName).subarray(0, cout), n);
    return out;
  }
  _acq(C, W, H) { const lay = this.nn.layout(C, W, H); return { t: this.nn.acquire(lay), lay, W, H, C }; }
  _full(prog, out) { const gl = this.gl; this.nn.attach(out.t.tex); gl.useProgram(prog.prog); gl.viewport(0, 0, out.lay.tw, out.lay.th); }
  _bindVec(prog, uniformName, tex, unit) { const gl = this.gl; gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(prog.u[uniformName], unit); }

  /** Set the input image from any RGBA texture (e.g. the tiny decoder's output): bilinear resize to 256×256. */
  setInput(tex, w, h) {
    const gl = this.gl, nn = this.nn, S = this.m.input.w; nn.begin();
    if (!this.inTex) this.inTex = this._acqPersistent(4, S, S);
    this._full(this.progs.resize, this.inTex); this._bindVec(this.progs.resize, 'uSrc', tex, 0); gl.uniform2i(this.progs.resize.u.uSrcSize, w, h); gl.uniform1i(this.progs.resize.u.uOut, S); nn.draw();
  }
  _acqPersistent(C, W, H) { const lay = this.nn.layout(C, W, H); const t = { tex: this.nn.tex(this.nn.rtFormat, lay.tw, lay.th, this.gl.RGBA, this.nn.rtType, null) }; return { t, lay, W, H, C }; }

  _run(stopAt = -1) {
    const gl = this.gl, nn = this.nn, T = new Map(), left = { ...this.uses };
    T.set(this.inputName, this.inTex);
    const get = (name) => { const t = T.get(name); if (!t) throw new Error('missing ' + name); return t; };
    const done = (name) => { if (name === this.inputName) return; if (--left[name] <= 0) { const t = T.get(name); if (t) nn.release(t.t); T.delete(name); } };
    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li], op = L.op;
      if (stopAt >= 0 && li > stopAt) break;
      if (op.op === 'conv' || op.op === 'head') {
        const x = get(op.in), k = op.k || 1, st = op.stride || 1, pd = op.pad || 0, oW = Math.floor((x.W + 2 * pd - k) / st) + 1, oH = Math.floor((x.H + 2 * pd - k) / st) + 1, y = this._acq(op.cout, oW, oH);
        nn.runConv(L.layer, x.t, x.lay, x.W, x.H, y.t, y.lay, oW, oH, null, oH); T.set(op.out, y); done(op.in);
      } else if (op.op === 'dwconv') {
        const x = get(op.in), oW = Math.floor((x.W + 2 * op.pad - op.k) / op.stride) + 1, oH = Math.floor((x.H + 2 * op.pad - op.k) / op.stride) + 1, y = this._acq(op.cout, oW, oH);
        this._full(L.prog, y); const p = L.prog; this._bindVec(p, 'uIn', x.t.tex, 0); this._bindVec(p, 'uW', L.w, 1); this._bindVec(p, 'uB', L.b, 2);
        gl.uniform2i(p.u.uInSize, x.W, x.H); gl.uniform1i(p.u.uInGpr, x.lay.gpr); gl.uniform2i(p.u.uOutSize, oW, oH); gl.uniform1i(p.u.uOutGpr, y.lay.gpr); gl.uniform1i(p.u.uG, L.G); nn.draw();
        T.set(op.out, y); done(op.in);
      } else if (op.op === 'scale_add') {
        const a = get(op.a), b = get(op.b), y = this._acq(a.C, a.W, a.H); this._full(this.progs.scaleAdd, y); const p = this.progs.scaleAdd;
        this._bindVec(p, 'uA', a.t.tex, 0); this._bindVec(p, 'uB', b.t.tex, 1); this._bindVec(p, 'uS', L.s, 2); gl.uniform2i(p.u.uSize, a.W, a.H); gl.uniform1i(p.u.uGpr, a.lay.gpr); nn.draw();
        T.set(op.out, y); done(op.a); done(op.b);
      } else if (op.op === 'gmean' || op.op === 'se') {
        const x = get(op.in), C = op.op === 'se' ? op.c : op.c, v = this._acq(C, 1, 1); this._full(this.progs.gmean, v); const p = this.progs.gmean;
        this._bindVec(p, 'uIn', x.t.tex, 0); gl.uniform2i(p.u.uSize, x.W, x.H); gl.uniform1i(p.u.uGpr, x.lay.gpr); nn.draw();
        if (op.op === 'gmean') { T.set(op.out, v); done(op.in); continue; }
        const h = this._acq(op.r, 1, 1); nn.runConv(L.fc1, v.t, v.lay, 1, 1, h.t, h.lay, 1, 1, null, 1); nn.release(v.t);
        const s = this._acq(op.c, 1, 1); nn.runConv(L.fc2, h.t, h.lay, 1, 1, s.t, s.lay, 1, 1, null, 1); nn.release(h.t);
        const y = this._acq(x.C, x.W, x.H); this._full(L.prog, y); const pc = L.prog; this._bindVec(pc, 'uIn', x.t.tex, 0); this._bindVec(pc, 'uS', s.t.tex, 1); gl.uniform2i(pc.u.uSize, x.W, x.H); gl.uniform1i(pc.u.uGpr, x.lay.gpr); nn.draw();
        nn.release(s.t); T.set(op.out, y); done(op.in);
      } else if (op.op === 'attn') {
        const x = get(op.in), N = x.W * x.H, heads = op.heads, DG = op.c / heads / 4;
        const qkv = this._acq(3 * op.c, x.W, x.H); nn.runConv(L.qkv, x.t, x.lay, x.W, x.H, qkv.t, qkv.lay, x.W, x.H, null, x.H);
        const sc = { t: nn.acquire({ tw: N / 4, th: N * heads }), lay: { tw: N / 4, th: N * heads } };
        this._full(this.progs.scores, sc); let p = this.progs.scores; this._bindVec(p, 'uQKV', qkv.t.tex, 0); gl.uniform2i(p.u.uSize, x.W, x.H); gl.uniform1i(p.u.uGpr, qkv.lay.gpr); gl.uniform1i(p.u.uHeads, heads); gl.uniform1i(p.u.uDG, DG); gl.uniform1f(p.u.uScale, op.scale); nn.draw();
        const o = this._acq(op.c, x.W, x.H); this._full(this.progs.apply, o); p = this.progs.apply; this._bindVec(p, 'uQKV', qkv.t.tex, 0); this._bindVec(p, 'uS', sc.t.tex, 1);
        gl.uniform2i(p.u.uSize, x.W, x.H); gl.uniform1i(p.u.uGpr, qkv.lay.gpr); gl.uniform1i(p.u.uOutGpr, o.lay.gpr); gl.uniform1i(p.u.uHeads, heads); gl.uniform1i(p.u.uDG, DG); gl.uniform1i(p.u.uN, N); nn.draw();
        nn.release(qkv.t); nn.release(sc.t);
        const y = this._acq(op.c, x.W, x.H); nn.runConv(L.proj, o.t, o.lay, x.W, x.H, y.t, y.lay, x.W, x.H, null, x.H); nn.release(o.t);
        T.set(op.out, y); done(op.in);
      } else if (op.op === 'gelu') {
        const x = get(op.in), y = this._acq(x.C, x.W, x.H); const prog = nn.program('cv-chanmul-gelu-id', VS, CHANMUL_FS('gelu'));
        if (!this.onesTex) this.onesTex = nn.tex(gl.RGBA16F, 512, 1, gl.RGBA, gl.HALF_FLOAT, new Uint16Array(2048).fill(0x3c00));
        this._full(prog, y); this._bindVec(prog, 'uIn', x.t.tex, 0); this._bindVec(prog, 'uS', this.onesTex, 1); gl.uniform2i(prog.u.uSize, x.W, x.H); gl.uniform1i(prog.u.uGpr, x.lay.gpr); nn.draw(); T.set(op.out, y); done(op.in);
      }
    }
    if (stopAt >= 0) { const name = this.layers[stopAt].op.out, t = T.get(name); return t; }
    return get(this.outputName);   // 1×1 map with 512 channels (G = 128 texels wide)
  }
  /** Debug: output of op `i` as Float32Array [C*H*W] channel-major (needs EXT_color_buffer_float). */
  debugTensor(i) {
    const t = this._run(i), raw = this.nn.readFloat(t.t.tex, t.lay.tw, t.lay.th), out = new Float32Array(t.C * t.W * t.H);
    for (let c = 0; c < t.C; c++) { const g = c >> 2, k = c & 3, bx = (g % t.lay.gpr) * t.W, by = ((g / t.lay.gpr) | 0) * t.H;
      for (let y = 0; y < t.H; y++) for (let x = 0; x < t.W; x++) out[(c * t.H + y) * t.W + x] = raw[((by + y) * t.lay.tw + bx + x) * 4 + k]; }
    for (const [name, tt] of this._lastT || []) {} ; this.nn.release(t.t); this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null); return out;
  }

  /** Unit embedding (Float32Array 512) of the image set with setInput(). */
  embed() {
    const gl = this.gl, nn = this.nn, t0 = performance.now(), e = this._run(), G = e.C / 4;
    if (!this.packTex) this.packTex = nn.tex(gl.RGBA8, e.C, 1, gl.RGBA, gl.UNSIGNED_BYTE, null);
    nn.attach(this.packTex); gl.useProgram(this.progs.pack.prog); gl.viewport(0, 0, e.C, 1); this._bindVec(this.progs.pack, 'uIn', e.t.tex, 0); gl.uniform1i(this.progs.pack.u.uG, G); nn.draw(); nn.release(e.t);
    const bytes = nn.readRGBA8(this.packTex, e.C, 1); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.stats.lastMs = performance.now() - t0; this.stats.embeds++;
    return new Float32Array(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice().buffer);
  }
  setTargets(targets) {
    const gl = this.gl, T = targets.length, G = 128, data = new Float32Array(G * 4 * T); for (let t = 0; t < T; t++) data.set(targets[t], t * 512);
    if (this.targetTex) gl.deleteTexture(this.targetTex); this.targetTex = this.nn.tex(gl.RGBA32F, G, T, gl.RGBA, gl.FLOAT, data); this.nTargets = T;
    if (this.scoreTex) gl.deleteTexture(this.scoreTex); this.scoreTex = this.nn.tex(gl.RGBA8, T, 1, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }
  /** Cosine of the current input's embedding with each target: Float32Array(nTargets). */
  score() {
    const gl = this.gl, nn = this.nn, t0 = performance.now(), e = this._run(), G = e.C / 4;
    nn.attach(this.scoreTex); gl.useProgram(this.progs.score.prog); gl.viewport(0, 0, this.nTargets, 1); this._bindVec(this.progs.score, 'uIn', e.t.tex, 0); this._bindVec(this.progs.score, 'uT', this.targetTex, 1); gl.uniform1i(this.progs.score.u.uG, G); nn.draw(); nn.release(e.t);
    const bytes = nn.readRGBA8(this.scoreTex, this.nTargets, 1); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.stats.lastMs = performance.now() - t0; this.stats.embeds++;
    return new Float32Array(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice().buffer);
  }
  release() {
    const gl = this.gl; for (const L of this.layers) for (const k of ['layer', 'fc1', 'fc2', 'qkv', 'proj']) { const l = L[k]; if (l) { if (l.ubo) gl.deleteBuffer(l.ubo); if (l.wtexture) gl.deleteTexture(l.wtexture); } }
    for (const L of this.layers) for (const k of ['w', 'b', 's']) if (L[k]) gl.deleteTexture(L[k]);
    for (const t of [this.inTex && this.inTex.t.tex, this.targetTex, this.scoreTex, this.packTex, this.onesTex]) if (t) gl.deleteTexture(t);
    if (!this.sharedNN) this.nn.destroy();
  }
}

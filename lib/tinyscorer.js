// Token-space CLIP scorer as WebGL2 passes: a batch of token grids -> cosine scores against target embeddings, no decode, no runtime.
// Weights: web/research/tiny/common.py export_scorer (fp16 bin + JSON). Batch = B grids of the same S×S, stacked vertically.
import { GLNN, VS, f16ToF32 } from './glnn.js';

// embedding lookup + bilinear resize of the S×S embedding grid to 16×16 (align_corners=false, like the app's resizeCHW); tile height = 16*B
const EMB_FS2 = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D; precision highp sampler2D;
uniform usampler2D uTok; uniform sampler2D uEmb; uniform int uS; uniform int uOutGpr; uniform int uG0; uniform int uTileH;
out vec4 o;
vec4 emb(int x, int y, int s, int g) { uint t = texelFetch(uTok, ivec2(x, s * uS + y), 0).r; return texelFetch(uEmb, ivec2(int(t % 128u) * uG0 + g, int(t / 128u)), 0); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int col = p.x / 16, row = p.y / uTileH; int g = row * uOutGpr + col;
  int x = p.x - col * 16, y = p.y - row * uTileH; int s = y / 16; int ly = y - s * 16;
  if (g >= uG0) { o = vec4(0.0); return; }
  float sc = float(uS) / 16.0;
  float fx = clamp((float(x) + 0.5) * sc - 0.5, 0.0, float(uS - 1)), fy = clamp((float(ly) + 0.5) * sc - 0.5, 0.0, float(uS - 1));
  int x0 = int(fx), y0 = int(fy); int x1 = min(x0 + 1, uS - 1), y1 = min(y0 + 1, uS - 1); float wx = fx - float(x0), wy = fy - float(y0);
  o = mix(mix(emb(x0, y0, s, g), emb(x1, y0, s, g), wx), mix(emb(x0, y1, s, g), emb(x1, y1, s, g), wx), wy);
}`;

// fully connected layer over the 2×2×C conv output (per sample) or over a B×G vector texture; weights as an RGBA32F texture, row = out group
const FC_FS = (spatial) => `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uIn; uniform sampler2D uW; uniform sampler2D uBias; uniform int uGI; uniform int uInGpr; uniform int uRelu;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int s = p.x, go = p.y;
  vec4 acc = texelFetch(uBias, ivec2(go, 0), 0);
  ${spatial ? `for (int g = 0; g < uGI; g++) for (int q = 0; q < 4; q++) {
    vec4 v = texelFetch(uIn, ivec2((g % uInGpr) * 2 + (q & 1), (g / uInGpr) * uTileH + s * 2 + (q >> 1)), 0);
    int wx = (g * 4 + q) * 4;
    acc += mat4(texelFetch(uW, ivec2(wx, go), 0), texelFetch(uW, ivec2(wx + 1, go), 0), texelFetch(uW, ivec2(wx + 2, go), 0), texelFetch(uW, ivec2(wx + 3, go), 0)) * v; }`
  : `for (int g = 0; g < uGI; g++) {
    vec4 v = texelFetch(uIn, ivec2(s, g), 0); int wx = g * 4;
    acc += mat4(texelFetch(uW, ivec2(wx, go), 0), texelFetch(uW, ivec2(wx + 1, go), 0), texelFetch(uW, ivec2(wx + 2, go), 0), texelFetch(uW, ivec2(wx + 3, go), 0)) * v; }`}
  o = uRelu == 1 ? max(acc, vec4(0.0)) : acc;
}`.replace('uniform int uRelu;', 'uniform int uRelu; uniform int uTileH;');

// normalise the 512-d vector of sample s and dot it with target t; the float score is packed into 4 bytes
const SCORE_FS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uIn; uniform sampler2D uT; uniform int uG;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int s = p.x, t = p.y; float n = 0.0, d = 0.0;
  for (int g = 0; g < uG; g++) { vec4 v = texelFetch(uIn, ivec2(s, g), 0); n += dot(v, v); d += dot(v, texelFetch(uT, ivec2(g, t), 0)); }
  float sc = d / sqrt(max(n, 1e-12));
  uint b = floatBitsToUint(sc); o = vec4(uvec4(b & 255u, (b >> 8) & 255u, (b >> 16) & 255u, (b >> 24) & 255u)) / 255.0;
}`;

// unit embedding component i of sample s, packed into 4 bytes (works without float readback)
const EMBPACK_FS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uIn; uniform int uG;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int s = p.x, i = p.y; float n = 0.0;
  for (int g = 0; g < uG; g++) { vec4 v = texelFetch(uIn, ivec2(s, g), 0); n += dot(v, v); }
  vec4 v = texelFetch(uIn, ivec2(s, i >> 2), 0); float c = (i & 3) == 0 ? v.x : (i & 3) == 1 ? v.y : (i & 3) == 2 ? v.z : v.w;
  uint b = floatBitsToUint(c / sqrt(max(n, 1e-12))); o = vec4(uvec4(b & 255u, (b >> 8) & 255u, (b >> 16) & 255u, (b >> 24) & 255u)) / 255.0;
}`;

export class TinyScorerGL {
  static async load(jsonUrl, binUrl, { fetchBuf = (u) => fetch(u).then((r) => r.arrayBuffer()), nn = null } = {}) {
    const [manifest, bin] = await Promise.all([fetchBuf(jsonUrl).then((b) => JSON.parse(new TextDecoder().decode(b))), fetchBuf(binUrl)]);
    return new TinyScorerGL(manifest, new Uint16Array(bin), nn);
  }
  constructor(manifest, f16, nn = null) {
    this.m = manifest; this.nn = nn || new GLNN(); this.sharedNN = !!nn; const gl = this.gl = this.nn.gl;
    const e = manifest.emb; this.G0 = e.c / 4;
    this.embTex = this.nn.tex(gl.RGBA16F, 128 * this.G0, 128, gl.RGBA, gl.HALF_FLOAT, f16.subarray(e.offset, e.offset + e.len));
    this.layers = manifest.layers.map((L) => this.nn.convLayer({ ...L, up: false, residual: null }, f16, { wtex: true }));
    this.embProg = this.nn.program('scorer-emb', VS, EMB_FS2); this.fcProg = [this.nn.program('scorer-fc-sp', VS, FC_FS(true)), this.nn.program('scorer-fc', VS, FC_FS(false))];
    this.scoreProg = this.nn.program('scorer-score', VS, SCORE_FS); this.packProg = this.nn.program('scorer-embpack', VS, EMBPACK_FS);
    // FC weights as RGBA32F textures. FC1 columns are permuted to our texture order: j = (g*4 + q)*4 + k  <->  torch i = (4g + k)*4 + q
    const [f1, f2] = manifest.fcs; const C = this.m.widths[this.m.widths.length - 1], G = C / 4;
    const w1 = f16ToF32(f16.subarray(f1.w, f1.w + f1.in * f1.out)), b1 = f16ToF32(f16.subarray(f1.b, f1.b + f1.out));
    const t1 = new Float32Array(f1.in * f1.out);   // texel (x = (g*4+q)*4 + k, y = go) = (W1[4go+0][i], .., W1[4go+3][i]) with i = (4g+k)*4+q
    for (let go = 0; go < f1.out / 4; go++) for (let g = 0; g < G; g++) for (let q = 0; q < 4; q++) for (let k = 0; k < 4; k++) {
      const x = (g * 4 + q) * 4 + k, i = (4 * g + k) * 4 + q;
      for (let r = 0; r < 4; r++) t1[(go * f1.in + x) * 4 + r] = w1[(4 * go + r) * f1.in + i];
    }
    this.fc1 = { tex: this.nn.tex(gl.RGBA32F, f1.in, f1.out / 4, gl.RGBA, gl.FLOAT, t1), bias: this.nn.tex(gl.RGBA32F, f1.out / 4, 1, gl.RGBA, gl.FLOAT, b1), GI: G, GO: f1.out / 4 };
    const w2 = f16ToF32(f16.subarray(f2.w, f2.w + f2.in * f2.out)), b2 = f16ToF32(f16.subarray(f2.b, f2.b + f2.out));
    const t2 = new Float32Array(f2.in * f2.out);   // texel (x = gi*4 + k, y = go) = W2[4go..4go+3][4gi+k]
    for (let go = 0; go < f2.out / 4; go++) for (let i = 0; i < f2.in; i++) for (let r = 0; r < 4; r++) t2[(go * f2.in + i) * 4 + r] = w2[(4 * go + r) * f2.in + i];
    this.fc2 = { tex: this.nn.tex(gl.RGBA32F, f2.in, f2.out / 4, gl.RGBA, gl.FLOAT, t2), bias: this.nn.tex(gl.RGBA32F, f2.out / 4, 1, gl.RGBA, gl.FLOAT, b2), GI: f2.in / 4, GO: f2.out / 4 };
    this.dim = f2.out;
    this.tokTex = this.nn.tex(gl.R16UI, 1, 1, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null); this.tokSize = [1, 1];
    this.targetTex = null; this.nTargets = 0; this.scoreTex = null; this.scoreSize = [0, 0];
    this.stats = { lastMs: 0, batches: 0, scored: 0 }; this.weightBytes = f16.byteLength;
  }

  /** targets: array of unit Float32Array(512). */
  setTargets(targets) {
    const gl = this.gl, T = targets.length, G = this.dim / 4, data = new Float32Array(G * 4 * T);
    for (let t = 0; t < T; t++) data.set(targets[t], t * this.dim);
    if (this.targetTex) gl.deleteTexture(this.targetTex);
    this.targetTex = this.nn.tex(gl.RGBA32F, G, T, gl.RGBA, gl.FLOAT, data); this.nTargets = T;
  }

  _forward(tokens, S, B) {
    const gl = this.gl, nn = this.nn;
    const u16 = tokens instanceof Uint16Array ? tokens : Uint16Array.from(tokens);
    gl.bindTexture(gl.TEXTURE_2D, this.tokTex);
    if (this.tokSize[0] !== S || this.tokSize[1] !== S * B) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, S, S * B, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16); this.tokSize = [S, S * B]; }
    else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, S, S * B, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16);
    nn.begin();
    let W = 16, H = 16 * B, lay = nn.layout(this.m.c0, W, H), cur = nn.acquire(lay);
    nn.attach(cur.tex); gl.viewport(0, 0, lay.tw, lay.th); gl.useProgram(this.embProg.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tokTex); gl.uniform1i(this.embProg.u.uTok, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.embTex); gl.uniform1i(this.embProg.u.uEmb, 1);
    gl.uniform1i(this.embProg.u.uS, S); gl.uniform1i(this.embProg.u.uOutGpr, lay.gpr); gl.uniform1i(this.embProg.u.uG0, this.G0); gl.uniform1i(this.embProg.u.uTileH, H);
    nn.draw();
    let sampleH = 16;
    for (const L of this.layers) {
      const oS = sampleH / L.stride, oW = W / L.stride, oH = oS * B, outLay = nn.layout(L.cout, oW, oH), out = nn.acquire(outLay);
      nn.runConv(L, cur, lay, W, H, out, outLay, oW, oH, null, oS);
      nn.release(cur); cur = out; lay = outLay; W = oW; H = oH; sampleH = oS;
    }
    // FC1 over the 2×2 tiles -> B × GO1, FC2 -> B × GO2
    const fc = (prog, inp, inLay, f, relu, spatial) => {
      const out = nn.acquire({ tw: B, th: f.GO }); nn.attach(out.tex); gl.useProgram(prog.prog); gl.viewport(0, 0, B, f.GO);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, inp.tex); gl.uniform1i(prog.u.uIn, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, f.tex); gl.uniform1i(prog.u.uW, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, f.bias); gl.uniform1i(prog.u.uBias, 2);
      gl.uniform1i(prog.u.uGI, f.GI); gl.uniform1i(prog.u.uInGpr, inLay ? inLay.gpr : 1); gl.uniform1i(prog.u.uRelu, relu ? 1 : 0); if (prog.u.uTileH) gl.uniform1i(prog.u.uTileH, spatial ? 2 * B : 0);
      nn.draw(); return out;
    };
    const h1 = fc(this.fcProg[0], cur, lay, this.fc1, true, true); nn.release(cur);
    const h2 = fc(this.fcProg[1], h1, null, this.fc2, false, false); nn.release(h1);
    return h2;   // B × (dim/4) texture of raw (unnormalised) embeddings
  }

  /** tokens: B grids of S×S concatenated (Uint16Array/Int32Array of B*S*S). Returns Float32Array(B * nTargets): cosine per (sample, target). */
  score(tokens, S, B) {
    const gl = this.gl, nn = this.nn, t0 = performance.now();
    if (!this.targetTex) throw new Error('setTargets first');
    const h2 = this._forward(tokens, S, B), T = this.nTargets;
    if (!this.scoreTex || this.scoreSize[0] !== B || this.scoreSize[1] !== T) { if (this.scoreTex) gl.deleteTexture(this.scoreTex); this.scoreTex = nn.tex(gl.RGBA8, B, T, gl.RGBA, gl.UNSIGNED_BYTE, null); this.scoreSize = [B, T]; }
    nn.attach(this.scoreTex); gl.useProgram(this.scoreProg.prog); gl.viewport(0, 0, B, T);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, h2.tex); gl.uniform1i(this.scoreProg.u.uIn, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.targetTex); gl.uniform1i(this.scoreProg.u.uT, 1);
    gl.uniform1i(this.scoreProg.u.uG, this.dim / 4); nn.draw(); nn.release(h2);
    const bytes = nn.readRGBA8(this.scoreTex, B, T); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Float32Array(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice().buffer);   // row t: samples 0..B-1
    this.stats.lastMs = performance.now() - t0; this.stats.batches++; this.stats.scored += B;
    return out;
  }

  /** Unit embeddings of B grids: Float32Array(B*dim), sample-major (packed-byte readback, no float extension needed). */
  embed(tokens, S, B) {
    const gl = this.gl, nn = this.nn, h2 = this._forward(tokens, S, B), G = this.dim / 4;
    const tex = nn.tex(gl.RGBA8, B, this.dim, gl.RGBA, gl.UNSIGNED_BYTE, null);
    nn.attach(tex); gl.useProgram(this.packProg.prog); gl.viewport(0, 0, B, this.dim);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, h2.tex); gl.uniform1i(this.packProg.u.uIn, 0); gl.uniform1i(this.packProg.u.uG, G); nn.draw(); nn.release(h2);
    const bytes = nn.readRGBA8(tex, B, this.dim); gl.deleteTexture(tex); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const col = new Float32Array(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice().buffer);   // row i: samples 0..B-1
    const out = new Float32Array(B * this.dim); for (let i = 0; i < this.dim; i++) for (let s = 0; s < B; s++) out[s * this.dim + i] = col[i * B + s];
    return out;
  }
  /** One square grid (tokens Int32Array(side*side)) -> unit Float32Array(dim). */
  embedOne(tokens, side) { return this.embed(tokens, side, 1); }

  release() {
    const gl = this.gl; for (const L of this.layers) { if (L.ubo) gl.deleteBuffer(L.ubo); if (L.wtexture) gl.deleteTexture(L.wtexture); }
    for (const t of [this.embTex, this.tokTex, this.fc1.tex, this.fc1.bias, this.fc2.tex, this.fc2.bias, this.targetTex, this.scoreTex]) if (t) gl.deleteTexture(t);
    if (!this.sharedNN) this.nn.destroy();
  }
}

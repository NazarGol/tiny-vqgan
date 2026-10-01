// Tiny VQGAN decoder as WebGL2 fragment-shader passes: tokens -> RGB, no ML runtime.
// Weights: fp16 binary + JSON manifest written by web/research/tiny/common.py export_tinydec.
import { GLNN, VS } from './glnn.js';

const EMB_FS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D; precision highp sampler2D;
uniform usampler2D uTok; uniform sampler2D uEmb; uniform ivec2 uOutSize; uniform int uOutGpr; uniform int uG0;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int col = p.x / uOutSize.x, row = p.y / uOutSize.y; int g = row * uOutGpr + col;
  int x = p.x - col * uOutSize.x, y = p.y - row * uOutSize.y;
  if (g >= uG0) { o = vec4(0.0); return; }
  uint t = texelFetch(uTok, ivec2(x, y), 0).r;
  o = texelFetch(uEmb, ivec2(int(t % 128u) * uG0 + g, int(t / 128u)), 0);
}`;

export class TinyDecoderGL {
  static async load(jsonUrl, binUrl, { fetchBuf = (u) => fetch(u).then((r) => r.arrayBuffer()), nn = null } = {}) {
    const [manifest, bin] = await Promise.all([fetchBuf(jsonUrl).then((b) => JSON.parse(new TextDecoder().decode(b))), fetchBuf(binUrl)]);
    return new TinyDecoderGL(manifest, new Uint16Array(bin), nn);
  }

  constructor(manifest, f16, nn = null) {
    this.m = manifest; this.F = 16; this.sharedNN = !!nn; this.nn = nn || new GLNN(); const gl = this.gl = this.nn.gl;
    if ((manifest.max_abs_activation || 0) > 30000) this.nn.useFloatRT(true);
    this.halfRT = this.nn.halfRT; this.maxTex = this.nn.maxTex; this.uboAlign = this.nn.uboAlign; this.maxUbo = this.nn.maxUbo;
    const e = manifest.emb; if (e.n !== 16384) throw new Error('embedding layout expects 16384 tokens');
    this.G0 = e.c / 4;
    this.embTex = this.nn.tex(gl.RGBA16F, 128 * this.G0, 128, gl.RGBA, gl.HALF_FLOAT, f16.subarray(e.offset, e.offset + e.len));   // token-major, 128 tokens per row
    this.layers = manifest.layers.map((L) => this.nn.convLayer(L, f16));
    this.embProg = this.nn.program('tinydec-emb', VS, EMB_FS);
    this.tokTex = this.nn.tex(gl.R16UI, 1, 1, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null); this.tokSize = [1, 1];
    this.outTex = null; this.outSize = [0, 0]; this.stats = { lastMs: 0, times: [] }; this.weightBytes = f16.byteLength;
  }

  /** tokens: Int32Array|Uint16Array (h*w, row-major). Returns {rgba: Uint8ClampedArray(16h*16w*4), w, h} in pixels. */
  decodeRGBA(tokens, h, w, into = null) {
    const gl = this.gl, nn = this.nn, t0 = performance.now();
    const u16 = tokens instanceof Uint16Array ? tokens : Uint16Array.from(tokens);
    gl.bindTexture(gl.TEXTURE_2D, this.tokTex);
    if (this.tokSize[0] !== w || this.tokSize[1] !== h) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, w, h, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16); this.tokSize = [w, h]; }
    else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16);
    nn.begin();
    let lay = nn.layout(this.m.emb.c, w, h), cur = nn.acquire(lay);
    nn.attach(cur.tex); gl.viewport(0, 0, lay.tw, lay.th); gl.useProgram(this.embProg.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tokTex); gl.uniform1i(this.embProg.u.uTok, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.embTex); gl.uniform1i(this.embProg.u.uEmb, 1);
    gl.uniform2i(this.embProg.u.uOutSize, w, h); gl.uniform1i(this.embProg.u.uOutGpr, lay.gpr); gl.uniform1i(this.embProg.u.uG0, this.G0);
    nn.draw();
    let prev = null, curLay = lay, W = w, H = h;   // prev = the tensor before the current one (a block's input, used by its residual layer)
    for (const L of this.layers) {
      const oW = L.up ? W * 2 : W, oH = L.up ? H * 2 : H;
      let outLay, out;
      if (L.out) {
        if (!this.outTex || this.outSize[0] !== oW || this.outSize[1] !== oH) { if (this.outTex) gl.deleteTexture(this.outTex); this.outTex = nn.tex(gl.RGBA8, oW, oH, gl.RGBA, gl.UNSIGNED_BYTE, null); this.outSize = [oW, oH]; }
        outLay = { W: oW, H: oH, G: 1, gpr: 1, tw: oW, th: oH }; out = { tex: this.outTex };
      } else { outLay = nn.layout(L.cout, oW, oH); out = nn.acquire(outLay); }
      nn.runConv(L, cur, curLay, W, H, out, outLay, oW, oH, L.res ? prev.tex : null, oH);
      if (L.res) { nn.release(prev); prev = null; nn.release(cur); } else { if (prev) nn.release(prev); prev = cur; }
      cur = L.out ? null : out; curLay = outLay; W = oW; H = oH;
    }
    if (prev) nn.release(prev);
    const rgba = nn.readRGBA8(this.outTex, W, H, into);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.stats.lastMs = performance.now() - t0; this.stats.times.push(this.stats.lastMs); if (this.stats.times.length > 50) this.stats.times.shift();
    return { rgba, w: W, h: H };
  }

  /** Same output shape as the ONNX Decoder: {data: Float32Array CHW 0..1, w, h}. */
  decode(tokens, h, w) {
    const { rgba, w: W, h: H } = this.decodeRGBA(tokens, h, w);
    const n = W * H, data = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { data[i] = rgba[i * 4] / 255; data[n + i] = rgba[i * 4 + 1] / 255; data[2 * n + i] = rgba[i * 4 + 2] / 255; }
    return { data, w: W, h: H };
  }

  release() {
    const gl = this.gl; for (const L of this.layers) { if (L.ubo) gl.deleteBuffer(L.ubo); if (L.wtexture) gl.deleteTexture(L.wtexture); }
    gl.deleteTexture(this.embTex); gl.deleteTexture(this.tokTex); if (this.outTex) gl.deleteTexture(this.outTex); this.layers = []; this.outTex = null;
    if (!this.sharedNN) this.nn.destroy();
  }
}

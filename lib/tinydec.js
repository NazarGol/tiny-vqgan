// Tiny VQGAN decoder as WebGL2 fragment-shader passes: tokens -> RGB, no ML runtime.
// Weights: fp16 binary + JSON manifest written by web/research/tiny/common.py export_tinydec.
// Tensor layout: a C-channel H×W activation is one RGBA16F texture; channel group g (4 channels) is the tile at
// ((g % gpr) * W, (g / gpr) * H). Each 3×3 conv = one draw per output group with that group's weights in a uniform block.

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

const VS = `#version 300 es
void main() { vec2 p = vec2((gl_VertexID & 1) * 4 - 1, (gl_VertexID & 2) * 2 - 1); gl_Position = vec4(p, 0.0, 1.0); }`;

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

function convFS(GI, up, relu, res, out) {
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uIn; uniform sampler2D uRes;
layout(std140) uniform W { mat4 w[${GI * 9}]; vec4 bias; };
uniform ivec2 uInSize; uniform int uInGpr; uniform ivec2 uOutSize; uniform int uOutGpr; uniform int uOutGroup;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int col = uOutGroup % uOutGpr, row = uOutGroup / uOutGpr;
  int x = p.x - col * uOutSize.x, y = p.y - row * uOutSize.y;
  vec4 acc = bias;
  for (int t = 0; t < 9; t++) {
    int sx = x + (t % 3) - 1, sy = y + (t / 3) - 1;
    ${up ? 'if (sx < 0 || sy < 0 || sx >= uOutSize.x || sy >= uOutSize.y) continue; int ix = sx >> 1, iy = sy >> 1;'
         : 'if (sx < 0 || sy < 0 || sx >= uInSize.x || sy >= uInSize.y) continue; int ix = sx, iy = sy;'}
    for (int g = 0; g < ${GI}; g++) acc += w[g * 9 + t] * texelFetch(uIn, ivec2((g % uInGpr) * uInSize.x + ix, (g / uInGpr) * uInSize.y + iy), 0);
  }
  ${res ? 'acc += texelFetch(uRes, p, 0);' : ''}
  ${relu ? 'acc = max(acc, vec4(0.0));' : ''}
  ${out ? 'o = vec4(acc.rgb, 1.0);' : 'o = acc;'}
}`;
}

export class TinyDecoderGL {
  /** Load manifest + weights (same base URL, names tiny_decoder_X.json / .bin) with the app's fetch helper or plain fetch. */
  static async load(jsonUrl, binUrl, { fetchBuf = (u) => fetch(u).then((r) => r.arrayBuffer()), gl = null } = {}) {
    const [manifest, bin] = await Promise.all([fetchBuf(jsonUrl).then((b) => JSON.parse(new TextDecoder().decode(b))), fetchBuf(binUrl)]);
    return new TinyDecoderGL(manifest, new Uint16Array(bin), gl);
  }

  constructor(manifest, f16, gl = null) {
    this.m = manifest; this.F = 16;
    if (!gl) {
      const c = typeof document !== 'undefined' ? document.createElement('canvas') : new OffscreenCanvas(1, 1);
      c.width = c.height = 1;
      gl = c.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: true, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
      this.canvas = c;
    }
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    const extF = gl.getExtension('EXT_color_buffer_float'), extH = gl.getExtension('EXT_color_buffer_half_float');
    if (!extF && !extH) throw new Error('no renderable float textures (EXT_color_buffer_half_float)');
    this.halfRT = !!extH && (manifest.max_abs_activation || 0) < 30000;   // RGBA16F when renderable, RGBA32F otherwise
    this.rtFormat = this.halfRT ? gl.RGBA16F : gl.RGBA32F; this.rtType = this.halfRT ? gl.HALF_FLOAT : gl.FLOAT;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.uboAlign = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT);
    this.maxUbo = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.vao = gl.createVertexArray(); this.fbo = gl.createFramebuffer();
    this.programs = new Map(); this.pool = new Map(); this.stats = { lastMs: 0, times: [] };
    this._buildEmbedding(f16); this._buildLayers(f16);
    this.tokTex = this._tex(gl.R16UI, 1, 1, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null); this.tokSize = [1, 1];
    this.outTex = null; this.outSize = [0, 0];
    this.weightBytes = f16.byteLength;
  }

  _tex(internal, w, h, format, type, data) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
    return t;
  }

  _buildEmbedding(f16) {
    const gl = this.gl, e = this.m.emb, G0 = e.c / 4;
    if (e.n !== 16384) throw new Error('embedding layout expects 16384 tokens');
    this.G0 = G0;
    const data = f16.subarray(e.offset, e.offset + e.len);   // token-major: 128 tokens per row, G0 texels per token
    this.embTex = this._tex(gl.RGBA16F, 128 * G0, 128, gl.RGBA, gl.HALF_FLOAT, data);
  }

  _buildLayers(f16) {
    const gl = this.gl, align = this.uboAlign;
    this.layers = this.m.layers.map((L) => {
      const GI = L.cin / 4, GO = L.cout / 4, per = GI * 9 * 16;                         // floats per output group
      const groupBytes = (per + 4) * 4, stride = Math.ceil(groupBytes / align) * align;
      if (groupBytes > this.maxUbo) throw new Error(`layer ${L.name}: ${groupBytes} B uniform block > ${this.maxUbo}`);
      const w = f16ToF32(f16.subarray(L.w, L.w + GO * per)), b = f16ToF32(f16.subarray(L.b, L.b + GO * 4));
      const buf = new Float32Array(stride / 4 * GO);
      for (let g = 0; g < GO; g++) { buf.set(w.subarray(g * per, (g + 1) * per), g * stride / 4); buf.set(b.subarray(g * 4, g * 4 + 4), g * stride / 4 + per); }
      const ubo = gl.createBuffer(); gl.bindBuffer(gl.UNIFORM_BUFFER, ubo); gl.bufferData(gl.UNIFORM_BUFFER, buf, gl.STATIC_DRAW);
      const prog = this._program(GI, !!L.up, !!L.relu, !!L.residual, L.name === 'out');
      return { ...L, GI, GO, ubo, stride, groupBytes, prog };
    });
  }

  _program(GI, up, relu, res, out) {
    const key = `${GI}|${up}|${relu}|${res}|${out}`;
    if (this.programs.has(key)) return this.programs.get(key);
    const p = this._link(VS, convFS(GI, up, relu, res, out));
    const gl = this.gl; gl.uniformBlockBinding(p.prog, gl.getUniformBlockIndex(p.prog, 'W'), 0);
    this.programs.set(key, p); return p;
  }

  _link(vs, fs) {
    const gl = this.gl, prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s) + '\n' + src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n'));
      gl.attachShader(prog, s);
    }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
    const u = {}; const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(prog, i); u[info.name] = gl.getUniformLocation(prog, info.name); }
    return { prog, u };
  }

  /** Activation texture for C channels at W×H: tiles of groups, as many per row as fit. */
  _layout(C, W, H) {
    const G = C / 4, gpr = Math.max(1, Math.min(G, Math.floor(this.maxTex / W)));
    return { C, W, H, G, gpr, tw: gpr * W, th: Math.ceil(G / gpr) * H };
  }
  _acquire(lay) {
    const key = `${lay.tw}x${lay.th}`; let list = this.pool.get(key); if (!list) this.pool.set(key, (list = []));
    const free = list.find((t) => !t.busy);
    if (free) { free.busy = true; return free; }
    const t = { tex: this._tex(this.rtFormat, lay.tw, lay.th, this.gl.RGBA, this.rtType, null), busy: true, key }; list.push(t); return t;
  }
  _release(t) { if (t) t.busy = false; }

  /** tokens: Int32Array|Uint16Array (h*w, row-major). Returns {rgba: Uint8ClampedArray(16h*16w*4), w, h} in pixels. */
  decodeRGBA(tokens, h, w) {
    const gl = this.gl, t0 = performance.now();
    const u16 = tokens instanceof Uint16Array ? tokens : Uint16Array.from(tokens);
    gl.bindTexture(gl.TEXTURE_2D, this.tokTex);
    if (this.tokSize[0] !== w || this.tokSize[1] !== h) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, w, h, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16); this.tokSize = [w, h]; }
    else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16);
    gl.bindVertexArray(this.vao); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo); gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.SCISSOR_TEST);
    // embedding pass
    let lay = this._layout(this.m.emb.c, w, h), cur = this._acquire(lay);
    if (!this.embProg) { this.embProg = this._link(VS, EMB_FS); }
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, cur.tex, 0);
    gl.viewport(0, 0, lay.tw, lay.th); gl.useProgram(this.embProg.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tokTex); gl.uniform1i(this.embProg.u.uTok, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.embTex); gl.uniform1i(this.embProg.u.uEmb, 1);
    gl.uniform2i(this.embProg.u.uOutSize, w, h); gl.uniform1i(this.embProg.u.uOutGpr, lay.gpr); gl.uniform1i(this.embProg.u.uG0, this.G0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    let prev = null, curLay = lay, prevLay = null;   // prev = tensor before the current one (the block input for residual layers)
    let W = w, H = h;
    for (const L of this.layers) {
      const oW = L.up ? W * 2 : W, oH = L.up ? H * 2 : H;
      const isOut = L.name === 'out';
      let outLay, out;
      if (isOut) {
        if (!this.outTex || this.outSize[0] !== oW || this.outSize[1] !== oH) { if (this.outTex) gl.deleteTexture(this.outTex); this.outTex = this._tex(gl.RGBA8, oW, oH, gl.RGBA, gl.UNSIGNED_BYTE, null); this.outSize = [oW, oH]; }
        outLay = { W: oW, H: oH, G: 1, gpr: 1, tw: oW, th: oH }; out = { tex: this.outTex };
      } else { outLay = this._layout(L.cout, oW, oH); out = this._acquire(outLay); }
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out.tex, 0);
      gl.useProgram(L.prog.prog);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, cur.tex); gl.uniform1i(L.prog.u.uIn, 0);
      if (L.residual) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, prev.tex); gl.uniform1i(L.prog.u.uRes, 1); }
      gl.uniform2i(L.prog.u.uInSize, W, H); gl.uniform1i(L.prog.u.uInGpr, curLay.gpr);
      gl.uniform2i(L.prog.u.uOutSize, oW, oH); gl.uniform1i(L.prog.u.uOutGpr, outLay.gpr);
      gl.enable(gl.SCISSOR_TEST);
      for (let g = 0; g < L.GO; g++) {
        const col = g % outLay.gpr, row = (g / outLay.gpr) | 0;
        gl.viewport(0, 0, outLay.tw, outLay.th); gl.scissor(col * oW, row * oH, oW, oH);
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, L.ubo, g * L.stride, L.groupBytes);
        gl.uniform1i(L.prog.u.uOutGroup, g);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      gl.disable(gl.SCISSOR_TEST);
      // rotate: residual layers consume `prev` (block input); after them the block input is free
      if (L.residual) { this._release(prev); prev = null; this._release(cur); }
      else { if (prev) this._release(prev); prev = cur; prevLay = curLay; }
      cur = isOut ? null : out; curLay = outLay; W = oW; H = oH;
    }
    if (prev) this._release(prev);
    const rgba = new Uint8ClampedArray(W * H * 4);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.outTex, 0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
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
    const gl = this.gl;
    for (const list of this.pool.values()) for (const t of list) gl.deleteTexture(t.tex);
    this.pool.clear(); for (const L of this.layers) gl.deleteBuffer(L.ubo);
    gl.deleteTexture(this.embTex); gl.deleteTexture(this.tokTex); if (this.outTex) gl.deleteTexture(this.outTex);
    for (const p of this.programs.values()) gl.deleteProgram(p.prog); this.programs.clear();
    const ext = gl.getExtension('WEBGL_lose_context'); if (ext && this.canvas) ext.loseContext();
  }
}

// Minimal WebGL2 "neural net" helpers shared by the tiny decoder and the token scorer: context, float render targets,
// a texture pool, and a generic 3×3-conv pass (zero padding, optional nearest 2× upsample or stride 2, ReLU, residual add,
// batch of samples stacked vertically). Weights fp16; per output channel group either a std140 uniform block or a texture row.

export function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
export function f32ToF16(v) {
  _f32[0] = v; const x = _u32[0], sign = (x >> 16) & 0x8000, e = ((x >> 23) & 0xff) - 112, m = x & 0x7fffff;
  if (e <= 0) return sign | (e < -10 ? 0 : ((m | 0x800000) >> (1 - e) + 13));
  if (e >= 31) return sign | 0x7c00;
  return sign | (e << 10) | (m >> 13);
}

export const VS = `#version 300 es
void main() { vec2 p = vec2((gl_VertexID & 1) * 4 - 1, (gl_VertexID & 2) * 2 - 1); gl_Position = vec4(p, 0.0, 1.0); }`;

export const GELU = `float gelu(float x) { float a = abs(x) / 1.4142135; float t = 1.0 / (1.0 + 0.3275911 * a); float e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a); e = x < 0.0 ? -e : e; return 0.5 * x * (1.0 + e); }
vec4 gelu4(vec4 v) { return vec4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w)); }`;
export const ACT = { none: '', relu: 'acc = max(acc, vec4(0.0));', gelu: 'acc = gelu4(acc);', sigmoid: 'acc = 1.0 / (1.0 + exp(-acc));' };

export function convFS({ GI, up = false, stride = 1, relu = true, res = false, out = false, wtex = false, k = 3, act = null }) {
  const taps = k * k, pad = (k - 1) >> 1, actCode = act != null ? ACT[act] : (relu ? ACT.relu : '');
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uIn; uniform sampler2D uRes;
${wtex ? 'uniform sampler2D uW; uniform vec4 uBias; uniform int uWRow;' : `layout(std140) uniform W { mat4 w[${GI * taps}]; vec4 bias; };`}
${GELU}
uniform ivec2 uInSize; uniform int uInGpr; uniform ivec2 uOutSize; uniform int uOutGpr; uniform int uOutGroup; uniform int uSampleH;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int col = uOutGroup % uOutGpr, row = uOutGroup / uOutGpr;
  int x = p.x - col * uOutSize.x, y = p.y - row * uOutSize.y;
  int s = y / uSampleH, ly = y - s * uSampleH;
  int inH = ${up ? 'uSampleH / 2' : `uSampleH * ${stride}`}; int inBase = s * inH;
  vec4 acc = ${wtex ? 'uBias' : 'bias'};
  for (int t = 0; t < ${taps}; t++) {
    ${up ? `int sx = x + (t % ${k}) - ${pad}, sy = ly + (t / ${k}) - ${pad}; if (sx < 0 || sy < 0 || sx >= uOutSize.x || sy >= uSampleH) continue; int ix = sx >> 1, iy = inBase + (sy >> 1);`
         : `int sx = x * ${stride} + (t % ${k}) - ${pad}, sy = ly * ${stride} + (t / ${k}) - ${pad}; if (sx < 0 || sy < 0 || sx >= uInSize.x || sy >= inH) continue; int ix = sx, iy = inBase + sy;`}
    for (int g = 0; g < ${GI}; g++) {
      vec4 v = texelFetch(uIn, ivec2((g % uInGpr) * uInSize.x + ix, (g / uInGpr) * uInSize.y + iy), 0);
      ${wtex ? `int wx = (g * ${taps} + t) * 4; acc += mat4(texelFetch(uW, ivec2(wx, uWRow), 0), texelFetch(uW, ivec2(wx + 1, uWRow), 0), texelFetch(uW, ivec2(wx + 2, uWRow), 0), texelFetch(uW, ivec2(wx + 3, uWRow), 0)) * v;`
             : `acc += w[g * ${taps} + t] * v;`}
    }
  }
  ${res ? 'acc += texelFetch(uRes, p, 0);' : ''}
  ${actCode}
  ${out ? 'o = vec4(acc.rgb, 1.0);' : 'o = acc;'}
}`;
}

export class GLNN {
  constructor(gl = null) {
    if (!gl) {
      const c = typeof document !== 'undefined' ? document.createElement('canvas') : new OffscreenCanvas(1, 1);
      c.width = c.height = 1;
      gl = c.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: true, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
      this.canvas = c;
    }
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    this.extF = !!gl.getExtension('EXT_color_buffer_float'); this.extH = !!gl.getExtension('EXT_color_buffer_half_float');
    if (!this.extF && !this.extH) throw new Error('no renderable float textures (EXT_color_buffer_half_float)');
    this.halfRT = this.extH;
    this.rtFormat = this.halfRT ? gl.RGBA16F : gl.RGBA32F; this.rtType = this.halfRT ? gl.HALF_FLOAT : gl.FLOAT;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE); this.uboAlign = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT); this.maxUbo = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE);
    this.renderer = (() => { try { const d = gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); } catch (_) { return ''; } })();
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    this.vao = gl.createVertexArray(); this.fbo = gl.createFramebuffer();
    this.programs = new Map(); this.pool = new Map(); this.texBytes = 0; this.poolBytes = 0; this.tick = 0;
  }
  useFloatRT(on) { const gl = this.gl; this.halfRT = !on && this.extH; this.rtFormat = this.halfRT ? gl.RGBA16F : gl.RGBA32F; this.rtType = this.halfRT ? gl.HALF_FLOAT : gl.FLOAT; }

  link(vs, fs) {
    const gl = this.gl, prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s) + '\n' + src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n'));
      gl.attachShader(prog, s);
    }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
    const u = {}, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(prog, i); u[info.name] = gl.getUniformLocation(prog, info.name); }
    const bi = gl.getUniformBlockIndex(prog, 'W'); if (bi !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, bi, 0);
    return { prog, u };
  }
  program(key, vs, fs) { let p = this.programs.get(key); if (!p) { p = this.link(vs, fs); this.programs.set(key, p); } return p; }

  tex(internal, w, h, format, type, data) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
    this.texBytes += w * h * (internal === gl.RGBA32F ? 16 : internal === gl.RGBA16F ? 8 : internal === gl.R16UI ? 2 : 4);
    return t;
  }
  /** Layout of a C-channel tensor of W×H (H includes the batch): groups of 4 channels tiled, as many per row as fit. */
  layout(C, W, H) { const G = C / 4, gpr = Math.max(1, Math.min(G, Math.floor(this.maxTex / W))); return { C, W, H, G, gpr, tw: gpr * W, th: Math.ceil(G / gpr) * H }; }
  acquire(lay) {
    const key = `${lay.tw}x${lay.th}`; let list = this.pool.get(key); if (!list) this.pool.set(key, (list = []));
    const free = list.find((t) => !t.busy); if (free) { free.busy = true; free.used = ++this.tick; return free; }
    const bytes = lay.tw * lay.th * (this.halfRT ? 8 : 16);
    const t = { tex: this.tex(this.rtFormat, lay.tw, lay.th, this.gl.RGBA, this.rtType, null), busy: true, key, bytes, used: ++this.tick }; list.push(t); this.poolBytes += bytes; return t;
  }
  release(t) { if (t) t.busy = false; }
  /** Delete the least recently used free pool textures until the pool is under `maxBytes` (crop sizes vary per stroke, so the pool would otherwise grow without bound). */
  trim(maxBytes = 32 * 2 ** 20) {
    if (this.poolBytes <= maxBytes) return;
    const all = []; for (const [key, list] of this.pool) for (const t of list) if (!t.busy) all.push(t);
    all.sort((a, b) => a.used - b.used);
    for (const t of all) { if (this.poolBytes <= maxBytes) break; this.gl.deleteTexture(t.tex); this.poolBytes -= t.bytes; this.texBytes -= t.bytes; const list = this.pool.get(t.key); list.splice(list.indexOf(t), 1); if (!list.length) this.pool.delete(t.key); }
  }
  attach(tex) { const gl = this.gl; gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0); }
  begin() { const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo); gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.SCISSOR_TEST); }
  draw() { this.gl.drawArrays(this.gl.TRIANGLES, 0, 3); }

  /** Prepare a conv layer from the manifest entry: weights in a uniform buffer when a group fits, else a texture (one row per out group). */
  convLayer(L, f16, opts = {}) {
    const gl = this.gl, k = L.k || 3, taps = k * k, GI = L.cin / 4, GO = L.cout / 4, per = GI * taps * 16, groupBytes = (per + 4) * 4;
    const wtex = groupBytes > this.maxUbo || opts.wtex;
    const b = f16ToF32(f16.subarray(L.b, L.b + GO * 4));
    const layer = { ...L, GI, GO, k, taps, wtex, bias: b, stride: L.stride || 1, up: !!L.up, relu: !!L.relu, res: !!L.residual, out: L.name === 'out', act: L.act || null };
    if (wtex) { layer.wtexture = this.tex(gl.RGBA16F, GI * taps * 4, GO, gl.RGBA, gl.HALF_FLOAT, f16.subarray(L.w, L.w + GO * per)); }
    else {
      const stride = Math.ceil(groupBytes / this.uboAlign) * this.uboAlign, w = f16ToF32(f16.subarray(L.w, L.w + GO * per)), buf = new Float32Array(stride / 4 * GO);
      for (let g = 0; g < GO; g++) { buf.set(w.subarray(g * per, (g + 1) * per), g * stride / 4); buf.set(b.subarray(g * 4, g * 4 + 4), g * stride / 4 + per); }
      layer.ubo = gl.createBuffer(); gl.bindBuffer(gl.UNIFORM_BUFFER, layer.ubo); gl.bufferData(gl.UNIFORM_BUFFER, buf, gl.STATIC_DRAW); layer.uboStride = stride; layer.groupBytes = groupBytes;
    }
    const key = `conv|${GI}|${layer.up}|${layer.stride}|${layer.relu}|${layer.res}|${layer.out}|${wtex}|${k}|${layer.act}`;
    layer.prog = this.program(key, VS, convFS({ GI, up: layer.up, stride: layer.stride, relu: layer.relu, res: layer.res, out: layer.out, wtex, k, act: layer.act }));
    return layer;
  }

  /** Run a conv layer. inp/out: {tex}, with layouts; W,H input size (H = sampleH_in * batch); sampleH = output sample height. */
  runConv(L, inp, inLay, W, H, out, outLay, oW, oH, resTex = null, sampleH = oH) {
    const gl = this.gl, p = L.prog;
    this.attach(out.tex); gl.useProgram(p.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, inp.tex); gl.uniform1i(p.u.uIn, 0);
    if (L.res) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, resTex); gl.uniform1i(p.u.uRes, 1); }
    if (L.wtex) { gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, L.wtexture); gl.uniform1i(p.u.uW, 2); }
    gl.uniform2i(p.u.uInSize, W, H); gl.uniform1i(p.u.uInGpr, inLay.gpr); gl.uniform2i(p.u.uOutSize, oW, oH); gl.uniform1i(p.u.uOutGpr, outLay.gpr); gl.uniform1i(p.u.uSampleH, sampleH);
    gl.viewport(0, 0, outLay.tw, outLay.th); gl.enable(gl.SCISSOR_TEST);
    for (let g = 0; g < L.GO; g++) {
      const col = g % outLay.gpr, row = (g / outLay.gpr) | 0;
      gl.scissor(col * oW, row * oH, oW, oH); gl.uniform1i(p.u.uOutGroup, g);
      if (L.wtex) { gl.uniform1i(p.u.uWRow, g); gl.uniform4f(p.u.uBias, L.bias[g * 4], L.bias[g * 4 + 1], L.bias[g * 4 + 2], L.bias[g * 4 + 3]); }
      else gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, L.ubo, g * L.uboStride, L.groupBytes);
      this.draw();
    }
    gl.disable(gl.SCISSOR_TEST);
  }

  readRGBA8(tex, w, h, into = null) { const gl = this.gl, out = into || new Uint8ClampedArray(w * h * 4); this.attach(tex); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out); return out; }
  readFloat(tex, w, h) {   // needs EXT_color_buffer_float (tests only)
    const gl = this.gl, out = new Float32Array(w * h * 4); this.attach(tex); gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, out); return out;
  }
  destroy() {
    const gl = this.gl;
    for (const list of this.pool.values()) for (const t of list) gl.deleteTexture(t.tex);
    this.pool.clear(); for (const p of this.programs.values()) gl.deleteProgram(p.prog); this.programs.clear();
    const ext = gl.getExtension('WEBGL_lose_context'); if (ext && this.canvas) ext.loseContext();
  }
}

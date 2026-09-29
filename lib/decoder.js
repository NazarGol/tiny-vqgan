// VQGAN token decoder (ONNX, onnxruntime-web). tokens int32 [1,h,w] -> image float32 [1,3,16h,16w] in 0..1.
import { ortRun } from './models.js';
export const F = 16; // pixels per token

export class Decoder {
  constructor(ort, session) { this.ort = ort; this.session = session; }

  static async create(ort, buf, { ep = 'webgpu' } = {}) {
    const session = await ort.InferenceSession.create(buf, { executionProviders: [ep], graphOptimizationLevel: 'all' });
    return new Decoder(ort, session);
  }

  /** tokens: Int32Array(h*w), row-major. Returns {data, w, h} in pixels. Serialised through the global ORT queue. */
  decode(tokens, h, w) {
    return ortRun(async () => {
      const t0 = performance.now();
      const t = new this.ort.Tensor('int32', tokens, [1, h, w]);
      const out = await this.session.run({ tokens: t });
      const img = out.image;
      this.lastMs = performance.now() - t0;   // pure decode time, excluding queue wait
      return { data: img.data, h: img.dims[2], w: img.dims[3] };
    });
  }
}

/** Token grid helpers. grid = {w, h, tokens: Int32Array(w*h)}. region = {x, y, w, h} in tokens. */
export function clampRegion(grid, r) {
  const x = Math.max(0, Math.min(grid.w - 1, r.x)), y = Math.max(0, Math.min(grid.h - 1, r.y));
  return { x, y, w: Math.max(1, Math.min(grid.w - x, r.w)), h: Math.max(1, Math.min(grid.h - y, r.h)) };
}

/** Region expanded by `margin` tokens on each side, clipped to the grid. */
export function expandRegion(grid, r, margin) {
  const x = Math.max(0, r.x - margin), y = Math.max(0, r.y - margin);
  const x1 = Math.min(grid.w, r.x + r.w + margin), y1 = Math.min(grid.h, r.y + r.h + margin);
  return { x, y, w: x1 - x, h: y1 - y };
}

export function readRegion(grid, r) {
  const out = new Int32Array(r.w * r.h);
  for (let yy = 0; yy < r.h; yy++) out.set(grid.tokens.subarray((r.y + yy) * grid.w + r.x, (r.y + yy) * grid.w + r.x + r.w), yy * r.w);
  return out;
}

export function writeRegion(grid, r, tokens) {
  for (let yy = 0; yy < r.h; yy++) grid.tokens.set(tokens.subarray(yy * r.w, yy * r.w + r.w), (r.y + yy) * grid.w + r.x);
}

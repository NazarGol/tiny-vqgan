// Model loading helpers: onnxruntime-web import, WebGPU check, fetch with progress + Cache Storage.

export async function loadOrt(base) {
  const ort = await import(/* @vite-ignore */ base + 'ort.webgpu.min.mjs');
  ort.env.wasm.wasmPaths = base;
  ort.env.logLevel = 'error';
  return ort;
}

/** Returns adapter info if WebGPU works, else null. */
export async function webgpuInfo() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const info = adapter.info || {};
    return { vendor: info.vendor || '', architecture: info.architecture || '', f16: adapter.features.has('shader-f16') };
  } catch (e) {
    return null;
  }
}

/**
 * Fetch a binary file, reporting progress, and keep a copy in Cache Storage so the
 * second visit does not download it again (works for cross-origin CORS responses too).
 */
export async function fetchCached(url, { onProgress, cacheName = 'vqpaint-models-v1' } = {}) {
  let cache = null;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, cached: true });
      return buf;
    }
  } catch (e) { cache = null; }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  const total = +resp.headers.get('content-length') || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ url, loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  if (cache) {
    try { await cache.put(url, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(loaded) } })); } catch (e) {}
  }
  return buf.buffer;
}

export async function fetchJsonCached(url) {
  const buf = await fetchCached(url);
  return JSON.parse(new TextDecoder().decode(buf));
}

/**
 * One global queue for every ORT session.run(): the asyncify wasm build cannot have two runs
 * in flight at once (it crashes with "memory access out of bounds"), even on different sessions.
 */
let ortBusy = Promise.resolve();
export function ortRun(fn) {
  const p = ortBusy.then(fn, fn);
  ortBusy = p.catch(() => {});
  return p;
}

// Model loading helpers: onnxruntime-web import, WebGPU check, fetch with progress + Cache Storage.

export async function loadOrt(base, entry = 'ort.webgpu.min.mjs') {
  const ort = await import(/* @vite-ignore */ base + entry);
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
  } catch (e) { console.warn('Cache Storage unavailable:', e && e.message); cache = null; }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  const total = +resp.headers.get('content-length') || 0;
  // store a clone in Cache Storage first (the browser streams it to disk), then read the body once into a single preallocated buffer
  if (cache) { try { cache.put(url, resp.clone()).catch((e) => console.warn('could not cache', url, e && e.message)); } catch (e) { console.warn('could not cache', url, e && e.message); } }
  const reader = resp.body.getReader();
  let buf = total ? new Uint8Array(total) : null, loaded = 0, chunks = buf ? null : [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf) { if (loaded + value.length > buf.length) { const bigger = new Uint8Array(Math.max(buf.length * 2, loaded + value.length)); bigger.set(buf.subarray(0, loaded)); buf = bigger; } buf.set(value, loaded); }
    else chunks.push(value);
    loaded += value.length;
    onProgress?.({ url, loaded, total, cached: false });
  }
  if (!buf) { buf = new Uint8Array(loaded); let off = 0; for (const c of chunks) { buf.set(c, off); off += c.length; } chunks = null; }
  else if (loaded !== buf.length) buf = buf.slice(0, loaded);
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

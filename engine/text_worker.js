// Worker: loads the tokenizer + tiny text encoder, answers {id, texts: [...]} with {id, embeddings: [Float32Array...]}; the caller terminates it.
import { CLIPTokenizer } from '../lib/clip_tokenizer.js';
import { TinyTextJS } from './text.js';
let ready = null;
/** Cache Storage first (the page already fetched these files), then the URL, then its mirror. */
async function loadBuf(url, alt) {
  let cache = null; try { cache = await caches.open('vqpaint-models-v1'); } catch (_) {}
  if (cache) { for (const u of [url, alt]) { if (!u) continue; const hit = await cache.match(u); if (hit) return hit.arrayBuffer(); } }
  for (const u of [url, alt]) { if (!u) continue; const r = await fetch(u); if (r.ok) return r.arrayBuffer(); }
  throw new Error('could not load ' + url);
}
self.onmessage = async (ev) => {
  const { id, init, texts } = ev.data;
  try {
    if (init) {
      ready = (async () => {
        const alt = init.alt || {}; const tj = init.tokenizerJson || JSON.parse(new TextDecoder().decode(await loadBuf(init.tokenizerUrl, alt.tokenizerUrl)));
        const tok = new CLIPTokenizer(tj); const enc = await TinyTextJS.load(init.jsonUrl, init.binUrl, { fetchBuf: (u) => loadBuf(u, u === init.jsonUrl ? alt.jsonUrl : alt.binUrl) }); return { tok, enc };
      })(); await ready; self.postMessage({ id, ok: true }); return; }
    const { tok, enc } = await ready; const t0 = performance.now();
    const embeddings = texts.map((t) => enc.encode(tok.encode(t).ids));
    self.postMessage({ id, embeddings, ms: performance.now() - t0 }, embeddings.map((e) => e.buffer));
  } catch (e) { self.postMessage({ id, error: e.message }); }
};

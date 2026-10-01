// Worker: loads the tokenizer + tiny text encoder, answers {id, texts: [...]} with {id, embeddings: [Float32Array...]}; the caller terminates it.
import { CLIPTokenizer } from '../lib/clip_tokenizer.js';
import { TinyTextJS } from './text.js';
let ready = null;
self.onmessage = async (ev) => {
  const { id, init, texts } = ev.data;
  try {
    if (init) { ready = (async () => { const tok = new CLIPTokenizer(init.tokenizerJson); const enc = await TinyTextJS.load(init.jsonUrl, init.binUrl); return { tok, enc }; })(); await ready; self.postMessage({ id, ok: true }); return; }
    const { tok, enc } = await ready; const t0 = performance.now();
    const embeddings = texts.map((t) => enc.encode(tok.encode(t).ids));
    self.postMessage({ id, embeddings, ms: performance.now() - t0 }, embeddings.map((e) => e.buffer));
  } catch (e) { self.postMessage({ id, error: e.message }); }
};

// Long text -> chunks under CLIP's 77-token context -> one blended target embedding. Never truncates silently.

const MAX = 75; // 77 minus BOS/EOS

function splitSentences(text) {
  return text.replace(/\s+/g, ' ').trim().split(/(?<=[.!?;:…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
}

/** Split `text` into chunks whose token count <= maxTokens. Whole sentences are merged greedily; an overlong sentence is split at commas, then words. */
export function chunkText(text, tokenizer, maxTokens = MAX) {
  const count = (s) => tokenizer.tokenize(s).length;
  const pieces = [];
  for (const sent of splitSentences(text)) {
    if (count(sent) <= maxTokens) { pieces.push({ text: sent, hard: false }); continue; }
    // too long: split at clause boundaries, then words
    const clauses = sent.split(/(?<=,|—|–|\(|\))\s*/).filter(Boolean);
    for (const cl of clauses) {
      if (count(cl) <= maxTokens) { pieces.push({ text: cl.trim(), hard: true }); continue; }
      let cur = [];
      for (const word of cl.split(' ')) {
        if (count([...cur, word].join(' ')) > maxTokens && cur.length) { pieces.push({ text: cur.join(' '), hard: true }); cur = []; }
        cur.push(word);
      }
      if (cur.length) pieces.push({ text: cur.join(' '), hard: true });
    }
  }
  // greedy merge of consecutive pieces
  const chunks = [];
  for (const p of pieces) {
    const last = chunks[chunks.length - 1];
    if (last && !p.hard && !last.hard && count(last.text + ' ' + p.text) <= maxTokens) { last.text += ' ' + p.text; last.tokens = count(last.text); }
    else chunks.push({ text: p.text, tokens: count(p.text), hard: p.hard });
  }
  return chunks;
}

/** Embed every chunk and blend them (weighted by token count) into one unit vector. */
export async function embedLongText(clip, text) {
  const chunks = chunkText(text, clip.tok);
  if (!chunks.length) throw new Error('empty text');
  const embs = [];
  for (const c of chunks) embs.push(await clip.embedText(c.text));
  const d = embs[0].length, target = new Float32Array(d);
  let wsum = 0;
  chunks.forEach((c, i) => { const w = Math.sqrt(c.tokens); wsum += w; for (let k = 0; k < d; k++) target[k] += w * embs[i][k]; });
  let n = 0; for (let k = 0; k < d; k++) { target[k] /= wsum; n += target[k] * target[k]; }
  n = Math.sqrt(n) + 1e-8; for (let k = 0; k < d; k++) target[k] /= n;
  return { target, chunks, embeddings: embs, hardSplits: chunks.filter((c) => c.hard).length };
}

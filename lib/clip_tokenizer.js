// CLIP BPE tokenizer (OpenAI / open_clip / HF CLIPTokenizer compatible) — zero dependencies.
// Usage: const tok = await CLIPTokenizer.load('.../tokenizer.json'); tok.encode('a red forest') -> Int32Array(77)
// Byte-level BPE with '</w>' end-of-word suffix, lowercase, BOS 49406, EOS 49407, padded to 77 with EOS
// Padded with 0 (this MobileCLIP export: pad_token "!" = id 0, like open_clip). The model pools at the EOS position.

function bytesToUnicode() {
  const bs = [];
  for (let i = '!'.charCodeAt(0); i <= '~'.charCodeAt(0); i++) bs.push(i);
  for (let i = '¡'.charCodeAt(0); i <= '¬'.charCodeAt(0); i++) bs.push(i);
  for (let i = '®'.charCodeAt(0); i <= 'ÿ'.charCodeAt(0); i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const map = new Map();
  bs.forEach((b, i) => map.set(b, String.fromCharCode(cs[i])));
  return map;
}

const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;

export class CLIPTokenizer {
  constructor(tokenizerJson) {
    const m = tokenizerJson.model;
    this.encoder = new Map(Object.entries(m.vocab));
    this.ranks = new Map(m.merges.map((s, i) => [Array.isArray(s) ? s.join(' ') : s, i]));
    this.byteEncoder = bytesToUnicode();
    this.cache = new Map();
    this.bos = this.encoder.get('<|startoftext|>');
    this.eos = this.encoder.get('<|endoftext|>');
    this.contextLength = 77;
  }

  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`tokenizer fetch ${url}: ${r.status}`);
    return new CLIPTokenizer(await r.json());
  }

  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);
    let word = [...token.slice(0, -1), token.slice(-1) + '</w>'];
    const getPairs = (w) => { const p = new Set(); for (let i = 0; i < w.length - 1; i++) p.add(w[i] + ' ' + w[i + 1]); return p; };
    let pairs = getPairs(word);
    if (!pairs.size) return token + '</w>';
    for (;;) {
      let best = null, bestRank = Infinity;
      for (const p of pairs) { const r = this.ranks.get(p); if (r !== undefined && r < bestRank) { bestRank = r; best = p; } }
      if (best === null) break;
      const [first, second] = best.split(' ');
      const out = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) { out.push(...word.slice(i)); break; }
        out.push(...word.slice(i, j)); i = j;
        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) { out.push(first + second); i += 2; }
        else { out.push(word[i]); i += 1; }
      }
      word = out;
      if (word.length === 1) break;
      pairs = getPairs(word);
    }
    const res = word.join(' ');
    this.cache.set(token, res);
    return res;
  }

  /** text -> array of token ids (no BOS/EOS) */
  tokenize(text) {
    text = text.normalize('NFC').replace(/\s+/g, ' ').trim().toLowerCase();
    const ids = [];
    const bytes = new TextEncoder();
    for (const tok of text.match(PAT) || []) {
      const enc = Array.from(bytes.encode(tok), (b) => this.byteEncoder.get(b)).join('');
      for (const piece of this.bpe(enc).split(' ')) {
        const id = this.encoder.get(piece);
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  /** text -> Int32Array(77): BOS ... EOS, zero padded. mask = 1 for real tokens. */
  encode(text) {
    const ids = this.tokenize(text).slice(0, this.contextLength - 2);
    const out = new Int32Array(this.contextLength); // pad id 0
    const mask = new Int32Array(this.contextLength);
    out[0] = this.bos; mask[0] = 1;
    ids.forEach((id, i) => { out[i + 1] = id; mask[i + 1] = 1; });
    out[ids.length + 1] = this.eos; mask[ids.length + 1] = 1;
    return { ids: out, mask, length: ids.length + 2 };
  }
}

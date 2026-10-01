// Real strokes from the app's own search (Chromium + WebGPU, local models): N prompts painted as lasso shapes in a fresh room,
// overlapping so some strokes land on painted canvas. Writes web/research/data/strokes.json after every stroke.
// Usage: node web/research/gen_strokes.mjs [--n 40] [--seconds 6]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { fileURLToPath } from 'node:url'; import { chromium } from 'playwright';
const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, s, i, arr) => { if (s.startsWith('--')) a.push([s.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return a; }, []));
const N = +(args.n || 40), seconds = +(args.seconds || 6), out = path.join(here, 'data', 'strokes.json');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css' };
const server = http.createServer((req, res) => { const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); });
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port, roomId = 'rs-' + Math.random().toString(36).slice(2, 8);
const url = `http://127.0.0.1:${port}/app/room.html?r=${roomId}&ort=/node_modules/onnxruntime-web/dist/&models=pages`;
const PROMPTS = ['a face', 'a red forest', 'the sea at night', 'a city street', 'green hills under a blue sky', 'a cat', 'deadline on Friday', "I miss my grandmother's kitchen", 'the smell of rain', 'a lighthouse in a storm', 'a bowl of oranges', 'snow on a mountain', 'a crowded market', 'a sleeping dog', 'fire', 'a glass of water on a table', 'we should talk about the budget', 'a yellow bicycle', 'an old library', "a child's drawing of a house", 'sunset over the plains', 'a portrait of a woman in blue', 'a horse in a field', 'mushrooms in the forest', 'the moon over the sea', 'a broken clock', 'a train station in winter', 'flowers in a vase', 'I feel tired today', 'a river through a canyon', 'a dark room with one candle', 'the first day of school', 'a bird on a wire', 'an abandoned factory', 'waves crashing on rocks', 'a wedding', 'a bridge in fog', 'coffee in the morning', 'a map of an imaginary island', 'the sound of a cello'];
const browser = await chromium.launch({ channel: 'chromium', headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=metal'] });
const page = await browser.newPage(); page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(url); await page.waitForFunction(() => window.__vqpaint && window.__vqpaint.ready && window.__vqpaint.grid, null, { timeout: 240000 });
await page.evaluate((s) => { window.__vqpaint.setEffortSeconds(s); window.__vqpaint.setTool('brush'); }, seconds);
console.log('ready, room', roomId);
let rnd = 7; const R = () => { rnd = (rnd * 1103515245 + 12345) % 2147483648; return rnd / 2147483648; };
const strokes = fs.existsSync(out) ? JSON.parse(fs.readFileSync(out, 'utf8')) : [];
for (let i = strokes.length; i < N; i++) {
  const prompt = PROMPTS[i % PROMPTS.length], realism = [0.2, 0.4, 0.6, 0.8][i % 4];
  const rx = 2.5 + R() * 5, ry = 2.5 + R() * 5;                      // lasso radii in tokens (desktop sizes up to ~12 tokens across)
  const cx = 116 + R() * 24, cy = 116 + R() * 24;                     // clustered so later strokes overlap earlier ones
  const pts = []; const n = 24; for (let k = 0; k < n; k++) { const a = k / n * Math.PI * 2, w = 1 + 0.25 * Math.sin(3 * a + i); pts.push([cx + rx * w * Math.cos(a), cy + ry * w * Math.sin(a)]); }
  const t0 = Date.now();
  try { await page.evaluate(({ pts, prompt, realism }) => window.__vqpaint.lassoPaint(pts, prompt, realism), { pts, prompt, realism }); }
  catch (e) { console.log('stroke failed', i, e.message); continue; }
  const s = await page.evaluate(() => { const v = window.__vqpaint, s = v.strokes[v.strokes.length - 1]; return { text: s.text, realism: s.realism, crop: s.crop, tokens: s.tokens, mask: s.mask, path: s.path, tries: v.stats.lastTries }; });
  strokes.push(s); fs.writeFileSync(out, JSON.stringify(strokes));
  console.log(`${i + 1}/${N} "${prompt}" r=${realism} crop ${s.crop.w}x${s.crop.h} ${s.tries} tries ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
await browser.close(); server.close(); console.log('wrote', out, strokes.length, 'strokes');

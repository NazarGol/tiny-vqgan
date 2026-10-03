// Headless run of research/compare_search.html (ORT real-CLIP search vs the light engine, both scored by real CLIP through the
// ORIGINAL decoder). Usage: node web/research/run_compare.mjs [--base /research/testmodels/] [--modes prefilter] [--n 10] [--seconds 8] [--out file.json]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { fileURLToPath } from 'node:url'; import { chromium } from 'playwright';
const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, s, i, arr) => { if (s.startsWith('--')) a.push([s.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return a; }, []));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => { const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); });
await new Promise((r) => server.listen(0, '127.0.0.1', r)); const port = server.address().port;
const url = `http://127.0.0.1:${port}/research/compare_search.html?base=${encodeURIComponent(args.base || '/research/testmodels/')}&modes=${args.modes || 'prefilter'}&n=${args.n || 10}&seconds=${args.seconds || 8}`;
const browser = await chromium.launch({ channel: 'chromium', headless: true, args: ['--ignore-gpu-blocklist', '--use-angle=metal', '--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const page = await browser.newPage(); page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(url); await page.waitForFunction(() => window.__result, null, { timeout: 1200000 });
const r = await page.evaluate(() => window.__result);
if (args.out) fs.writeFileSync(args.out, JSON.stringify(r, null, 1));
console.log(JSON.stringify({ ok: r.ok, error: r.error, summary: r.summary }));
for (const row of r.rows || []) console.log(`${row.prompt.padEnd(30)} ORT ${row.A.real.toFixed(3)} (${row.A.tries}) | ` + Object.entries(row.modes).map(([m, v]) => `${m} ${v.real.toFixed(3)} (${v.tries} tries, ${v.sec.toFixed(1)}s)`).join(' | '));
await browser.close(); server.close();

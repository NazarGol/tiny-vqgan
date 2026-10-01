// Runs test_tinydec.html in Chromium and WebKit: shader output vs PyTorch, timing. Usage: node web/research/test_tinydec.mjs [--variant A] [--base ./tiny/test/] [--browser chromium|webkit|all]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { fileURLToPath } from 'node:url'; import { chromium, webkit } from 'playwright';
const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, s, i, arr) => { if (s.startsWith('--')) a.push([s.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return a; }, []));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => { const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); });
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const url = `http://127.0.0.1:${port}/research/test_tinydec.html?variant=${args.variant || 'A'}&base=${encodeURIComponent(args.base || './tiny/test/')}`;
const which = args.browser === 'all' ? ['chromium', 'webkit'] : [args.browser || 'chromium'];
for (const b of which) {
  const browser = b === 'webkit' ? await webkit.launch({ headless: true }) : await chromium.launch({ channel: 'chromium', headless: true, args: ['--ignore-gpu-blocklist', '--use-angle=metal', ...(args.swiftshader ? ['--use-gl=angle', '--use-angle=swiftshader'] : [])] });
  const page = await browser.newPage(); page.on('pageerror', (e) => console.log('[pageerror]', e.message)); page.on('console', (m) => { if (m.type() === 'error') console.log('[console]', m.text()); });
  await page.goto(url); await page.waitForFunction(() => window.__result, null, { timeout: 120000 });
  const r = await page.evaluate(() => window.__result);
  console.log(b, JSON.stringify(r));
  await browser.close();
}
server.close();

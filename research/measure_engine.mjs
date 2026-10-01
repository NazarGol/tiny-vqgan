// Peak process-tree RSS of the engine test page over N strokes (text worker per stroke): WebKit with an iPhone profile, or Chromium.
// Usage: node web/research/measure_engine.mjs [--browser webkit|chromium] [--device "iPhone 11"] [--strokes 20] [--seconds 3]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { execSync } from 'node:child_process'; import { fileURLToPath } from 'node:url'; import { chromium, webkit, devices } from 'playwright';
const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, s, i, arr) => { if (s.startsWith('--')) a.push([s.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return a; }, []));
const browserName = args.browser || 'webkit', deviceName = args.device || 'iPhone 11', strokes = +(args.strokes || 20), seconds = +(args.seconds || 3);
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => { const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); });
await new Promise((r) => server.listen(0, '127.0.0.1', r)); const port = server.address().port;
const url = `http://127.0.0.1:${port}/research/test_engine.html?base=${encodeURIComponent(args.base || './testmodels/')}&strokes=${strokes}&seconds=${seconds}&text=${args.text ?? 1}&batch=${args.batch || 32}${args.hold ? '&hold=' + args.hold : ''}`;
const pattern = browserName === 'webkit' ? 'ms-playwright/webkit' : 'ms-playwright/chromium';
const rss = () => { try { return execSync(`ps -axo rss=,command= | grep -F '${pattern}' | grep -v grep | awk '{s+=$1} END {print s+0}'`).toString().trim() * 1 || 0; } catch { return 0; } };
const wc = () => { try { return execSync(`ps -axo rss=,command= | grep -F '${pattern}' | grep -F -e WebContent -e 'type=renderer' | grep -v grep | sort -rn | head -1 | awk '{print $1+0}'`).toString().trim() * 1 || 0; } catch { return 0; } };
const browser = browserName === 'webkit' ? await webkit.launch({ headless: true }) : await chromium.launch({ channel: 'chromium', headless: true, args: ['--ignore-gpu-blocklist', '--use-angle=metal'] });
const ctx = await browser.newContext({ ...(devices[deviceName] || {}) }); const page = await ctx.newPage();
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('about:blank'); await page.waitForTimeout(1500); const base = rss(), baseWc = wc();
let peak = 0, peakWc = 0; const timer = setInterval(() => { peak = Math.max(peak, rss() - base); peakWc = Math.max(peakWc, wc()); }, 250);
const t0 = Date.now(); await page.goto(url); await page.waitForFunction(() => window.__result, null, { timeout: 600000 }); clearInterval(timer);
const r = await page.evaluate(() => window.__result);
const tries = r.strokes.map((s) => s.perSec); const avg = tries.length ? tries.reduce((a, b) => a + b, 0) / tries.length : 0;
console.log(JSON.stringify({ browser: browserName, device: deviceName, ok: r.ok, error: r.error, strokes: r.strokes.length, triesPerSec: Math.round(avg), decodeMs: r.strokes.map((s) => +s.decodeMs.toFixed(1)).slice(0, 5), textMs: r.textMs, loadMs: r.loadMs, weightMB: +r.weightMB.toFixed(1), texMB: +r.texMB.toFixed(1), peakRssAboveEmptyMB: Math.round(peak / 1024), peakWebContentMB: Math.round(peakWc / 1024), baseWebContentMB: Math.round(baseWc / 1024), totalSec: +((Date.now() - t0) / 1000).toFixed(0) }));
await browser.close(); server.close();

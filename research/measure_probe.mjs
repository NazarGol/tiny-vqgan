// RSS growth of the WebContent process for each engine component (research/test_memprobe.html). Usage: node measure_probe.mjs [--browser webkit] [--device "iPhone 11"]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { execSync } from 'node:child_process'; import { fileURLToPath } from 'node:url'; import { chromium, webkit, devices } from 'playwright';
const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, s, i, arr) => { if (s.startsWith('--')) a.push([s.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return a; }, []));
const browserName = args.browser || 'webkit', deviceName = args.device || 'iPhone 11';
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => { const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); });
await new Promise((r) => server.listen(0, '127.0.0.1', r)); const port = server.address().port;
const pattern = browserName === 'webkit' ? 'ms-playwright/webkit' : 'ms-playwright/chromium';
const wc = () => { try { return execSync(`ps -axo rss=,command= | grep -F '${pattern}' | grep -F -e WebContent -e 'type=renderer' | grep -v grep | sort -rn | head -1 | awk '{print $1+0}'`).toString().trim() * 1 || 0; } catch { return 0; } };
const all = () => { try { return execSync(`ps -axo rss=,command= | grep -F '${pattern}' | grep -v grep | awk '{s+=$1} END {print s+0}'`).toString().trim() * 1 || 0; } catch { return 0; } };
const out = {};
for (const what of (args.what || 'empty,gl,programs,decoder,scorer,data,worker,decode,score,all').split(',')) {
  const browser = browserName === 'webkit' ? await webkit.launch({ headless: true }) : await chromium.launch({ channel: 'chromium', headless: true, args: ['--ignore-gpu-blocklist', '--use-angle=metal'] });
  const ctx = await browser.newContext({ ...(devices[deviceName] || {}) }); const page = await ctx.newPage();
  await page.goto(`http://127.0.0.1:${port}/research/test_memprobe.html?what=none`); await page.waitForTimeout(1500); const b = wc(), ba = all();
  if (what !== 'empty') { await page.goto(`http://127.0.0.1:${port}/research/test_memprobe.html?what=${what}`); await page.waitForFunction(() => window.__result, null, { timeout: 120000 }); }
  let peak = 0, peakA = 0; for (let i = 0; i < 8; i++) { await page.waitForTimeout(250); peak = Math.max(peak, wc() - b); peakA = Math.max(peakA, all() - ba); }
  out[what] = { webContentMB: Math.round(peak / 1024), treeMB: Math.round(peakA / 1024) }; console.log(what, JSON.stringify(out[what]));
  await browser.close();
}
server.close();

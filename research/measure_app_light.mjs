// Stage-by-stage RSS of the app with the light engine on a phone profile: page ready → engine loaded → text → search → stroke done → settled.
// Usage: node research/measure_app_light.mjs [--browser webkit] [--device "iPhone 11"] [--seconds 5] [--engine ort]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { execSync } from 'node:child_process'; import { fileURLToPath } from 'node:url'; import { chromium, webkit, devices } from 'playwright';
const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, s, i, arr) => { if (s.startsWith('--')) a.push([s.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return a; }, []));
const browserName = args.browser || 'webkit', deviceName = args.device || 'iPhone 11', seconds = +(args.seconds || 5);
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => { const p = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); });
await new Promise((r) => server.listen(0, '127.0.0.1', r)); const port = server.address().port, roomId = 'mem-' + Math.random().toString(36).slice(2, 7);
const url = `http://127.0.0.1:${port}/app/room.html?r=${roomId}&models=pages&ort=/node_modules/onnxruntime-web/dist/${args.engine ? '&engine=' + args.engine : ''}${args.extra || ''}`;
const pattern = browserName === 'webkit' ? 'ms-playwright/webkit' : 'ms-playwright/chromium';
const wc = () => { try { return execSync(`ps -axo rss=,command= | grep -F '${pattern}' | grep -F -e WebContent -e 'type=renderer' | grep -v grep | sort -rn | head -1 | awk '{print $1+0}'`).toString().trim() * 1 || 0; } catch { return 0; } };
const gpu = () => { try { return execSync(`ps -axo rss=,command= | grep -F '${pattern}' | grep -F -e 'WebKit.GPU' -e 'type=gpu-process' | grep -v grep | sort -rn | head -1 | awk '{print $1+0}'`).toString().trim() * 1 || 0; } catch { return 0; } };
const browser = browserName === 'webkit' ? await webkit.launch({ headless: true }) : await chromium.launch({ channel: 'chromium', headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=metal'] });
const ctx = await browser.newContext({ ...(devices[deviceName] || {}) }); const page = await ctx.newPage(); await page.goto('about:blank'); await page.waitForTimeout(1000);
const b0 = wc(), g0 = gpu(); const mark = (label) => console.log(`${label.padEnd(26)} WebContent +${Math.round((wc() - b0) / 1024)} MB   GPU process +${Math.round((gpu() - g0) / 1024)} MB`);
let peak = 0; const timer = setInterval(() => { peak = Math.max(peak, wc() - b0); }, 200);
await page.goto(url); await page.waitForFunction(() => window.__vqpaint && window.__vqpaint.ready && window.__vqpaint.grid, null, { timeout: 120000 }); await page.waitForTimeout(1500); mark('page ready (viewing)');
await page.evaluate((s) => { window.__vqpaint.setEffortSeconds(s); return window.__vqpaint.ensureBrush(); }, seconds); await page.waitForTimeout(1500); mark('engine loaded');
const info = await page.evaluate(() => ({ engine: window.__vqpaint.stats.engine, variant: window.__vqpaint.stats.engineVariant, decodeMs: window.__vqpaint.stats.fullDecodeMs }));
await page.evaluate(() => window.__vqpaint.tapPaint(140, 120, 'a red forest at night', 0.6));
await page.waitForFunction(() => window.__vqpaint.painting, null, { timeout: 60000 }).catch(() => {}); await page.waitForTimeout(2500); mark('during search');
await page.waitForFunction(() => window.__vqpaint.strokes.length >= 1 && !window.__vqpaint.painting, null, { timeout: 180000 }); mark('stroke done');
const r = await page.evaluate(() => ({ tries: window.__vqpaint.stats.lastTries, crop: window.__vqpaint.strokes[0].crop }));
await page.waitForTimeout(6000); mark('settled 6 s later'); clearInterval(timer);
console.log(JSON.stringify({ browser: browserName, device: deviceName, ...info, tries: r.tries, crop: `${r.crop.w}x${r.crop.h}`, peakWebContentMB: Math.round(peak / 1024) }));
await browser.close(); server.close();

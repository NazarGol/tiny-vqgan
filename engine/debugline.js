// Hidden debug line: tap the room bar 5 times within 3 s to toggle a small monospace line with engine mode, tries/s,
// seconds per stroke and memory (where the browser exposes it). Read-only; nothing else changes.
export function mountDebugLine(tapEl, getInfo) {
  const line = document.createElement('div');
  line.hidden = true; line.style.cssText = 'position:fixed;left:8px;right:8px;bottom:8px;z-index:9999;padding:6px 8px;border-radius:8px;background:rgba(0,0,0,.72);color:#eee;font:11px/1.35 ui-monospace,Menlo,monospace;white-space:pre-wrap;pointer-events:none';
  document.body.appendChild(line);
  let taps = [], timer = null;
  const render = () => { try { const i = getInfo(); line.textContent = Object.entries(i).map(([k, v]) => `${k} ${v}`).join('  ·  '); } catch (e) { line.textContent = 'debug: ' + e.message; } };
  const show = (on) => { line.hidden = !on; if (timer) clearInterval(timer); timer = on ? setInterval(render, 1000) : null; if (on) render(); };
  tapEl.addEventListener('pointerdown', () => { const now = Date.now(); taps = taps.filter((t) => now - t < 3000); taps.push(now); if (taps.length >= 5) { taps = []; show(line.hidden); } }, { passive: true });
  return { show, render };
}
/** Memory as the browser reports it: Chrome's JS heap, or n/a (Safari exposes nothing). */
export function memoryInfo() {
  const m = performance.memory; const parts = [];
  if (m && m.usedJSHeapSize) parts.push(`js heap ${Math.round(m.usedJSHeapSize / 2 ** 20)} MB`);
  if (navigator.deviceMemory) parts.push(`device ${navigator.deviceMemory} GB`);
  return parts.length ? parts.join(', ') : 'n/a';
}

// Remote log: compact JSON entries batched to the rooms worker (POST /room/:id/log). Works on the page and in workers.
// Nothing personal is sent: loader stages, engine init facts, errors, user agent, memory hint, GPU renderer string.
let queue = [], timer = null, endpoint = null, clientTag = '', sending = false, disabled = false;
function flush() {
  if (!endpoint || !queue.length || sending) return;
  const batch = queue.splice(0, 50); sending = true;
  try { fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(batch), keepalive: true }).catch(() => {}).finally(() => { sending = false; if (queue.length) schedule(); }); }
  catch (_) { sending = false; }
}
function schedule(ms = 2500) { if (timer) return; timer = setTimeout(() => { timer = null; flush(); }, ms); }
export function rlog(entry) {
  if (disabled) return;
  try { const e = { ts: Date.now(), c: clientTag, ...entry }; queue.push(e); if (queue.length >= 20) flush(); else schedule(); } catch (_) {}
}
/** url: POST endpoint; tag: 'page' | 'worker'; also hooks errors and unhandled rejections and logs device facts once. */
export function installRemoteLog({ url, tag = 'page', device = true } = {}) {
  endpoint = url; clientTag = tag; disabled = !url; globalThis.__rlog = rlog;
  if (disabled) return rlog;
  try {
    globalThis.addEventListener?.('error', (e) => rlog({ t: 'error', msg: String(e.message || e), src: e.filename ? `${e.filename.split('/').pop()}:${e.lineno}` : undefined }));
    globalThis.addEventListener?.('unhandledrejection', (e) => rlog({ t: 'rejection', msg: String(e.reason && (e.reason.message || e.reason)).slice(0, 300) }));
  } catch (_) {}
  if (device) {
    const d = { t: 'device', ua: (globalThis.navigator && navigator.userAgent) || '', mem: globalThis.navigator && navigator.deviceMemory, cores: globalThis.navigator && navigator.hardwareConcurrency, webgpu: !!(globalThis.navigator && navigator.gpu), offscreen: typeof OffscreenCanvas !== 'undefined', worker: typeof document === 'undefined' };
    try { const c = typeof document !== 'undefined' ? document.createElement('canvas') : new OffscreenCanvas(1, 1); const gl = c.getContext('webgl2'); if (gl) { const dbg = gl.getExtension('WEBGL_debug_renderer_info'); d.gl = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); d.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE); d.halfRT = !!gl.getExtension('EXT_color_buffer_half_float'); d.floatRT = !!gl.getExtension('EXT_color_buffer_float'); gl.getExtension('WEBGL_lose_context')?.loseContext(); } else d.gl = null; } catch (e) { d.gl = 'error: ' + (e && e.message); }
    rlog(d);
  }
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') flush(); });
  return rlog;
}
export function flushRemoteLog() { flush(); }

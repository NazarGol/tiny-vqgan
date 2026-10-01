"""Apply the light-engine wiring to web/app/room.js (idempotent; asserts each anchor). Usage: python patch_room.py [path/to/room.js] [--check]"""
import sys
path = next((a for a in sys.argv[1:] if not a.startswith("--")), "web/app/room.js"); check = "--check" in sys.argv
s = open(path).read(); n = 0
REPS = [
 ("import { Painter } from '../lib/search.js';", "import { Painter } from '../lib/search.js';\nimport { loadEngineBridge } from './engine_bridge.js';"),
 ("const lowMem = isPhone || safeMode || params.get('lowmem') === '1';", "const lowMem = isPhone || safeMode || params.get('lowmem') === '1';\nlet useTiny = params.get('engine') === 'tiny';   // the light engine (tiny decoder + token scorer, WebGL2): default on phones and without WebGPU, ?engine=ort forces the ONNX path"),
 # either the lasso cap (older room.js) or the hold-to-grow radius (no-modes room.js)
 (("const max = lowMem ? 8 : 40;", "const max = useTiny ? 24 : lowMem ? 8 : 40;"), ("const MAX_R = lowMem ? 4 : 12, BASE_R = lowMem ? 3 : 4.5;", "const MAX_R = useTiny ? 8 : lowMem ? 4 : 12, BASE_R = useTiny ? 4 : lowMem ? 3 : 4.5;")),
 ("if (h && (forceNoPaint || lowMem || !caps.gpu)) return requestHelp(", "if (h && (forceNoPaint || (!useTiny && (lowMem || !caps.gpu)))) return requestHelp("),
 # the brush-on-select line only exists in the older room.js (optional)
 (("if (name === 'brush' && ready && !safeMode && !lowMem) ensureBrush()", "if (name === 'brush' && ready && !safeMode && (!lowMem || useTiny)) ensureBrush()"), None),
 ("    if (lowMem) await releaseBrush();\n", "    if (lowMem && !useTiny) await releaseBrush();\n"),
 ("    if (photo && photo.chw) {   // the photo guides CLIP too", "    if (photo && (useTiny ? photo.tokens : photo.chw)) {   // the photo guides CLIP too (light engine: the scorer's embedding of the photo's tokens)"),
 ("      const [pe] = await clip.embedImages(photo.chw.length", "      const [pe] = useTiny ? [clip.embedTokens(photo.tokens, photo.side)] : await clip.embedImages(photo.chw.length"),
 ("grid, mask, target, seconds: rp.seconds, margin: MARGIN,", "grid, mask, target, seconds: rp.seconds, margin: useTiny ? 2 : MARGIN,"),
 ("let ensuring = null;", "let ensuring = null, engineBridge = null;"),
 ("""  ensuring ||= (async () => {
    mode = 'brush'; sessionStorage.setItem('vqpaint.boot', 'painting');""", """  ensuring ||= (async () => {
    mode = 'brush'; sessionStorage.setItem('vqpaint.boot', 'painting');
    if (useTiny) {   // light engine: ~16 MB of files, no ONNX Runtime
      setStage('engine'); const total = 16 * 2 ** 20, seen = {};
      const onP = (p) => { seen[p.url] = p.loaded; const loaded = Object.values(seen).reduce((a, b) => a + b, 0); loading.set(t('load.brush', { pct: Math.min(99, Math.round(loaded / total * 100)) })); stats.modelBytes = loaded; stats.cached = !!p.cached; };
      loading.set(t('load.brush', { pct: 0 }));
      engineBridge = await loadEngineBridge({ base: M, onProgress: onP, bank: /^[a-z_]+$/.test(params.get('bank') || '') ? params.get('bank') : 'bank' });
      decoder = engineBridge.decoder; clip = engineBridge.clip; painter = engineBridge.painter; bank = engineBridge.engine.bank;
      layers.setDecoder(decoder); caps.speed = stats.fullDecodeMs = Math.round(engineBridge.probe()); stats.engine = 'tiny'; stats.engineVariant = engineBridge.decoder.variant;
      setStage('brush-ready'); beacon('brush-ready'); modelsLoaded = true; caps.paint = true; room?.setCaps(caps);
      loading.hide(); sessionStorage.setItem('vqpaint.boot', 'ok');
      return;
    }"""),
 ("""  painter = null; modelsLoaded = false; mode = 'view'; setStage('releasing');
  if (clip) { await clip.release(); clip = null; }""", """  painter = null; modelsLoaded = false; mode = 'view'; setStage('releasing');
  if (engineBridge) { engineBridge.release(); engineBridge = null; }
  if (clip) { await clip.release(); clip = null; }"""),
 ("""  beacon('gpu', { gpu }); caps.gpu = !!gpu;""", """  beacon('gpu', { gpu }); caps.gpu = !!gpu;
  if (params.get('engine') !== 'ort' && (lowMem || !gpu)) useTiny = true;"""),
]
missing = []
for item in REPS:
    alts = [x for x in (item if isinstance(item[0], tuple) else (item,)) if x is not None]
    if any(b in s for a, b in alts): continue
    hit = next((x for x in alts if s.count(x[0]) == 1), None)
    if hit is None:
        if isinstance(item[0], tuple) and item[1] is None: continue   # optional anchor
        missing.append(alts[0][0][:70]); continue
    s = s.replace(hit[0], hit[1]); n += 1
if check: print(f"{len(REPS) - len(missing)}/{len(REPS)} anchors found"); [print("  missing:", m) for m in missing]; sys.exit(1 if missing else 0)
if missing: sys.exit("anchors not found: " + " | ".join(missing))
open(path, "w").write(s); print(f"room.js: {n} edits applied")

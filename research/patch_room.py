"""Apply the light-engine wiring to web/app/room.js (the worker-engine version, 2026-10-02+). Idempotent; asserts each anchor.
Usage: python patch_room.py [path/to/room.js] [--check]"""
import sys
path = next((a for a in sys.argv[1:] if not a.startswith("--")), "web/app/room.js"); check = "--check" in sys.argv
s = open(path).read(); n = 0
REPS = [
 ("import { Engine } from '../lib/engine/client.js';", "import { Engine } from '../lib/engine/client.js';\nimport { LightEngine } from '../lib/engine/light.js';\nimport { mountDebugLine, memoryInfo } from './debugline.js';"),
 # crash levels: first crash -> lightest light engine (int8 CLIP, no scorer); second -> the note waits for a computer (their safe mode)
 ("let safeMode = params.get('safe') === '1' || (lastBoot === 'loading' || lastBoot === 'painting');\nif (safeMode) localStorage.setItem('vqpaint.crashes', String((+localStorage.getItem('vqpaint.crashes') || 0) + 1));",
  "const crashedLastTime = lastBoot === 'loading' || lastBoot === 'painting';\nif (crashedLastTime) localStorage.setItem('vqpaint.crashes', String((+localStorage.getItem('vqpaint.crashes') || 0) + 1));\nif (params.get('reset') === '1') localStorage.setItem('vqpaint.crashes', '0');\nconst crashes = +localStorage.getItem('vqpaint.crashes') || 0;\nlet safeMode = params.get('safe') === '1' || (crashedLastTime && crashes >= 2);   // light engine: one crash -> lightest mode, two -> the note waits for a computer\nconst lightest = params.get('light') === '1' || (crashes >= 1 && !safeMode);"),
 ("const lowMem = isPhone || safeMode || params.get('lowmem') === '1';", "const lowMem = isPhone || safeMode || params.get('lowmem') === '1';\nlet useTiny = params.get('engine') !== 'ort';   // the light engine (tiny decoder + MobileCLIP as WebGL2 shaders, no ONNX Runtime) is the default on every device; ?engine=ort forces the ONNX worker"),
 ("const engine = new Engine();", "let engine = new Engine();"),
 ("function canPaintHere() { if (forceNoPaint || safeMode || engine.broken) return false; if (!lowMem) return true;", "function canPaintHere() { if (forceNoPaint || safeMode || engine.broken) return false; if (useTiny) return true; if (!lowMem) return true;"),
 ("if (helpersOn && (forceNoPaint || lowMem || !caps.gpu) && (bestHelper()", "if (helpersOn && (forceNoPaint || (!useTiny && (lowMem || !caps.gpu))) && (bestHelper()"),
 ("if (h && (forceNoPaint || lowMem || !caps.gpu)) return requestReaction(", "if (h && (forceNoPaint || (!useTiny && (lowMem || !caps.gpu)))) return requestReaction("),
 ("margin: MARGIN,", "margin: useTiny ? 2 : MARGIN,"),
 ("if (lowMem) await releaseBrush();", "if (lowMem && !useTiny) await releaseBrush();"),
 ("  beacon('gpu', { gpu }); caps.gpu = !!gpu;", "  beacon('gpu', { gpu }); caps.gpu = !!gpu;\n  if (useTiny) { engine = new LightEngine(lightest ? { mode: 'clip', clip: 'clip_vision_i8', scorer: null, text: 'S' } : { mode: params.get('mode') || 'prefilter' }); stats.engine = 'tiny'; stats.engineMode = lightest ? 'lightest (int8 CLIP, no scorer)' : (params.get('mode') || 'prefilter'); stats.crashes = crashes; }\n  mountDebugLine($('roombar'), () => { const secs = stats.strokeSeconds[stats.strokeSeconds.length - 1]; return { engine: (stats.engine || 'ort') + (stats.engineVariant ? ' ' + stats.engineVariant : ''), mode: stats.engineMode || ep || '-', crashes: stats.crashes ?? 0, 'last stroke': secs ? `${stats.lastTries} tries in ${secs.toFixed(1)} s (${(stats.lastTries / secs).toFixed(1)}/s)` : '-', 'decode ms': stats.fullDecodeMs ?? '-', memory: memoryInfo() }; });"),
 ("  caps.helper = !!gpu && !lowMem && !forceNoPaint && helpersOn;", "  caps.helper = (useTiny || !!gpu) && !lowMem && !forceNoPaint && helpersOn;"),
]
missing = []
for a, b in REPS:
    if b in s: continue
    c = s.count(a)
    if c == 0: missing.append(a[:70]); continue
    s = s.replace(a, b); n += c
if check: print(f"{len(REPS) - len(missing)}/{len(REPS)} anchors found"); [print("  missing:", m) for m in missing]; sys.exit(1 if missing else 0)
if missing: sys.exit("anchors not found: " + " | ".join(missing))
open(path, "w").write(s); print(f"room.js: {n} edits applied")

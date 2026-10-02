"""Apply the light-engine wiring to web/app/room.js (the worker-engine version, 2026-10-02+). Idempotent; asserts each anchor.
Usage: python patch_room.py [path/to/room.js] [--check]"""
import sys
path = next((a for a in sys.argv[1:] if not a.startswith("--")), "web/app/room.js"); check = "--check" in sys.argv
s = open(path).read(); n = 0
REPS = [
 ("import { Engine } from '../lib/engine/client.js';", "import { Engine } from '../lib/engine/client.js';\nimport { LightEngine } from '../lib/engine/light.js';"),
 ("const lowMem = isPhone || safeMode || params.get('lowmem') === '1';", "const lowMem = isPhone || safeMode || params.get('lowmem') === '1';\nlet useTiny = params.get('engine') === 'tiny';   // the light engine (tiny decoder + MobileCLIP as WebGL2 shaders, no ONNX Runtime): default on phones and without WebGPU, ?engine=ort forces the ONNX worker"),
 ("const engine = new Engine();", "let engine = new Engine();"),
 ("function canPaintHere() { if (forceNoPaint || safeMode || engine.broken) return false; if (!lowMem) return true;", "function canPaintHere() { if (forceNoPaint || safeMode || engine.broken) return false; if (useTiny) return true; if (!lowMem) return true;"),
 ("const MAX_R = lowMem ? 4 : 12, BASE_R = lowMem ? 3 : 4.5;", "const MAX_R = useTiny ? 8 : lowMem ? 4 : 12, BASE_R = useTiny ? 4 : lowMem ? 3 : 4.5;"),
 ("if (h && (forceNoPaint || lowMem || !caps.gpu)) return requestHelp(", "if (h && (forceNoPaint || (!useTiny && (lowMem || !caps.gpu)))) return requestHelp("),
 ("if (h && (forceNoPaint || lowMem || !caps.gpu)) return requestReaction(", "if (h && (forceNoPaint || (!useTiny && (lowMem || !caps.gpu)))) return requestReaction("),
 ("margin: MARGIN,", "margin: useTiny ? 2 : MARGIN,"),
 ("if (lowMem) await releaseBrush();", "if (lowMem && !useTiny) await releaseBrush();"),
 ("  beacon('gpu', { gpu }); caps.gpu = !!gpu;", "  beacon('gpu', { gpu }); caps.gpu = !!gpu;\n  if (params.get('engine') !== 'ort' && (lowMem || !gpu)) useTiny = true;\n  if (useTiny) { engine = new LightEngine({ mode: params.get('mode') || 'prefilter' }); stats.engine = 'tiny'; }"),
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

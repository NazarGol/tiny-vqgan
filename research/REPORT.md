# Tiny VQGAN engine — report (draft, numbers filled in as runs finish)

**Goal.** Paint on the minimum phones (`web/DEVICES.md`: iPhone XR/11/SE2 on iOS 15+, 3 GB Androids from 2020) with no server and no ML runtime in the painting loop.

## What changed
- **Before**: ONNX Runtime + the 45 MB packed VQGAN decoder + MobileCLIP (12 + 41 MB): ~900 MB just to load in WebKit, 1.4–1.9 GB per stroke, 4–6 tries/s. Phones had to ask a laptop to paint.
- **After**: three small models as WebGL2 shader passes / plain JS (`web/engine/`), ~17 MB of files in total:
  - tiny decoder (tokens → RGB, distilled from the original decoder): 3.1 MB, 256 px in ~10 ms on the M1;
  - token scorer (tokens → CLIP image embedding, no decode): 5.2 MB, ~1 700–2 000 scored candidates/s (batch 32);
  - tiny text encoder (distilled MobileCLIP text tower): 6.6 MB, 15–90 ms per note in a worker;
  - plus the existing palette (4 MB) and bank (2.4 MB).
- The search scores candidates in token space and decodes only the preview (every 300 ms).

## Numbers (emulation on the M1 Pro; real-phone numbers: see "what to test")
| | before (ORT path) | after (light engine) |
|---|---|---|
| download to paint | 108 MB | 17.5 MB |
| painting peak memory, iPhone profile (WebKit, above an empty tab) | 1 777 MB | 280 MB (app), 250 MB (engine test page, 20 strokes) |
| tries per second (desktop Chromium / WebKit) | 4–6 | ~2 000 / ~1 750 |
| brush ready (files cached) | 1.3–1.7 s | 1.3–2.5 s |
| 256 px decode | 220–420 ms | 7–15 ms |

## Quality
- Phase 1 side-by-side (original vs tiny decoder, 30 grids incl. real strokes): _pending Kaggle run_.
- Phase 2 scorer vs real CLIP (correlation, pair agreement; search side by side): _pending_.
- Phase 3 text encoder vs MobileCLIP text (cosine, retrieval overlap): _pending_.
- Phase 5 one-pass seed: _pending_.

## PR
_pending_

## What to test on the real phones
See `web/DEVICES.md` → "What to test on a real old phone".

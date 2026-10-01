# Research progress (tiny VQGAN engine)

Goal: paint on the minimum phones (`web/DEVICES.md`) with no ML runtime in the loop: tiny decoder + token-space CLIP scorer as WebGL2 shader passes, text encoder once per note.

## Phase 1 — tiny decoder (2026-10-01)
- Training code: `tiny/common.py` (on-the-fly token grids: bank photos/paintings crops, 32×32 COCO/CelebA crops, random, mosaics, strokes on blank/painted canvas, search-style mutations; students; export), `tiny/train_decoder.py` (L1 + LPIPS, EMA, time-based cosine, checkpoints, eval sheet with real strokes, CLIP-agreement metric). Smoke-tested locally on MPS end to end.
- Kaggle: notebook `kaggle/tiny_decoder/` (clones this branch, checkpoint from the `vqpaint-assets` dataset, token grids from `vqpaint-tokens`). 21.7 GPU hours left this week.
- Browser runtime `lib/tinydec.js` (WebGL2, no runtime): matches PyTorch (max err 0.002 = 8-bit rounding) in Chromium and WebKit; 256 px decode 15 ms Chromium / 12 ms WebKit on the M1 Pro, 3.1 MiB of weights. Test: `node web/research/test_tinydec.mjs --browser all`.
- Data: 40 real strokes from the app's search (`data/strokes.json`), 10 000 16×16 grids, 6 500 32×32 grids.

## Phase 2 — token-space scorer
- Training code `tiny/train_scorer.py` + notebook `kaggle/tiny_scorer/` written and smoke-tested; runtime in progress.
- Scorer runtime `lib/tinyscorer.js` (batched: B grids of S×S in one texture; convs with texture weights; FC + cosine + float packing on the GPU): matches PyTorch (score err 3e-5, cos 0.9999999) in Chromium and WebKit; 64 grids of 12×12 scored in 13–20 ms on the M1 Pro (≈ 3 000–5 000 candidates/s vs 4–6 decode+CLIP tries/s today).
- `engine/search.js` (token-space hill-climb, batch of 32 mutations per generation, preview decode every 300 ms) and `engine/engine.js` (load / encodeText / paintStroke / decode / release) written; mechanics to be tested with the smoke weights, quality once Kaggle weights land.

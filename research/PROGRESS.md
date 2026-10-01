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

## Phase 3 — text encoder (2026-10-01)
- Student `TinyText` (CLIP-style causal transformer, token table 49408×32 + width 192/256, 4 layers; 3.5 M / 4.9 M params = 6.6 / 9.3 MB fp16), trainer `tiny/train_text.py` (COCO captions + painting prompts + metaphors + synthetic notes + random id sequences; 1−cos; eval: held-out caption cosine, top-10 neighbour overlap, prompt/metaphor/note cosine). Smoke-tested; notebook `kaggle/tiny_text/` waits for a free GPU slot (Kaggle allows 2 concurrent GPU kernels).
- Runtime `engine/text.js` (plain JS, no runtime) + `engine/text_worker.js` (Worker, terminated by the caller): matches PyTorch (cos 0.99999999), 15 ms per short note / 68 ms for a 25-word note on the M1 Pro, worker load+encode 150 ms. Test: `node web/research/test_tinydec.mjs --page test_tinytext.html --variant S`.

## Notes
- Kaggle kernels v1 failed: datasets were not found at `/kaggle/input/<slug>`; v2 searches recursively and falls back to heibox for the checkpoint. Kernel logs are only readable after a kernel ends.
- Uploading the token grids to Hugging Face as a fallback data source was blocked by the Claude Code permission classifier (data exfiltration rule); Kaggle datasets remain the only data channel. Nothing secret was involved (token indices, 17 MB).

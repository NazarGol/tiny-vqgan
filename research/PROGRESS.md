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

## Engine memory (WebKit headless, iPhone 11 profile, process RSS above an empty tab; `research/measure_engine.mjs`, `measure_probe.mjs`)
- 20 strokes in a row with a text encode per stroke: peak **250 MB**, flat across strokes (was 426 MB before: the texture pool grew with every new crop size → LRU trim to 24 MB after each stroke; and WebKit never returns a terminated Worker's memory → one persistent text worker instead of one per note).
- Breakdown of a load: GL context 6, shader programs 4, tiny decoder 28, scorer 42, palette + bank 59 (Cache Storage machinery; the app already pays this while viewing), text worker 84 → 73 with fp16-only weights (JSON tokenizer tables + a second JS VM dominate). Chromium (Pixel 5 profile) renderer: +180 MB over 20 strokes.
- Speed on the M1 Pro: ~1 750 tries/s (WebKit) / ~2 000 (Chromium) with batch 32, 256 px preview decode 7–15 ms, text encode 40–95 ms. Real-phone numbers are unknown until the device test.

## Phase 4 — wiring (2026-10-01)
- `app/engine_bridge.js` presents the engine with the decoder / clip / painter shapes room.js uses; room.js: `useTiny` = `?engine=tiny`, or by default on phones (lowMem) and without WebGPU (`?engine=ort` forces the ONNX path). With the light engine phones paint themselves (no helper request), keep the engine loaded between strokes, lasso cap 24 tokens, margin 2. Photos: tokens still seed the shape; the photo's CLIP embedding is not mixed in (no image tower in the light engine).
- `research/test_app_tiny.mjs`: desktop Chromium (`?engine=tiny`) and the iPhone 11 WebKit profile paint two strokes locally: brush ready in 1.3–2.5 s (17.5 MB of files), ~6 000–6 800 tries per 4 s stroke, notes carry tokens + lasso path, previews upload, layers render. PASS on both.
- `tools/measure_memory.mjs --browser webkit --device "iPhone 11"` (the product's tool, phone paints itself with the light engine): viewing 108 MB, **painting peak 280 MB** above the empty browser (WebContent process 298 MB total; was 1 777 MB with ORT), stroke 7.6 s including model load. ORT path: `app/test_app.mjs` ALL PASS after the patch.

## Phase 4 test matrix (`research/test_matrix.sh`: engine test page, 20 strokes × 3 s, text encode per stroke, M1 Pro emulation)
| profile | tries/s | 256 px decode | text encode | peak RSS above empty tab | 20 strokes |
|---|---|---|---|---|---|
| Chrome desktop | 2 230 | 6–15 ms | 190 ms | 296 MB | ok |
| Chrome, JS heap capped at 384 MB (engine never uses WebGPU) | 2 190 | 6–9 ms | 195 ms | 302 MB | ok |
| Safari desktop (WebKit) | 1 705 | 7–22 ms | 146 ms | 293 MB | ok |
| iPhone 11 profile (WebKit) | 1 800 | 9–17 ms | 299 ms | 277 MB | ok |
| Pixel 5 profile (Chromium) | 2 070 | 6–12 ms | 193 ms | 303 MB | ok |

Also: decoder variant B (3.0 MB) matches PyTorch like A; `variant: 'auto'` loads A, probes a 256 px decode and switches to B above 60 ms; the app's no-WebGPU path (`?nogpu=1`) picks the light engine by itself and paints (7 400 tries in 4.5 s).

## 2026-10-02 morning — first Kaggle runs were NaN (6 GPU hours lost)
- Both kernels trained for their full budget with NaN losses from step 1: the VQGAN teacher overflows under fp16 autocast (its residual stream reaches ~1e5; the ONNX export already works around this with a 1/16 tail rescale). The MPS smoke test never ran fp16, so it did not catch it. Kaggle kernel logs are only readable after the run, so the NaN went unnoticed for 3.5 h.
- Fix: `load_teacher` applies `rescale_tail(dec, 16)` (output identical in fp32; reproduced locally in half precision: raw → 50 % NaN, rescaled → finite, max diff 0.013 vs fp32), `teach_safe` falls back to fp32 for a non-finite batch, non-finite losses are skipped, non-finite CLIP embeddings dropped. Scorer notebook pins `onnxruntime-gpu==1.20.1` (the latest needs CUDA 13; Kaggle has 12), CPU fallback kept.
- Relaunched decoder (3.5 h) and scorer (2.5 h); text and one-pass follow when slots free up. Quota left before relaunch: 15.6 h.
- Scorer v3 crashed at step 948 (a non-finite loss hit the skip guard, then the AMP scaler asserted). At 0.5 h it had embedding cosine 0.65 but prompt-score Pearson 0.10 and pair-sign agreement 0.54 (chance level) — too early to judge, full budget needed. v4 relaunched 11:00 with the student in fp32 (no normalisation layers → fp16 overflow), a bounded pair loss, and the scaler fix (also applied to the decoder trainer for the next run).

## Direction received 2026-10-02 (for after Phase 4 + app PR)
The engine becomes a standalone public repo (`tiny-vqgan` if free) with history, notebooks, test page/matrix, DEVICES.md, an outsider README with numbers/usage/side-by-side/retraining/limitations, a licence review of all upstream sources (weights stay unpublished until Nazar confirms), the app vendoring a pinned copy, and a "numbers for the white paper" block in the final report. Phase 5 may follow later.

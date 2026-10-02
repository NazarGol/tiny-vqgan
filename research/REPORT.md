# Tiny VQGAN engine — report (2026-10-02)

**Goal.** Paint on the minimum phones (`web/DEVICES.md`: iPhone XR/11/SE2 on iOS 15+, 3 GB Androids from 2020) with no server and no ML runtime in the painting loop.

## What was built
- **Tiny decoder** (`lib/tinydec.js`, 3.1 MB): VQGAN f16 tokens → RGB, distilled on Kaggle from the original decoder (L1 + LPIPS, 3.5 h). Gate approved by Nazar on the side-by-side sheet (`research/results/decoder_eval_sheet.png`, pairs `decoder_pair_*.png`): "looks like the original, just softer". Metrics vs the original on held-out grids: L1 0.044, LPIPS 0.28, PSNR 25.1, CLIP cosine 0.79. Variant B (3.0 MB) for slow GPUs, picked by a probe.
- **Real MobileCLIP-S0 image tower as WebGL2 passes** (`lib/clipvision.js`, 21.7 MB fp16): the ONNX graph converted to 118 shader ops; cosine 0.9995–0.9999 vs ONNX Runtime; 13 ms per 256 px image on the M1. Real CLIP scores the tiny decoder's output with no readback.
- **Token scorer** (`lib/tinyscorer.js`, 5.2 MB): approximate CLIP from tokens. As the only scorer it is gamed by the search (rejected); as a pre-filter (ranks 32 mutations, real CLIP judges 4) it is the best mode at equal time.
- **Text encoder** (`engine/text.js`, 9.3 MB, plain JS in a Worker): distilled MobileCLIP text tower, cosine 0.955 to the real one on held-out captions (0.94 on our prompts, 0.98 on notes).
- **Engine** (`engine/engine.js`: load / encodeTexts / paintStroke / decode / release) and the app facade `lib/engine/light.js` (same calls as the ORT engine worker). Default on phones and without WebGPU; desktops keep ONNX. Standalone repo: https://github.com/NazarGol/tiny-vqgan (vendored, pinned in `web/engine/VENDOR.md`).
- **PR #3**: https://github.com/NazarGol/VQPAINT/pull/3 (not merged). Model files on gh-pages `models/tiny/`.

## Side-by-side
- Decoder, original vs tiny A vs tiny B (36 grids incl. 12 real strokes): `research/results/decoder_eval_sheet.png`; full-resolution pairs `research/results/decoder_pair_*.png`.
- Search modes at 8 s per stroke, 10 prompts, scored by real CLIP through the original decoder: `research/results/search_modes_8s_m1.png` (columns: ORT path, tiny + real CLIP, pre-filter + real CLIP, token-only, ORT result through the tiny decoder).

## Numbers for the white paper (Apple M1 Pro emulation unless stated; real phones pending)
| measure | ONNX Runtime path (before) | light engine (after) |
|---|---|---|
| download needed to paint | 108 MB | 42 MB (decoder 3.1 + CLIP 21.7 + text 9.3 + scorer 5.2 + palette/bank 2.4) |
| ML runtime | onnxruntime-web (~10 MB wasm + JSEP) | none (WebGL2 shaders + JS) |
| painting peak memory, iPhone 11 WebKit profile, above an empty tab | 1 777 MB (web process) | ~375 MB WebContent + ~95 MB GPU process (download path dominates; ~250 MB for the engine without the CLIP tower) |
| real-CLIP evaluations per second, desktop Chromium / WebKit | 4–6 | 40–50 (tiny decode 5–8 ms + CLIP 13 ms) |
| candidates screened per second with the pre-filter | — | ~2 000 (token scorer), 32 per real-CLIP evaluation |
| seconds per stroke | 10 (default effort) | 8–10 (same budget; quality at equal time below) |
| 10 prompts × 8 s, mean real-CLIP score (original decoder) | 0.174 (≈50 tries) | 0.179 pre-filter (≈320 CLIP tries, 4/10 prompts better), 0.159 CLIP-only, 0.128 token-only |
| quality gate (decoder) | — | approved; L1 0.044 / LPIPS 0.28 / PSNR 25.1 / CLIP cosine 0.79 vs original |
| CLIP image tower vs ONNX Runtime | — | cosine 0.9995–0.9999, 13 ms/image |
| token scorer vs real CLIP | — | embedding cosine 0.72; per-prompt score Pearson 0.42 / Spearman 0.41; sign agreement on small mutations 0.55 (chance 0.5), on different grids 0.63 |
| text encoder vs MobileCLIP text tower | — | cosine 0.955 (held-out captions), top-10 retrieval overlap 0.74 |
| 256 px decode | 220–420 ms (ORT WebGPU) | 7–15 ms |
| five-profile matrix (Chrome, Chrome capped heap, Safari, iPhone 11, Pixel 5), 20 strokes each | — | all pass, 1 700–2 230 token tries/s, ~280–300 MB peak (engine test page, before the CLIP tower) |

## Licences (details in `research/LICENSES.md`)
Code MIT. Tiny decoder weights: derived from the CompVis VQGAN ImageNet checkpoint (MIT repo, no separate weight licence; ImageNet grey area shared by everything built on it) — publishable. CLIP image tower, scorer and text encoder: Apple MobileCLIP-S0 → **Apple ML Research Model terms, research use only**; commercial use needs a re-distillation from a permissive CLIP (OpenAI ViT-B/32 is MIT) plus re-exported palette/bank embeddings. CelebA grids (training inputs, 500 faces in the bank) are non-commercial. **No weights were published** (gh-pages holds the app's deployment only).

## What is still open
- Real-phone numbers (USB scripts: `research/phone_usb.md`; checklist: `web/DEVICES.md`). The engine on the old iPhone decides whether the memory target holds.
- Decoder round 2 with a CLIP-faithfulness loss (training now on Kaggle; closes the 0.05–0.08 gap between CLIP on the tiny render and on the original render).
- Scorer round 2 (pair-focused loss, hard negatives later), one-pass starting model (phase 5, not started: the token scorer must be trustworthy first).
- Photos on the light engine still use the ORT VQGAN encoder (delegated to the ORT worker; heavy on the minimum phones).

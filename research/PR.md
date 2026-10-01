# PR draft: light painting engine for phones (tiny decoder + token scorer + tiny text encoder, no ONNX Runtime)

**What**: painting on the minimum phones (`web/DEVICES.md`) without a server and without an ML runtime in the painting loop.
Three small models distilled on Kaggle (free GPU) run as WebGL2 shader passes / plain JS:
- `lib/tinydec.js` — tiny VQGAN decoder (3.1 MB), 256 px in ~10 ms on the M1;
- `lib/tinyscorer.js` — token → CLIP-image-embedding scorer (5.2 MB): the search scores ~2 000 candidates/s without decoding;
- `engine/text.js` (+ worker) — distilled MobileCLIP text tower (6.6 MB), once per note.
`engine/engine.js` is the self-contained interface (load / encodeText / paintStroke / decode / release); `app/engine_bridge.js` adapts it to the
decoder / clip / painter shapes `room.js` uses. **Default on phones and devices without WebGPU; desktops keep the ONNX path** (`?engine=tiny` / `?engine=ort` to switch).

**Numbers (emulation, M1 Pro)**: _filled from research/REPORT.md_

**Quality**: _side-by-side sheets in web/research/results/_

**Not in this PR**: the one-pass starting model (phase 5, separate PR); a tiny photo encoder (photos on the minimum phones still need a helper for the token encoding step).

**How to test**: `node web/research/test_app_tiny.mjs` (desktop), `--browser webkit --device "iPhone 11"`; `node app/test_app.mjs` (ONNX path unchanged); real phones: `web/DEVICES.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

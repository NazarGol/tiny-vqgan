# Minimum devices (replaces "iPhone 13 mini is the minimum" everywhere)

The app must paint, not just view, on these phones, with no server:

- **iPhone XR / iPhone 11 / iPhone SE (2nd gen)**, iOS 15+, Safari. 3 GB RAM. No WebGPU on iOS 15–17 (WebGPU arrives with iOS 26 Safari).
- **A budget Android phone from ~2020 with 3 GB RAM**, Chrome (for example a Redmi 9A / Galaxy A21s / Moto G8 class: Mali-G52 / Adreno 610 GPU). Chrome on these has WebGL2; WebGPU is not guaranteed.

Rules that follow from this:

- **WebGL2 is the main path.** Every painting-time model runs as WebGL2 shader passes (no ONNX Runtime, no wasm ML runtime in the loop). WebGPU is only a speed-up when it is available and must never be required.
- **Painting peak memory under ~250 MB** on these phones (web process, above an empty tab). iOS kills a tab well under 1 GB on a 3 GB phone; the old path peaked at 1.4–1.9 GB and reloaded.
- **Everything needed to paint: under ~30 MB download in total** (tiny decoder + token scorer + text encoder + bank/palette data + code). The 89 MB decoder and the 85 MB CLIP text tower stay only for desktop print export.
- **The 256 px decode must be fast enough for a live preview** on these GPUs (target: well under a second per preview frame, hundreds of candidate scores per second without decoding).

Emulation on the laptop proves the flow, sizes and the absence of crashes under a memory limit; it cannot prove speed or memory on these GPUs. See `research/PROGRESS.md` ("what to test on a real old phone") for the exact checks to run on a borrowed device.

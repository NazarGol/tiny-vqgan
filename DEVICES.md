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

## What to test on a real old phone (10 minutes, laptop closed)

Open the live link on the phone, create a room, and note the answers. The page shows numbers in the ⋯ menu → "stats" (or add `&stats=1`).

1. **Opens and shows the canvas** within ~30 s on Wi-Fi? (The brush downloads ~17 MB the first time.)
2. **Brush → draw a lasso → type "a red forest" → Enter.** Does a painting appear inside the shape within ~15 s, with a live preview (fog → clearer) while it paints? Note the "tries" count shown in the stats line (expected: hundreds per second on a 2018–2020 phone; the laptop does ~1 700/s).
3. **Repeat 10 strokes in a row**, some overlapping. Does Safari/Chrome ever reload the page ("a problem repeatedly occurred" on iOS)? That is the memory test; the emulated peak is 280 MB above an empty tab, the budget is ~250 MB, and only the real device tells.
4. **Lock the phone mid-stroke, unlock**: the stroke should continue and finish.
5. **Long note** (3–4 sentences): it should paint without an error toast (text is encoded in a worker, then chunked).
6. **Join from the phone** a room painted on the laptop: it should open looking at the painting (previews, no model).
7. **Compare with the old engine** by adding `&engine=ort` to the room URL: on the iPhone XR/11/SE2 that path should fail to paint (or reload) — this confirms the new one is what makes the phone work.
8. Tell me: phone model + iOS/Android version, the tries-per-second line, whether any reload happened, and how the paintings looked next to the laptop's (the laptop still uses the full decoder for its own strokes unless you add `&engine=tiny`).

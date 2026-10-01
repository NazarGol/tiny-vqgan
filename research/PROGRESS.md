# Research progress (tiny VQGAN engine)

Goal: paint on the minimum phones (`web/DEVICES.md`) with no ML runtime in the loop: tiny decoder + token-space CLIP scorer as WebGL2 shader passes, text encoder once per note.

## Phase 1 — tiny decoder (in progress, 2026-10-01)
- Worktree + notes created. Kaggle: CLI works, 21.7 GPU hours left this week (refresh 2026-10-03), checkpoint already on Kaggle in `noi3noi3/vqpaint-assets`.
- Data: encoding COCO val2017 + CelebA at 512 px → 32×32 token grids locally (MPS) for the training set; 16×16 grids come from the two banks and the paintings run.

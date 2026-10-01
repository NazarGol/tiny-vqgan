# Research decisions (tiny VQGAN engine), one line each, newest last

- 2026-10-01 Work happens in the worktree `~/VQPAINT-research` on `research/tiny-vqgan`, branched from `web-spikes` (main is 36 commits behind it; PR #2 is pending). Product files are never edited here; model/data dirs are symlinked read-only from the main checkout.
- 2026-10-01 Minimum devices are in `web/DEVICES.md` (iPhone XR/11/SE2 on iOS 15+, 3 GB Android from 2020). Historic "13 mini" numbers in the product notes stay as records; new work follows DEVICES.md.
- 2026-10-01 Training data is generated on the fly from token grids only (no pixels on Kaggle): bank grids (6500 photos + 1500 paintings at 16×16), 32×32 grids from COCO/CelebA at 512 px encoded locally, uniform-random grids, and search-style mutations (palette samples, neighbour copies, block moves, swaps, mosaics). Targets are the original decoder's output computed in the kernel.

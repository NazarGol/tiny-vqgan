# Notice: upstream licences

The code in this repository is MIT (see LICENSE). The model weights it runs are **not** included and are not MIT:


Short answer: **the code can be published under MIT. The tiny decoder weights can be published. The token scorer and the tiny text
encoder are distilled from MobileCLIP-S0, whose weights are under Apple's "ML Research Model" terms: research only, no commercial
use, derivatives must carry Apple's notice — so those two can only be published as research-only, and the engine as a whole is
not commercially usable until they are re-distilled from a permissively licensed CLIP.** Nothing is published until you confirm.

| component | used for | licence | commercial use | source |
|---|---|---|---|---|
| taming-transformers code | teacher decoder, encoder | MIT | yes | github.com/CompVis/taming-transformers (License.txt) |
| `vqgan_imagenet_f16_16384` weights | teacher for the tiny decoder; token grids (encoder) | no separate licence stated (released from the MIT repo); trained on ImageNet | grey: widely treated as MIT; ImageNet's own terms bind the dataset user (CompVis), not weight users — a lawyer's call if commercial use matters | repo README, image-net.org terms ("non-commercial research and educational purposes") |
| MobileCLIP-S0 code | — | MIT | yes | github.com/apple/ml-mobileclip LICENSE |
| **MobileCLIP-S0 weights** (teacher of the token scorer and the text encoder; the app's current CLIP) | targets for the scorer and text encoder; the app's painting target | **Apple Machine Learning Research Model TOU** (LICENSE_MODELS) | **no**: "exclusively for Research Purposes … does not include any commercial exploitation, product development or use in any commercial product or service"; Model Derivatives (incl. retraining/fine-tuning, and by a conservative reading distillation from its outputs) are limited to Research Purposes, must be identified as derivatives and carry the notice "Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement." | github.com/apple/ml-mobileclip LICENSE_MODELS |
| MobileCLIP data (DataCompDR) | not used by us directly | CC-BY-NC-ND (Apple's metadata); DataComp samples CC-BY-4.0 | n/a | ml-mobileclip LICENSE_DATA, HF dataset card |
| Xenova/mobileclip_s0 (ONNX conversion) | the app's CLIP files | "other" (inherits Apple's terms) | no | huggingface.co/Xenova/mobileclip_s0 |
| LPIPS (richzhang/PerceptualSimilarity) | training loss only | BSD-2-Clause | yes | repo LICENSE |
| VGG16 weights (torchvision) inside LPIPS | training loss only | BSD-3 (torchvision); trained on ImageNet | training-time only, nothing shipped | pytorch/vision |
| COCO val2017 images | token grids for training (never shipped as pixels) | annotations CC-BY-4.0; images: Flickr terms, per-image CC licences (several are NC) | grey for derived data; filter to CC-BY / CC-BY-SA images (licence ids in the annotations) for a commercial retrain | cocodataset.org terms, cocoapi issue #551 |
| CelebA-HQ | token grids for training; 500 faces in the app's bank | **non-commercial research only**; "not … exploit for any commercial purposes … any portion of derived data"; no redistribution | **no** | mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| our VQGAN+CLIP paintings (web/export/paintings) | token grids for training; 1500 in the app's bank | ours (generated with VQGAN + OpenAI CLIP ViT-B/32, MIT) | yes (subject to the VQGAN grey area) | — |
| onnxruntime-web (app's old path only) | not in the engine | MIT | yes | — |

## What this means

1. **Research-only release (what we have now)**: publish code (MIT) + weights with a clear notice: decoder "MIT, distilled from
   CompVis VQGAN (ImageNet)"; scorer + text encoder "research use only — Model Derivatives of Apple MobileCLIP-S0 under the Apple
   ML Research Model License Agreement" (include the agreement text and the attribution line; identify them as derivatives and
   describe the distillation). Training data note: COCO/CelebA token grids were used as inputs; CelebA forbids commercial
   exploitation of derived data.
2. **Commercial use** would need, in this order of cost:
   - re-distil the scorer and text encoder from a permissive CLIP (OpenAI CLIP ViT-B/32 is MIT; open_clip LAION weights are MIT) —
     same student code, ~4 GPU hours; the app's palette, bank and metaphor embeddings must then be recomputed in that CLIP space
     (export scripts exist), and the app's full-size CLIP path would switch to the same model;
   - retrain without CelebA grids and with COCO filtered to CC-BY/CC-BY-SA images (or our own paintings + random grids only) —
     ~6 GPU hours; drop the 500 CelebA faces from the shipped bank;
   - accept (or get advice on) the VQGAN-weights/ImageNet grey area, which every project built on these checkpoints shares.
3. The app today already ships MobileCLIP-S0 (Xenova ONNX) and CelebA-derived bank grids, so the research-only status is not new
   to the engine; it is inherited from the app's current models.

"""Generates the Kaggle notebooks (decoder, scorer, text) with one shared, robust setup cell. Run: python build_notebooks.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = '''# setup: repo (research branch), taming-transformers, checkpoint (vqpaint-assets dataset or heibox), token grids (vqpaint-tokens dataset)
import glob, os, shutil, subprocess, sys, time
TMP, WORK = "/kaggle/tmp", "/kaggle/working"; os.makedirs(TMP, exist_ok=True)
REPO, DATA = f"{TMP}/VQPAINT", f"{TMP}/data"
def sh(cmd):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
def find_input(name):
    for root in ("/kaggle/input", "/kaggle/input/datasets"):
        for pat in (f"{root}/{name}*", f"{root}/*/{name}*", f"{root}/**/{name}*"):
            hits = sorted(glob.glob(pat, recursive=True))
            if hits: return hits[0]
    return None
sh("find /kaggle/input -maxdepth 4 | head -50 || true")
if not os.path.isdir(REPO): sh(f"git clone -q --branch {S['BRANCH']} --depth 1 https://github.com/NazarGol/tiny-vqgan.git {REPO}")
TAM = f"{REPO}/export/taming-transformers"
if not os.path.isdir(TAM): sh(f"git clone -q --depth 1 https://github.com/CompVis/taming-transformers.git {TAM}")
CK = f"{REPO}/export/checkpoints"; os.makedirs(CK, exist_ok=True)
ASSETS = find_input("vqpaint-assets"); print("assets dataset:", ASSETS)
for f in ("vqgan_imagenet_f16_16384.yaml", "vqgan_imagenet_f16_16384.ckpt"):
    dst = f"{CK}/{f}"
    if os.path.exists(dst): continue
    src = f"{ASSETS}/checkpoints/{f}" if ASSETS else None
    if src and os.path.exists(src): os.symlink(src, dst)
    else:
        url = "https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1" if f.endswith("yaml") else "https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1"
        sh(f"curl -sSL '{url}' -o {dst}")
TOK = find_input("vqpaint-tokens"); print("tokens dataset:", TOK); assert TOK, "vqpaint-tokens dataset not mounted (see the listing above)"
if not os.path.isdir(DATA): shutil.copytree(TOK, DATA)
sh(f"ls -la {DATA} {CK}/ && nvidia-smi --query-gpu=name,memory.total --format=csv")
'''
def nb(title, intro, config, extra_setup, train_cmd, out_name, resume_glob):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": f"# {title}\n\n{intro}"}]
    def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
    code(config + f'\nOUT = "/kaggle/working/{out_name}"\nprint(S)')
    code(SETUP + extra_setup)
    code(f'''os.makedirs(OUT, exist_ok=True)
prev = sorted(glob.glob("{resume_glob}"))
if prev and not os.path.exists(f"{{OUT}}/ckpt.pt"): shutil.copy(prev[-1], f"{{OUT}}/ckpt.pt"); print("resuming from", prev[-1])''')
    code(f'''cmd = {train_cmd}
print("$", cmd, flush=True)
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
with open(f"{{OUT}}/train.log", "a") as log:
    for line in p.stdout:
        if "Warning" in line or "warn(" in line: continue
        print(line, end="", flush=True); log.write(line)
print("exit", p.wait())''')
    code('''import json
print(json.dumps(json.load(open(f"{OUT}/eval.json")), indent=1)); sh(f"ls -la {OUT}")
if os.path.exists(f"{OUT}/eval_sheet.png"):
    from IPython.display import Image, display; display(Image(f"{OUT}/eval_sheet.png", width=700))''')
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
def write(dirname, slug, title, notebook):
    d = os.path.join(HERE, dirname); os.makedirs(d, exist_ok=True)
    json.dump(notebook, open(os.path.join(d, f"{slug}.ipynb"), "w"), indent=1)
    json.dump({"id": f"noi3noi3/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
               "dataset_sources": ["noi3noi3/vqpaint-assets", "noi3noi3/vqpaint-tokens"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)

CLIP_DL = '''sh(f"{sys.executable} -m pip install -q lpips tokenizers scipy")
sh(f"{sys.executable} -m pip install -q onnxruntime-gpu==1.19.2 || {sys.executable} -m pip install -q onnxruntime")
for f in ("onnx/vision_model.onnx", "onnx/text_model.onnx", "tokenizer.json"):
    dst = f"{DATA}/{os.path.basename(f)}"
    if not os.path.exists(dst): sh(f"curl -sSL https://huggingface.co/Xenova/mobileclip_s0/resolve/main/{f} -o {dst}")
import onnxruntime as ort; print("ORT providers:", ort.get_available_providers())
'''
DEC_DL = CLIP_DL + '''sh(f"{sys.executable} {REPO}/research/tiny/export_clip_vision.py --onnx {DATA}/vision_model.onnx --out {DATA}/clipx")   # differentiable CLIP for the faithfulness loss
'''
write("tiny_decoder", "vqpaint-tiny-decoder", "vqpaint tiny decoder", nb("VQPAINT tiny decoder (distilled VQGAN f16 decoder)",
    "Small conv decoders from `vqgan_imagenet_f16_16384` tokens to RGB, distilled from the original decoder on on-the-fly token grids. Output: `/kaggle/working/tiny/`.",
    '''S = dict(HOURS=2.0, BATCH=16, VARIANTS="A:64,64,64,32,16:2,2,2,1,1;B:64,64,48,24,12:2,2,2,1,1", LPIPS=1.0, LR=5e-4, GAN_W=0.0, GAN_START=0.4, CLIP_W=0.5, BRANCH="main")
# round 2 = resume from the previous version's output (this kernel is its own kernel source) + CLIP-faithfulness loss; HOURS are added''',
    DEC_DL,
    '''f"{sys.executable} -u {REPO}/research/tiny/train_decoder.py --data {DATA} --out {OUT} --hours {S['HOURS']} --batch {S['BATCH']} --variants '{S['VARIANTS']}' --lpips {S['LPIPS']} --lr {S['LR']} --gan-w {S['GAN_W']} --gan-start {S['GAN_START']} --clip-w {S['CLIP_W']} --clip-ir {DATA}/clipx --clip {DATA}/vision_model.onnx"''',
    "tiny", "/kaggle/input/**/vqpaint-tiny-decoder*/tiny/ckpt.pt"))
write("tiny_scorer", "vqpaint-tiny-scorer", "vqpaint tiny scorer", nb("VQPAINT token scorer (tokens → MobileCLIP image embedding)",
    "Distils teacher decoder + MobileCLIP-S0 vision into a small conv net on the token grid. Output: `/kaggle/working/scorer/`.",
    '''S = dict(HOURS=3.0, BATCH=32, VARIANTS="S:64:64,96,128,192;M:64:96,128,192,256", PAIRS=0.6, PAIR_W=10.0, LR=5e-4, WORKERS=3, BRANCH="main")
# round 2: resumes from the previous version's ckpt (this kernel's own output is attached as a kernel source); HOURS are added''',
    CLIP_DL,
    '''f"{sys.executable} -u {REPO}/research/tiny/train_scorer.py --data {DATA} --out {OUT} --hours {S['HOURS']} --batch {S['BATCH']} --variants '{S['VARIANTS']}' --pairs {S['PAIRS']} --pair-w {S['PAIR_W']} --lr {S['LR']} --workers {S['WORKERS']} --clip-vision {DATA}/vision_model.onnx --clip-text {DATA}/text_model.onnx --tokenizer {DATA}/tokenizer.json"''',
    "scorer", "/kaggle/input/**/vqpaint-tiny-scorer*/scorer/ckpt.pt"))
TEXT_DL = CLIP_DL + '''ANN = f"{DATA}/annotations"
if not os.path.exists(f"{ANN}/captions_train2017.json"):
    sh(f"curl -sSL http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o {TMP}/ann.zip && cd {DATA} && unzip -qo {TMP}/ann.zip annotations/captions_train2017.json annotations/captions_val2017.json && rm {TMP}/ann.zip")
'''
write("tiny_text", "vqpaint-tiny-text", "vqpaint tiny text", nb("VQPAINT tiny text encoder (distilled MobileCLIP-S0 text tower)",
    "Small causal transformer that maps CLIP token ids to the MobileCLIP-S0 text embedding space, distilled on COCO captions, painting prompts, metaphors and synthetic notes. Output: `/kaggle/working/text/`.",
    '''S = dict(HOURS=1.5, BATCH=256, VARIANTS="S:32,192,4,4;M:32,256,4,4", BRANCH="main")''',
    TEXT_DL,
    '''f"{sys.executable} -u {REPO}/research/tiny/train_text.py --data {DATA} --out {OUT} --hours {S['HOURS']} --batch {S['BATCH']} --variants '{S['VARIANTS']}' --clip-text {DATA}/text_model.onnx --tokenizer {DATA}/tokenizer.json --captions {DATA}/annotations"''',
    "text", "/kaggle/input/**/vqpaint-tiny-text*/text/ckpt.pt"))
ONEPASS_DL = CLIP_DL + '''SC = find_input("vqpaint-tiny-scorer"); print("scorer kernel output:", SC); assert SC, "attach the vqpaint-tiny-scorer kernel output"
SCK = sorted(glob.glob(f"{SC}/**/ckpt.pt", recursive=True)); assert SCK, "scorer ckpt.pt not found"; SCK = SCK[0]; print("scorer ckpt", SCK)
ANN = f"{DATA}/annotations"
if not os.path.exists(f"{ANN}/captions_train2017.json"):
    sh(f"curl -sSL http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o {TMP}/ann.zip && cd {DATA} && unzip -qo {TMP}/ann.zip annotations/captions_train2017.json annotations/captions_val2017.json && rm {TMP}/ann.zip")
'''
write("onepass", "vqpaint-onepass", "vqpaint onepass", nb("VQPAINT one-pass starting model (phase 5)",
    "Runs the token-space search over many prompts/contexts with the trained scorer, trains a one-pass model that predicts a starting grid for a masked region from the text embedding + context, and compares search-from-one-pass vs search-from-mosaic at equal budget. Output: `/kaggle/working/onepass/`.",
    '''S = dict(GEN_HOURS=1.2, N=8000, TRAIN_HOURS=1.0, SCORER="S", BRANCH="main")''',
    ONEPASS_DL,
    '''f"{sys.executable} -u {REPO}/research/tiny/gen_onepass_data.py --scorer-ckpt {SCK} --variant {S['SCORER']} --data {DATA} --clip-text {DATA}/text_model.onnx --tokenizer {DATA}/tokenizer.json --captions {DATA}/annotations --out {OUT}/onepass_data.npz --n {S['N']} --hours {S['GEN_HOURS']} && {sys.executable} -u {REPO}/research/tiny/train_onepass.py --npz {OUT}/onepass_data.npz --scorer-ckpt {SCK} --variant {S['SCORER']} --data {DATA} --out {OUT} --hours {S['TRAIN_HOURS']}"''',
    "onepass", "/kaggle/input/**/vqpaint-onepass*/onepass/nothing"))
import json as _j
m3 = _j.load(open(os.path.join(HERE, "tiny_decoder", "kernel-metadata.json"))); m3["kernel_sources"] = ["noi3noi3/vqpaint-tiny-decoder"]; _j.dump(m3, open(os.path.join(HERE, "tiny_decoder", "kernel-metadata.json"), "w"), indent=1)
m2 = _j.load(open(os.path.join(HERE, "tiny_scorer", "kernel-metadata.json"))); m2["kernel_sources"] = ["noi3noi3/vqpaint-tiny-scorer"]; _j.dump(m2, open(os.path.join(HERE, "tiny_scorer", "kernel-metadata.json"), "w"), indent=1)   # round 2 resumes from its own previous output
# the onepass kernel also needs the scorer kernel's output
import json as _j
m = _j.load(open(os.path.join(HERE, "onepass", "kernel-metadata.json"))); m["kernel_sources"] = ["noi3noi3/vqpaint-tiny-scorer"]; _j.dump(m, open(os.path.join(HERE, "onepass", "kernel-metadata.json"), "w"), indent=1)
print("notebooks written")

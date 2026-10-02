"""Reference for the browser CLIP-vision test: 4 decoded images as PNG + their onnxruntime embeddings. Usage: python dump_ref_clip.py OUT_DIR"""
import base64, json, os, sys
import numpy as np, torch, onnxruntime as ort
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common as C
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
sess = ort.InferenceSession(os.path.join(C.WEB, "models", "mobileclip_s0", "onnx", "vision_model.onnx"), providers=["CPUExecutionProvider"])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); teacher, _ = C.load_teacher(dev); g16, g32 = C.load_grids(os.path.join(C.HERE, "..", "data")); s = C.GridSampler(g16, g32, seed=3)
cases = []
for i, S in enumerate((16, 16, 12, 20)):
    tok = torch.from_numpy(s.sample(S))[None]
    with torch.no_grad(): img = teacher(tok.to(dev)).float().clamp(0, 1).cpu()
    u8 = (img[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8); Image.fromarray(u8).save(os.path.join(out, f"clip_img_{i}.png"))
    # the embedding of the 8-bit image resized to 256 (what the browser will feed), bilinear without antialias like the page
    x = torch.from_numpy(u8.astype(np.float32) / 255).permute(2, 0, 1)[None]; x = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    e = sess.run(None, {"pixel_values": x.numpy()})[0][0]; e /= np.linalg.norm(e)
    cases.append({"png": f"clip_img_{i}.png", "w": int(u8.shape[1]), "h": int(u8.shape[0]), "emb": base64.b64encode(e.astype(np.float32).tobytes()).decode()})
json.dump({"cases": cases}, open(os.path.join(out, "ref_clip.json"), "w")); print("wrote", len(cases), "cases")

"""Metaphor bank: short visual prompts + their MobileCLIP text embeddings, so practical notes can borrow imagery.
Writes ../models/metaphors/metaphors.json (list) and metaphors.f16 (N x 512 fp16, unit vectors)."""
import json, os, numpy as np, onnxruntime as ort
from transformers import CLIPTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.normpath(os.path.join(HERE, "..", "models"))
OUT = os.path.join(M, "metaphors"); os.makedirs(OUT, exist_ok=True)
groups = {
 "weather": ["storm", "storm clouds gathering", "lightning over a field", "heavy rain on a window", "fog at dawn", "clear sky after rain", "snowfall at night", "hail on a tin roof", "a rainbow over wet fields", "hot summer haze", "wind bending tall grass", "frost on glass", "thunder over the sea", "drizzle in a grey city", "a sandstorm", "sun breaking through clouds", "first snow", "a heatwave on asphalt", "mist over a lake", "an ice storm on branches"],
 "light": ["sunrise", "sunset over water", "golden hour light", "candlelight in a dark room", "neon reflections in rain", "moonlight on snow", "a single lamp in a window", "harsh noon light", "light through dusty air", "a lighthouse beam at night", "embers glowing", "stage light on an empty stage", "torchlight in a cave", "light at the end of a tunnel", "a match flaring", "blue hour in a city", "sparks from a fire", "sunlight through leaves", "a flashlight in the dark", "city lights from above"],
 "materials": ["cracked wall", "rusted iron", "wet clay", "polished marble", "rough concrete", "woven rope", "broken glass", "melting wax", "old wood grain", "torn paper", "a tangled knot", "frayed fabric", "smooth river stones", "shattered ceramic", "molten metal", "layers of peeling paint", "a spider web with dew", "chalk on a blackboard", "wax seal on a letter", "crumpled foil"],
 "landscape": ["river delta from above", "a mountain pass", "a crossroads in the desert", "a narrow bridge over a gorge", "a maze of hedges", "a cliff edge", "an open field at dawn", "a dense forest path", "a frozen lake", "a volcano at night", "a quiet harbor", "an island in a storm", "terraced rice fields", "a dry riverbed", "a canyon", "rolling hills", "a swamp at dusk", "a glacier", "a ruined city", "a garden in bloom"],
 "objects": ["clock gears", "an hourglass", "a locked door", "an open door with light behind it", "a compass", "a broken mirror", "a ladder", "a knot of cables", "an empty chair", "a stack of books", "a burning candle", "a key", "a bridge", "a map with pins", "scales balancing", "a chess board mid-game", "an anchor", "a kite in wind", "a fence", "a telescope"],
 "motion": ["waves crashing", "a river in flood", "a bird taking off", "falling leaves", "a train leaving", "smoke rising", "a spinning top", "an avalanche", "a slow drip of water", "dancers in motion", "a crowd rushing", "a rocket launch", "a pendulum", "ripples on still water", "a tug of war", "a race at the finish line", "wheels turning", "a door slamming", "a flag in wind", "a seed sprouting"],
 "people": ["two people talking at a table", "a hug", "a crowd of raised hands", "a lonely figure on a bench", "children playing", "hands holding a cup", "a handshake", "someone reading by a window", "a face in shadow", "friends laughing", "a person walking away", "a couple under one umbrella", "an old woman knitting", "a child looking up", "a musician playing", "people around a campfire", "a silhouette at a door", "a tired worker", "a teacher at a board", "a protest march"],
 "food": ["bread and coffee on a table", "a bowl of fruit", "snacks on a tray", "a cup of tea steaming", "a shared dinner", "a picnic blanket", "ripe apples", "spilled milk", "a birthday cake", "wine glasses clinking"],
 "abstract": ["a spiral", "a labyrinth", "a knot untying", "two roads diverging", "a crack spreading", "a wall with a gap", "a scale tipping", "a clock running out", "a fork in a river", "a mosaic of broken tiles", "a seed under snow", "an unfinished puzzle", "a stone in a stream", "a shadow growing", "a door half open", "threads weaving together", "a wave about to break", "an echo in a canyon", "a bright window in a dark house", "a tightrope"],
 "seasons": ["autumn leaves", "spring blossoms", "midsummer meadow", "winter branches", "harvest fields", "a cold morning", "cherry trees", "a warm evening", "bare trees in fog", "a snow-covered village"],
 "night": ["the sea at night", "a city at night", "stars over a desert", "a campfire in the dark", "a night train window", "the moon behind clouds", "a lantern on a path", "a dark forest", "fireworks", "the northern lights"],
 "work": ["a desk full of papers", "a calendar with a circled day", "a whiteboard of arrows", "a clock on a wall", "a to-do list", "a meeting room", "a long corridor", "a race against time", "an empty inbox", "a finish line", "gears meshing", "a construction site", "a toolbox", "a road under construction", "a marathon runner"],
 "feelings": ["a warm blanket", "a cold empty room", "a heart of glass", "a storm inside a bottle", "a bright balloon", "a dark cloud", "a garden after rain", "a locked box", "sunlight on a wall", "a distant shore", "a bird in a cage", "a wide open window", "a heavy stone", "a feather floating", "a lit lantern", "a broken bridge", "a mirror in fog", "a warm kitchen", "a cliff over the sea", "a path through snow"],
}
prompts = [p for g in groups.values() for p in g]
prompts = list(dict.fromkeys(prompts))
tok = CLIPTokenizer.from_pretrained(os.path.join(M, "mobileclip_s0"))
ts = ort.InferenceSession(os.path.join(M, "mobileclip_s0", "onnx", "text_model_fp16.onnx"), providers=["CPUExecutionProvider"])
embs = []
for i in range(0, len(prompts), 32):
    ids = tok(prompts[i:i+32], padding="max_length", max_length=77, return_tensors="np")["input_ids"].astype(np.int64)
    e = ts.run(None, {"input_ids": ids})[0]; embs.append(e / np.linalg.norm(e, axis=1, keepdims=True))
E = np.concatenate(embs).astype(np.float16)
E.tofile(os.path.join(OUT, "metaphors.f16"))
json.dump({"n": len(prompts), "dim": 512, "prompts": prompts, "groups": {g: len(v) for g, v in groups.items()}}, open(os.path.join(OUT, "metaphors.json"), "w"), ensure_ascii=False)
print(len(prompts), "metaphors,", os.path.getsize(os.path.join(OUT, "metaphors.f16")) // 1024, "KB")
# quick check: nearest metaphors for practical notes
tests = ["deadline on Friday", "we disagree about chapter 3", "buy snacks for next meeting", "I felt lonely reading this"]
ids = tok(tests, padding="max_length", max_length=77, return_tensors="np")["input_ids"].astype(np.int64)
t = ts.run(None, {"input_ids": ids})[0]; t /= np.linalg.norm(t, axis=1, keepdims=True)
S = t @ E.astype(np.float32).T
for i, q in enumerate(tests):
    top = np.argsort(-S[i])[:4]; print(f"{q!r} -> " + ", ".join(f"{prompts[j]} ({S[i][j]:.2f})" for j in top))

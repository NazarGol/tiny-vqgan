"""Text sources shared by the text-encoder distillation and the one-pass data generator."""
import json, os, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); EXPORT = os.path.normpath(os.path.join(HERE, "..", "..", "export"))
sys.path.insert(0, os.path.join(EXPORT, "paintings"))
PROMPTS40 = ['a face', 'a red forest', 'the sea at night', 'a city street', 'green hills under a blue sky', 'a cat', 'deadline on Friday', "I miss my grandmother's kitchen", 'the smell of rain', 'a lighthouse in a storm', 'a bowl of oranges', 'snow on a mountain', 'a crowded market', 'a sleeping dog', 'fire', 'a glass of water on a table', 'we should talk about the budget', 'a yellow bicycle', 'an old library', "a child's drawing of a house", 'sunset over the plains', 'a portrait of a woman in blue', 'a horse in a field', 'mushrooms in the forest', 'the moon over the sea', 'a broken clock', 'a train station in winter', 'flowers in a vase', 'I feel tired today', 'a river through a canyon', 'a dark room with one candle', 'the first day of school', 'a bird on a wire', 'an abandoned factory', 'waves crashing on rocks', 'a wedding', 'a bridge in fog', 'coffee in the morning', 'a map of an imaginary island', 'the sound of a cello']
ADJ = ["tired", "happy", "anxious", "calm", "angry", "hopeful", "lost", "excited", "bored", "grateful", "sad", "nervous", "proud", "lonely", "fine", "overwhelmed", "curious", "relieved"]
THING = ["my grandmother's kitchen", "the sea", "the old house", "my friends", "summer", "the city", "the mountains", "the garden", "the train", "home", "the river", "that song", "school", "the market", "the forest"]
TOPIC = ["the budget", "the plan", "the deadline", "the schedule", "our trip", "the project", "the garden", "dinner", "the meeting", "the report", "the move", "the party", "the exam", "the book", "the painting"]
VERB = ["call mom", "water the plants", "finish the report", "book the tickets", "fix the bike", "clean the kitchen", "write back", "buy bread", "pay the rent", "walk the dog", "read more", "sleep early", "start again", "say thank you", "take a break"]
DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "tomorrow", "next week", "the 5th", "noon", "midnight", "morning", "evening"]
NOTE_T = ["I feel {adj} today", "deadline on {day}", "we should talk about {topic}", "remember to {verb}", "{topic} meeting on {day}", "I miss {thing}", "what if we {verb}?", "{topic} is {adj}", "let's {verb} on {day}", "thinking about {thing}", "{thing} at {day}", "I am {adj} about {topic}", "don't forget: {verb}", "{adj} and {adj}", "a note about {topic}", "{thing}", "{topic}", "{verb}", "{adj}", "why is {topic} so {adj}", "see you on {day}", "{thing} makes me {adj}", "plan: {verb}, then {verb}", "question about {topic}"]

def notes(n=20000, seed=0):
    rng = random.Random(seed)
    def one():
        t = rng.choice(NOTE_T); return t.format(adj=rng.choice(ADJ), day=rng.choice(DAY), topic=rng.choice(TOPIC), verb=rng.choice(VERB), thing=rng.choice(THING))
    return list(dict.fromkeys(one() for _ in range(n)))

def metaphors():
    src = open(os.path.join(EXPORT, "make_metaphors.py")).read()
    return sorted(set(re.findall(r'"([^"\n]{3,60})"', src.split("groups = {")[1].split("\n}")[0])))

def painting_prompts(n=30000):
    try:
        from prompts import get_prompts; return list(dict.fromkeys(get_prompts(n, seed=1)))
    except Exception as e: print("prompts.py unavailable", e); return []

def captions(captions_dir):
    out = {"coco": [], "val": []}
    if captions_dir and os.path.exists(os.path.join(captions_dir, "captions_train2017.json")):
        out["coco"] = [a["caption"].strip() for a in json.load(open(os.path.join(captions_dir, "captions_train2017.json")))["annotations"]]
        out["val"] = [a["caption"].strip() for a in json.load(open(os.path.join(captions_dir, "captions_val2017.json")))["annotations"]]
    return out

def all_texts(captions_dir=""):
    t = captions(captions_dir); t["prompts"] = painting_prompts(); t["metaphors"] = metaphors(); t["notes"] = notes(); return t

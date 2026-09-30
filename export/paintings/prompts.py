"""Deterministic painting prompts for the offline VQGAN+CLIP painting generator.

Subjects are things people talk and write about (book clubs, discussions, workshops, notes):
feelings, memories, ideas, relationships, places, weather, objects, animals, plants, cities, seasons,
abstract concepts and plain nature. Templates are painting media/styles; about half the prompts get a
palette or composition hint so colours and layouts vary. get_prompts(n, seed) is stable for a given seed,
so a resumed run gets the same prompt for the same index.
"""
import random

SUBJECTS = {
    "feelings": [
        "a quiet joy", "the ache of missing someone", "nervous excitement before a first meeting", "relief after a hard week",
        "a slow-burning anger", "the calm after crying", "boredom on a long afternoon", "a sudden burst of courage",
        "homesickness", "the warmth of being understood", "loneliness in a crowded room", "gratitude at the end of the day",
        "jealousy dressed up as concern", "the lightness of forgiving someone", "a shy kind of pride", "dread on a Sunday evening",
        "the fear of being misunderstood", "tenderness toward a sleeping child", "restlessness", "feeling small under the sky",
        "the thrill of a new idea", "grief that comes in waves", "contentment by a warm stove", "embarrassment fading into laughter",
    ],
    "memories": [
        "a childhood kitchen on a winter morning", "the first day at a new school", "a grandmother's hands folding dough",
        "the smell of rain on a summer road", "a long-gone summer house by the lake", "the last conversation before a move",
        "a train journey taken alone at seventeen", "the sound of a parent's car returning at night", "an old photo album left open",
        "a birthday nobody remembered", "the bicycle we learned to ride on", "a hospital corridor at dawn", "a wedding under plum trees",
        "the day the river froze", "a lost letter finally read", "the garden behind the first apartment", "a snowball fight in the schoolyard",
        "a song that played in a borrowed car", "the attic full of forgotten toys", "fishing with an uncle at first light",
    ],
    "ideas": [
        "the idea that stories keep the dead alive", "an argument about free will", "the pull between routine and adventure",
        "a plan that changed halfway through", "the question of what we owe strangers", "learning to say no",
        "an unfinished thought", "the difference between solitude and loneliness", "a theory whispered late at night",
        "the invention of the calendar", "the border between memory and imagination", "an experiment that failed beautifully",
        "the weight of an unread book", "a map of everything we do not know", "the notion of home as a person",
        "a language nobody speaks anymore", "the rules of a game invented by children", "the moment a habit becomes a self",
    ],
    "relationships": [
        "two old friends who no longer need to talk", "a mother and daughter walking apart then together", "brothers arguing over a chessboard",
        "a couple sharing one umbrella", "a teacher and the student who surpassed her", "neighbours talking over a fence",
        "a long-distance friendship kept alive by letters", "a first date in a noisy cafe", "a reunion at a railway station",
        "strangers sharing a bench in the rain", "a grandfather teaching a boy to whistle", "a marriage of forty years at breakfast",
        "sisters braiding each other's hair", "a book club arguing about the ending", "an apology said with flowers",
        "a goodbye at an airport", "a family dinner with one empty chair", "three friends on a rooftop at midnight",
        "a dog waiting by the door", "a mentor's last lesson",
    ],
    "places": [
        "a village library with a leaking roof", "a lighthouse at the edge of the world", "a rooftop garden above the traffic",
        "an empty classroom after the bell", "a night train crossing a plain", "a bathhouse in the mountains",
        "a bookshop that never closes", "a harbour full of sleeping boats", "a hospital garden", "a chapel in a wheat field",
        "the waiting room of a small station", "a greenhouse in winter", "a kitchen table covered in maps", "a monastery on a cliff",
        "a caravan parked by the sea", "an abandoned amusement park", "a courtyard with a single fig tree", "a cabin above the treeline",
        "a ferry crossing at dusk", "a market square before the stalls open", "a bridge over a slow river", "a hotel corridor at 3 a.m.",
    ],
    "weather": [
        "the first snow of the year", "a thunderstorm rolling over the hills", "fog swallowing a harbour", "heat shimmering over asphalt",
        "a rainbow after a hailstorm", "drizzle on a grey Tuesday", "wind bending a field of wheat", "sunlight breaking through after rain",
        "a dust storm approaching a town", "sleet against a window", "a warm front arriving over the sea", "the still air before a storm",
        "monsoon rain on tin roofs", "frost on a spider's web", "a heatwave at noon", "clouds racing over the moor",
    ],
    "objects": [
        "a chipped teacup", "an old typewriter", "a bowl of lemons", "a pair of muddy boots by the door", "a stack of unread letters",
        "a violin in its open case", "a jar of buttons", "a wooden chair by a window", "a kettle on a stove", "a bicycle leaning on a wall",
        "a bundle of dried lavender", "an umbrella turned inside out", "a pocket watch stopped at noon", "a loaf of bread and a knife",
        "a candle burning down", "a suitcase packed in a hurry", "a globe with faded oceans", "a radio on a windowsill",
        "keys on a table", "a half-finished puzzle", "a red scarf on a hook", "a fishbowl on a bookshelf", "a wilting bouquet in a vase",
    ],
    "animals": [
        "a fox crossing a snowy field", "a heron standing in the shallows", "a cat asleep in a patch of sun", "an owl in a barn at night",
        "a flock of starlings at dusk", "a horse in a rainy paddock", "a dog running along the shore", "bees in a lavender bush",
        "a whale surfacing under a grey sky", "a hare in tall grass", "swans on a black lake", "a bear in a berry thicket",
        "an old tortoise in a garden", "crows on a telephone wire", "a deer at the edge of the forest", "a goldfish in a rain barrel",
        "a moth circling a lamp", "sheep on a green hillside", "a wolf howling on a ridge", "pigeons in a city square",
    ],
    "plants": [
        "a sunflower field", "an apple tree in blossom", "moss on old stones", "a cactus on a windowsill", "wild poppies in a wheat field",
        "a birch forest in autumn", "a single dandelion in cracked concrete", "ivy climbing a brick wall", "water lilies on a pond",
        "a willow trailing in the river", "tulips in a kitchen jar", "an olive grove in the heat", "ferns in a damp gorge",
        "a cherry tree losing its petals", "a pine forest after rain", "a wisteria over a doorway", "a bowl of peaches",
        "an oak tree in a storm", "a vegetable garden in July", "a bonsai on a shelf", "reeds along a canal", "a rose bush in frost",
    ],
    "cities": [
        "a rainy street with neon reflections", "rooftops under a full moon", "a tram at dawn", "an old town square with pigeons",
        "a canal city in winter", "a night market full of lanterns", "a subway platform at rush hour", "a cathedral in fog",
        "laundry hanging between balconies", "a city seen from a hill at sunset", "a bridge lit up over a dark river",
        "a bus stop in the snow", "cranes over a construction site", "a cafe terrace in the evening", "a skyline dissolving in haze",
        "a narrow alley with a single lamp", "a harbour city at dusk", "a highway interchange at night", "a quiet suburb in the morning",
    ],
    "seasons": [
        "the last warm day of autumn", "a spring morning after rain", "midsummer twilight", "the middle of winter",
        "an early thaw", "harvest time", "the first cold morning of the year", "late summer haze over the fields",
        "a snowy afternoon", "the equinox", "leaves falling on a pond", "an orchard in late spring", "a long June evening",
        "the dark days before the solstice", "the first green of April", "October fog in the valley",
    ],
    "concepts": [
        "time passing", "silence", "doubt", "hope", "forgetting", "attention", "patience", "chaos and order", "the passing of a year",
        "waiting", "trust", "beginnings", "endings", "curiosity", "uncertainty", "belonging", "distance", "change", "stillness",
        "the weight of words", "an unanswered question", "the space between two people", "inheritance", "boredom", "wonder",
        "the shape of a week", "the sound of thinking", "a promise", "a secret", "resilience", "the feeling of almost remembering",
    ],
    "nature": [
        "a mountain lake at dawn", "waves breaking on black rocks", "a river bend in autumn", "a desert under the stars",
        "a meadow full of wildflowers", "cliffs above a green sea", "a valley filled with morning mist", "the northern lights over a forest",
        "a waterfall in the jungle", "sand dunes at sunset", "a frozen lake with cracks", "a volcano at night", "rolling hills after rain",
        "a canyon at midday", "a moonlit beach", "a glacier meeting the sea", "a storm over the ocean", "a marsh at dusk",
        "a rocky coast in winter light", "a bamboo forest", "a hot spring in the snow", "a sky full of cumulus clouds",
        "rain over a lake", "a field of snow with a single tree", "a red sunset over the plains",
    ],
}

TEMPLATES = [
    "an oil painting of {s}", "an oil painting about {s}, {mood}", "watercolor of {s}", "a watercolour and ink sketch of {s}, {mood}",
    "an abstract expressionist painting about {s}", "a gouache sketch of {s}, {mood}", "an impressionist painting of {s} at {time}",
    "ink wash painting of {s}", "a child's crayon drawing of {s}", "a surreal painting of {s}", "a pastel drawing of {s}, {mood}",
    "a fauvist painting of {s}", "a cubist painting of {s}", "an acrylic painting of {s} at {time}", "a naive folk art painting of {s}",
    "a fresco of {s}", "an expressionist painting of {s}, {mood}", "a tempera painting of {s}", "a minimalist painting of {s}",
    "a pointillist painting of {s} at {time}", "a charcoal and pastel sketch of {s}", "a woodblock print of {s}",
    "a colour field painting about {s}", "a collage painting of {s}", "a symbolist painting of {s}", "a post-impressionist painting of {s}",
    "an art brut painting of {s}", "a palette knife painting of {s}", "a quick plein air oil sketch of {s} at {time}",
    "a dreamy painting of {s}, {mood}", "a bold graphic poster painting of {s}", "an encaustic painting of {s}",
]

MOODS = [
    "melancholic", "hopeful", "serene", "restless", "tender", "eerie", "playful", "solemn", "luminous", "brooding", "nostalgic",
    "joyful", "wistful", "quiet", "feverish", "gentle", "stormy", "dreamlike", "bittersweet", "bright and airy",
]
TIMES = [
    "dawn", "sunrise", "noon", "golden hour", "dusk", "twilight", "night", "midnight", "a rainy afternoon", "a foggy morning",
    "the blue hour", "a snowy evening", "high summer", "the end of autumn",
]
PALETTES = [
    "in warm reds and ochres", "in cool blues and greys", "in muted earth tones", "in pale pastels", "in deep greens and gold",
    "in black, white and one red", "in violet and amber", "in sunlit yellows", "in teal and rust", "in rose and slate",
    "in monochrome indigo", "in vivid saturated colours", "in faded sepia tones", "in icy whites and silver", "in emerald and coral",
    "in burnt orange and navy", "in soft greys with a spark of pink", "in olive and cream", "in crimson and gold", "in lilac and moss green",
    "in dusty pink and charcoal", "in ultramarine and lemon", "in warm browns and turquoise", "in mint and terracotta",
]
COMPOSITIONS = [
    "close-up", "wide view", "seen from above", "with a lone figure", "with a big empty sky", "with thick brushstrokes",
    "with visible canvas texture", "with loose brushwork", "with fine detail", "framed by a window", "with a low horizon",
    "with swirling shapes", "with bold outlines", "with drips and splatters", "with soft edges", "with strong diagonals",
    "with a single light source", "with flat shapes", "with heavy impasto", "with a wide empty foreground",
]
GROUPS = sorted(SUBJECTS)


def _one(rng):
    group = rng.choice(GROUPS)
    s = rng.choice(SUBJECTS[group])
    p = rng.choice(TEMPLATES).format(s=s, mood=rng.choice(MOODS), time=rng.choice(TIMES))
    r = rng.random()
    if r < 0.45:
        p += ", " + rng.choice(PALETTES)
    elif r < 0.7:
        p += ", " + rng.choice(COMPOSITIONS)
    elif r < 0.8:
        p += ", " + rng.choice(PALETTES) + ", " + rng.choice(COMPOSITIONS)
    return p


def get_prompts(n, seed=0):
    """n distinct prompts; the first k prompts are the same for any n >= k (same seed)."""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        p = _one(rng)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for i, p in enumerate(get_prompts(n)):
        print(f"{i:05d} {p}")

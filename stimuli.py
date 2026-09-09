#!/usr/bin/env python3
"""Generate matched OCR stimuli that differ only in how plausible the text is.

The whole experiment rests on one control: **visual difficulty must be identical
across conditions.** If the implausible strings were also blurrier, a model doing
worse on them would prove nothing. So each item is rendered from the same font,
at the same size, with the same seeded degradation — blur, contrast, noise,
rotation — and only the characters change.

Five conditions, ordered by how much a language model can help:

    word        a real Turkish word                     strong prior
    perturbed   the same word, one glyph swapped        prior actively wrong
    pseudo      Turkish phonotactics, not a word        weak prior
    random      random glyphs                           no prior
    plate       Turkish licence plate format            structural prior

`perturbed` is the measurement that matters. The swap is chosen to be visually
unambiguous — K to M, not O to 0 — so a model that outputs the *original* word
is not misreading blurred pixels. It is overriding clear pixels with what it
expected to see. That behaviour has a name in an annotation pipeline: a label
that looks correct and is not.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Common Turkish words, mixed length, covering every Turkish-specific glyph.
# ponytail: hand-curated rather than frequency-ranked from a corpus; a corpus
# ranking would let "prior strength" be a measured variable instead of an
# assumed one, and that is the obvious upgrade if the effect turns out subtle.
WORDS = [
    "KAVŞAK", "TRAFİK", "ARAÇ", "YOLCU", "DURAK", "GEÇİT", "ŞERİT", "KÖPRÜ",
    "TÜNEL", "SİNYAL", "LEVHA", "PARK", "GİRİŞ", "ÇIKIŞ", "DUR", "YAVAŞ",
    "OKUL", "HASTANE", "MERKEZ", "SANAYİ", "İSTASYON", "OTOGAR", "LİMAN",
    "BULVAR", "CADDE", "SOKAK", "MAHALLE", "İLÇE", "KÖY", "ŞEHİR",
    "GÜVENLİK", "KAMERA", "HIZ", "SINIR", "TAŞIT", "KAMYON", "OTOBÜS",
    "MOTOSİKLET", "BİSİKLET", "YAYA", "GEÇİŞ", "YASAK", "SERBEST", "AÇIK",
    "KAPALI", "ÇALIŞMA", "BAKIM", "ONARIM", "TEHLİKE", "DİKKAT",
]

# Swaps chosen for visual distance: a reader cannot confuse the pair, so any
# "repair" back to the original is a language prior at work, not bad eyesight.
SWAPS = {"K": "M", "T": "L", "R": "P", "A": "E", "S": "F", "N": "H", "L": "T",
         "M": "N", "İ": "U", "E": "O", "O": "A", "Ş": "B", "Ç": "G", "Ü": "A",
         "Ö": "E", "I": "Y", "G": "Ç", "D": "B", "P": "R", "U": "V", "Y": "K",
         "B": "D", "C": "S", "F": "T", "H": "K", "V": "Z", "Z": "V"}

SYLLABLES = ["ka", "de", "mi", "ru", "sö", "tü", "la", "be", "çı", "gö",
             "ne", "pa", "şe", "vi", "ya", "zu", "ba", "ce", "dö", "fı"]

# Recombining syllables at random will eventually spell something crude. A
# benchmark that renders it is a benchmark nobody puts on a slide, so candidates
# containing any of these are discarded and redrawn.
BLOCKED = ("GÖT", "SİK", "AMC", "OROS", "PUŞ", "YARR", "MAL", "APTA", "SALA")

PROVINCES_VALID = [1, 6, 7, 16, 27, 34, 35, 41, 55, 61, 81]
PROVINCES_INVALID = [0, 82, 90, 97, 99]

# Turkish plates are two digits of province, then one to three letters, then a
# run of digits whose length depends on how many letters came before it. The
# table is the structural prior itself: a model that "knows plates" knows that
# 99 is not a province and that 3 letters are never followed by 5 digits.
PLATE_DIGITS = {1: (4, 5), 2: (3, 4), 3: (2, 3, 4)}
PLATE_LETTERS = "ABCDEFGHJKLMNPRSTUVYZ"        # no I, O, Q, W, X, Ö, Ü on plates

# 520 × 110 mm, with a 40 mm blue band down the left carrying a white TR in its
# lower third. No EU stars: Turkey is not a member.
PLATE_ASPECT = 520 / 110
PLATE_BAND = 40 / 520
PLATE_BLUE = (0, 51, 153)

CANVAS = (900, 200)
BACKGROUND = (238, 238, 234)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def find_font() -> str:
    for path in FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            return path
    raise SystemExit("no usable font found; install fonts-dejavu")


def perturb(word: str, rng: random.Random) -> tuple[str, int]:
    """Swap one interior glyph for a visually distant one. Returns (string, index)."""
    positions = [i for i, ch in enumerate(word) if ch in SWAPS]
    interior = [i for i in positions if 0 < i < len(word) - 1] or positions
    if not interior:
        return word, -1
    index = rng.choice(interior)
    return word[:index] + SWAPS[word[index]] + word[index + 1:], index


def pseudo_word(rng: random.Random, syllables: int = 3) -> str:
    for _ in range(50):
        candidate = "".join(rng.choice(SYLLABLES) for _ in range(syllables)).upper()
        if not any(bad in candidate for bad in BLOCKED):
            return candidate
    return "KADEMİ"                                    # deterministic fallback


def random_string(rng: random.Random, length: int = 6) -> str:
    """Letters only, on purpose.

    An earlier version mixed in digits and the condition scored far worse than
    the others — but that was glyph ambiguity (O/0, I/1, S/5), not absence of a
    language prior. Matching the character inventory to the word conditions is
    what makes the comparison about priors instead of about typography.
    """
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(length))


def plate(rng: random.Random, valid: bool) -> str:
    """A plate string that is well-formed apart from the province, when invalid.

    Letter count and digit count are drawn together from PLATE_DIGITS so that
    both arms are legal Turkish layouts. Only the province separates them: an
    invalid plate must be wrong in exactly one way, or a model rejecting it
    would not tell us which cue it used.
    """
    province = rng.choice(PROVINCES_VALID if valid else PROVINCES_INVALID)
    count = rng.choice(list(PLATE_DIGITS))
    letters = "".join(rng.choice(PLATE_LETTERS) for _ in range(count))
    digits = "".join(rng.choice("0123456789") for _ in range(rng.choice(PLATE_DIGITS[count])))
    return f"{province:02d} {letters} {digits}"


def draw_plate(text: str, font_path: str, size: tuple[int, int],
               background=BACKGROUND) -> Image.Image:
    """Draw `text` inside a Turkish plate rather than on a bare field.

    The plate conditions test a *structural* prior — that 99 is not a province,
    that three letters are not followed by five digits. A prior can only fire if
    the model knows what it is looking at, and a string on a grey background does
    not say "plate". The frame, the proportions and the blue TR band are what put
    the reader into the regime this condition means to measure.
    """
    width, height = size
    canvas = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(canvas)

    plate_w = min(int(width * 0.94), int(height * 0.86 * PLATE_ASPECT))
    plate_h = int(plate_w / PLATE_ASPECT)
    x0, y0 = (width - plate_w) // 2, (height - plate_h) // 2
    radius = max(2, plate_h // 12)

    outline = [x0, y0, x0 + plate_w, y0 + plate_h]
    draw.rounded_rectangle(outline, radius=radius, fill=(252, 252, 250))

    # The band shares the plate's rounded left corners, so it is drawn as a
    # rounded rectangle that overshoots to the right and the overshoot is painted
    # back out. Cheaper than compositing through a mask, and pixel-identical.
    band_w = max(6, int(plate_w * PLATE_BAND))
    draw.rounded_rectangle([x0, y0, x0 + band_w + radius, y0 + plate_h],
                           radius=radius, fill=PLATE_BLUE)
    draw.rectangle([x0 + band_w, y0 + 1, x0 + band_w + radius, y0 + plate_h - 1],
                   fill=(252, 252, 250))

    # Border last: on a real plate it frames the band too, not just the white.
    draw.rounded_rectangle(outline, radius=radius, outline=(20, 20, 20),
                           width=max(2, plate_h // 26))

    # TR sits in the lower third of the band, centred within that third.
    band_font = ImageFont.truetype(font_path, max(7, int(plate_h * 0.20)))
    box = draw.textbbox((0, 0), "TR", font=band_font)
    third = plate_h / 3
    draw.text((x0 + (band_w - (box[2] - box[0])) // 2,
               y0 + 2 * third + (third - (box[3] - box[1])) / 2 - box[1]),
              "TR", font=band_font, fill=(255, 255, 255))

    # Shrink to fit: plates run from 7 to 9 characters and the widest must still
    # sit inside the same frame.
    inner_x = x0 + band_w + int(plate_w * 0.035)
    inner_w = x0 + plate_w - inner_x - int(plate_w * 0.035)
    for points in range(int(plate_h * 0.62), 6, -2):
        font = ImageFont.truetype(font_path, points)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= inner_w:
            break
    draw.text((inner_x + (inner_w - (box[2] - box[0])) // 2,
               y0 + (plate_h - (box[3] - box[1])) // 2 - box[1]),
              text, font=font, fill=(12, 12, 12))
    return canvas


def render(text: str, font_path: str, rng: random.Random, difficulty: float,
           as_plate: bool = False) -> Image.Image:
    """Render `text` with a seeded degradation of the given strength.

    `difficulty` runs 0 (clean) to 1 (hard). Every knob moves together so a
    single number describes how legible the item is, which is what makes the
    conditions comparable: the same difficulty means the same pixels-per-glyph
    quality, whatever the glyphs spell.

    `as_plate` draws the string inside a plate instead of on a bare field. The
    degradation that follows is identical either way, so the two renderings of
    the same plate string differ only in whether the structural cue is present.
    """
    if as_plate:
        canvas = draw_plate(text, font_path, CANVAS)
    else:
        font = ImageFont.truetype(font_path, 64)
        canvas = Image.new("RGB", CANVAS, BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((CANVAS[0] - (box[2] - box[0])) // 2,
                   (CANVAS[1] - (box[3] - box[1])) // 2 - box[1]),
                  text, font=font, fill=(18, 18, 18))

    if difficulty > 0:
        # Resolution first: real OCR difficulty is mostly small or distant text,
        # and blur alone on a 64pt glyph barely registers. Downscaling and
        # scaling back is what a far-away sign actually does to a sensor.
        scale = 1 - 0.86 * difficulty
        small = canvas.resize((max(60, int(canvas.width * scale)),
                               max(20, int(canvas.height * scale))), Image.BILINEAR)
        canvas = small.resize(canvas.size, Image.BILINEAR)
        canvas = canvas.rotate(rng.uniform(-2.5, 2.5) * difficulty, resample=Image.BICUBIC,
                               fillcolor=BACKGROUND)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=2.6 * difficulty))
        canvas = ImageEnhance.Contrast(canvas).enhance(1 - 0.55 * difficulty)
        spread = int(38 * difficulty)
        if spread:
            # Same noise on all three channels, so the degradation is luminance
            # only — a colour cast would be a second uncontrolled variable.
            array = np.asarray(canvas, dtype=np.int16)
            noise = np.random.default_rng(rng.randrange(1 << 30)).integers(
                -spread, spread + 1, size=array.shape[:2])[:, :, None]
            canvas = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    return canvas


def build(out: pathlib.Path, items: int, difficulties: list[float], seed: int,
          plate_style: str = "plate") -> dict:
    font_path = find_font()
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)

    stimuli = []
    for index in range(items):
        word = WORDS[index % len(WORDS)]
        perturbed, position = perturb(word, rng)
        variants = {
            "word": word,
            "perturbed": perturbed,
            "pseudo": pseudo_word(rng),
            "random": random_string(rng),
            "plate_valid": plate(rng, valid=True),
            "plate_invalid": plate(rng, valid=False),
        }
        for difficulty in difficulties:
            # One seed per (item, difficulty): every condition in this cell gets
            # exactly the same rotation, blur, contrast and noise.
            for condition, text in variants.items():
                cell_seed = hash((seed, index, difficulty)) & 0xFFFFFFF
                image = render(text, font_path, random.Random(cell_seed), difficulty,
                               as_plate=plate_style == "plate" and condition.startswith("plate"))
                name = f"{index:04d}_{int(difficulty * 100):03d}_{condition}.png"
                image.save(out / "images" / name)
                stimuli.append({
                    "file": name, "condition": condition, "text": text,
                    "difficulty": difficulty, "item": index,
                    "source_word": word if condition in ("word", "perturbed") else None,
                    "swap_index": position if condition == "perturbed" else None,
                })

    manifest = {"seed": seed, "items": items, "difficulties": difficulties,
                "font": font_path, "plate_style": plate_style,
                "count": len(stimuli), "stimuli": stimuli}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="stimuli")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--difficulties", type=float, nargs="+",
                    default=[0.0, 0.75, 0.9, 0.95, 1.0])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--plate-style", choices=["plate", "text"], default="plate",
                        help="draw plate conditions inside a plate, or as bare text")
    args = parser.parse_args()

    manifest = build(pathlib.Path(args.out), args.items, args.difficulties, args.seed,
                     args.plate_style)
    print(f"{manifest['count']} images across {len(manifest['difficulties'])} difficulties")
    for condition in ("word", "perturbed", "pseudo", "random", "plate_valid", "plate_invalid"):
        sample = next(s for s in manifest["stimuli"] if s["condition"] == condition)
        print(f"  {condition:<14} e.g. {sample['text']}")

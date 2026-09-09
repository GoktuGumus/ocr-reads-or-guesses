#!/usr/bin/env python3
"""Checks for the two pieces of this generator that can be quietly wrong.

A stimulus set is not like other code: a bug does not raise, it just produces a
number. If a plate is malformed in a way real plates never are, the model will
reject it for the wrong reason and `plate_repair` measures something other than
what it claims. If the blue band is missing, the structural prior never fires
and the condition measures nothing at all.

    python test_stimuli.py
"""
from __future__ import annotations

import pathlib
import random
import re
import tempfile

import numpy as np

from stimuli import (PLATE_BLUE, PLATE_DIGITS, PLATE_LETTERS, PROVINCES_INVALID,
                     PROVINCES_VALID, BACKGROUND, build, draw_plate, find_font,
                     plate, render)

SHAPE = re.compile(r"^(\d{2}) ([A-Z]{1,3}) (\d{2,5})$")


def test_plate_shapes() -> None:
    rng = random.Random(7)
    for valid in (True, False):
        for _ in range(400):
            text = plate(rng, valid=valid)
            match = SHAPE.match(text)
            assert match, text
            province, letters, digits = match.groups()
            assert len(digits) in PLATE_DIGITS[len(letters)], text
            assert not set(letters) - set(PLATE_LETTERS), text
            # The only thing separating the arms. Anything else different and a
            # model could reject an invalid plate without using the province.
            pool = PROVINCES_VALID if valid else PROVINCES_INVALID
            assert int(province) in pool, text


def test_both_arms_span_every_layout() -> None:
    """A layout that only ever appears in one arm would be a free giveaway."""
    rng = random.Random(11)
    seen = {True: set(), False: set()}
    for valid in (True, False):
        for _ in range(400):
            seen[valid].add(len(SHAPE.match(plate(rng, valid=valid)).group(2)))
    assert seen[True] == seen[False] == set(PLATE_DIGITS)


def test_plate_is_drawn_as_a_plate() -> None:
    image = np.asarray(draw_plate("34 ABC 123", find_font(), (900, 200)))

    blue = np.abs(image.astype(int) - PLATE_BLUE).sum(axis=2) < 60
    assert blue.any(), "no blue band"
    # Mostly-blue rows and columns, not merely blue-somewhere: the rounded
    # corners leave stray blue a few pixels into the white body of the plate.
    rows = np.nonzero(blue.mean(axis=1) > 0.5 * blue.mean(axis=1).max())[0]
    columns = np.nonzero(blue.mean(axis=0) > 0.6 * blue.mean(axis=0).max())[0]
    top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
    plate = np.nonzero((np.abs(image.astype(int) - BACKGROUND).sum(axis=2) > 30)
                       .any(axis=0))[0]
    assert left - plate.min() < 5, "the band is not at the plate's left edge"

    # TR is white ink inside the band, and it sits low: the band's top third must
    # be empty and its bottom third must not be.
    band = image[top:bottom + 1, left:right + 1]
    white = band.min(axis=2) > 200
    third = band.shape[0] // 3
    # A ratio, not a zero: the band's rounded top corner lets a few background
    # pixels into the bounding box, and they are light too.
    assert white[2 * third:].sum() > 10 * white[:third].sum(), \
        "TR is not sitting in the lower third of the band"
    assert white[third:2 * third].sum() == 0, "the band is inked across its middle"


def glyph_height(image, inside_plate: bool) -> int:
    """Height of the tallest run of ink, ignoring a plate's frame and band."""
    grey = np.asarray(image.convert("L"))
    if inside_plate:
        mask = np.abs(np.asarray(image).astype(int) - BACKGROUND).sum(axis=2) > 30
        rows, columns = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        inset = int((rows.max() - rows.min()) * 0.18)
        span = columns.max() - columns.min()
        grey = grey[rows.min() + inset:rows.max() - inset,
                    columns.min() + int(span * 0.20):columns.max() - int(span * 0.05)]
    ink = np.nonzero((grey < 100).any(axis=1))[0]
    return int(ink.max() - ink.min() + 1)


def test_every_condition_gets_the_same_glyph_size() -> None:
    """The one control the whole benchmark rests on.

    Same difficulty must mean the same pixels per character, whatever the
    characters spell. An earlier draft sized the plate to the canvas and shrank
    the text to fit, which made plate glyphs 1.6x taller than word glyphs — the
    plate conditions would have scored better for having more resolution, and
    the result would have been read as a prior.
    """
    font = find_font()
    heights = {}
    for text, as_plate in [("HASTANE", False), ("MERKEZ", False), ("KAMYON", False),
                           ("34 ABC 123", True), ("99 X 12345", True), ("06 TF 449", True)]:
        image = render(text, font, random.Random(1), 0.0, as_plate=as_plate)
        heights[text] = glyph_height(image, as_plate)
    assert max(heights.values()) - min(heights.values()) <= 3, heights


def test_a_subset_run_is_the_same_experiment() -> None:
    """Rendering only some conditions must not move the ones it keeps.

    The random stream is shared across conditions, so a subset build that
    skipped the draws instead of the renders would produce different plates —
    and a plate-only run could then not be compared to a full one at all.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        full = build(root / "full", items=3, difficulties=[0.0, 0.9], seed=13)
        part = build(root / "part", items=3, difficulties=[0.0, 0.9], seed=13,
                     conditions=["plate_valid", "plate_invalid"])

        kept = {(s["file"], s["text"]) for s in full["stimuli"]
                if s["condition"].startswith("plate")}
        assert kept == {(s["file"], s["text"]) for s in part["stimuli"]}
        assert len(part["stimuli"]) == len(kept) > 0
        for name, _ in kept:
            assert (root / "full" / "images" / name).read_bytes() == \
                   (root / "part" / "images" / name).read_bytes(), name


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"ok  {name}")

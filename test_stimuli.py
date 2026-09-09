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

import random
import re

import numpy as np

from stimuli import (PLATE_BLUE, PLATE_DIGITS, PLATE_LETTERS, PROVINCES_INVALID,
                     PROVINCES_VALID, draw_plate, find_font, plate)

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
    columns = np.nonzero(blue.mean(axis=0) > 0.5)[0]
    top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
    assert left < image.shape[1] * 0.1, "the band is not on the left"

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


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"ok  {name}")

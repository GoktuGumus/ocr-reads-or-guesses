#!/usr/bin/env python3
"""Printable sheets that carry the whole experiment inside one photograph.

The synthetic half of this benchmark could render KAVŞAK and KAVBAK with
byte-identical degradation. A photograph cannot: there is no picture of a sign
that says KAVBAK, and editing one into a real photo replaces the thing being
controlled — the perturbed glyph would carry different light, focus and grain
from its neighbours, and any difference in the reading could be blamed on that.

So the matching moves into the capture. Every condition is printed on the same
sheet, the sheet is photographed once, and each string is cropped out of that one
frame. Illumination, focus, motion blur, sensor noise, exposure and white balance
are then identical across conditions by construction, because they are the same
photograph. Distance and lighting become the difficulty ladder, and they vary
between photographs rather than within them.

Four ArUco markers sit at the corners. They turn crop extraction into a
homography instead of a manual labelling job: `extract.py` finds the markers,
rectifies the sheet to its known geometry, and cuts each cell at coordinates that
were fixed when the sheet was generated. No hand annotation, and no chance of a
crop being nudged toward what the annotator expected to see.

    python sheets.py --sheets 3 --out sheets/

Print at 100% scale on A4, no "fit to page" — the marker geometry is what makes
the extraction exact.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stimuli import (BLOCKED, PROVINCES_INVALID, PROVINCES_VALID, SWAPS, WORDS,
                     draw_plate, find_font, perturb, plate, pseudo_word,
                     random_string)

# A4 at 150 dpi. Print scaling must be 100%: the marker separation is the ruler
# that `extract.py` measures everything else against.
DPI = 150
PAGE = (int(8.27 * DPI), int(11.69 * DPI))
# 28 mm. The marker size sets how far away the sheet can be photographed: ArUco
# needs roughly 25 pixels across to decode, so a marker that shrinks below that
# in the frame takes the whole photo with it. On a 12 MP phone this is a
# non-issue at any sane distance; on a low-resolution capture it bites early.
MARKER_PX = int(1.10 * DPI)
MARGIN = int(0.39 * DPI)                    # 10 mm

CONDITIONS = ["word", "perturbed", "pseudo", "random", "plate_valid", "plate_invalid"]
ROWS, COLUMNS = 6, 2                        # 12 cells: two items × six conditions


def markers(canvas: Image.Image) -> list[dict]:
    """Four distinct ArUco markers, one per corner, ids 0-3 clockwise from top-left."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    placements = [(MARGIN, MARGIN), (PAGE[0] - MARGIN - MARKER_PX, MARGIN),
                  (PAGE[0] - MARGIN - MARKER_PX, PAGE[1] - MARGIN - MARKER_PX),
                  (MARGIN, PAGE[1] - MARGIN - MARKER_PX)]
    out = []
    for marker_id, (x, y) in enumerate(placements):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)
        canvas.paste(Image.fromarray(image).convert("RGB"), (x, y))
        out.append({"id": marker_id, "x": x, "y": y, "size": MARKER_PX})
    return out


def cell_boxes() -> list[tuple[int, int, int, int]]:
    """Where each string is printed, in sheet pixel coordinates."""
    top = MARGIN + MARKER_PX + int(0.3 * DPI)
    bottom = PAGE[1] - MARGIN - MARKER_PX - int(0.3 * DPI)
    left, right = MARGIN, PAGE[0] - MARGIN
    height = (bottom - top) // ROWS
    width = (right - left) // COLUMNS
    return [(left + column * width, top + row * height, width, height)
            for row in range(ROWS) for column in range(COLUMNS)]


def build_sheet(index: int, rng: random.Random, font_path: str) -> tuple[Image.Image, dict]:
    words = rng.sample(WORDS, COLUMNS)
    variants: list[tuple[str, str, str | None]] = []
    for word in words:
        perturbed, _ = perturb(word, rng)
        variants.append(("word", word, word))
        variants.append(("perturbed", perturbed, word))
        variants.append(("pseudo", pseudo_word(rng), None))
        variants.append(("random", random_string(rng), None))
        variants.append(("plate_valid", plate(rng, valid=True), None))
        variants.append(("plate_invalid", plate(rng, valid=False), None))

    # Shuffle placement so that a condition never sits at the same distance from
    # the lens axis across sheets — corner softness and focus curvature would
    # otherwise be confounded with the condition.
    rng.shuffle(variants)

    canvas = Image.new("RGB", PAGE, "white")
    draw = ImageDraw.Draw(canvas)
    placed = markers(canvas)
    font = ImageFont.truetype(font_path, int(0.42 * DPI))

    cells = []
    for (condition, text, source), (x, y, width, height) in zip(variants, cell_boxes()):
        if condition.startswith("plate"):
            # Printed as a plate, for the same reason the synthetic half renders
            # one: the structural prior cannot fire on a string that does not
            # look like a plate, and this condition exists to make it fire.
            canvas.paste(draw_plate(text, font_path, (width, height), "white"), (x, y))
        else:
            box = draw.textbbox((0, 0), text, font=font)
            draw.text((x + (width - (box[2] - box[0])) // 2,
                       y + (height - (box[3] - box[1])) // 2 - box[1]),
                      text, font=font, fill=(15, 15, 15))
        # A hairline frame gives the extractor a visual check and the reader
        # nothing: it never touches the glyphs.
        draw.rectangle([x + 4, y + 4, x + width - 4, y + height - 4],
                       outline=(205, 205, 205), width=1)
        cells.append({"condition": condition, "text": text, "source_word": source,
                      "x": x, "y": y, "w": width, "h": height})

    draw.text((MARGIN + MARKER_PX + 12, PAGE[1] - MARGIN - int(0.22 * DPI)),
              f"sheet {index:02d} · print at 100% · A4", font=ImageFont.truetype(font_path, 18),
              fill=(150, 150, 150))

    return canvas, {"sheet": index, "page_px": PAGE, "dpi": DPI,
                    "markers": placed, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", type=int, default=4)
    parser.add_argument("--out", default="sheets")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    font_path = find_font()

    manifest = {"dpi": DPI, "page_px": PAGE, "seed": args.seed, "sheets": []}
    pages = []
    for index in range(args.sheets):
        canvas, record = build_sheet(index, rng, font_path)
        canvas.save(out / f"sheet_{index:02d}.png")
        pages.append(canvas)
        manifest["sheets"].append(record)

    pages[0].save(out / "sheets.pdf", "PDF", resolution=DPI,
                  save_all=True, append_images=pages[1:])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    total = sum(len(s["cells"]) for s in manifest["sheets"])
    print(f"{args.sheets} sheets, {total} strings → {out}/sheets.pdf")
    for condition in CONDITIONS:
        example = next(c["text"] for s in manifest["sheets"]
                       for c in s["cells"] if c["condition"] == condition)
        print(f"  {condition:<14} e.g. {example}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Photograph the printed sheets with a simulated camera.

    python simulate_capture.py --sheets sheets/ --out photos_sim/

This is **not** a substitute for the real capture. It exists because the
extraction path — marker detection, homography, cell cutting, legibility
measurement — is the part of this benchmark most likely to be quietly wrong, and
finding that out after 64 photographs have been taken is the expensive order to
find it out in.

So the sheets are put through a camera model instead: placed at a distance in a
12 MP frame, tilted, lit unevenly, blurred, given sensor noise and JPEG artefacts.
The output is dimensionally honest — a real phone at these distances produces
glyphs of these pixel heights — which is what makes it useful for choosing the
distances to shoot at. It is not honest about optics, print texture, paper
reflectance or motion, and no result measured on it is reported as a result on
photographs.

The distance model is the reason this script earned its place. A 26 mm-equivalent
lens sees 1.32 m of width per metre of distance, so an A4 sheet at 4 m is 120 px
wide in a 3024 px frame and its glyphs are four pixels tall. The first version of
the capture protocol asked for 1, 2, 4 and 6 m; two of those four distances would
have produced nothing but unreadable crops.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

import cv2
import numpy as np

FRAME = (3024, 4032)                    # a 12 MP phone, portrait
SHEET_MM = 210.0                        # A4 width
FOV_PER_METRE = 1.324                   # 26 mm-equivalent lens, horizontal

# gain, colour gain (B, G, R), noise sigma, extra blur, gradient strength
LIGHTING = {
    "daylight": (1.00, (1.00, 1.00, 1.00), 2.0, 0.0, 0.10),
    "shade":    (0.72, (1.12, 1.02, 0.92), 4.0, 0.3, 0.18),
    "indoor":   (0.62, (0.88, 0.98, 1.14), 6.0, 0.5, 0.30),
    "lowlight": (0.38, (0.94, 0.99, 1.08), 14.0, 1.2, 0.40),
}


def sheet_width_px(distance_m: float) -> int:
    """How wide the sheet lands in the frame, from the lens geometry alone."""
    return int(SHEET_MM / 1000 / (FOV_PER_METRE * distance_m) * FRAME[0])


def place(sheet: np.ndarray, distance_m: float, rng: random.Random) -> np.ndarray:
    """Put the sheet in the frame at `distance_m`, at a hand-held angle."""
    width = sheet_width_px(distance_m)
    height = int(width * sheet.shape[0] / sheet.shape[1])

    # A hand-held shot is never square to the page. The corners move by up to 4%
    # of the sheet, which is a few degrees of tilt.
    jitter = width * 0.04
    centre_x = FRAME[0] / 2 + rng.uniform(-1, 1) * (FRAME[0] - width) * 0.25
    centre_y = FRAME[1] / 2 + rng.uniform(-1, 1) * (FRAME[1] - height) * 0.25
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                        [width / 2, height / 2], [-width / 2, height / 2]], np.float32)
    corners += [centre_x, centre_y]
    corners += np.array([[rng.uniform(-jitter, jitter) for _ in range(2)]
                         for _ in range(4)], np.float32)

    source = np.array([[0, 0], [sheet.shape[1], 0],
                       [sheet.shape[1], sheet.shape[0]], [0, sheet.shape[0]]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners)

    # A desk, not a void: mid-grey with a little texture, so the marker detector
    # has to separate the page from a background instead of from nothing.
    frame = np.full((FRAME[1], FRAME[0], 3), 118, np.uint8)
    frame = cv2.add(frame, np.random.default_rng(rng.randrange(1 << 30))
                    .integers(-12, 13, frame.shape, dtype=np.int16).astype(np.int16)
                    .clip(-255, 255).astype(np.uint8))
    return cv2.warpPerspective(sheet, matrix, FRAME, dst=frame,
                               borderMode=cv2.BORDER_TRANSPARENT)


def expose(frame: np.ndarray, lighting: str, rng: random.Random) -> np.ndarray:
    """Apply the light, the lens and the sensor, in that order."""
    gain, colour, sigma, blur, gradient = LIGHTING[lighting]
    image = frame.astype(np.float32)

    # Light falls across the page rather than onto it evenly. A linear ramp in a
    # random direction is the cheapest version of that, and it is the thing the
    # RMS-contrast proxy has to survive.
    ys, xs = np.mgrid[0:frame.shape[0], 0:frame.shape[1]].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * xs / frame.shape[1] + np.sin(angle) * ys / frame.shape[0])
    image *= (gain * (1 - gradient / 2 + gradient * ramp))[:, :, None]
    image *= np.array(colour, np.float32)

    radius = 1.1 + blur + rng.uniform(0, 0.4)
    image = cv2.GaussianBlur(image, (0, 0), radius)
    image += np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", default="sheets")
    parser.add_argument("--out", default="photos_sim")
    parser.add_argument("--distances", type=float, nargs="+", default=[0.5, 0.8, 1.2, 1.8])
    parser.add_argument("--lighting", nargs="+", default=sorted(LIGHTING))
    parser.add_argument("--quality", type=int, default=88, help="JPEG quality")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    root = pathlib.Path(args.sheets)
    manifest = json.loads((root / "manifest.json").read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"{FRAME[0]}x{FRAME[1]} frame, {SHEET_MM:.0f} mm sheet")
    for distance in args.distances:
        width = sheet_width_px(distance)
        print(f"  {distance:>4.1f} m → sheet {width:>4} px wide, "
              f"glyphs ~{46 * width / manifest['page_px'][0]:.0f} px")

    written = 0
    for record in manifest["sheets"]:
        sheet = cv2.imread(str(root / f"sheet_{record['sheet']:02d}.png"))
        for distance in args.distances:
            for lighting in args.lighting:
                frame = expose(place(sheet, distance, rng), lighting, rng)
                name = (f"sheet{record['sheet']:02d}_{str(distance).replace('.', 'p')}m"
                        f"_{lighting}.jpg")
                cv2.imwrite(str(out / name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                written += 1

    print(f"{written} simulated photographs → {out}/")


if __name__ == "__main__":
    main()

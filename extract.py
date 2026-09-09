#!/usr/bin/env python3
"""Turn photographs of the printed sheets into a labelled crop set.

    python extract.py --photos photos/ --sheets sheets/ --out real/

Each photo contains every condition at once, so the crops that come out of one
frame share illumination, focus, exposure and grain exactly. The four ArUco
markers give a homography from the photo back to the sheet's known geometry, so
every cell is cut at coordinates that were decided before the shutter — nobody
draws a box, and nobody's expectation about the text can nudge one.

The synthetic half of this benchmark had a difficulty knob. A photograph does
not, so difficulty is measured instead of set:

    x_height_px    how many pixels tall the glyphs are — the dominant factor
    contrast       RMS contrast inside the crop
    sharpness      variance of the Laplacian, the standard blur proxy

Those three are proxies, and a proxy needs validating. The CTC reader is the
instrument for that: it never repairs, so its accuracy is a measure of legibility
uncontaminated by priors. If accuracy falls monotonically as the proxy says
difficulty rises, the axis is real and the VLM's repair rate can be read against
it.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
CROP_PAD = 0.06                     # fraction of cell size kept around the text


def detect_markers(image: np.ndarray) -> dict[int, np.ndarray]:
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTIONARY), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    return {int(marker_id): corner[0] for marker_id, corner in zip(ids.flatten(), corners)}


def homography(found: dict[int, np.ndarray], sheet: dict) -> np.ndarray | None:
    """Map sheet pixels to photo pixels using whichever markers were detected.

    Each marker contributes four correspondences, so three visible markers are
    enough — useful, because a hand-held shot at an angle loses one corner more
    often than you would like.
    """
    source, target = [], []
    for marker in sheet["markers"]:
        detected = found.get(marker["id"])
        if detected is None:
            continue
        x, y, size = marker["x"], marker["y"], marker["size"]
        source.extend([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
        target.extend(detected.tolist())
    if len(source) < 8:                                  # fewer than two markers
        return None
    matrix, _ = cv2.findHomography(np.array(source, np.float32),
                                   np.array(target, np.float32), cv2.RANSAC, 5.0)
    return matrix


def legibility(crop: np.ndarray) -> dict:
    """Three cheap, objective proxies for how readable this crop is."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    contrast = float(grey.std())

    # x-height from the ink itself: threshold, then take the median height of the
    # connected components that look like glyphs rather than specks or the frame.
    _, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, count)
               if 3 < stats[i, cv2.CC_STAT_HEIGHT] < crop.shape[0] * 0.9
               and stats[i, cv2.CC_STAT_AREA] > 12]
    x_height = float(np.median(heights)) if heights else 0.0

    return {"x_height_px": round(x_height, 1), "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 1), "crop_px": [crop.shape[1], crop.shape[0]]}


def crop_cell(photo: np.ndarray, matrix: np.ndarray, cell: dict) -> np.ndarray | None:
    """Warp one cell out of the photo, keeping its printed aspect ratio."""
    x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
    pad_x, pad_y = w * CROP_PAD, h * CROP_PAD
    corners = np.array([[x + pad_x, y + pad_y], [x + w - pad_x, y + pad_y],
                        [x + w - pad_x, y + h - pad_y], [x + pad_x, y + h - pad_y]],
                       np.float32)
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    # Output size follows the cell's size *in this photo*, so a distant sheet
    # yields a small crop and stays hard. Resampling it up would hand the reader
    # back the resolution the distance took away.
    width = int(max(np.linalg.norm(projected[1] - projected[0]),
                    np.linalg.norm(projected[2] - projected[3])))
    height = int(max(np.linalg.norm(projected[3] - projected[0]),
                     np.linalg.norm(projected[2] - projected[1])))
    if width < 16 or height < 8:
        return None
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    return cv2.warpPerspective(photo, cv2.getPerspectiveTransform(projected, destination),
                               (width, height))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", required=True, help="directory of sheet photographs")
    parser.add_argument("--sheets", default="sheets", help="where sheets.py wrote its manifest")
    parser.add_argument("--out", default="real")
    args = parser.parse_args()

    sheets = json.loads((pathlib.Path(args.sheets) / "manifest.json").read_text())
    by_sheet = {s["sheet"]: s for s in sheets["sheets"]}

    out = pathlib.Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    stimuli, skipped = [], []
    photos = sorted(p for p in pathlib.Path(args.photos).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})

    for photo_path in photos:
        photo = cv2.imread(str(photo_path))
        if photo is None:
            skipped.append({"photo": photo_path.name, "why": "unreadable"})
            continue

        found = detect_markers(photo)
        if len(found) < 2:
            skipped.append({"photo": photo_path.name, "why": f"{len(found)} markers"})
            continue

        # The sheet index is not in the photo, so it comes from the filename:
        # sheet00_3m_daylight.jpg → sheet 0.
        stem = photo_path.stem.lower()
        sheet_id = next((i for i in by_sheet if f"sheet{i:02d}" in stem or f"sheet_{i:02d}" in stem), None)
        if sheet_id is None:
            skipped.append({"photo": photo_path.name, "why": "no sheetNN in filename"})
            continue

        sheet = by_sheet[sheet_id]
        matrix = homography(found, sheet)
        if matrix is None:
            skipped.append({"photo": photo_path.name, "why": "homography failed"})
            continue

        for index, cell in enumerate(sheet["cells"]):
            crop = crop_cell(photo, matrix, cell)
            if crop is None:
                continue
            name = f"{photo_path.stem}_{index:02d}_{cell['condition']}.png"
            cv2.imwrite(str(out / "images" / name), crop)
            stimuli.append({"file": name, "photo": photo_path.name, "sheet": sheet_id,
                            "condition": cell["condition"], "text": cell["text"],
                            "source_word": cell["source_word"],
                            "markers_seen": len(found), **legibility(crop)})

    manifest = {"source": "printed sheets, photographed", "photos": len(photos),
                "count": len(stimuli), "skipped": skipped, "stimuli": stimuli}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"{len(stimuli)} crops from {len(photos) - len(skipped)}/{len(photos)} photos")
    if stimuli:
        heights = [s["x_height_px"] for s in stimuli]
        print(f"x-height px: min {min(heights):.0f}  median {np.median(heights):.0f}  "
              f"max {max(heights):.0f}")
    for entry in skipped:
        print(f"  skipped {entry['photo']}: {entry['why']}")


if __name__ == "__main__":
    main()

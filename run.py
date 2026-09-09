#!/usr/bin/env python3
"""Run one reader over the stimulus set and record every transcription.

    python run.py --reader easyocr --stimuli stimuli --out predictions/easyocr.json

Raw outputs are stored verbatim alongside the normalised form. Scoring happens
separately, in `score.py`, so a change to a metric never means re-running a model.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from readers import READERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", required=True, choices=sorted(READERS))
    parser.add_argument("--stimuli", default="stimuli")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", help="override the reader's default checkpoint")
    parser.add_argument("--limit", type=int, help="stop after this many stimuli")
    parser.add_argument("--prompt", choices=["neutral", "strict", "primed"],
                        help="prompt style, for readers that take one")
    args = parser.parse_args()

    root = pathlib.Path(args.stimuli)
    manifest = json.loads((root / "manifest.json").read_text())
    stimuli = manifest["stimuli"][:args.limit] if args.limit else manifest["stimuli"]

    options = {}
    if args.model_id:
        options["model_id"] = args.model_id
    if args.prompt:
        options["prompt"] = args.prompt
    reader = READERS[args.reader](**options)
    print(f"reader: {reader.name} ({reader.family}) · {len(stimuli)} stimuli", flush=True)

    records, started = [], time.perf_counter()
    for index, item in enumerate(stimuli, start=1):
        path = root / "images" / item["file"]
        began = time.perf_counter()
        try:
            prediction = reader(str(path))
            error = None
        except Exception as exc:                       # a reader failing one image
            prediction, error = "", f"{type(exc).__name__}: {exc}"
        records.append({**item, "prediction": prediction, "error": error,
                        "ms": round((time.perf_counter() - began) * 1000, 1)})
        if index % 25 == 0 or index == len(stimuli):
            print(f"  {index}/{len(stimuli)}  {(time.perf_counter() - started):.0f}s", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reader": reader.name, "family": reader.family,
                               "stimuli": args.stimuli, "records": records},
                              indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

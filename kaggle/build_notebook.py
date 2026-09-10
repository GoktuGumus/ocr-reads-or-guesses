#!/usr/bin/env python3
"""Generate the Kaggle notebook from source, and compile every cell first.

    python kaggle/build_notebook.py

Notebooks are JSON with source code stored as lists of strings, which makes them
miserable to review and easy to break in ways that only appear when a cell runs
twenty minutes into a session. Writing them from a script means the code lives in
one place, and `compile()` catches a syntax error here rather than on Kaggle.

The notebook clones this repository rather than pasting `score.py` into a cell:
one definition of the metrics, and a notebook that cannot drift from the results
in the README.
"""
from __future__ import annotations

import json
import pathlib

REPO = "https://github.com/GoktuGumus/ocr-reads-or-guesses"
DATA = "/kaggle/input/ocr-reads-or-guesses"

CELLS: list[tuple[str, str]] = [
    ("markdown", f"""# Reads or guesses?

A generative OCR model that misreads a word tells you it failed. One that
*rewrites* it does not — the output is a well-formed word, spelled correctly,
arriving with the same confidence as every correct answer. In an annotation
pipeline that is the worse failure, because nothing downstream can see it.

Shown `KAVBAK` — one glyph off the real word `KAVŞAK`, and unambiguous in the
pixels — a reader returns `KAVBAK` and a guesser returns `KAVŞAK`. The rate at
which that happens is the **repair rate**.

This notebook reproduces every number in [{REPO.split('/')[-1]}]({REPO}) from the
raw readings: 36,000 transcriptions across a CTC baseline and two model sizes ×
three prompts, plus a second arm on plates drawn as plates and a third on sheets
put through a camera model.

Nothing here needs a GPU. The models already ran; this is the analysis."""),

    ("code", f"""import json, pathlib, sys, subprocess

# Clone the repository so the metrics come from score.py rather than from a copy
# of it pasted into this notebook. One definition, no drift.
if not pathlib.Path("ocr-reads-or-guesses").exists():
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
sys.path.insert(0, "ocr-reads-or-guesses")

from score import summarise

DATA = pathlib.Path("{DATA}")
ARMS = {{"synthetic": "predictions", "drawn plates": "predictions_plate",
        "camera model": "predictions_sim"}}

reports = {{}}
for arm, folder in ARMS.items():
    reports[arm] = {{}}
    for path in sorted((DATA / folder).glob("*.json")):
        data = json.loads(path.read_text())
        reports[arm][data["reader"]] = summarise(data["records"])
    print(f"{{arm:<14}} {{len(reports[arm])}} readers, "
          f"{{sum(r['n'] for r in reports[arm].values()):,}} readings")"""),

    ("code", '''NAMES = {"easyocr": "EasyOCR (CTC)"}
def label(reader):
    if reader in NAMES:
        return NAMES[reader]
    _, model, prompt = reader.split(":")
    return f"{model.replace('Qwen2.5-VL-', '').replace('-Instruct', '')} {prompt}"

ORDER = ["easyocr", "qwen-vl:Qwen2.5-VL-3B-Instruct:strict",
         "qwen-vl:Qwen2.5-VL-3B-Instruct:neutral", "qwen-vl:Qwen2.5-VL-3B-Instruct:primed",
         "qwen-vl:Qwen2.5-VL-7B-Instruct:strict", "qwen-vl:Qwen2.5-VL-7B-Instruct:primed"]

def table(report, title):
    print(f"\\n{title}")
    print(f"{'reader':<16}{'exact':>8}{'CER':>8}{'format':>8}{'repair':>9}"
          f"{'plates':>9}{'diacritic':>11}")
    print("-" * 69)
    for reader in ORDER:
        entry = report.get(reader)
        if not entry:
            continue
        o, c = entry["overall"], entry["by_condition"]
        plate = c.get("plate_valid", {}).get("exact")
        print(f"{label(reader):<16}{o['exact']:>8.1%}{o['cer']:>8.3f}{o['format_ok']:>8.0%}"
              f"{c.get('perturbed', {}).get('repair', 0):>9.1%}"
              f"{(f'{plate:.1%}' if plate is not None else '-'):>9}"
              f"{o['diacritic_loss']:>11.1%}")

for arm, report in reports.items():
    table(report, arm)'''),

    ("markdown", """## Prior-pull scales with the model

Repair is not a fixed property of a model. It moves with the prompt, and it moves
with the size — in the same direction, and it is invisible in the accuracy column
that most people would read instead."""),

    ("code", '''def repair_curve(report, reader):
    entry = report.get(reader)
    if not entry:
        return [], []
    band = entry["by_difficulty"] or entry["by_legibility"]
    keys = list(band)
    return keys, [(band[k]["repair"] or 0) for k in keys]

print(f"{'reader':<16}" + "".join(f"{k:>9}" for k in
      list(reports["synthetic"]["easyocr"]["by_difficulty"])))
print("-" * 61)
for reader in ORDER:
    keys, values = repair_curve(reports["synthetic"], reader)
    if keys:
        print(f"{label(reader):<16}" + "".join(f"{v:>9.1%}" for v in values))'''),

    ("code", '''import matplotlib.pyplot as plt

SERIES = [("qwen-vl:Qwen2.5-VL-3B-Instruct:strict", "3B strict", "#2a78d6", "solid"),
          ("qwen-vl:Qwen2.5-VL-3B-Instruct:primed", "3B primed", "#2a78d6", (0, (1, 1.6))),
          ("qwen-vl:Qwen2.5-VL-7B-Instruct:strict", "7B strict", "#d1521f", "solid"),
          ("qwen-vl:Qwen2.5-VL-7B-Instruct:primed", "7B primed", "#d1521f", (0, (1, 1.6)))]

figure, axis = plt.subplots(figsize=(8, 4.4))
for reader, name, colour, dash in SERIES:
    keys, values = repair_curve(reports["synthetic"], reader)
    axis.plot([float(k) for k in keys], values, color=colour, linestyle=dash,
              marker="o", linewidth=2, label=name)
axis.set_xlabel("image degradation")
axis.set_ylabel("repair rate")
axis.set_title("The reader returns the word it expected", loc="left")
axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axis.spines[["top", "right"]].set_visible(False)
axis.legend(frameon=False)
plt.tight_layout()
plt.show()'''),

    ("markdown", """## The prior does not wait for the pixels to fail

The smaller model only overrides a glyph once the glyph stops being legible. The
larger one, primed with its domain, does not wait — it rewrites clean, undegraded
words. Those are the cases an annotation pipeline is least likely to review."""),

    ("code", '''clean = json.loads((DATA / "predictions/qwen-7b-primed.json").read_text())
shown = [r for r in clean["records"]
         if r["condition"] == "perturbed" and r["difficulty"] == 0.0
         and r["prediction"].strip() == r["source_word"]]
print(f"{len(shown)} of 200 undegraded perturbed words came back as the original\\n")
for record in shown[:8]:
    print(f"  shown {record['text']:<12} → {record['prediction']}")'''),

    ("markdown", """## Priming does not degrade the task — it replaces it

Told *"this is a Turkish road sign, read the Turkish word on it"*, the model finds
the only word-shaped token on a licence plate and hands that back. Not a misread:
an obeyed instruction, with the digits dropped."""),

    ("code", '''plates = [r for r in clean["records"]
          if r["condition"].startswith("plate") and r["difficulty"] == 0.0]
for record in plates[:6]:
    print(f"  {record['text']:<14} → {record['prediction']}")
print(f"\\nexact match on plates: "
      f"{sum(r['prediction'].strip() == r['text'] for r in plates) / len(plates):.1%}")'''),

    ("markdown", """## Through a camera model

The third arm puts the printed sheets through a simulated camera — placed at a
distance in a 12 MP frame, tilted, lit unevenly, blurred, with sensor noise and
JPEG artefacts — and cuts the cells back out using the ArUco markers on the page.

Difficulty is no longer a knob, so it is measured: every crop carries the height
of its glyphs in pixels. If accuracy tracks that height, the proxy is real."""),

    ("code", '''sim = reports["camera model"]
bands = list(sim["easyocr"]["by_legibility"])
counts = [sim["easyocr"]["by_legibility"][b]["n"] for b in bands]
print(f"{'reader':<16}" + "".join(f"{b:>10}" for b in bands))
print(f"{'(crops)':<16}" + "".join(f"{n:>10}" for n in counts))
print("-" * 66)
for reader in ORDER:
    entry = sim.get(reader)
    if entry:
        print(f"{label(reader):<16}" + "".join(
            f"{entry['by_legibility'][b]['exact']:>10.0%}" for b in bands))'''),

    ("code", '''from PIL import Image

manifest = json.loads((DATA / "real_sim/manifest.json").read_text())
picks = []
for low, high in [(30, 60), (20, 26), (13, 17), (8, 12)]:
    matches = [s for s in manifest["stimuli"]
               if s["condition"] == "word" and low <= s["x_height_px"] < high]
    if matches:
        picks.append(max(matches, key=lambda s: s["x_height_px"]))

figure, axes = plt.subplots(len(picks), 1, figsize=(7, 1.5 * len(picks)))
for axis, item in zip(axes, picks):
    axis.imshow(Image.open(DATA / "real_sim/images" / item["file"]))
    axis.set_title(f"{item['text']} — x-height {item['x_height_px']:.0f} px, "
                   f"sharpness {item['sharpness']:.0f}", loc="left", fontsize=9)
    axis.axis("off")
plt.tight_layout()
plt.show()'''),

    ("markdown", f"""## What to take from this

1. **Do not prime the model with domain context.** The costliest thing measured
   here, in both accuracy and hallucination, and the damage grows with size.
2. **Constrain the output format explicitly.** The strict prompt was best on every
   axis, including the ones it was not aimed at — and held at 0.0% repair in every
   legibility band of the camera-model arm.
3. **A bigger model is not a safer one.** 7B strict reads better than 3B strict
   and repairs six times as often.
4. **Legibility is a useful review trigger, not a sufficient one.** Repair is zero
   until legibility fails — except for the primed 7B model, where it is 4.5% on
   clean images.
5. **Disagreement is the cheap flag.** The CTC reader never repaired once in
   36,000 readings. Where it and a VLM disagree, one of them is guessing.

Code, stimulus generator and printable sheets: [{REPO}]({REPO})"""),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        if kind == "code":
            compile(source, "<cell>", "exec")          # fail here, not on Kaggle
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": source.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": source.splitlines(keepends=True)})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                        "language": "python"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "reads-or-guesses.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    code = sum(c["cell_type"] == "code" for c in build()["cells"])
    print(f"wrote {out} · {code} code cells, all compiled")

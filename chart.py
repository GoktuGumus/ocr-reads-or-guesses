#!/usr/bin/env python3
"""The two curves that carry the result.

Left: how often the reader returns the word it expected instead of the word it
was shown, against how degraded the image is. Right: what that costs in plain
accuracy. Both against the same x-axis, because the point is that they move
together — the prompt that induces the most repair is also the one that reads
worst, and neither number means much without the other.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Each series carries a dash pattern as well as a hue. Two reasons: the strict
# and neutral curves sit on top of each other for most of the range, and a hue
# alone would hide one behind the other; and identity that survives a colourblind
# reader should not rest on colour.
SERIES = [("strict", "strict — transcribe exactly", "#2a78d6", "solid"),
          ("neutral", "neutral — what text is this?", "#eb6834", (0, (5, 2))),
          ("primed", "primed — it's a Turkish sign", "#1baf7a", (0, (1, 1.6)))]


def pick(report: dict, needle: str) -> dict | None:
    for name, entry in report.items():
        if needle in name:
            return entry
    return None


def draw(report: dict, out: pathlib.Path, model_label: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)

    for axis in axes:
        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=1)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=9, length=0)
        axis.set_xlabel("image degradation", color=MUTED, fontsize=9)

    baseline = pick(report, "easyocr")
    for panel, (metric, label) in enumerate([("repair", "repair rate"),
                                             ("exact", "exact match")]):
        axis = axes[panel]
        endpoints = []
        for key, label_text, colour, dash in SERIES:
            entry = pick(report, key)
            if not entry:
                continue
            points = sorted(entry["by_difficulty"].items(), key=lambda kv: float(kv[0]))
            x = [float(k) for k, _ in points]
            y = [(v[metric] or 0) for _, v in points]
            axis.plot(x, y, color=colour, linewidth=2, marker="o", markersize=8,
                      linestyle=dash, markeredgecolor=SURFACE, markeredgewidth=2,
                      label=label_text, clip_on=False, zorder=3)
            endpoints.append([y[-1], x[-1], key])

        # Curves that finish at the same value would stack their labels on top of
        # each other; push them apart in the order they end.
        span = max((e[0] for e in endpoints), default=1) or 1
        endpoints.sort()
        for index in range(1, len(endpoints)):
            gap = endpoints[index][0] - endpoints[index - 1][0]
            if gap < span * 0.07:
                endpoints[index][0] = endpoints[index - 1][0] + span * 0.07
        for label_y, label_x, key in endpoints:
            axis.annotate(key, xy=(label_x, label_y), xytext=(7, 0),
                          textcoords="offset points", color=MUTED, fontsize=9,
                          va="center", zorder=4)

        if baseline:
            points = sorted(baseline["by_difficulty"].items(), key=lambda kv: float(kv[0]))
            axis.plot([float(k) for k, _ in points],
                      [(v[metric] or 0) for _, v in points],
                      color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2,
                      label="easyocr (CTC baseline)")

        axis.set_ylabel(label, color=MUTED, fontsize=9)
        axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axis.set_xlim(-0.03, 1.12)

    axes[0].set_title("the reader returns the word it expected", color=INK,
                      fontsize=11, loc="left", pad=10)
    axes[1].set_title("and reads worse while doing it", color=INK,
                      fontsize=11, loc="left", pad=10)
    axes[1].legend(frameon=False, labelcolor=MUTED, fontsize=8.5, loc="lower left")

    figure.suptitle(f"One model, three prompts: {model_label}", color=INK, fontsize=13,
                    x=0.006, ha="left", y=1.01)
    figure.tight_layout()
    figure.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="report.json")
    parser.add_argument("--out", default="docs/prompt-effect.png")
    parser.add_argument("--label", default="Qwen2.5-VL-3B")
    parser.add_argument("--filter", default="3B", help="only readers whose name contains this")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    subset = {k: v for k, v in report.items() if args.filter in k or "easyocr" in k}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    draw(subset, pathlib.Path(args.out), args.label)

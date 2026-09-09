#!/usr/bin/env python3
"""Score transcriptions, with the metrics the question actually needs.

Exact match and character error rate are table stakes. They tell you *how often*
a reader is wrong, and nothing about *how* — and in an annotation pipeline the
how is the entire risk. Three failure modes are worth separating:

  repair          the reader was shown KAVBAK and returned KAVŞAK — a real word,
                  one edit away, and the one the language model expected. Nothing
                  downstream can catch this: the label is well-formed, confident
                  and wrong.

  diacritic loss  Ş read as S, Ö as O. Common, quiet, and specific to writing
                  systems the model saw less of. In Turkish it changes meaning.

  plate repair    an invalid province code corrected into a valid one. The same
                  failure as `repair`, in a format where the prior is structural
                  rather than lexical, and where the consequence is a plate that
                  belongs to a different vehicle.

A reader that fails by emitting noise is safe: the error is visible. A reader
that fails by emitting plausible text is dangerous, and these metrics separate
the two.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from stimuli import PROVINCES_VALID, WORDS

LEXICON = {w.upper() for w in WORDS}
DIACRITICS = str.maketrans({"Ş": "S", "Ç": "C", "Ğ": "G", "Ü": "U", "Ö": "O", "İ": "I"})


TURKISH_CHARS = set("ŞÇĞÜÖİ")

# Openers a chatty model puts in front of the answer. Matched against the
# diacritic-stripped text, because normalising Turkish casing turns "IN" into
# "İN" and the pattern would miss it otherwise. Every rule here is independent
# of the target string — an extractor that peeked at the answer would be
# marking its own homework.
PREAMBLES = (
    "THE TEXT IN THE IMAGE IS", "THE TEXT IN THE IMAGE READS",
    "THE TEXT ON THE SIGN IS", "THE TEXT READS", "THE IMAGE SHOWS",
    "THE IMAGE CONTAINS THE TEXT", "THE WORD IS", "IT SAYS", "THE TEXT IS",
    "GORUNTUDEKI METIN", "RESIMDEKI METIN",
)


def strip_diacritics(text: str) -> str:
    return text.translate(DIACRITICS)


def extract(raw: str) -> tuple[str, bool]:
    """Pull the transcription out of a conversational answer.

    Returns (text, was_already_clean). The second value is a metric in its own
    right: in an annotation pipeline, an answer that needs parsing is an answer
    that can be parsed wrongly, and half of them needing it is a pipeline
    problem regardless of how well the model read.
    """
    flat = strip_diacritics(raw)

    quote = re.search(r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{1,40})", flat)
    if quote:
        return raw[quote.start(1):quote.end(1)].strip(), False

    for preamble in PREAMBLES:
        if flat.startswith(preamble):
            return raw[len(preamble):].strip(" :.-\u2014"), False

    return raw, True


def strip_band(text: str) -> tuple[str, bool]:
    """Drop the country band from a plate reading.

    A plate carries a blue TR band that is part of the object and not part of
    the registration. Readers with a text detector find it and return it, which
    would fail exact match on a plate whose digits were read perfectly. That is
    worth knowing — it is why `band_read` is a metric — but it is not a reading
    error, so it is removed before the string is compared.

    Only a standalone TR at either end goes, and never the last two tokens: a
    plate really can read `34 TR 123`, and a reader that returns nothing but
    `TR` has failed and must keep failing.
    """
    tokens, stripped = text.split(), False
    while len(tokens) > 2 and tokens[0] == "TR":
        tokens.pop(0)
        stripped = True
    while len(tokens) > 2 and tokens[-1] == "TR":
        tokens.pop()
        stripped = True
    return (" ".join(tokens), stripped) if stripped else (text, False)


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(truth: str, prediction: str) -> float:
    return edit_distance(truth, prediction) / max(len(truth), 1)


def province_of(text: str) -> str | None:
    head = text.strip().split(" ")[0] if " " in text else text[:2]
    return head if head.isdigit() else None


def turkish_count(text: str) -> int:
    return sum(ch in TURKISH_CHARS for ch in text)


def classify(record: dict) -> dict:
    """One record in, its failure modes out."""
    truth = record["text"]
    prediction, clean = extract(record["prediction"])
    band_read = False
    if record["condition"].startswith("plate"):
        prediction, band_read = strip_band(prediction)
    exact = prediction == truth
    flags = {
        "exact": exact,
        "cer": cer(truth, prediction),
        "empty": prediction == "",
        "format_ok": clean,
        "band_read": band_read,
        # A wrong answer that is nevertheless a real word: plausible, so invisible.
        "plausible_error": (not exact and prediction in LEXICON),
        # Same letters, different diacritics. Splitting the direction matters:
        # dropping them is a recognition failure, adding them is the model
        # making the string more Turkish than the pixels were.
        "diacritic_loss": (not exact
                           and strip_diacritics(prediction) == strip_diacritics(truth)
                           and turkish_count(prediction) < turkish_count(truth)),
        "turkified": (not exact
                      and strip_diacritics(prediction) == strip_diacritics(truth)
                      and turkish_count(prediction) > turkish_count(truth)),
        "repair": False,
        "plate_repair": False,
    }
    if record["condition"] == "perturbed" and record.get("source_word"):
        # The single measurement this benchmark exists for.
        flags["repair"] = prediction == record["source_word"]
    if record["condition"] == "plate_invalid":
        shown, read = province_of(truth), province_of(prediction)
        flags["plate_repair"] = bool(read and read != shown
                                     and int(read) in PROVINCES_VALID)
    return flags


def rate(scored, key):
    return round(sum(s[key] for s in scored) / max(len(scored), 1), 4)


def summarise(records: list[dict]) -> dict:
    scored_all = [classify(r) for r in records]

    by_condition = {}
    for condition in sorted({r["condition"] for r in records}):
        pairs = [(r, s) for r, s in zip(records, scored_all) if r["condition"] == condition]
        subset = [s for _, s in pairs]
        entry = {"n": len(subset)}
        for key in ("exact", "cer", "format_ok", "band_read", "diacritic_loss",
                    "turkified", "plausible_error", "empty"):
            entry[key] = rate(subset, key)
        if condition == "perturbed":
            entry["repair"] = rate(subset, "repair")
        if condition == "plate_invalid":
            entry["plate_repair"] = rate(subset, "plate_repair")
        by_condition[condition] = entry

    by_difficulty = {}
    for difficulty in sorted({r["difficulty"] for r in records}):
        pairs = [(r, s) for r, s in zip(records, scored_all) if r["difficulty"] == difficulty]
        subset = [s for _, s in pairs]
        perturbed = [s for r, s in pairs if r["condition"] == "perturbed"]
        by_difficulty[str(difficulty)] = {
            "exact": rate(subset, "exact"), "cer": rate(subset, "cer"),
            "repair": rate(perturbed, "repair") if perturbed else None,
            "turkified": rate(subset, "turkified"),
        }

    overall = {key: rate(scored_all, key) for key in
               ("exact", "cer", "format_ok", "band_read", "diacritic_loss",
                "turkified", "plausible_error")}
    return {"overall": overall, "by_condition": by_condition,
            "by_difficulty": by_difficulty, "n": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args()

    report = {}
    for path in args.predictions:
        data = json.loads(pathlib.Path(path).read_text())
        report[data["reader"]] = {"family": data["family"], **summarise(data["records"])}

    def short(name):
        return name.replace("Qwen2.5-VL-", "").replace("-Instruct", "").replace("qwen-vl:", "qwen ")

    print(f"{'reader':<20}{'exact':>8}{'CER':>7}{'format':>8}{'repair':>8}"
          f"{'plate fix':>10}{'band':>7}{'türkçeleş':>11}{'aksan yit':>11}")
    print("-" * 90)
    for reader, r in report.items():
        o, c = r["overall"], r["by_condition"]
        print(f"{short(reader):<20}{o['exact']:>8.1%}{o['cer']:>7.3f}{o['format_ok']:>8.0%}"
              f"{c.get('perturbed', {}).get('repair', 0):>8.1%}"
              f"{c.get('plate_invalid', {}).get('plate_repair', 0):>10.1%}"
              f"{o['band_read']:>7.1%}"
              f"{o['turkified']:>11.1%}{o['diacritic_loss']:>11.1%}")

    print(f"\n{'reader':<20}{'condition':<16}{'exact':>8}{'CER':>7}{'plausible':>11}")
    print("-" * 62)
    for reader, r in report.items():
        for condition, c in r["by_condition"].items():
            print(f"{short(reader):<20}{condition:<16}{c['exact']:>8.1%}{c['cer']:>7.3f}"
                  f"{c['plausible_error']:>11.1%}")

    print(f"\n{'reader':<20}{'difficulty':<12}{'exact':>8}{'repair':>9}{'türkçeleş':>11}")
    print("-" * 60)
    for reader, r in report.items():
        for difficulty, d in r["by_difficulty"].items():
            repair = f"{d['repair']:.1%}" if d["repair"] is not None else "-"
            print(f"{short(reader):<20}{difficulty:<12}{d['exact']:>8.1%}{repair:>9}"
                  f"{d['turkified']:>11.1%}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

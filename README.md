# Reads or guesses?

A generative OCR model that misreads a word tells you it failed. A generative OCR
model that *rewrites* it does not: the output is a well-formed word, spelled
correctly, arriving with the same confidence as every correct answer. In an
automatic annotation pipeline that is the worse failure by a wide margin, because
nothing downstream can see it.

This measures how often that happens, and what makes it happen.

## The instrument

Take a real word, swap one glyph for a visually distant one, and render both the
same way:

| shown | what a reader returns | what a guesser returns |
|---|---|---|
| `KAVŞAK` | KAVŞAK | KAVŞAK |
| `KAVBAK` | KAVBAK | **KAVŞAK** |

The swap is chosen so the pixels are unambiguous — Ş to B, not O to 0. A model
that returns `KAVŞAK` for the second image is not misreading a blurred glyph. It
is overriding a clear one with what it expected to see. The rate at which that
happens is the **repair rate**, and it is the number this repository exists to
produce.

Six conditions per item, all rendered with byte-identical degradation so that
only the linguistic plausibility of the string varies:

| condition | example | prior available |
|---|---|---|
| `word` | KAVŞAK | a real Turkish word |
| `perturbed` | KAVBAK | one glyph off a real word — the prior is *wrong* |
| `pseudo` | SÖBETÜ | Turkish phonotactics, not a word |
| `random` | XQKZMR | none |
| `plate_valid` | 34 ABC 123 | valid Turkish plate format |
| `plate_invalid` | 99 ABC 123 | invalid province code — structural prior is wrong |

The plate conditions are drawn as plates, not as text on a field:

![plate stimulus](docs/plate-stimulus.png)

That is not decoration. A structural prior — *99 is not a province, three letters
are never followed by five digits* — can only fire if the reader knows it is
looking at a plate, and a bare string does not say so. Letter and digit counts
are drawn from the real layout table (`99 X 9999`, `99 XX 999`, `99 XXX 99`, and
so on up to nine characters), and both arms span every layout, so the province is
the only thing separating a valid plate from an invalid one. `--plate-style text`
renders them bare, which turns the frame itself into a second axis.

Five degradation levels, from clean to barely legible. Degradation is applied by
downscaling and scaling back before blur and noise, because real OCR difficulty
is mostly small or distant text, and blur alone on a large glyph barely registers.

## What it measures

Beyond exact match and CER, four things that separate *how* a reader fails:

| metric | what it catches |
|---|---|
| **repair** | shown a perturbed word, returned the original — the invisible failure |
| **turkified** | added diacritics that were not in the image, making the string more Turkish than the pixels were |
| **diacritic loss** | dropped Ş, Ç, Ğ, Ü, Ö, İ — quiet, common, and meaning-changing in Turkish |
| **format compliance** | the answer arrived as the text, not wrapped in `The text in the image is "…"` |

Format compliance matters for a reason that is not academic: an answer that needs
parsing is an answer that can be parsed wrongly.

## Results

*Preliminary run: 50 items × 6 conditions × 6 degradation levels = 1,800 images
per configuration, with plates rendered as bare text. A 200-item run with plates
drawn as plates is in progress; these numbers will be replaced by it.*

![prompt effect](docs/prompt-effect.png)

| reader | exact | CER | format | **repair** | turkified | diacritic loss |
|---|---|---|---|---|---|---|
| EasyOCR (CTC) | 69.8% | 0.117 | 100% | **0.0%** | 0.3% | 7.6% |
| Qwen2.5-VL-3B · strict | **84.4%** | **0.040** | 100% | 0.3% | 0.3% | 5.7% |
| Qwen2.5-VL-3B · neutral | 78.4% | 0.060 | **58%** | 0.3% | 0.6% | 6.6% |
| Qwen2.5-VL-3B · primed | 62.6% | 0.195 | 100% | **2.0%** | **1.3%** | 5.4% |

Three prompts, one model, the same images:

```
strict    Transcribe the text exactly as it appears… do not correct spelling.
neutral   What text is in this image?
primed    This is a Turkish road sign. Read the Turkish word on it.
```

### Repair only happens where the pixels stop deciding

| degradation | EasyOCR | strict | neutral | primed |
|---|---|---|---|---|
| 0.0 – 0.75 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.9 | 0.0% | 0.0% | 0.0% | **4.0%** |
| 1.0 | 0.0% | 2.0% | 2.0% | **8.0%** |

Nothing repairs anything while the text is legible. The prior takes over exactly
at the point where the evidence runs out — which is the behaviour you would want
from a Bayesian reasoner and the behaviour you must not have in a labelling
pipeline, because the pipeline cannot tell the two regimes apart.

### Prior-pull is a setting, not a property

The same model repairs at 0.3% or 2.0% depending on one sentence of context.
Priming it with the domain — the thing most people do to *help* — quadrupled the
repair rate and cost 22 points of exact match.

It also breaks plates:

| condition | strict | primed |
|---|---|---|
| `plate_valid` | 90.0% | **28.3%** |
| CER | 0.015 | 0.457 |

Told it is looking at a Turkish road sign, the model tries to read `34 EYA 382`
as a word.

### The CTC baseline never repairs, and never survives

EasyOCR's repair rate is 0.0% everywhere. It has no lexical prior strong enough
to override a glyph — which is exactly why it is safe, and why it collapses to
9.7% exact match at the hardest level where the VLM still manages 38%.

Robustness to degradation and willingness to hallucinate turn out to be the same
capability seen from two sides.

## What to do with this

For an annotation pipeline that uses a VLM as an OCR engine:

1. **Do not prime the model with domain context.** It is the single most costly
   thing measured here, in both accuracy and hallucination.
2. **Constrain the output format explicitly.** The strict prompt was best on
   every axis, including the ones it was not aimed at.
3. **Treat degraded crops differently.** Repair rate is zero until legibility
   fails; a legibility estimate is therefore a usable trigger for human review.
4. **Consider disagreement as a flag.** A CTC reader and a VLM fail in different
   directions. Where they disagree is where one of them is guessing.

## Running it

```bash
pip install -r requirements.txt

python stimuli.py --items 200 --out stimuli          # --plate-style text for bare plates
python run.py --reader easyocr --out predictions/easyocr.json
python run.py --reader qwen-vl --prompt strict --out predictions/qwen-strict.json
python score.py predictions/*.json --json report.json
python chart.py --report report.json
python test_stimuli.py                              # the generator's own checks
```

Stimulus generation needs nothing but Pillow and NumPy. Each reader pulls its own
dependencies only when used.

## Prior work

The counterfactual-perturbation method is not new. [Do VLMs Read or
Rewrite?](https://arxiv.org/html/2607.21617), [Reading or
Guessing?](https://arxiv.org/pdf/2605.27750) on Ancient Greek editions, and
[Visual Merit or Linguistic Crutch?](https://arxiv.org/html/2601.03714v2) on
DeepSeek-OCR all establish that end-to-end models lean on language priors where
pipeline OCR does not.

What is added here: the measurement in **Turkish**, where agglutination makes the
space of plausible strings enormous and the diacritics give a second, quieter
failure mode; the **licence-plate** conditions, where the prior is structural
rather than lexical; the **prompt as an experimental variable** rather than a
fixed detail; and a framing aimed at the person running an annotation pipeline
rather than at the leaderboard.

## What this is, and what it isn't

It **is** a controlled measurement with matched stimuli and a reproducible
generator.

It **is not** a claim about VLMs in general: it tests two model sizes from one
family, at one point in time, on synthetic renderings. Synthetic text is cleaner
and more uniform than a photograph of a sign, and the effect on real imagery is
the obvious next thing to measure. Repair rates in the low single digits also
rest on small cell counts — the direction is consistent across every difficulty
level, the exact percentages are not precise.

## Licence

MIT.

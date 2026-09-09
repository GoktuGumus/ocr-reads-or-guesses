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

![difficulty ladder](docs/difficulty-ladder.png)

Every condition is set at the same point size, so the same difficulty means the
same pixels per character whatever the characters spell. That is the control the
whole benchmark rests on, and `test_stimuli.py` asserts it: a first draft of the
plate rendering sized the frame to the canvas and shrank the text to fit, which
made plate glyphs 1.6x taller than word glyphs. The plate conditions would have
scored better for having more resolution, and it would have read as a prior.

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

200 items × 6 conditions × 5 degradation levels = 6,000 images per configuration.
Six configurations: a CTC baseline, and two model sizes × three prompts.

![prompt effect](docs/prompt-effect.png)

| reader | exact | CER | format | **repair** | turkified | diacritic loss |
|---|---|---|---|---|---|---|
| EasyOCR (CTC) | 56.7% | 0.153 | 100% | **0.0%** | 1.0% | 13.6% |
| Qwen2.5-VL-3B · strict | 76.8% | 0.057 | 100% | 0.2% | 0.7% | 10.1% |
| Qwen2.5-VL-3B · neutral | 71.6% | 0.074 | **59%** | 0.3% | 1.1% | 10.9% |
| Qwen2.5-VL-3B · primed | 58.9% | 0.186 | 100% | 3.4% | 2.0% | 8.4% |
| Qwen2.5-VL-7B · strict | **83.9%** | **0.037** | 100% | 1.2% | 0.8% | 6.6% |
| Qwen2.5-VL-7B · primed | 49.5% | 0.294 | 100% | **7.1%** | **3.2%** | 4.3% |

Three prompts, the same images:

```
strict    Transcribe the text exactly as it appears… do not correct spelling.
neutral   What text is in this image?
primed    This is a Turkish road sign. Read the Turkish word on it.
```

### Prior-pull scales with the model

Repair rate, by how degraded the image is:

| degradation | EasyOCR | 3B strict | 3B neutral | 3B primed | 7B strict | 7B primed |
|---|---|---|---|---|---|---|
| 0.00 | 0.0% | 0.0% | 0.0% | 0.5% | 0.0% | **4.5%** |
| 0.75 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **6.5%** |
| 0.90 | 0.0% | 0.0% | 0.5% | 3.5% | 2.0% | **8.0%** |
| 0.95 | 0.0% | 0.0% | 0.0% | 3.0% | 1.5% | **7.0%** |
| 1.00 | 0.0% | 1.0% | 1.0% | **10.0%** | 2.5% | 9.5% |

The bigger model repairs more than the smaller one at every prompt — 1.2% against
0.2% under the strict prompt, 7.1% against 3.4% under the primed one. Whatever
makes 7B the better reader is the same thing that makes it the more confident
rewriter, and it does not show up in any accuracy number.

### The prior does not wait for the pixels to fail

A smaller model only overrides a glyph once the glyph stops being legible: 3B's
repair rate is zero everywhere until 0.90. The 7B model primed with a domain does
not wait. On **undegraded** images it repairs 4.5% of perturbed words:

```
shown PAPK      → PARK
shown ÇIKYŞ     → ÇIKIŞ
shown GEÇUŞ     → GEÇİŞ
shown ÇALYŞMA   → ÇALIŞMA
```

Nothing is blurred in those. The glyphs are 64pt and clean, and the model returns
a different word. This is the failure that an annotation pipeline cannot see, on
the images an annotation pipeline would consider easy.

### Priming does not degrade the task — it replaces it

Told *"this is a Turkish road sign, read the Turkish word on it"*, the 7B model
scores **0.1%** on plates. Not misread — obeyed:

| shown | 7B primed returns |
|---|---|
| `01 PEY 114` | `PEY` |
| `35 MPZ 421` | `MPZ` |
| `90 ECJ 946` | `ECJ` |

It finds the only word-shaped token on the plate and hands it back. On plates
drawn with the blue band it returns `TR`. The 3B model does the same thing more
mildly (34.7%). One sentence of well-meant context turned an OCR engine into a
word detector, and every output was well-formed.

### The frame helps the model and hurts the pipeline

The plate conditions were run twice: as bare strings, and drawn as plates.

| reader | `plate_valid` bare | drawn | band read as text |
|---|---|---|---|
| EasyOCR (CTC) | 71.9% | 67.9% | **19.9%** |
| 3B strict | 86.7% | **90.7%** | 3.0% |
| 3B neutral | 74.6% | **84.6%** | 0.1% |
| 7B strict | 89.8% | **91.1%** | 1.7% |

Drawing the frame helps every VLM — up to 10 points for the neutral prompt — and
costs the CTC reader, which reads the country band as part of the registration on
a fifth of all plates (`34 ABC 123 TR`). Those readings are scored after the band
is removed; the column is what the raw output looked like. If you run a detector
plus recogniser over plate crops, that is a real 20% contamination rate on a
string that was otherwise read perfectly.

### The structural prior never fires

The headline conditions of this benchmark, `plate_invalid`, produced almost
nothing: correcting an invalid province code into a valid one happened at **0.0%
to 0.9%** in every configuration, both renderings, at every difficulty. The
lexical prior is strong enough to override clear pixels; the structural one is
not, or is not represented at all. That is a negative result and it is reported as
one — a model that knows Turkish words evidently does not, in the same way, know
Turkish province codes.

### Word errors in Turkish are mostly diacritics

`random` — six letters, no diacritics, no prior whatsoever — is read *better* than
`word` by every VLM (81.4% against 73.0% for 3B strict). The word conditions are
not harder because of ambiguity; they are harder because they contain Ş, Ç, Ğ, Ü,
Ö and İ, and 7–26% of readings drop one. In Turkish that changes the word.

## What to do with this

For an annotation pipeline that uses a VLM as an OCR engine:

1. **Do not prime the model with domain context.** It is the single most costly
   thing measured here, in both accuracy and hallucination, and the damage grows
   with model size: the primed 7B model scored 0.1% on plates and repaired 7.1%
   of perturbed words.
2. **Constrain the output format explicitly.** The strict prompt was best on
   every axis, including the ones it was not aimed at.
3. **Do not assume a bigger model is a safer one.** 7B strict reads better than
   3B strict by seven points and repairs six times as often. Accuracy and
   prior-pull rose together, and only one of them was visible in the metrics.
4. **A legibility estimate is a useful trigger, not a sufficient one.** For the
   3B model, repair is zero until the text stops being legible. For 7B primed it
   is 4.5% on clean images. Routing only the blurry crops to review would have
   caught none of those.
5. **Consider disagreement as a flag.** A CTC reader and a VLM fail in different
   directions. Where they disagree is where one of them is guessing — and the
   CTC reader never repaired once in 36,000 readings.
6. **Strip the country band before comparing plate strings.** A detector plus
   recogniser returned it as part of the registration on 19.9% of drawn plates.

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

The plate arm re-runs only the two plate conditions, so it is a third of the
work and directly comparable to the full run:

```bash
python stimuli.py --items 200 --out stimuli_plate --conditions plate_valid plate_invalid
python run.py --reader qwen-vl --stimuli stimuli_plate --prompt strict \
    --out predictions_plate/qwen-3b-strict.json
python score.py predictions_plate/*.json --json report_plate.json
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
and more uniform than a photograph of a sign, which is what `sheets.py` and
`extract.py` exist to fix — printed sheets, photographed at four distances under
four lighting conditions, with every condition inside the same frame.

Repair rates in the low single digits rest on 200 perturbed items per difficulty
level. The direction is consistent — across both model sizes, all three prompts
and every difficulty — but a cell reading 1.5% and one reading 2.0% are not
distinguishable at this sample size. The differences the results lean on are the
large ones: 0.2% against 7.1%, 90% against 0.1%.

## Licence

MIT.

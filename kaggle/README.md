# Kaggle

- **Dataset** — [gktugm/ocr-reads-or-guesses](https://www.kaggle.com/datasets/gktugm/ocr-reads-or-guesses):
  every reading from all three arms, 51,540 transcriptions, plus the 708 crops cut
  out of the simulated photographs.
- **Notebook** — [reads-or-guesses-ocr-prior-pull](https://www.kaggle.com/code/gktugm/reads-or-guesses-ocr-prior-pull):
  rebuilds every table and chart in the top-level README from those readings, on
  CPU, in about a minute. The models already ran; this is the analysis.

The notebook clones this repository and imports `score.py` rather than carrying a
copy of the metrics, so it cannot drift from the published results. It is written
by `build_notebook.py`, which compiles every code cell before writing the file.

```bash
python kaggle/build_notebook.py
```

## Publishing

Three things about the API that cost an afternoon:

- **`KGAT_` tokens need the 2.x client.** Kaggle 1.7.x sends the key as HTTP basic
  auth and gets a flat 401; 2.x sends it as a bearer token. The client checks
  credentials lazily, so `datasets list` succeeds against a public endpoint and
  everything that actually needs auth fails.
- **The 2.x client needs Python 3.11+.** On an older interpreter pip silently
  resolves to 1.7.4.5 and you are back to the 401.
- **Datasets are created private.** `-u` makes one public, and there is no
  command to flip an existing one — it has to be deleted and recreated.

```bash
python3.12 -m venv .kgl && .kgl/bin/pip install kaggle
export KAGGLE_API_TOKEN="KGAT_…"        # Kaggle → Settings → API → Create New Token

mkdir -p kaggle/data && cp -r predictions predictions_plate predictions_sim kaggle/data/
mkdir -p kaggle/data/real_sim && cp -r real_sim/manifest.json real_sim/images kaggle/data/real_sim/
cp kaggle/dataset-metadata.json kaggle/data/

.kgl/bin/kaggle datasets create -p kaggle/data -r zip -u    # -v for a new version
.kgl/bin/kaggle kernels push -p kaggle
```

The notebook locates the dataset by searching `/kaggle/input` for a `predictions`
directory rather than hard-coding the mount path — Kaggle currently nests it at
`/kaggle/input/datasets/<owner>/<slug>`, and a path that is one segment off fails
as an empty result rather than an error, which looks exactly like a run that
worked.

`kaggle/data/` is gitignored: 39 MB of predictions and crops this repository can
regenerate.

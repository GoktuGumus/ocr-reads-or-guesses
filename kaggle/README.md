# Kaggle

Two artefacts: the raw readings as a dataset, and a notebook that rebuilds every
table and chart in the top-level README from them. Neither needs a GPU — the
models already ran.

The notebook clones this repository and imports `score.py` rather than carrying a
copy of the metrics, so it cannot drift from the published results. It is written
by `build_notebook.py`, which compiles every code cell before writing the file.

```bash
python kaggle/build_notebook.py
```

## Publishing

Needs a valid `~/.kaggle/kaggle.json` — download it from Kaggle → Settings → API →
Create New Token. Never paste the key anywhere else; the file is the only place it
belongs.

Assemble the dataset from a finished run, then push both:

```bash
mkdir -p kaggle/data && cp -r predictions predictions_plate predictions_sim kaggle/data/
mkdir -p kaggle/data/real_sim && cp -r real_sim/manifest.json real_sim/images kaggle/data/real_sim/
cp kaggle/dataset-metadata.json kaggle/data/

kaggle datasets create -p kaggle/data -r zip     # first time; -v for a new version
kaggle kernels push -p kaggle
```

`kaggle/data/` is gitignored: it is 39 MB of predictions and crops that this
repository can regenerate.

# fall-2026-AI-financial-models
Team project: fall-2026-AI-financial-models

Environment for neural option pricing, numerical model validation, historical
option-data analysis, and structured LLM test proposals. This repository currently
contains environment configuration, local Nifty data, and an EDA workflow; the earlier
synthetic benchmark described in the project handoff is not included here.

## Setup

Run these commands from the repository root with Conda installed:

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip check
python -m jupyterlab
```

The environment lives in `.conda/` inside this repository and is ignored by Git.
In VS Code, select `.conda/bin/python` as the Python interpreter and notebook
kernel (Windows: `.conda/python.exe`). JupyterLab launched from this environment
also uses its Python kernel; no global kernel registration is required.

To apply dependency changes to an existing environment:

```bash
conda env update --prefix ./.conda --file environment.yml
```

If you already have Python 3.11 and prefer `venv`, the same dependencies work with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`environment.yml` selects Python 3.11 and loads `requirements.txt` as the single
Python dependency list. It includes PyTorch, NumPy/SciPy, pandas/PyArrow,
scikit-learn, plotting, schema validation, JupyterLab, pytest, and Ruff.

On **Intel macOS / x86_64 Python**, PyTorch is pinned to **2.2.2** and NumPy to
**1.26.4**. The [PyTorch 2.2 release announcement](https://pytorch.org/blog/pytorch2-2/)
identifies 2.2.x as the last macOS x86 release series. Other platforms resolve
PyTorch 2.6 or newer within major version 2. CPU execution is sufficient for the
small pricing network; GPU support depends on the installed PyTorch build.

These dependency ranges are not an exact cross-platform lockfile. Record the
resolved packages with each experiment, along with its seeds and configuration:

```bash
python -m pip freeze > requirements-resolved.txt
```

QuantLib is optional for independent formula checks (`python -m pip install
QuantLib`). An LLM SDK or CLI can be added once the provider is chosen; no paid
API calls or model downloads are part of setup. Store any future API keys in a
local `.env` file and share only sanitized `.env.example` templates.

## Local data and experiment artifacts

Keep the downloaded Nifty ZIPs and extracted files in `Nifty Options Data/`,
which is ignored except for its root README and schema. Use `data/raw/`,
`data/interim/`, and `data/processed/`
for other local datasets; their README files and `.gitkeep` placeholders may be
committed. Keep small public reference cases in `data/reference/`.

Model weights, generated binary arrays, caches, logs, and local environments are
ignored. Notebooks, source code, small reference fixtures, experiment protocols,
and text/JSON/CSV reports remain eligible for version control. Keep bulk CSVs in
the ignored data directories. `.gitignore` does not untrack files already
committed to Git.

## Nifty data EDA

Open `notebooks/01_nifty_eda.ipynb` for the walkthrough, actual results,
charts, interpretation, and difficulties encountered. A rendered copy is saved
as `reports/nifty_eda/eda.html`. To rebuild the full audit from the raw files:

```bash
conda activate ./.conda
python scripts/nifty_eda.py --refresh-inventory
python -m pytest -q tests/test_nifty_eda.py
```

The workflow hashes all source files, excludes only verified byte-identical
CSV/TXT representations within the same archive/file group, then scans every
selected row. It checks schema changes, numeric parsing, OHLC consistency,
timestamps, within-file duplicates, coverage, and observation gaps. All flagged
records remain in the audited data. Price/volume summary statistics use the full
selected dataset; the distribution plot uses a reproducible uniform 100,000-row
sample.

The raw files have no headers. Eight fields resemble symbol/date/time/OHLC/volume;
some later files add a ninth field whose meaning is not verified. The scripts
call it `extra_field_9`. Archive months are provenance labels, not confirmed
expiry dates. These files do not provide a verified complete contract identifier,
synchronized underlying prices, or bid/ask quotes, so this EDA does not compute
implied volatility, Greeks, moneyness, or Black–Scholes pricing errors.

Small tables, figures, and summaries are in `reports/nifty_eda/`. File-level
provenance, quality examples, audited Parquet partitions, and the uniform sample
are local outputs under the Git-ignored `data/processed/nifty_eda/`. Keep the
original files unchanged. After rerunning the terminal workflow, rerun the
Notebook cells to refresh its displayed results.

# fall-2026-AI-financial-models
Team project: fall-2026-AI-financial-models

Environment for neural option pricing, numerical model validation, historical
option-data analysis, and structured LLM test proposals. This repository currently
contains environment configuration, local Nifty data, EDA, preprocessing, and
reproducible neural forecasting baselines; the earlier
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

## Training data: next-minute option return

Open `notebooks/02_nifty_preprocessing.ipynb` for the completed cleaning workflow,
actual split counts, feature definitions, and copy-ready assignment answers.
The executed walkthrough is also available as
`reports/nifty_preprocessing/preprocessing.html`.
The prediction task is the next-minute log price change,
`log(close[t+1] / close[t])`, using six consecutive observed bars ending at `t`.
The corresponding predicted price is `close[t] * exp(predicted_log_return)`.
This prepares an empirical forecasting baseline. A theoretical option-pricing
model still needs verified expiry dates and synchronized underlying inputs.

```bash
.conda/bin/python scripts/nifty_preprocess.py
.conda/bin/python scripts/verify_nifty_training.py
.conda/bin/python -m pytest -q
```

The pipeline verifies the raw-file hashes against the EDA inventory, rejects
invalid essential values and inconsistent OHLC bars, and selects source-clock
times from 09:15 through 15:29. Outside-window special sessions are outside this
experiment's scope, not automatically erroneous. Valid zero-volume, flat-price,
low-price, and weekend observations remain included. The unverified ninth field
and expiry candidates are omitted from model inputs; no gaps are filled.

Every example contains six consecutive input bars and a target exactly one
minute later, within one source file and date. Source file IDs preserve
provenance, but do not establish verified exchange contract identities. Dates
are split chronologically into 70% training, 15% validation, and 15% testing.
Nine continuous features are standardized using training examples only; six
cyclic/binary features retain their original encoding. The target is unscaled.
The existing EDA examined all years, so the test set is held out from fitting
and tuning but was not unseen during data exploration.

Local outputs are under `data/processed/nifty_training/`:

- `train/`, `valid/`, `test/`: `X.npy` (15 float32 features), `y.npy` (float32
  log-return targets), and aligned `metadata.parquet` with source rows, window
  timestamps, current close, and target close. Metadata and targets are not
  extra model inputs. Array rows are in source-cohort/file order, not globally
  chronological order within a split.
- `feature_spec.json`, `scaler.json`, `source_manifest.csv`, `manifest.json`:
  definitions, training-only scaling, provenance, and output hashes. Require
  `manifest.json` with `status == "complete"` before consuming the arrays.
- `clean_bars/`, `rejected/`, `examples/`: traceable intermediate partitions.

Small reproducible results are saved to `reports/nifty_preprocessing/`.
With the current data, there are **17,576,228 clean bars** and **11,469,619
supervised examples**. The split summary records exact dates and counts.
Load arrays without copying the complete dataset into RAM:

```python
from pathlib import Path
import json
import numpy as np

data = Path("data/processed/nifty_training")
assert json.loads((data / "manifest.json").read_text())["status"] == "complete"
X_train = np.load(data / "train/X.npy", mmap_mode="r")
y_train = np.load(data / "train/y.npy", mmap_mode="r")
X_batch = np.array(X_train[:1024], copy=True)
y_batch = np.array(y_train[:1024], copy=True)
```

Start model development with the zero-return/persistence baseline and a small
network. Tune only on validation data, use early stopping and regularization,
and evaluate both return error and reconstructed price error. Overlapping
windows and shared market conditions mean millions of rows do not represent
millions of independent observations; check performance across dates and
liquidity regimes. No model has been trained by this preprocessing workflow.

## Baseline model architecture study

`notebooks/03_nifty_models.ipynb` contains the architecture decisions, executed
comparisons, learning curves, validation diagnostics, and check-in answers.
Read the rendered version at `reports/nifty_models/models.html`.

```bash
.conda/bin/python scripts/train_nifty_models.py
.conda/bin/python scripts/verify_nifty_models.py
.conda/bin/python -m pytest -q
```

Training refuses to replace a completed study unless `--overwrite` is supplied.
The saved `reports/nifty_models/protocol.json` records the settings before
fitting. It compares persistence (zero return), ridge regression, and three
conventional ReLU MLPs: `15→32→16→1`, `15→64→32→1`, and
`15→128→64→32→1`. Each neural model uses dropout 0.1 and the same full
training/validation data, optimizer, and three seeds. Test arrays are not opened
by training or model verification.

The neural loss is Smooth L1 on returns divided by their training standard
deviation, without centering or clipping. AdamW uses learning rate 0.001 and
weight decay 0.0001; batches contain 4,096 examples. Training permits up to 12
epochs, with validation early stopping, learning-rate reduction, and gradient
clipping. The primary selection metric averages MAE equally over validation
dates. The smallest network within 0.5% of the best three-seed mean is selected;
seed 42 is the predeclared reference checkpoint. Pooled return and reconstructed
price errors remain secondary metrics. Ridge and MLP use different fitting
losses, so their comparison does not isolate architecture effects.

The completed study selected **15→32→16→1 (1,057 parameters)**. Its frozen
seed-42 checkpoint has equal-date validation MAE **0.0149292**, compared with
**0.0148986** for persistence. Pooled log-return RMSE is **5.1% lower**, but
absolute return error and reconstructed-price errors are slightly worse.
The RMSE gain is concentrated in low-price options. This is a fixed neural
benchmark, without a demonstrated overall advantage over persistence.

`reports/nifty_models/frozen_config.json` identifies the fixed neural baseline.
Weights and validation predictions are local, Git-ignored files under
`data/processed/nifty_models/`; small tables and diagnostics are in the report
directory. The verifier reloads every checkpoint and reproduces every saved
validation prediction and metric. Read `verification.json` for its result.

Example inference on an already prepared feature batch:

```python
import numpy as np
import torch
from scripts.nifty_models import ReturnMLP, predict_batches

checkpoint = torch.load(
    "data/processed/nifty_models/frozen_mlp.pt",
    map_location="cpu", weights_only=True,
)
model = ReturnMLP(
    checkpoint["input_dim"], checkpoint["hidden"], checkpoint["dropout"]
)
model.load_state_dict(checkpoint["state_dict"])
X_valid = np.load("data/processed/nifty_training/valid/X.npy", mmap_mode="r")
predicted_log_returns = predict_batches(model, X_valid[:256], checkpoint["target_scale"])
```

These are chronological validation results, not final test or trading results.
The notebook explicitly distinguishes a useful fixed neural benchmark from
evidence of beating persistence on a particular metric. Warm CPU inference
timings exclude data loading, bar preparation, and queuing.

The future LLM-assisted error-analysis/stress-testing experiment is documented
in the same notebook but **has not started**. Its proposed tests use verified
historical labels first, deterministic execution, and equal-budget fixed-rule
and random comparators. LLM prompts must not consume final test data; generated
market paths cannot be assigned invented ground-truth prices.

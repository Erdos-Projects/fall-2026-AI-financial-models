"""Build the next check-in from verified fixed-baseline validation results.

No fitting, model selection, or train/test array access is performed. The
saved seed-42 predictions were independently verified in the earlier study.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import mistune
import numpy as np
import pandas as pd
from plot_nifty_baseline_performance import make_figures

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/nifty_mlp_study"
OUT = ROOT / "reports/nifty_baseline_performance"
INPUTS = ["summary.json", "protocol.json", "runs.csv", "architecture_comparison.csv",
          "history.csv", "subgroups.csv", "daily_validation.csv", "inference.csv"]
METRICS = {"val_day_mae": "Equal-date return MAE", "val_mae": "Pooled return MAE",
           "val_rmse": "Pooled return RMSE", "val_price_mae": "Price MAE",
           "val_price_rmse": "Price RMSE"}
GROUPS = {"calls": "Calls", "puts": "Puts", "close_at_most_1": "Current price ≤ 1",
          "close_above_1": "Current price > 1"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change(value, reference):
    return 100 * (value / reference - 1)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def build():
    verified = json.loads((STUDY / "verification.json").read_text())
    if verified["status"] != "passed" or verified["selected_architecture"] != "compact":
        raise ValueError("This performance check-in requires the verified compact baseline")
    hashes = {}
    for name in INPUTS:
        path = STUDY / name
        relative = str(path.relative_to(ROOT))
        hashes[relative] = digest(path)
        if hashes[relative] != verified["saved_artifact_hashes"][relative]:
            raise ValueError(f"Changed verified evidence: {name}")
    frozen = ROOT / "data/processed/nifty_models/frozen_mlp.pt"
    reference = "data/processed/nifty_models/compact_seed42.pt"
    if digest(frozen) != verified["saved_artifact_hashes"][reference]:
        raise ValueError("Frozen model differs from the verified reference seed")
    summary = json.loads((STUDY / "summary.json").read_text())
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if summary["reference_seed"] != 42 or summary["test_evaluated"]:
        raise ValueError("Unexpected model or evaluation scope")
    model, persistence = summary["selected_reference_metrics"], summary["persistence_metrics"]
    daily = pd.read_csv(STUDY / "daily_validation.csv", parse_dates=["day"])
    daily = daily.loc[daily.model.isin(["compact", "persistence"])]
    compact = daily.loc[daily.model.eq("compact")].sort_values("day").set_index("day")
    baseline = daily.loc[daily.model.eq("persistence")].sort_values("day").set_index("day")
    if not compact.index.equals(baseline.index) or len(compact) != 150:
        raise ValueError("Unaligned daily diagnostics")
    # Reconstruct all five headline metrics from the independently saved daily table.
    for frame, expected in [(compact, model), (baseline, persistence)]:
        actual = {"val_day_mae": frame.mae.mean(), "val_mae": np.average(frame.mae, weights=frame.rows),
                  "val_rmse": np.sqrt(np.average(frame.mse, weights=frame.rows)),
                  "val_price_mae": np.average(frame.price_mae, weights=frame.rows),
                  "val_price_rmse": np.sqrt(np.average(frame.price_mse, weights=frame.rows))}
        for metric, value in actual.items():
            if not np.isclose(value, expected[metric], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Daily aggregates do not reproduce {metric}")
    subgroup = pd.read_csv(STUDY / "subgroups.csv")
    subgroup = subgroup.loc[subgroup.model.isin(["compact", "persistence"])].set_index(["model", "group"])
    runs = pd.read_csv(STUDY / "runs.csv")
    seeds = runs.loc[runs.model.eq("compact")]
    inference = pd.read_csv(STUDY / "inference.csv")
    latency = inference.loc[inference.model.eq("compact")].set_index("batch_size")
    changes = {key: change(model[key], persistence[key]) for key in METRICS}
    difference = compact.mae - baseline.mae
    low = subgroup.loc[("compact", "close_at_most_1")]
    high = subgroup.loc[("compact", "close_above_1")]
    low_base = subgroup.loc[("persistence", "close_at_most_1")]
    high_base = subgroup.loc[("persistence", "close_above_1")]
    low_change, high_change = change(low.val_rmse, low_base.val_rmse), change(high.val_rmse, high_base.val_rmse)
    low_reduction = low.rows * (low_base.val_rmse ** 2 - low.val_rmse ** 2)
    high_reduction = high.rows * (high_base.val_rmse ** 2 - high.val_rmse ** 2)
    zero_rows = int(subgroup.loc[("compact", "zero_volume"), "rows"])
    daily_output = pd.DataFrame({"rows": compact.rows, "compact_mae": compact.mae,
                                 "persistence_mae": baseline.mae, "difference": difference})
    OUT.mkdir(parents=True, exist_ok=True)
    daily_output.to_csv(OUT / "daily_comparison.csv")
    metric_rows = [[label, f"{model[key]:.7f}", f"{persistence[key]:.7f}", f"{changes[key]:+.3f}%"]
                   for key, label in METRICS.items()]
    metric_table = table(["Metric", "Compact MLP, seed 42", "Persistence", "Relative error change"], metric_rows)
    groups = []
    for group, label in GROUPS.items():
        a, b = subgroup.loc[("compact", group)], subgroup.loc[("persistence", group)]
        groups.append([label, f"{int(a.rows):,}", f"{a.val_mae:.6f}", f"{a.val_rmse:.6f}",
                       f"{change(a.val_mae, b.val_mae):+.3f}%", f"{change(a.val_rmse, b.val_rmse):+.3f}%"])
    group_table = table(["Subset", "Rows", "Return MAE", "Return RMSE", "MAE change", "RMSE change"], groups)
    figures = make_figures(ROOT, OUT / "figures")
    short_answer = f"""1. We applied the selected `15→32→16→1` MLP to 7,069,922 training examples and evaluated its frozen seed-42 checkpoint on 2,009,818 validation examples. The final test set remains reserved.
2. Validation return MAE was **{model['val_mae']:.5f}**, RMSE **{model['val_rmse']:.5f}**, and equal-date MAE **{model['val_day_mae']:.7f}**; price MAE/RMSE were **{model['val_price_mae']:.3f}/{model['val_price_rmse']:.3f}**. Warm CPU inference took approximately **{latency.loc[1, 'median_batch_ms'] * 1000:.0f} µs per example**. Diagnostic figures show learning curves, daily errors, and subgroup comparisons.
3. Return RMSE improved **{-changes['val_rmse']:.2f}%** over persistence, especially for low-price options (**{-low_change:.2f}%** improvement). However, low-price options still had much higher absolute RMSE (**{low.val_rmse:.4f}** versus **{high.val_rmse:.4f}** above one price unit). Overall MAE and price errors were slightly worse, so the model has not demonstrated an overall forecasting advantage."""
    (OUT / "check_in.md").write_text("# Baseline performance — short check-in answer\n\n" + short_answer + "\n")
    report = f"""# Baseline model performance check-in

Prepared October 22, 2026. This follow-up evaluates the **previously selected compact MLP, seed 42**, using the completed, verified experiment. It introduces no new fit, tuning, architecture selection, or test-set evaluation.

## 1. Requirement audit and model application

| Check-in requirement | Status and evidence |
| --- | --- |
| Use the project data in last week's baseline | Already completed: full training data used in the `15→32→16→1` MLP; fixed seed-42 checkpoint retained |
| List metrics and useful diagnostic figures | Metrics already measured and verified; this report adds focused baseline learning, daily-error, and subgroup figures |
| Describe where the model works well and poorly | Existing subgroup evidence is organized below, separating absolute difficulty from improvement over persistence |

The model has **1,057 trainable parameters**, ReLU hidden activations, dropout 0.1, and a signed linear output. Its 15 features summarize six consecutive observed option bars. The target is `log(close[t+1] / close[t])`. Training used **7,069,922 examples** from 700 dates (2016-12-14–2019-10-17); validation used **2,009,818 examples** from 150 later dates (2019-10-18–2020-05-29).

Only training data updated model weights and fitted scalers. Validation guided early stopping and architecture selection, so these results are development estimates. The **2,389,879 reserved test examples were not evaluated**. Earlier EDA inspected all years; the test period was held out from fitting/tuning rather than unseen during exploration. The frozen checkpoint selected epoch 1, stopping after epoch 4. Its recorded fitting time was **{seeds.loc[seeds.seed.eq(42), 'train_seconds'].iloc[0]:.1f} seconds**, including validation and checkpoint work.

## 2. Performance metrics and inference time

This is regression, so classification “accuracy” is not an appropriate metric. Lower is better for every error metric. Return errors use log-return units; price errors use reported price units. Persistence predicts zero return, or no change in the next close. Price predictions are reconstructed as `close[t] * exp(predicted_return)`.

{metric_table}

Positive relative changes mean worse error; negative changes mean improvement. Equal-date MAE averages the 150 daily MAEs equally; pooled MAE/RMSE weight every observation equally. The model improves return RMSE but does not improve overall absolute error or reconstructed-price accuracy.

All headline and subgroup results above refer to **one frozen checkpoint (seed 42)**. Separately, compact-architecture equal-date MAE across seeds 17, 42, and 2026 was **{seeds.val_day_mae.mean():.7f} ± {seeds.val_day_mae.std(ddof=1):.7f}** (mean ± sample SD). Across dates for the frozen checkpoint, daily MAE was **{compact.mae.mean():.7f} ± {compact.mae.std(ddof=1):.7f}**. Seed variability and variation across dates describe different things; neither is an independent final-test confidence interval.

Warm CPU batch-1 inference: **{latency.loc[1, 'median_batch_ms'] * 1000:.3f} µs median**, **{latency.loc[1, 'p95_batch_ms'] * 1000:.3f} µs p95** over 500 measurements. A prepared batch of 4,096 took **{latency.loc[4096, 'median_batch_ms']:.6f} ms median** over 50 measurements. Both follow 20 warm-ups, using four CPU threads, PyTorch {protocol['hardware']['versions']['torch']} on {protocol['hardware']['system']} {protocol['hardware']['machine']}.

These timings include the forward pass and target rescaling, but exclude loading, feature preparation, tensor construction, and queuing. They do not measure total inference time for the entire validation dataset. The timing measurements are reused from the completed study; this check-in does not rerun the benchmark.

## 3. Useful diagnostic figures

**Learning curves:** validation date-MAE over epochs for the three compact-model seeds, with selected checkpoints marked. Early stopping keeps the selected checkpoint rather than the last epoch.

![Compact MLP learning curves](figures/learning_curve.png)

**Daily errors:** frozen MLP versus persistence and their difference across validation dates. Below-zero differences favor the MLP. It improves date-MAE on **{int((difference < 0).sum())}/150 dates** and worsens it on **{int((difference > 0).sum())}/150 dates**.

![Daily baseline errors and differences](figures/daily_error.png)

**Subgroup errors:** absolute return RMSE and relative MAE/RMSE changes across price bands and option types. These figures use existing predefined groups, not newly selected thresholds or synthetic noise labels.

![Subgroup baseline performance](figures/subgroup_error.png)

## 4. Where the baseline worked well and poorly

{group_table}

**Useful relative improvement:** return RMSE falls **{-changes['val_rmse']:.2f}%** overall. For prices at or below one, RMSE falls **{-low_change:.2f}%**; for prices above one, it falls only **{-high_change:.2f}%**. The {int(low.rows):,} low-price rows are **{100 * low.rows / (low.rows + high.rows):.2f}%** of validation observations and account for **{100 * low_reduction / (low_reduction + high_reduction):.2f}%** of the squared-return-error reduction. This is a concentrated benefit on large return errors, not a broad gain in typical forecasts.

**Absolute difficulty:** low-price observations remain much harder in return units: RMSE **{low.val_rmse:.6f}**, versus **{high.val_rmse:.6f}** above one. Small absolute price changes can produce large log returns when the starting price is very low. Conversely, their absolute price errors can be small: return error and price error describe different behavior.

**Option type:** puts have lower absolute return MAE than calls, but both have slightly higher MAE than their own persistence comparator. This does not establish that the model adds more value for puts. Counts and relative changes are shown in the table.

**Weaknesses:** pooled MAE increases **{changes['val_mae']:.3f}%**, equal-date MAE increases **{changes['val_day_mae']:.3f}%**, and price MAE/RMSE increase **{changes['val_price_mae']:.3f}%/{changes['val_price_rmse']:.3f}%**. The model beats persistence on only {int((difference < 0).sum())} of 150 dates. Its best absolute daily MAE is **{compact.mae.min():.7f}** on **{compact.mae.idxmin():%Y-%m-%d}**, while its worst is **{compact.mae.max():.7f}** on **{compact.mae.idxmax():%Y-%m-%d}**. A hard date can still show an improvement over persistence; absolute error and relative value should not be confused.

**Insufficient evidence:** there are only **{zero_rows} zero-volume validation observations**, so no reliable zero-volume performance claim is possible. The raw dataset's many zero-volume bars do not imply a large eligible validation subgroup. We have not introduced “clean/noisy” labels because this financial dataset has no validated noise-type annotation.

The baseline is a useful small neural reference, but it has not demonstrated an overall forecasting advantage. These bar-price errors do not establish trading profitability. Keep the model frozen for comparisons; investigate the causes of return/price-metric disagreement before spending more compute on larger networks.

## 5. Evidence and reproducibility

The builder checks hashes of all eight saved evidence tables/configurations against the [completed study verification](../nifty_mlp_study/verification.json), confirms the frozen model matches the verified seed-42 checkpoint, and independently reconstructs all five headline metrics from the daily table. It reads no train/test arrays, performs no fitting, and does not alter the earlier reports. The original verifier reproduced all 24 model checkpoints' validation predictions, including this baseline.

```bash
MPLCONFIGDIR=/private/tmp/nifty-mlp-matplotlib .conda/bin/python scripts/build_nifty_baseline_performance.py
```

Sources: [model runs](../nifty_mlp_study/runs.csv), [subgroup metrics](../nifty_mlp_study/subgroups.csv), [inference measurements](../nifty_mlp_study/inference.csv), and [daily comparison](daily_comparison.csv). The new [summary and evidence hashes](summary.json) preserve the provenance of this presentation. The [architecture report](../nifty_mlp_study/report.md) remains the full eight-model comparison.

## 6. Short answers to the check-in questions

{short_answer}
"""
    (OUT / "report.md").write_text(report)
    html = mistune.create_markdown(plugins=["table"])(report)
    for path in figures:
        path = Path(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        html = html.replace(f'src="figures/{path.name}"', f'src="data:image/png;base64,{encoded}"')
    css = """body{margin:0;background:#f1f4f1;color:#21312e;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1060px;margin:32px auto;background:white;padding:40px 48px;border-top:6px solid #16665d}
    h1,h2{color:#16665d;line-height:1.3}h2{margin-top:40px}a{color:#16665d}img{max-width:100%;height:auto}
    table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}th,td{padding:9px;border-bottom:1px solid #d8e1dc;text-align:left}
    th{background:#e5efeb}tr:nth-child(even){background:#f5f8f5}code,pre{background:#eef2f0}pre{padding:16px;overflow:auto}
    @media(max-width:800px){main{padding:20px;margin:0}table{display:block;overflow:auto}}
    @media print{main{border:0;margin:0;padding:0}img,table{break-inside:avoid}h2{break-after:avoid}}"""
    (OUT / "report.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                                   '<meta name="viewport" content="width=device-width, initial-scale=1">'
                                   f'<title>Baseline performance check-in</title><style>{css}</style></head>'
                                   f'<body><main>{html}</main></body></html>')
    result = {"status": "complete", "baseline": "compact", "seed": 42,
              "new_fits": 0, "test_evaluated": False, "validation_rows": int(compact.rows.sum()),
              "validation_dates": len(compact), "metrics": model, "persistence_metrics": persistence,
              "relative_error_change_percent": changes, "daily_mae_sd": float(compact.mae.std(ddof=1)),
              "days_better_than_persistence": int((difference < 0).sum()),
              "zero_volume_rows": zero_rows, "all_input_hashes_matched": True,
              "headline_metrics_reconstructed_from_daily": True, "input_hashes": hashes,
              "prior_verification_sha256": digest(STUDY / "verification.json"),
              "frozen_checkpoint_sha256": digest(frozen),
              "builder_source_sha256": digest(Path(__file__)),
              "plot_source_sha256": digest(ROOT / "scripts/plot_nifty_baseline_performance.py")}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    update_readme(model, changes)
    print("Baseline performance report, three diagnostic figures, check-in answers, and README completed.")


def update_readme(model, changes):
    section = f"""<!-- BASELINE_PERFORMANCE_START -->
## Baseline performance check-in

The [performance report](reports/nifty_baseline_performance/report.md)
([HTML](reports/nifty_baseline_performance/report.html)) and
[short check-in answers](reports/nifty_baseline_performance/check_in.md)
address the next assignment's three questions using the **frozen compact MLP,
seed 42**. Training and evaluation were already complete; this follow-up adds
focused learning, daily-error, and subgroup figures without retraining.

On 2,009,818 validation examples, pooled return MAE/RMSE are
**{model['val_mae']:.5f}/{model['val_rmse']:.5f}**, equal-date MAE is
**{model['val_day_mae']:.7f}**, and price MAE/RMSE are
**{model['val_price_mae']:.3f}/{model['val_price_rmse']:.3f}**. Return RMSE improves
**{-changes['val_rmse']:.2f}%** over persistence, concentrated in low-price options;
absolute and price errors remain slightly worse. These are fixed-checkpoint
validation results, not the three-seed averages in the architecture table.
Only two validation rows have zero volume, precluding a reliable subgroup claim.
The final test set remains reserved.

```bash
.conda/bin/python scripts/build_nifty_baseline_performance.py
```

The builder checks the saved evidence hashes and reconstructs the headline
metrics from daily diagnostics. See its [audit summary](reports/nifty_baseline_performance/summary.json).
<!-- BASELINE_PERFORMANCE_END -->"""
    path = ROOT / "README.md"
    original = path.read_text()
    start, end = "<!-- BASELINE_PERFORMANCE_START -->", "<!-- BASELINE_PERFORMANCE_END -->"
    if start in original:
        before, tail = original.split(start, 1)
        _, after = tail.split(end, 1)
        updated = before + section + after
    else:
        updated = original.rstrip() + "\n\n" + section + "\n"
    path.write_text(updated)


if __name__ == "__main__":
    build()

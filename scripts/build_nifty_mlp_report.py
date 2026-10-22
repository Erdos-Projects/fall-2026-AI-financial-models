"""Render the verified MLP comparison as Markdown/HTML and update the README.

Reads small saved experiment tables only; never fits models or opens test data.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/nifty_mlp_study"
LABELS = {"compact": "Compact", "standard": "Standard + dropout", "wide": "Wide / tapered",
          "shallow": "Shallow", "standard_plain": "Standard, no dropout", "deep": "Deep",
          "batchnorm": "BatchNorm", "residual": "Residual + LayerNorm"}
STRUCTURES = {"compact": "15 → 32 → 16 → 1", "standard": "15 → 64 → 32 → 1",
              "wide": "15 → 128 → 64 → 32 → 1", "shallow": "15 → 64 → 1",
              "standard_plain": "15 → 64 → 32 → 1", "deep": "15 → 64 → 64 → 64 → 64 → 1",
              "batchnorm": "15 → 64 → 32 → 1 + BatchNorm",
              "residual": "15 → 64; 2 × residual blocks at width 64; → 1"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def signed_change(value, baseline):
    return 100 * (value / baseline - 1)


def plots(comparison, inference, history, persistence):
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fafbf9", "axes.facecolor": "#fafbf9"})
    ordered = comparison.sort_values("val_day_mae_mean", ascending=False)
    names = [LABELS[row.model] + (" (selected)" if row.selected else "")
             for row in ordered.itertuples()]
    colors = ["#16665d" if v else "#647e92" for v in ordered.selected]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    y = np.arange(len(ordered))
    axes[0].errorbar(ordered.val_day_mae_mean, y, xerr=ordered.val_day_mae_std,
                    fmt="none", ecolor="#334155", capsize=3)
    axes[0].scatter(ordered.val_day_mae_mean, y, c=colors, s=48, zorder=3)
    axes[0].axvline(persistence["val_day_mae"], color="#b35735", linestyle="--", label="Persistence")
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("Equal-date validation MAE (lower is better)")
    axes[0].set_title("Accuracy: mean ± one seed SD")
    axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].legend(loc="upper right", fontsize=9)
    latency = inference.loc[inference.batch_size.eq(1)].set_index("model")
    axes[1].barh(y, [latency.loc[n, "median_batch_ms"] * 1000 for n in ordered.model], color=colors)
    axes[1].set_yticks(y, names)
    axes[1].set_xlabel("Warm CPU latency per example (µs)")
    axes[1].set_title("Inference: median, batch size 1")
    fig.suptitle("Nifty option returns · eight MLP architectures", fontsize=15, weight="bold")
    fig.savefig(figures / "accuracy_latency.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained", sharey=True)
    for ax, name in zip(axes.flat, LABELS, strict=True):
        for seed, group in history.loc[history.model.eq(name)].groupby("seed"):
            ax.plot(group.epoch, group.val_day_mae, marker=".", label=str(seed), linewidth=1.4)
        ax.axhline(persistence["val_day_mae"], color="#b35735", linestyle="--", linewidth=1)
        ax.set_title(LABELS[name])
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Equal-date validation MAE")
    axes[1, 0].set_ylabel("Equal-date validation MAE")
    axes[0, 0].legend(title="Seed", fontsize=8)
    fig.suptitle("Learning curves · checkpoints selected using validation", fontsize=15, weight="bold")
    fig.savefig(figures / "learning_curves.png", dpi=170)
    plt.close(fig)


def diagnostics(daily, summary, runs):
    pivot = daily.pivot(index="day", columns="model", values="mae")
    result = {"method": "2,000 circular moving-block resamples of 150 validation dates; block length 5; seed 8801",
              "caveat": "Descriptive post-selection validation intervals; not independent test confidence intervals",
              "models": {}}
    for name in dict.fromkeys([summary["selected_architecture"], summary["best_mean_architecture"]]):
        diff = (pivot[name] - pivot.persistence).to_numpy()
        rng = np.random.default_rng(8801)
        starts = rng.integers(0, len(diff), size=(2000, int(np.ceil(len(diff) / 5))))
        indices = ((starts[..., None] + np.arange(5)) % len(diff)).reshape(2000, -1)[:, :len(diff)]
        interval = np.quantile(diff[indices].mean(axis=1), [.025, .975])
        result["models"][name] = {"seed": 42, "mean_day_mae_difference": float(diff.mean()),
                                   "descriptive_95pct_interval": interval.tolist(),
                                   "days_better_than_persistence": int((diff < 0).sum()), "days": len(diff)}
    pairs = []
    for left, right, label in [("standard_plain", "standard", "Add dropout 0.1"),
                                ("standard", "batchnorm", "Add BatchNorm")]:
        a = runs.loc[runs.model.eq(left)].set_index("seed")
        b = runs.loc[runs.model.eq(right)].set_index("seed")
        for seed in [17, 42, 2026]:
            pairs.append({"comparison": label, "seed": seed, "from_model": left, "to_model": right,
                          "day_mae_difference": float(b.loc[seed, "val_day_mae"] - a.loc[seed, "val_day_mae"])})
    pd.DataFrame(pairs).to_csv(REPORT / "paired_comparisons.csv", index=False)
    (REPORT / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def render_html(markdown):
    body = mistune.create_markdown(plugins=["table"])(markdown)
    for image in (REPORT / "figures").glob("*.png"):
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        body = body.replace(f'src="figures/{image.name}"', f'src="data:image/png;base64,{encoded}"')
    css = """body{margin:0;background:#f1f4f1;color:#21312e;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1140px;margin:40px auto;padding:48px 56px;background:#fff;border-top:7px solid #16665d}
    h1{font-size:36px;line-height:1.2;color:#124e48}h2{margin-top:42px;color:#16665d;font-size:24px}
    h3{font-size:19px}a{color:#16665d}p,li{max-width:100ch}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45;margin:24px 0}
    th{text-align:left;background:#e5efeb}td,th{padding:10px 9px;border-bottom:1px solid #d9e1dc;vertical-align:top}
    tr:nth-child(even){background:#f6f8f6}img{max-width:100%;height:auto}code{background:#eef2f0;padding:2px 5px;border-radius:3px;font-size:.88em}
    pre{background:#eef2f0;padding:18px;overflow:auto}pre code{padding:0}blockquote{margin:24px 0;padding:10px 22px;background:#edf4f0;border-left:4px solid #16665d}
    @media(max-width:800px){main{margin:0;padding:24px 18px}table{display:block;overflow:auto}h1{font-size:28px}}
    @media print{body{background:white}main{margin:0;padding:0;border:0}h2{break-after:avoid}img,table{break-inside:avoid}}"""
    (REPORT / "report.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Nifty MLP architecture comparison · October 22, 2026</title>'
        f'<style>{css}</style></head><body><main>{body}</main></body></html>')


def build():
    summary = json.loads((REPORT / "summary.json").read_text())
    verified = json.loads((REPORT / "verification.json").read_text())
    if summary["status"] != "complete" or verified["status"] != "passed":
        raise ValueError("Only render a complete, verified study")
    protocol = json.loads((REPORT / "protocol.json").read_text())
    comparison = pd.read_csv(REPORT / "architecture_comparison.csv")
    runs = pd.read_csv(REPORT / "runs.csv")
    inference = pd.read_csv(REPORT / "inference.csv")
    history = pd.read_csv(REPORT / "history.csv")
    daily = pd.read_csv(REPORT / "daily_validation.csv")
    subgroups = pd.read_csv(REPORT / "subgroups.csv")
    baselines = pd.read_csv(ROOT / "reports/nifty_models/baseline_comparison.csv")
    persistence = summary["persistence_metrics"]
    indexed = comparison.set_index("model")
    chosen, best = summary["selected_architecture"], summary["best_mean_architecture"]
    plots(comparison, inference, history, persistence)
    diag = diagnostics(daily, summary, runs)
    content = report_text(summary, protocol, comparison, runs, inference, history,
                          subgroups, baselines, indexed, persistence, chosen, best, diag)
    (REPORT / "report.md").write_text(content)
    render_html(content)
    update_readme(indexed, persistence, chosen, best, inference)
    print("Created reports/nifty_mlp_study/report.md, report.html, figures and README comparison.")


# Narrative generation is kept below the numerical analysis for easy review.
def report_text(summary, protocol, comparison, runs, inference, history,
                subgroups, baselines, indexed, persistence, chosen, best, diag):
    accuracy = []
    for row in comparison.itertuples():
        accuracy.append([LABELS[row.model], f"{row.parameters:,}",
                         f"{row.val_day_mae_mean:.7f} ± {row.val_day_mae_std:.7f}",
                         f"{row.val_mae_mean:.7f}", f"{row.val_rmse_mean:.7f}",
                         f"{row.val_price_mae_mean:.4f}", f"{row.val_price_rmse_mean:.4f}"])
    accuracy_table = table(["MLP", "Parameters", "Date MAE, mean ± SD", "Pooled MAE",
                            "Pooled RMSE", "Price MAE", "Price RMSE"], accuracy)
    baseline_table = table(["Comparator", "Date MAE", "Pooled MAE", "Pooled RMSE", "Price MAE", "Price RMSE"],
                           [[row.model, f"{row.val_day_mae:.7f}", f"{row.val_mae:.7f}",
                             f"{row.val_rmse:.7f}", f"{row.val_price_mae:.4f}", f"{row.val_price_rmse:.4f}"]
                            for row in baselines.itertuples()])
    macs = {"compact": 1008, "standard": 3040, "wide": 12192, "shallow": 1024,
            "standard_plain": 3040, "deep": 13312, "batchnorm": 3040, "residual": 17408}
    costs = []
    for row in comparison.itertuples():
        one = inference.loc[inference.model.eq(row.model) & inference.batch_size.eq(1)].iloc[0]
        batch = inference.loc[inference.model.eq(row.model) & inference.batch_size.eq(4096)].iloc[0]
        cost_origin = "prior" if row.origin == "prior_study" else "new"
        costs.append([LABELS[row.model], f"{macs[row.model]:,}", f"{row.epochs_run_mean:.1f}",
                      f"{row.train_seconds_mean:.1f} ({cost_origin})", f"{one.median_batch_ms * 1000:.1f}",
                      f"{one.p95_batch_ms * 1000:.1f}", f"{batch.median_batch_ms:.3f}"])
    cost_table = table(["MLP", "Dense MACs / row", "Mean epochs", "Mean fit seconds",
                        "Batch-1 median µs", "Batch-1 p95 µs", "Batch-4096 median ms"], costs)
    arch_table = table(["MLP", "Structure", "Regularization / normalization", "Evidence origin"],
                       [[LABELS[name], STRUCTURES[name],
                         "No dropout" if name == "standard_plain" else
                         ("Dropout 0.1 + BatchNorm" if name == "batchnorm" else
                          "Dropout 0.1 + LayerNorm + skips" if name == "residual" else "Dropout 0.1"),
                         "Prior: 3 verified fits" if name in ("compact", "standard", "wide") else "New: 3 fits"]
                        for name in LABELS])
    dropout_change = signed_change(indexed.loc["standard", "val_day_mae_mean"], indexed.loc["standard_plain", "val_day_mae_mean"])
    bn_change = signed_change(indexed.loc["batchnorm", "val_day_mae_mean"], indexed.loc["standard", "val_day_mae_mean"])
    best_change = signed_change(indexed.loc[best, "val_day_mae_mean"], persistence["val_day_mae"])
    selected_change = signed_change(indexed.loc[chosen, "val_day_mae_mean"], persistence["val_day_mae"])
    best_vs_selected = signed_change(indexed.loc[chosen, "val_day_mae_mean"], indexed.loc[best, "val_day_mae_mean"])
    selected_rmse_gain = -signed_change(indexed.loc[chosen, "val_rmse_mean"], persistence["val_rmse"])
    selected_price_change = signed_change(indexed.loc[chosen, "val_price_mae_mean"], persistence["val_price_mae"])
    shallow_change = signed_change(indexed.loc["shallow", "val_day_mae_mean"], indexed.loc["compact", "val_day_mae_mean"])
    deep_change = signed_change(indexed.loc["deep", "val_day_mae_mean"], indexed.loc["wide", "val_day_mae_mean"])
    residual_change = signed_change(indexed.loc["residual", "val_day_mae_mean"], indexed.loc["wide", "val_day_mae_mean"])
    subgroup_index = subgroups.set_index(["model", "group"])
    low_rmse_change = signed_change(subgroup_index.loc[(chosen, "close_at_most_1"), "val_rmse"],
                                    subgroup_index.loc[("persistence", "close_at_most_1"), "val_rmse"])
    high_rmse_change = signed_change(subgroup_index.loc[(chosen, "close_above_1"), "val_rmse"],
                                     subgroup_index.loc[("persistence", "close_above_1"), "val_rmse"])
    low = subgroup_index.loc[("persistence", "close_at_most_1")]
    high = subgroup_index.loc[("persistence", "close_above_1")]
    low_row_share = 100 * low.rows / (low.rows + high.rows)
    low_error_share = 100 * low.rows * low.val_rmse ** 2 / (
        low.rows * low.val_rmse ** 2 + high.rows * high.val_rmse ** 2)
    reductions = {group: subgroup_index.loc[("persistence", group), "rows"] * (
        subgroup_index.loc[("persistence", group), "val_rmse"] ** 2
        - subgroup_index.loc[(chosen, group), "val_rmse"] ** 2)
        for group in ["close_at_most_1", "close_above_1"]}
    low_reduction_share = 100 * reductions["close_at_most_1"] / sum(reductions.values())
    zero_volume_rows = int(subgroup_index.loc[("persistence", "zero_volume"), "rows"])
    deep_price_rmse_change = signed_change(indexed.loc["deep", "val_price_rmse_mean"], persistence["val_price_rmse"])
    uncertainty = []
    for name, result in diag["models"].items():
        lower, upper = result["descriptive_95pct_interval"]
        uncertainty.append(f"- **{LABELS[name]}, seed 42:** date-MAE difference versus persistence "
                           f"{result['mean_day_mae_difference']:+.7f}; descriptive 95% interval "
                           f"[{lower:+.7f}, {upper:+.7f}]; better on {result['days_better_than_persistence']} "
                           f"of {result['days']} validation dates.")
    group_table = table(["Subset", "Model, seed 42", "Rows", "Date MAE", "Pooled RMSE", "Price MAE"],
                        [[row.group, "Persistence" if row.model == "persistence" else LABELS[row.model],
                          f"{row.rows:,}", f"{row.val_day_mae:.7f}", f"{row.val_rmse:.7f}", f"{row.val_price_mae:.4f}"]
                         for row in subgroups.itertuples() if row.group in ("close_at_most_1", "close_above_1")])
    checkin = checkin_text(indexed, persistence, chosen, best, inference)
    (REPORT / "check_in.md").write_text("# Weekly check-in — October 22, 2026\n\n" + checkin + "\n")
    return f"""# Nifty options: MLP baseline architecture comparison

**Weekly check-in · October 22, 2026 · Completed experiment**

We compared **eight MLP architectures across three seeds each** using the same
15 prepared features and chronological split. Fifteen fits are new; nine are
verified results from the October 14 study. **{LABELS[best]} has the lowest mean
validation date-MAE among the MLPs ({indexed.loc[best, 'val_day_mae_mean']:.7f}).**
The predeclared complexity rule selects **{LABELS[chosen]}**, with
{int(indexed.loc[chosen, 'parameters']):,} parameters, because its mean is only
{best_vs_selected:.3f}% above the best neural mean. Persistence has date-MAE
{persistence['val_day_mae']:.7f}; the best MLP's error is
{abs(best_change):.3f}% {'higher' if best_change > 0 else 'lower'}.
These are development-set results; the final test partition was not evaluated.

## 1. How last week's data exploration informs the models

The EDA audited 17,611,169 selected rows. Preprocessing retained 17,576,228 clean
bars and produced 11,469,619 supervised examples. Prices and volumes are skewed;
the retained data include 374,544 zero-volume bars, 490,741 prices at or below
one price unit, and 3,345,913 flat OHLC bars. Missing minutes are common and are
not filled. The unverified ninth field and inferred expiry dates are excluded.
See the existing [EDA summary](../nifty_eda/summary.json) and
[preprocessing summary](../nifty_preprocessing/summary.json).

The task is **next-minute option log-return forecasting**:
`y = log(close[t+1] / close[t])`. Each observation uses six consecutive bars
ending at `t`, and the target must occur exactly one minute later within the
same source file and date. Inputs are **15 engineered tabular features**, not
a `6 × 15` raw sequence: logged price/strike/volume, range and candle change,
one- and five-minute returns, recent volatility and volume, cyclical time, and
call/put, zero-volume, low-price, and flat-bar indicators. Nine continuous
features use training-only standardization; the others retain their encoding.

This representation supports small MLPs directly. Correlated windows and
options reduce the effective sample size, motivating dropout and compact
capacity despite millions of rows. Extreme returns motivate a robust training
loss and reporting both absolute and squared errors. Sparse and low-price
observations motivate subgroup checks. The dataset lacks verified complete
contract identities, expiry inputs, synchronized underlying prices, and
executable quotes; this experiment does not measure theoretical pricing error
or trading profitability.

| Partition | Dates | Observed dates | Examples | Use |
| --- | --- | --- | --- | --- |
| Train | 2016-12-14–2019-10-17 | 700 | 7,069,922 | Fit parameters and scalers |
| Validation | 2019-10-18–2020-05-29 | 150 | 2,009,818 | Early stopping and architecture comparison |
| Reserved test | 2020-06-01–2020-12-31 | 150 | 2,389,879 | No model evaluation in this study |

The split is chronological by date, not random by row. Earlier EDA examined
all periods, so the test period is held out from model development rather than
unseen during exploration. This extension also reuses the earlier validation
period and therefore adds selection exposure.

## 2. Baselines and literature-backed architecture choices

**Persistence** predicts zero return, equivalent to the next close equaling
the current close. **Ridge regression** is a linear comparator using the same
features; its penalty is selected from 0.0001, 0.01, and 1.0. The **compact MLP**
is the established neural baseline against which the other MLPs are compared.

Plain MLP and residual tabular networks are established comparison families in
[Gorishniy et al. (2021)](https://arxiv.org/html/2106.11959v5). We test ordinary
ReLU networks at different depths and widths, a matched dropout comparison,
BatchNorm, and an adapted residual network. The exact widths below are project
choices; they are not claims to reproduce published tuned architectures.

{arch_table}

All heads are unconstrained linear scalars, allowing positive and negative
returns. Ordinary hidden blocks are `Linear → ReLU → Dropout`; BatchNorm is
inserted between Linear and ReLU. The residual model has a `15 → 64` ReLU stem
and two blocks computing
`ReLU(h + Linear(Dropout(ReLU(Linear(LayerNorm(h))))))`, followed by a scalar
head. Each block contains two width-64 dense layers. This LayerNorm design
adapts the residual family; the cited tabular ResNet uses BatchNorm instead.

Dropout is motivated by [Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html);
normalization choices by [Ioffe and Szegedy (2015)](https://proceedings.mlr.press/v37/ioffe15.html)
and [Ba et al. (2016)](https://arxiv.org/abs/1607.06450).
The standard/plain pair holds dense widths fixed while adding dropout.
The standard/BatchNorm pair adds normalization at the same widths. Other
comparisons change several architectural properties and parameter counts.
The [literature companion](literature.md) explains expected performance and
cost tradeoffs and provides eight primary references verified online.

## 3. Controlled training and selection

- Full training and validation arrays for every model; no downsampled results.
- Seeds **17, 42, 2026**, batch size **4,096**, AdamW learning rate **0.001**,
  weight decay **0.0001**, gradient norm clipping **1.0**.
- Smooth L1 loss (`beta=1`) on returns divided by their training-only standard
  deviation **{summary['target_scale']:.10f}**, without centering or clipping.
- At most **12 epochs**; stop after three epochs without date-MAE improvement
  of at least `1e-6`. Halve learning rate on a validation plateau, with patience
  one and minimum learning rate `1e-5`. Restore the best eligible checkpoint.
- Default PyTorch hidden initialization and zero output weights/bias, giving
  a persistence prediction before training. All training rows are shuffled
  within the training partition each epoch.
- **Primary metric:** compute MAE on each validation date, then average equally
  across dates. Secondary metrics pool rows: return MAE/RMSE and reconstructed
  price MAE/RMSE, where `predicted_price = close[t] * exp(predicted_return)`.
- **Selection:** smallest parameter count within **0.5%** of the lowest
  three-seed mean date-MAE; predeclared seed **42** is the reference checkpoint.

The [protocol](protocol.json) was saved before fitting and fixes these choices.
The old nine runs are reused only after validating their artifact and input
hashes; their provenance remains visible in [runs.csv](runs.csv). This avoids
claiming they were trained again. The extension leaves the prior study and
its frozen checkpoint unchanged. Interrupted work is resumed only after
checking saved run hashes; a partially completed seed is restarted.

This is a common-recipe comparison, not a separately optimized hyperparameter
search for every family. Training loss weights rows equally, whereas selection
weights dates equally. Ridge uses squared loss, so ridge-versus-MLP results
compare complete pipelines rather than isolating architecture alone.

## 4. Measured validation accuracy

All neural table entries are means across the three runs; **± SD** is the sample
standard deviation across seeds for date-MAE. It measures optimization
variability, not uncertainty about future market conditions. Return errors are
in log-return units; price errors use the source's reported price units.
Lower is better for every metric.

{accuracy_table}

Non-neural comparators from the verified prior full-data study:

{baseline_table}

![Validation accuracy and CPU inference latency](figures/accuracy_latency.png)

**Accuracy versus complexity.** The selected {LABELS[chosen]} has mean date-MAE
{indexed.loc[chosen, 'val_day_mae_mean']:.7f}, a {selected_change:+.3f}% change
versus persistence (positive means worse). Its mean pooled return RMSE improves
by {selected_rmse_gain:.2f}%, while price MAE changes by {selected_price_change:+.3f}%.
Different metrics therefore need separate interpretation. The tolerance rule
chooses a practical neural benchmark; it is not a significance test and does
not require a neural model to beat persistence.
Smooth L1 is not the MAE selection objective: better performance on large
return errors can coexist with worse typical absolute or price errors.

**Dropout at fixed widths.** Adding dropout 0.1 to `64 → 32` changes mean
date-MAE by **{dropout_change:+.3f}%** relative to the no-dropout model.
**BatchNorm at fixed widths.** Adding BatchNorm to the dropout model changes
mean date-MAE by **{bn_change:+.3f}%**. Positive changes mean higher error.
[Paired seed differences](paired_comparisons.csv) preserve the individual
comparisons; three seeds are insufficient for a broad claim about superiority.

**Depth and width.** Shallow, compact, wide, deep, and residual results test
useful practical structures, but capacity and optimization change together.
Despite similar parameter counts (1,089 versus 1,057), the shallow model's
mean date-MAE changes by **{shallow_change:+.3f}%** relative to compact.
The four-hidden-layer model changes date-MAE by **{deep_change:+.3f}%** relative
to the tapered wide model (13,569 versus 12,417 parameters). The residual model
changes it by **{residual_change:+.3f}%** relative to wide, using 17,985 parameters.
These are measured common-recipe comparisons, not claims about all possible
widths or tuning budgets. A network's additional operations are justified only
if the observed accuracy benefit merits the cost in the next table.

Deep has the lowest neural mean price MAE and price RMSE in this study. Its
price RMSE changes by **{deep_price_rmse_change:+.4f}%** versus persistence,
while its date-MAE and price MAE remain worse. This small secondary-metric gain
does not change the predeclared date-MAE selection.

![Validation learning curves for all three seeds](figures/learning_curves.png)

The plot shows all completed epochs, including epochs after the selected
checkpoint. These curves and [history.csv](history.csv) document early stopping;
an early best epoch is possible even when training loss continues improving.

## 5. Compute and measured inference time

All fits ran on CPU. The recorded environment is {protocol['hardware']['system']}
{protocol['hardware']['machine']}, {protocol['hardware']['logical_cpus']} logical CPUs,
**four PyTorch/BLAS threads**, PyTorch {protocol['hardware']['versions']['torch']},
NumPy {protocol['hardware']['versions']['numpy']}, and pandas {protocol['hardware']['versions']['pandas']}.
Reported training durations include epoch validation, training probes, final
prediction, and checkpoint saving; they exclude initial input verification.
The new 15 completed fits total **{runs.loc[runs.origin.eq('new_run'), 'train_seconds'].sum() / 60:.1f}
minutes** of recorded run time, excluding abandoned partial work and setup.

{cost_table}

Training times marked **prior** were recorded in the earlier session; **new**
times were recorded in this extension. Shared hardware/software settings help
comparability, but system load and early-stopping epochs differ, so these are
not controlled training-speed ratios. Inference was measured afresh for all
eight architectures together: 20 warm-ups, 500 repeats at batch 1, and 50 repeats
at batch 4,096, using the seed-42 checkpoints in evaluation mode.

Inference timings include the forward pass and rescaling to return units;
they exclude data loading, tensor construction, preprocessing, and queuing.
They are warm local latency measurements, not end-to-end service guarantees.
Dropout is inactive during inference; BatchNorm uses training-fitted running
statistics. LayerNorm normalizes each example. Dense MAC estimates count
matrix multiply-accumulates only, excluding nonlinearities and normalization.
Float32 parameter storage ranges from roughly 4 to 70 KiB, excluding buffers,
optimizer state, and activations; fitting the full dataset is the larger cost.

## 6. Dates, low-price options, and uncertainty

For the selected model and lowest-mean-error model, we compare predeclared
seed-42 checkpoints with persistence on individual dates:

{chr(10).join(uncertainty)}

These intervals use 2,000 circular moving-block resamples of validation dates,
block length five, seed 8801. They are descriptive **after selection on the same
validation period**, not independent test confidence intervals. Nonstationary
market regimes can invalidate the block assumptions. Daily metrics for every
MLP's seed-42 checkpoint are saved in [daily_validation.csv](daily_validation.csv).

{group_table}

Low option prices can turn a small absolute price change into a large log
return. The selected model's seed-42 return RMSE changes by **{low_rmse_change:+.2f}%**
versus persistence for prices at or below one, and **{high_rmse_change:+.2f}%**
above one. Comparing both subsets helps determine whether pooled return RMSE
is dominated by cheap options. Low-price rows are **{low_row_share:.2f}%** of
validation observations but contribute **{low_error_share:.2f}%** of persistence's
squared return error and **{low_reduction_share:.2f}%** of the selected seed-42
model's squared-error reduction. The overall RMSE gain is therefore concentrated
in cheap options. The zero-volume validation subset contains only
**{zero_volume_rows} observations** and cannot support a meaningful volume-regime
conclusion. Additional call/put and zero/positive-volume
diagnostics are in [subgroups.csv](subgroups.csv). Subgroup results above refer
to seed 42, while the architecture table averages three seeds; they should not
be conflated.

## 7. Other deep learning techniques and the next experiment

The following techniques were researched but **not trained in this study**.
Their performance and timing on Nifty remain unmeasured.

| Technique | Expected value to test | Compute and inference implications |
| --- | --- | --- |
| LSTM | Learn temporal order that six-bar summaries discard; no guaranteed accuracy gain | Requires causal per-bar sequence inputs; recurrent time steps introduce sequential work |
| Causal TCN | Learn short local patterns and a chosen receptive field using temporal convolutions | Requires sequence inputs; convolutions parallelize positions, with cost set by channels, kernels, and depth |
| FT-Transformer | Learn interactions among the existing tabular features through feature tokens | Can use the current features; embeddings and attention add work, with quadratic attention interactions in token count |

These proposals follow [Hochreiter and Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735),
[Bai et al. (2018)](https://arxiv.org/abs/1803.01271), and
[Gorishniy et al. (2021)](https://arxiv.org/html/2106.11959v5).
The original [Transformer paper](https://arxiv.org/html/1706.03762v7) explains
attention's token-count cost and parallelism. With short windows or few feature
tokens, measured overhead and hidden width also matter. Published rankings
on other datasets do not predict the ranking on these option returns.

For the next check-in, retain persistence and the selected compact neural
benchmark, improve the causal sequence representation before adding an
LSTM/TCN, and use validation-only date/regime checks. A later final comparison
should freeze choices first and evaluate the reserved test partition once.
No claim of profitable trading follows from the current error metrics.

## 8. Reproducibility and verification

From the repository root:

```bash
# Start the extension when its output directory has no existing protocol.
.conda/bin/python scripts/train_nifty_mlp_study.py
# Continue an interrupted extension while preserving completed fits.
.conda/bin/python scripts/resume_nifty_mlp_study.py
# Independently reproduce predictions and metrics; then rebuild this report.
.conda/bin/python scripts/verify_nifty_mlp_study.py
.conda/bin/python scripts/build_nifty_mlp_report.py
.conda/bin/python -m pytest -q
```

The runner protects existing study outputs. To intentionally start a new full
study, first archive both `reports/nifty_mlp_study/` and
`data/processed/nifty_mlp_study/`; preserve the literature companion for report
navigation. Model weights and full validation predictions are Git-ignored
local artifacts. Small results and reports remain versionable.

The independent [verification record](verification.json) checks the 24
checkpoint reloads against saved predictions, the five metrics per run,
parameter counts, seeds, source/input hashes, early-stopping choices,
aggregation, and the selection rule. Synthetic tests cover signed outputs,
checkpoint restoration, inference determinism, BatchNorm statistics, residual
gradients, and learning on a known small nonlinear relationship. The report
builder reads saved tables and does not fit models or access test data.

Key files: [protocol.json](protocol.json), [runs.csv](runs.csv),
[architecture_comparison.csv](architecture_comparison.csv),
[inference.csv](inference.csv), [diagnostics.json](diagnostics.json),
[literature.md](literature.md), and the [short check-in answer](check_in.md).

## 9. Short check-in answer

{checkin}
"""


def checkin_text(indexed, persistence, chosen, best, inference):
    one = inference.loc[inference.model.eq(chosen) & inference.batch_size.eq(1)].iloc[0]
    best_change = signed_change(indexed.loc[best, "val_day_mae_mean"], persistence["val_day_mae"])
    comparison = (f"The best MLP's mean date-MAE was {abs(best_change):.2f}% "
                  f"{'higher' if best_change > 0 else 'lower'} than persistence.")
    return ("Based on last week's exploration, we used 15 cleaned features to predict next-minute "
            "Nifty option log returns. We compared eight MLP architectures—including shallow, deeper, "
            "dropout, BatchNorm, and residual networks—across three seeds, with persistence and ridge "
            f"as reference baselines. The {best.replace('_', ' ')} MLP achieved the lowest neural validation date-MAE "
            f"({indexed.loc[best, 'val_day_mae_mean']:.7f}), while our size-versus-accuracy rule selected "
            f"the {chosen.replace('_', ' ')} MLP ({int(indexed.loc[chosen, 'parameters']):,} parameters; approximately "
            f"{one.median_batch_ms * 1000:.0f} µs warm CPU inference per example). {comparison} "
            "We will retain this small neural benchmark and investigate sequence models next; "
            "these are validation findings, with final test evaluation still reserved.")


def update_readme(indexed, persistence, chosen, best, inference):
    rows = [[LABELS[name], STRUCTURES[name], f"{int(row.parameters):,}",
             f"{row.val_day_mae_mean:.7f}", f"{row.val_rmse_mean:.7f}"]
            for name, row in indexed.iterrows()]
    results = table(["MLP", "Structure", "Parameters", "Mean date MAE", "Mean return RMSE"], rows)
    section = f"""<!-- MLP_STUDY_START -->
## Expanded MLP comparison — October 22, 2026

Read the [full report](reports/nifty_mlp_study/report.md),
[rendered HTML](reports/nifty_mlp_study/report.html),
[research references](reports/nifty_mlp_study/literature.md), and
[short weekly check-in](reports/nifty_mlp_study/check_in.md).

The completed extension compares **eight common MLP architectures × three seeds**
on the full 7,069,922 training examples and 2,009,818 chronological validation
examples. It adds 15 new fits and reuses the nine verified October 14 fits with
explicit provenance. All models use the same 15 features, target scaling,
Smooth L1 loss, AdamW, batch size, early-stopping rules, and seed set.
The reserved test data were not evaluated.

{results}

Persistence date-MAE is **{persistence['val_day_mae']:.7f}**. The lowest MLP mean
belongs to **{LABELS[best]}**; the predeclared smallest-model-within-0.5% rule
selects **{LABELS[chosen]}**. Means across seeds describe architecture performance;
seed 42 remains the reference checkpoint. A lower return RMSE alone does not
establish better price accuracy or trading performance.

The report includes seed variability, price errors, measured training times,
fresh CPU inference benchmarks for all eight models, learning curves,
date/price/volume diagnostics, and a sourced discussion of LSTM, TCN, and
FT-Transformer alternatives. Those alternatives have not been trained here.
Plain MLP and residual families are motivated by
[Gorishniy et al. (2021)](https://arxiv.org/html/2106.11959v5); exact widths are
project choices. Prior training times come from the earlier session; all
inference timings are newly measured together.

```bash
# Fresh study only; protects existing outputs.
.conda/bin/python scripts/train_nifty_mlp_study.py
# Resume unfinished runs after an interruption.
.conda/bin/python scripts/resume_nifty_mlp_study.py
# Verify saved results and regenerate report/README without retraining.
.conda/bin/python scripts/verify_nifty_mlp_study.py
.conda/bin/python scripts/build_nifty_mlp_report.py
.conda/bin/python -m pytest -q
```

New checkpoints and prediction arrays are local under
`data/processed/nifty_mlp_study/`; reused checkpoints remain in
`data/processed/nifty_models/`. The prior frozen model and notebook are preserved.
The [verification record](reports/nifty_mlp_study/verification.json) records
independent full-validation checkpoint and metric checks.
<!-- MLP_STUDY_END -->"""
    path = ROOT / "README.md"
    readme = path.read_text()
    start, end = "<!-- MLP_STUDY_START -->", "<!-- MLP_STUDY_END -->"
    if start in readme:
        before, remaining = readme.split(start, 1)
        _, after = remaining.split(end, 1)
        readme = before + section + after
    else:
        readme = readme.rstrip() + "\n\n" + section + "\n"
    path.write_text(readme)


if __name__ == "__main__":
    build()

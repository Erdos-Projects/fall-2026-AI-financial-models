"""Configuration proposers compared under one experiment budget.

Random and Gaussian-process Bayesian search sample the shared parameter space;
GPT's three predeclared hypotheses form a small fixed arm. LLM proposers reach
the runner only through numbered text files: the runner writes a prompt, a
chat model answers (relayed by a person or any other client), and the reply is
parsed and validated before training. LLM text never supplies labels or
metric values, and prompts never contain test-period information.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
from nifty_search_space import (
    BASELINE,
    PARAMETERS,
    SearchConfig,
    parse_proposals,
    sample_random,
)

# The shared baseline plus three random proposals precede Gaussian-process proposals.
BAYES_STARTUP_TRIALS = 4

# Paraphrased from the three first hypotheses in the project's GPT planning notes.
GPT_HYPOTHESES = [
    {"hypothesis": ("Training weights individual observations equally, while the selection metric weights "
                    "dates equally; aligning the training weights with the metric should lower equal-date MAE."),
     "change": "sample_weighting: none -> date_balanced",
     "expected_effect": "Lower equal-date MAE, possibly with a small pooled-RMSE cost.",
     "rejection_criterion": "Objective not lower than the baseline configuration.",
     "config": {**BASELINE.model_dump(), "sample_weighting": "date_balanced"}},
    {"hypothesis": ("The MLP improves large-error behaviour but slightly worsens MAE, so its predictions need "
                    "shrinkage toward persistence: adjusted = alpha * prediction with 0 <= alpha <= 1."),
     "change": "calibration: none -> global_scale",
     "expected_effect": "Lower equal-date MAE while keeping part of the RMSE gain.",
     "rejection_criterion": "Objective not lower than baseline; alpha = 0 would instead support persistence.",
     "config": {**BASELINE.model_dump(), "calibration": "global_scale"}},
    {"hypothesis": ("Almost all squared-error improvement comes from options priced at or below one, so the "
                    "two price groups need separate calibration coefficients."),
     "change": "calibration: none -> price_group_scale",
     "expected_effect": "Lower equal-date MAE than one global coefficient.",
     "rejection_criterion": "Objective not lower than the global_scale hypothesis or the baseline.",
     "config": {**BASELINE.model_dump(), "calibration": "price_group_scale"}},
]


def random_proposal(repeat: int, index: int) -> SearchConfig:
    return sample_random(np.random.default_rng([repeat, index]))


def bayes_proposal(repeat: int, index: int, observed: list[tuple[SearchConfig, float]]) -> SearchConfig:
    """Rebuild a Gaussian-process study from every observed objective and ask for one new point."""
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    distributions = {name: CategoricalDistribution(spec["choices"]) if "choices" in spec
                     else FloatDistribution(spec["low"], spec["high"], log=spec["log"])
                     for name, spec in PARAMETERS.items()}
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.GPSampler(seed=1000 * repeat + index, n_startup_trials=BAYES_STARTUP_TRIALS)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for config, value in observed:
        study.add_trial(optuna.trial.create_trial(params=config.model_dump(), distributions=distributions,
                                                  value=value))
    return SearchConfig(**study.ask(distributions).params).canonical()


def _pct(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def describe_config(values: dict) -> str:
    shown = []
    for name in PARAMETERS:
        if name == "smooth_l1_beta" and values["loss"] != "smooth_l1":
            continue
        value = values[name]
        shown.append(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}")
    return ", ".join(shown)


def result_block(result: dict, baseline_objective: float | None) -> str:
    """Measured diagnostics for one evaluated configuration as compact Markdown.

    ``baseline_objective`` is None when the block describes the baseline itself.
    """
    if result["status"] != "evaluated":
        return f"Training failed: {result['error']}"
    windows = result["windows"]
    versus_baseline = ("" if baseline_objective is None else
                       f"{_pct(result['objective'], baseline_objective):+.3f}% vs the baseline configuration, ")
    headline = (f"Objective {result['objective']:.7f}: {versus_baseline}"
                f"{_pct(result['objective'], result['persistence_objective']):+.3f}% vs persistence.")
    header = ("| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than "
              "persistence | best / run epochs | calibration alpha |")
    lines = [headline, "", header, "|---|---|---|---|---|---|---|---|---|"]
    for number, window in enumerate(windows, 1):
        metric, base = window["metrics"], window["persistence"]
        changes = " | ".join(f"{_pct(metric[key], base[key]):+.2f}%"
                             for key in ["day_mae", "mae", "rmse", "price_mae", "price_rmse"])
        better = sum(m < p for m, p in zip(metric["daily_mae"], base["daily_mae"]))
        alpha = ", ".join(f"{group} {value:.2f}" for group, value in window["calibration"].items()) or "-"
        lines.append(f"| {number} | {changes} | {better}/{len(metric['daily_mae'])} | "
                     f"{window['best_epoch']} / {window['epochs_run']} | {alpha} |")
    lines += ["", "Metric columns are changes relative to persistence on the same score rows (negative is better)."]
    groups = []
    for group, label in [("close_at_most_1", "close <= 1"), ("close_above_1", "close > 1"),
                         ("calls", "calls"), ("puts", "puts")]:
        pairs = [(w["metrics"]["groups"][group], w["persistence"]["groups"][group])
                 for w in windows if group in w["metrics"]["groups"]]
        if pairs:
            mae = np.mean([_pct(m["mae"], p["mae"]) for m, p in pairs])
            rmse = np.mean([_pct(m["rmse"], p["rmse"]) for m, p in pairs])
            groups.append(f"{label}: MAE {mae:+.2f}%, RMSE {rmse:+.2f}%")
    lines.append("Subgroups vs persistence (mean over folds): " + "; ".join(groups) + ".")
    ratio = np.mean([w["metrics"]["mean_abs_prediction"] / w["metrics"]["mean_abs_target"] for w in windows])
    correlation = np.mean([w["metrics"]["correlation"] for w in windows])
    lines.append(f"Prediction size: mean |prediction| / mean |target| = {ratio:.3f}; "
                 f"correlation(prediction, target) = {correlation:.3f} (means over folds).")
    lines.append("Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; "
                 "MAE on a fixed 65,536-row training sample):")
    for number, window in enumerate(windows, 1):
        tune = ", ".join(f"{_pct(h['tune_day_mae'], window['tune_persistence_day_mae']):+.2f}%"
                         for h in window["history"])
        probe = ", ".join(f"{h['train_probe_mae']:.5f}" for h in window["history"])
        lines.append(f"- fold {number}: tune [{tune}]; training sample [{probe}]")
    return "\n".join(lines)


def brief(protocol: dict, baseline: dict) -> str:
    """The diagnostic package shared by both LLM arms: task, rules, objective, search space and diagnostics."""
    inputs = protocol["brief_inputs"]
    data, target, earlier = inputs["data"], inputs["target_profile"], inputs["earlier_findings"]
    features = "\n".join(f"| {name} | {definition} |" for name, definition in inputs["features"].items())
    folds = "\n".join(
        f"| {number} | " + " | ".join(
            f"{w[key]['first_date']} to {w[key]['last_date']} ({w[key]['dates']} dates; {w[key]['rows']:,} rows)"
            for key in ["fit", "tune", "score"]) + " |"
        for number, w in enumerate(protocol["search"]["windows"], 1))
    space = []
    for name, spec in PARAMETERS.items():
        allowed = (", ".join(f'"{choice}"' for choice in spec["choices"]) if "choices" in spec
                   else f"{spec['low']:g} to {spec['high']:g}" + (" (log scale)" if spec["log"] else ""))
        baseline_value = f'"{spec["baseline"]}"' if "choices" in spec else f"{spec['baseline']:g}"
        space.append(f"| {name} | {allowed} | {baseline_value} | {spec['meaning']} |")
    quantiles = "; ".join(f"{float(q):.1%}: {value:.4f} ({value / target['std']:.3f} std)"
                          for q, value in target["abs_quantiles"].items())
    groups = "\n".join(
        f"- {label}: {g['share_rows']:.1%} of rows; std(y) {g['std']:.4f}; mean |y| {g['mean_abs']:.4f}; "
        f"exactly zero in {g['share_exact_zero']:.1%}."
        for label, g in [("close <= 1", target["groups"]["close_at_most_1"]),
                         ("close > 1", target["groups"]["close_above_1"])])
    example = json.dumps({"proposals": [{
        "hypothesis": "...", "change": "...", "predicted_change_pct": -0.2, "expected_effect": "...",
        "rejection_criterion": "...", "config": BASELINE.model_dump()}]})
    return f"""# Experiment brief: improving a small next-minute option-return forecaster

You are the research assistant in a controlled, budgeted experiment. You propose training and calibration settings for a fixed small neural network. A Python program validates each proposal, trains the network, and measures the results; you never see raw rows or labels, and you do not compute any metric yourself. Use only the information in this brief. Do not browse, run code, or draw on earlier conversations about this project.

## Prediction task

- Data: one-minute bars (open, high, low, close, volume) of Nifty index options. This search uses {data['dates']} trading dates from {data['first_date']} to {data['last_date']}.
- Target: y = ln(close[t+1] / close[t]), the log return over the next minute, predicted after bar t is complete.
- Persistence predicts y = 0, i.e. an unchanged next close. It is the reference forecast.
- Each example needs six consecutive observed bars of one option within one date. There are {data['rows']:,} examples; rows per date range from {data['rows_per_date_min']:,} to {data['rows_per_date_max']:,} (median {data['rows_per_date_median']:,}).
- Inputs (15):

| feature | definition |
|---|---|
{features}

The first nine inputs are standardized with fit-window statistics; the last six are cyclic or binary.

## Fixed (cannot be changed)

- Network 15 -> 32 -> 16 -> 1 with ReLU hidden layers and a signed linear output. Output weights and bias start at zero, so training starts from the persistence forecast.
- The training target is y / std(y over the fit window), not centered or clipped. Predictions are converted back to log-return units.
- AdamW, batch size 4,096, at most 12 epochs. After each epoch the tune-window objective is computed: training stops after 3 epochs without an improvement of at least 1e-6, the learning rate halves after each epoch without improvement (minimum 1e-5), and the best epoch is restored. Gradient-norm clipping 1.0. Training seed 42.
- Inputs, data windows and evaluation code.

## Objective (lower is better)

Equal-date MAE: for each date, the mean absolute error of the predicted log return across that date's rows; then the plain average of the daily values, so each date counts equally regardless of its number of rows.

The search objective is the mean equal-date MAE over the score dates of three expanding forward folds:

| fold | fit dates (training) | tune dates | score dates |
|---|---|---|---|
{folds}

The network trains only on fit dates. Tune dates drive early stopping, learning-rate reduction and any calibration. Score dates only produce the metrics you receive.

## Search space

Every proposal sets all eight values.

| parameter | allowed values | baseline | meaning |
|---|---|---|---|
{chr(10).join(space)}

Continuous values are rounded to three significant digits. A configuration identical to the baseline or to an earlier configuration in this search uses up an evaluation without producing new information.

## Diagnostics

### Target distribution in the fold-1 fit dates ({target['first_date']} to {target['last_date']}; {target['rows']:,} rows)

- std(y) = {target['std']:.4f} log-return units (the target scaling unit); mean = {target['mean']:.5f}; exactly zero in {target['share_exact_zero']:.1%} of rows.
- |y| quantiles: {quantiles}; maximum {target['abs_max']:.3f} ({target['abs_max'] / target['std']:.1f} std).
{groups}

### Baseline configuration on the three folds

{result_block(baseline, None)}

### Earlier findings with the baseline configuration

Previously, the same configuration was trained on all {data['dates']} search dates and evaluated on {earlier['dates']} later dates ({earlier['first_date']} to {earlier['last_date']}), which this search does not use. Relative to persistence: equal-date MAE {earlier['day_mae']:+.2f}%, pooled MAE {earlier['mae']:+.2f}%, pooled RMSE {earlier['rmse']:+.2f}%, price MAE {earlier['price_mae']:+.2f}%, price RMSE {earlier['price_rmse']:+.2f}%. It beat persistence on {earlier['dates_better']} of {earlier['dates']} dates. Rows with close <= 1 ({earlier['low_price_share']:.2%} of rows) had MAE {earlier['low_price_mae']:+.2f}% and RMSE {earlier['low_price_rmse']:+.2f}%, and accounted for {earlier['low_price_share_of_squared_error_reduction']:.2%} of the total reduction in squared error; rows with close > 1 had MAE {earlier['high_price_mae']:+.2f}% and RMSE {earlier['high_price_rmse']:+.2f}%.

## Reply format

Reply with one JSON object and nothing else, for example:

{example}

- hypothesis: what you expect and why, citing the diagnostics.
- change: which parameters differ from the configuration you build on, and how.
- predicted_change_pct: your predicted percentage change of the objective relative to the baseline configuration (negative means better).
- expected_effect: what should happen to the objective and the diagnostics.
- rejection_criterion: a measurable result that would reject the hypothesis.
- config: all eight parameters. The example values are the baseline configuration.
"""


def batch_prompt(protocol: dict, baseline: dict, budget: int) -> str:
    return brief(protocol, baseline) + f"""
## Your task

Propose exactly {budget} configurations now. All {budget} will be trained and scored, and you will not see any results before proposing. Afterwards, the configuration with the lowest objective is selected. Every configuration must be complete, different from the others, and different from the baseline.
"""


def feedback_prompt(protocol: dict, baseline: dict, history: list[dict], budget: int) -> str:
    """``history`` items are trial records, each with its full ``result`` when one exists."""
    entries = []
    for record in history:
        proposal = record.get("proposal") or {}
        lines = [f"### Trial {record['trial']}"]
        if proposal:
            lines += [f"Hypothesis: {proposal.get('hypothesis')}", f"Change: {proposal.get('change')}",
                      f"Predicted change: {proposal.get('predicted_change_pct')}%"]
        if record["status"] == "invalid":
            lines.append(f"Invalid proposal, not evaluated: {record['error']}")
        else:
            lines.append(f"Configuration: {describe_config(record['config'])}")
            if record["status"] == "duplicate":
                lines.append(f"Duplicate of trial {record['duplicate_of']} (trial 0 is the baseline); "
                             "no new information.")
            else:
                lines.append(result_block(record["result"], baseline["objective"]))
        entries.append("\n".join(lines))
    scored = [record for record in history if record.get("objective") is not None]
    if scored:
        best = min(scored, key=lambda record: (record["objective"], record["trial"]))
        entries.append(f"Lowest objective so far: trial {best['trial']} ({best['objective']:.7f}, "
                       f"{_pct(best['objective'], baseline['objective']):+.3f}% vs the baseline configuration).")
    elif history:
        entries.append("No proposal in this search has produced a measured objective yet.")
    results = "\n\n".join(entries) if entries else "No evaluations yet."
    return brief(protocol, baseline) + f"""
## Results so far in this search

Evaluations used: {len(history)} of {budget}.

{results}

## Your task

Propose exactly 1 configuration for the next evaluation ({budget - len(history)} remaining, including this one). When the budget is used up, the configuration with the lowest objective in this search is selected.
"""


def repair_prompt(prompt: str, reply: str, errors: list[str], expected: int) -> str:
    problems = "\n".join(f"- {error}" for error in errors)
    return f"""# Correction needed

Your previous reply could not be used as submitted:
{problems}

Send a corrected reply: one JSON object in the required format with exactly {expected} proposal(s), and nothing else. The original request and your previous reply follow.

=== ORIGINAL REQUEST ===
{prompt}
=== YOUR PREVIOUS REPLY ===
{reply}
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


class Handoff:
    """Numbered prompt/reply files for one LLM search; a person or client fills in the replies."""

    def __init__(self, directory: Path, wait: bool, poll_seconds: float = 5.0):
        self.directory = directory
        self.wait = wait
        self.poll_seconds = poll_seconds
        self.count = 0

    def paths(self, number: int) -> tuple[Path, Path, Path]:
        return tuple(self.directory / f"{number:02d}_{name}" for name in ["prompt.md", "reply.txt", "meta.json"])

    def ask(self, text: str) -> dict | None:
        """Write (or verify) the next prompt; return its reply, or None while the reply is pending."""
        self.count += 1
        prompt, reply, meta = self.paths(self.count)
        if prompt.exists():
            if prompt.read_text() != text:
                raise ValueError(f"{prompt} differs from the regenerated prompt; the search state is inconsistent")
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            prompt.write_text(text)
            meta.write_text(json.dumps({"prompt_created_at_utc": now(), "prompt_chars": len(text)}, indent=1) + "\n")
        while not reply.exists():
            if not self.wait:
                return None
            time.sleep(self.poll_seconds)
        details = json.loads(meta.read_text()) if meta.exists() else {}
        answer = reply.read_text()
        return {"number": self.count, "prompt_chars": len(text), "reply_chars": len(answer), "reply": answer,
                "model": details.get("model"), "prompt_created_at_utc": details.get("prompt_created_at_utc"),
                "reply_submitted_at_utc": details.get("reply_submitted_at_utc")}


def converse(handoff: Handoff, prompt: str, expected: int) -> dict | None:
    """One request with at most one repair round. Returns parsed records and message metadata."""
    first = handoff.ask(prompt)
    if first is None:
        return None
    records, problems = parse_proposals(first["reply"], expected)
    errors = problems + [f"Proposal {number}: {record['error']}"
                         for number, record in enumerate(records, 1) if record["error"]]
    messages = [first]
    if errors:
        second = handoff.ask(repair_prompt(prompt, first["reply"], errors, expected))
        if second is None:
            return None
        messages.append(second)
        records, _ = parse_proposals(second["reply"], expected)
    for message in messages:
        message.pop("reply")
    return {"records": records, "messages": messages, "repairs": len(messages) - 1}

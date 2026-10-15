"""Compare and freeze a Nifty MLP baseline using train/validation data only.

No test-partition array or metadata is opened. No LLM is called. The protocol
is written before fitting; all architectures receive the same data and seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nifty_models import (
    MODEL_CONFIGS,
    daily_metrics,
    fit_ridge,
    make_model,
    parameter_count,
    predict_batches,
    predict_ridge,
    regression_metrics,
)
from threadpoolctl import threadpool_limits
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "task": "Next-minute log option-price return from the existing 15 engineered features",
    "architectures": MODEL_CONFIGS,
    "seeds": [17, 42, 2026],
    "reference_seed": 42,
    "training_rows": "All 7,069,922 prepared training examples",
    "validation_rows": "All 2,009,818 prepared validation examples",
    "batch_size": 4096,
    "max_epochs": 12,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-6,
    "learning_rate": 1e-3,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "loss": "SmoothL1Loss(beta=1) on y / training_target_std; no target centering or clipping",
    "gradient_clip_norm": 1.0,
    "lr_schedule": "ReduceLROnPlateau: factor=.5, patience=1, min_lr=1e-5 on validation day MAE",
    "initialization": "PyTorch Linear defaults for hidden layers; zero final weights/bias for zero-return starting forecast",
    "primary_metric": "val_day_mae: mean of the 150 daily mean absolute log-return errors",
    "secondary_metrics": ["val_mae", "val_rmse", "val_price_mae", "val_price_rmse"],
    "selection_rule": "Smallest parameter count within 0.5% of the lowest three-seed mean val_day_mae",
    "ridge_alphas": [1e-4, 1e-2, 1.0],
    "ridge_objective": "mean squared error + alpha * squared coefficient norm, unpenalized intercept",
    "test_evaluated": False,
    "llm_executed": False,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validation_scores(y, predictions, days, close) -> dict:
    return {f"val_{key}": value for key, value in regression_metrics(y, predictions, days, close).items()}


def train_one(name, seed, X, y, valid_X, valid_y, days, close, target_scale, train_probe, artifact, history):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(name, X.shape[1])
    nn.init.zeros_(model.network[-1].weight)
    nn.init.zeros_(model.network[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=1, min_lr=1e-5)
    criterion = nn.SmoothL1Loss(beta=1.0)
    best, best_epoch, bad_epochs, best_state = float("inf"), 0, 0, None
    start_time = time.perf_counter()
    for epoch in range(1, PROTOCOL["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(X))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), PROTOCOL["batch_size"]):
            indices = order[start:start + PROTOCOL["batch_size"]]
            batch_x = torch.from_numpy(np.array(X[indices], dtype=np.float32, copy=True))
            batch_y = torch.from_numpy(np.array(y[indices] / target_scale, dtype=np.float32, copy=True))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss: {name}, seed {seed}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        predictions = predict_batches(model, valid_X, target_scale)
        metrics = validation_scores(valid_y, predictions, days, close)
        probe_predictions = predict_batches(model, X[train_probe], target_scale)
        probe_mae = float(np.mean(np.abs(probe_predictions - y[train_probe])))
        score = metrics["val_day_mae"]
        history.append({"model": name, "seed": seed, "epoch": epoch,
                        "train_scaled_smooth_l1": loss_sum / len(X), "train_probe_mae": probe_mae,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": time.perf_counter() - epoch_start, **metrics})
        if score < best - PROTOCOL["early_stopping_min_delta"]:
            best, best_epoch, bad_epochs = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        scheduler.step(score)
        print(f"{name} seed={seed} epoch={epoch:02d} day_MAE={score:.7f} RMSE={metrics['val_rmse']:.7f} best_epoch={best_epoch}", flush=True)
        if bad_epochs >= PROTOCOL["early_stopping_patience"]:
            break
    if best_state is None:
        raise RuntimeError("No valid model checkpoint")
    model.load_state_dict(best_state)
    predictions = predict_batches(model, valid_X, target_scale)
    metrics = validation_scores(valid_y, predictions, days, close)
    checkpoint = artifact / f"{name}_seed{seed}.pt"
    payload = {"state_dict": model.state_dict(), "architecture": name, "input_dim": X.shape[1],
               "hidden": MODEL_CONFIGS[name]["hidden"], "dropout": MODEL_CONFIGS[name]["dropout"],
               "target_scale": target_scale, "target_center": 0.0, "seed": seed, "best_epoch": best_epoch,
               "validation_metrics": metrics}
    torch.save(payload, checkpoint)
    np.save(artifact / f"{name}_seed{seed}_valid_predictions.npy", predictions)
    row = {"model": name, "seed": seed, "parameters": parameter_count(model), "best_epoch": best_epoch,
           "epochs_run": epoch, "train_seconds": time.perf_counter() - start_time,
           "checkpoint": str(checkpoint.relative_to(ROOT)), **metrics}
    return row


def benchmark_model(model, X, name, target_scale):
    """Warm CPU inference only; excludes I/O, feature engineering and queuing."""
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch_size, repeats in [(1, 500), (4096, 50)]:
            tensor = torch.tensor(np.array(X[:batch_size], copy=True))
            for _ in range(20):
                model(tensor) * target_scale
            times = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                model(tensor) * target_scale
                times.append((time.perf_counter_ns() - start) / 1e6)
            median = float(np.median(times))
            rows.append({"model": name, "batch_size": batch_size, "repeats": repeats,
                         "median_batch_ms": median, "p95_batch_ms": float(np.quantile(times, .95)),
                         "rows_per_second": batch_size * 1000 / median})
    return rows


def run(root=ROOT, overwrite=False, threads=4):
    if root.resolve() != ROOT:
        raise ValueError("Run against this repository's prepared dataset")
    report = root / "reports/nifty_models"
    artifact = root / "data/processed/nifty_models"
    report.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    if (report / "summary.json").exists() and not overwrite:
        raise ValueError("A completed run exists; use --overwrite only for an intentional rerun")
    (report / "summary.json").unlink(missing_ok=True)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data = root / "data/processed/nifty_training"
    manifest = json.loads((data / "manifest.json").read_text())
    verified = json.loads((root / "reports/nifty_preprocessing/verification.json").read_text())
    if (manifest["status"] != "complete" or verified.get("status") != "passed"
            or not verified.get("all_output_hashes_verified")
            or verified["manifest_sha256"] != digest(data / "manifest.json")):
        raise ValueError("Prepared data must have a completed matching verification")
    # Only train and validation artifacts are accessed. Test values remain sealed.
    checked_inputs = {}
    for split in ["train", "valid"]:
        for filename in ["X.npy", "y.npy", "metadata.parquet"]:
            relative = f"{split}/{filename}"
            checksum = digest(data / relative)
            if checksum != manifest["outputs"][relative]["sha256"]:
                raise ValueError(f"Prepared artifact hash changed: {relative}")
            checked_inputs[relative] = checksum
    for filename in ["feature_spec.json", "scaler.json"]:
        checksum = digest(data / filename)
        if checksum != manifest["outputs"][filename]["sha256"]:
            raise ValueError(f"Prepared preprocessing definition changed: {filename}")
        checked_inputs[filename] = checksum
    spec = json.loads((data / "feature_spec.json").read_text())
    X, y = np.load(data / "train/X.npy", mmap_mode="r"), np.load(data / "train/y.npy", mmap_mode="r")
    valid_X = np.load(data / "valid/X.npy", mmap_mode="r")
    valid_y = np.load(data / "valid/y.npy", mmap_mode="r")
    if X.shape[1] != len(spec["feature_columns"]) or X.shape[1] != 15:
        raise ValueError("Unexpected feature dimension")
    metadata = pd.read_parquet(data / "valid/metadata.parquet", columns=["timestamp", "close", "option_type"])
    days = metadata.timestamp.dt.normalize().to_numpy()
    close = metadata.close.to_numpy()
    target_scale = float(np.std(y, dtype=np.float64))
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training target must have a positive finite standard deviation")
    target_scaler = {"scale": target_scale, "center": 0.0, "observed_train_mean": float(np.mean(y, dtype=np.float64)),
                     "fit_rows": len(y), "fit_split": "train", "clipping": False}
    hardware = {"system": platform.system(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                "device": "cpu", "torch_threads": torch.get_num_threads(),
                "versions": {name: version(name) for name in ["torch", "numpy", "pandas"]}}
    protocol = {**PROTOCOL, "created_at_utc": datetime.now(UTC).isoformat(), "train_rows": len(y),
                "valid_rows": len(valid_y), "feature_columns": spec["feature_columns"],
                "model_source_sha256": digest(root / "scripts/nifty_models.py"),
                "runner_source_sha256": digest(Path(__file__)), "input_hashes": checked_inputs,
                "preprocessing_manifest_sha256": digest(data / "manifest.json"),
                "feature_scaler_sha256": digest(data / "scaler.json"), "hardware": hardware,
                "train_probe_rows": 65536, "train_probe_seed": 9102}
    write_json(report / "protocol.json", protocol)
    write_json(artifact / "target_scaler.json", target_scaler)
    print("Protocol saved. Fitting on all training rows; test arrays are not opened.", flush=True)

    baseline_rows = []
    zero_predictions = np.zeros(len(valid_y), dtype=np.float64)
    baseline_rows.append({"model": "persistence", "alpha": None, "parameters": 0, "fit_seconds": 0.0,
                          **validation_scores(valid_y, zero_predictions, days, close)})
    ridge_predictions = {}
    for alpha in PROTOCOL["ridge_alphas"]:
        start = time.perf_counter()
        coefficients = fit_ridge(X, y, alpha)
        elapsed = time.perf_counter() - start
        predictions = predict_ridge(valid_X, coefficients)
        name = f"ridge_{alpha:g}"
        ridge_predictions[name] = predictions
        np.save(artifact / f"{name}_coefficients.npy", coefficients)
        baseline_rows.append({"model": name, "alpha": alpha, "parameters": len(coefficients), "fit_seconds": elapsed,
                              **validation_scores(valid_y, predictions, days, close)})
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(report / "baseline_comparison.csv", index=False)
    best_ridge = baselines.loc[baselines.model.ne("persistence")].sort_values("val_day_mae").iloc[0]
    print(baselines.to_string(index=False), flush=True)
    train_probe = np.sort(np.random.default_rng(9102).choice(len(y), size=min(65536, len(y)), replace=False))
    runs, history = [], []
    for name in MODEL_CONFIGS:
        for seed in PROTOCOL["seeds"]:
            runs.append(train_one(name, seed, X, y, valid_X, valid_y, days, close, target_scale, train_probe, artifact, history))
            pd.DataFrame(runs).to_csv(report / "runs.csv", index=False)
            pd.DataFrame(history).to_csv(report / "history.csv", index=False)
    runs_frame = pd.DataFrame(runs)
    comparison = []
    for name, group in runs_frame.groupby("model", sort=False):
        row = {"model": name, "hidden": "-".join(map(str, MODEL_CONFIGS[name]["hidden"])),
               "parameters": int(group.parameters.iloc[0]), "runs": len(group),
               "train_seconds_mean": float(group.train_seconds.mean())}
        for metric in PROTOCOL["secondary_metrics"] + ["val_day_mae"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        comparison.append(row)
    comparison_frame = pd.DataFrame(comparison)
    threshold = comparison_frame.val_day_mae_mean.min() * 1.005
    eligible = comparison_frame.loc[comparison_frame.val_day_mae_mean <= threshold]
    chosen = eligible.sort_values(["parameters", "val_day_mae_mean"]).iloc[0]
    chosen_name = chosen.model
    comparison_frame["within_selection_tolerance"] = comparison_frame.val_day_mae_mean.le(threshold)
    comparison_frame["selected"] = comparison_frame.model.eq(chosen_name)
    comparison_frame.to_csv(report / "architecture_comparison.csv", index=False)
    reference = runs_frame.loc[runs_frame.model.eq(chosen_name) & runs_frame.seed.eq(PROTOCOL["reference_seed"])].iloc[0]
    reference_checkpoint = root / reference.checkpoint
    checkpoint = artifact / "frozen_mlp.pt"
    shutil.copyfile(reference_checkpoint, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected_model = make_model(payload["architecture"], payload["input_dim"])
    selected_model.load_state_dict(payload["state_dict"])
    selected_predictions = predict_batches(selected_model, valid_X, target_scale)
    saved_predictions = np.load(artifact / f"{chosen_name}_seed42_valid_predictions.npy")
    if not np.array_equal(selected_predictions, saved_predictions):
        raise ValueError("Reloaded frozen checkpoint predictions differ from saved validation predictions")
    if not all(torch.isfinite(weight).all() for weight in selected_model.parameters()):
        raise ValueError("Nonfinite frozen model weights")
    selected_metrics = validation_scores(valid_y, selected_predictions, days, close)

    inference = []
    for name in MODEL_CONFIGS:
        payload_for_timing = torch.load(artifact / f"{name}_seed42.pt", map_location="cpu", weights_only=True)
        model_for_timing = make_model(name)
        model_for_timing.load_state_dict(payload_for_timing["state_dict"])
        inference.extend(benchmark_model(model_for_timing, valid_X, name, target_scale))
    pd.DataFrame(inference).to_csv(report / "inference.csv", index=False)

    daily_frames = []
    for name, predictions in [("persistence", zero_predictions), (best_ridge.model, ridge_predictions[best_ridge.model]),
                              ("frozen_mlp", selected_predictions)]:
        frame = daily_metrics(valid_y, predictions, days, close)
        frame.insert(0, "model", name)
        daily_frames.append(frame)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily.to_csv(report / "daily_validation.csv", index=False)
    subgroup_rows = []
    masks = {"calls": metadata.option_type.eq("CE").to_numpy(), "puts": metadata.option_type.eq("PE").to_numpy(),
             "zero_volume": valid_X[:, spec["feature_columns"].index("zero_volume")] == 1,
             "positive_volume": valid_X[:, spec["feature_columns"].index("zero_volume")] == 0,
             "close_at_most_1": close <= 1, "close_above_1": close > 1,
             "flat_bar": valid_X[:, spec["feature_columns"].index("flat_ohlc")] == 1,
             "nonflat_bar": valid_X[:, spec["feature_columns"].index("flat_ohlc")] == 0}
    for group, mask in masks.items():
        if not mask.any():
            continue
        for name, predictions in [("persistence", zero_predictions), ("frozen_mlp", selected_predictions)]:
            subgroup_rows.append({"group": group, "model": name, "rows": int(mask.sum()),
                                  **validation_scores(valid_y[mask], predictions[mask], days[mask], close[mask])})
    pd.DataFrame(subgroup_rows).to_csv(report / "subgroups.csv", index=False)
    daily_comparison = daily.pivot(index="day", columns="model", values="mae")
    difference = (daily_comparison.frozen_mlp - daily_comparison.persistence).to_numpy()
    rng = np.random.default_rng(8801)
    block_length = 5
    starts = rng.integers(0, len(difference), size=(2000, int(np.ceil(len(difference) / block_length))))
    indices = ((starts[..., None] + np.arange(block_length)) % len(difference)).reshape(2000, -1)[:, :len(difference)]
    bootstrapped = difference[indices].mean(axis=1)
    uncertainty = {"comparison": "frozen MLP day MAE minus persistence day MAE; negative favors MLP",
                   "mean_difference": float(difference.mean()), "descriptive_95pct_interval": np.quantile(bootstrapped, [.025, .975]).tolist(),
                   "method": "2000 circular moving-block resamples of validation days; block length 5; seed 8801",
                   "caveat": "Exploratory validation comparison after architecture selection, not an unbiased test confidence interval; nonstationarity may invalidate block assumptions"}
    frozen_config = {"architecture": chosen_name, "seed": 42, "input_dim": 15,
                     "hidden": MODEL_CONFIGS[chosen_name]["hidden"], "dropout": .1,
                     "activation": "ReLU", "output": "unconstrained linear scalar",
                     "parameters": int(chosen.parameters), "checkpoint": str(checkpoint.relative_to(root)),
                     "checkpoint_sha256": digest(checkpoint), "best_epoch": int(reference.best_epoch),
                     "feature_columns": spec["feature_columns"], "target_scale": target_scale,
                     "target_center": 0.0, "preprocessing_manifest_sha256": digest(data / "manifest.json"),
                     "selection_rule": PROTOCOL["selection_rule"], "reference_seed_predeclared": True,
                     "test_evaluated": False, "llm_executed": False}
    write_json(report / "frozen_config.json", frozen_config)
    summary = {"status": "complete", "finished_at_utc": datetime.now(UTC).isoformat(),
               "selected_model": frozen_config, "selected_validation_metrics": selected_metrics,
               "best_ridge": best_ridge.to_dict(), "persistence_validation_metrics": baseline_rows[0],
               "architecture_mean_day_mae_improvement_percent": float(100 * (1 - chosen.val_day_mae_mean / baseline_rows[0]["val_day_mae"])),
               "reference_day_mae_improvement_percent": float(100 * (1 - selected_metrics["val_day_mae"] / baseline_rows[0]["val_day_mae"])),
               "protocol": protocol, "hardware": hardware, "target_scaler": target_scaler,
               "selection_rule": PROTOCOL["selection_rule"], "uncertainty": uncertainty,
               "validation_days": int(pd.Series(days).nunique()), "checkpoint_reload_exact_match": True,
               "test_evaluated": False, "llm_executed": False,
               "limitations": ["Validation guided architecture selection, early stopping and learning-rate scheduling",
                               "Earlier EDA inspected all periods, but this modeling workflow did not open test arrays",
                               "Reported bar prices are not executable quotes; errors do not establish trading profitability",
                               "Input features summarize six bars; recurrent/convolutional/attention sequence models require a separate sequence adapter",
                               "Target scaling is fitted on training only; absolute errors are reported in original log-return or reported price units"]}
    write_json(report / "summary.json", summary)
    print(comparison_frame.to_string(index=False), flush=True)
    print(f"Frozen architecture: {chosen_name}, seed 42, {int(chosen.parameters):,} parameters. Test and LLM stages not run.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=args.threads):
        run(overwrite=args.overwrite, threads=args.threads)

"""Resume interrupted MLP fits without changing the frozen study protocol.

Completed seeds are hash-checked and reused; an interrupted seed restarts from
its original seed using the original fit_one function. No test data are read.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_nifty_mlp_study import (
    CONFIGS,
    METRICS,
    MODEL_CONFIGS,
    PROTOCOL,
    REPORT,
    ROOT,
    benchmark_model,
    daily_metrics,
    digest,
    fit_one,
    load_model,
    predict_batches,
    validation_scores,
    write_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_state():
    """Verify the frozen inputs and every completed seed before reuse."""
    require(not (REPORT / "summary.json").exists(), "Study is already complete")
    protocol = json.loads((REPORT / "protocol.json").read_text())
    require(protocol["architectures"] == {**MODEL_CONFIGS, **CONFIGS}, "Architecture drift")
    for key, value in PROTOCOL.items():
        if key != "architectures":
            require(protocol[key] == value, f"Training protocol drift: {key}")
    for path, checksum in {**protocol["source_hashes"], **protocol["reused_report_hashes"]}.items():
        require(digest(ROOT / path) == checksum, f"Frozen file changed: {path}")
    hardware = protocol["hardware"]
    require(platform.system() == hardware["system"] and platform.machine() == hardware["machine"],
            "Resume host platform differs")
    for package, expected in hardware["versions"].items():
        require(version(package) == expected, f"Package version changed: {package}")
    data = ROOT / "data/processed/nifty_training"
    require(digest(data / "manifest.json") == protocol["preprocessing_manifest_sha256"],
            "Preprocessing manifest changed")
    for path, checksum in protocol["input_hashes"].items():
        require(path in {f"{split}/{name}" for split in ("train", "valid")
                         for name in ("X.npy", "y.npy", "metadata.parquet")}
                | {"feature_spec.json", "scaler.json"}, "Unexpected input path")
        require(digest(data / path) == checksum, f"Training/validation input changed: {path}")
    runs = pd.read_csv(REPORT / "runs.csv")
    history = pd.read_csv(REPORT / "history.csv")
    pairs = set(zip(runs.model, runs.seed))
    expected = {(name, seed) for name in protocol["architectures"] for seed in protocol["seeds"]}
    prior = {(name, seed) for name in MODEL_CONFIGS for seed in protocol["seeds"]}
    require(len(pairs) == len(runs) and prior <= pairs <= expected, "Invalid completed run set")
    require(set(zip(history.model, history.seed)) == pairs, "History/run pairs differ")
    for row in runs.itertuples(index=False):
        origin = "nifty_models" if row.model in MODEL_CONFIGS else "nifty_mlp_study"
        checkpoint = ROOT / f"data/processed/{origin}/{row.model}_seed{row.seed}.pt"
        require(row.checkpoint == str(checkpoint.relative_to(ROOT)), "Unexpected checkpoint path")
        prediction = checkpoint.with_name(checkpoint.stem + "_valid_predictions.npy")
        require(digest(checkpoint) == row.checkpoint_sha256, f"Checkpoint changed: {checkpoint.name}")
        require(digest(prediction) == row.predictions_sha256, f"Predictions changed: {prediction.name}")
        epochs = history.loc[history.model.eq(row.model) & history.seed.eq(row.seed)].sort_values("epoch")
        require(epochs.epoch.tolist() == list(range(1, int(row.epochs_run) + 1)), "Incomplete saved history")
        require(1 <= row.best_epoch <= row.epochs_run <= protocol["max_epochs"], "Invalid epoch count")
        require(np.isfinite(np.asarray([getattr(row, metric) for metric in METRICS])).all(),
                "Nonfinite saved metrics")
    return protocol, data, runs, history


def finalize(runs, protocol, vx, vy, metadata, days, close, scale):
    """The original runner's explicit post-training aggregation and diagnostics."""
    aggregate = []
    for name, group in runs.groupby("model", sort=False):
        row = {"model": name, "parameters": int(group.parameters.iloc[0]), "runs": len(group),
               "origin": group.origin.iloc[0], "train_seconds_mean": float(group.train_seconds.mean()),
               "epochs_run_mean": float(group.epochs_run.mean())}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        aggregate.append(row)
    comparison = pd.DataFrame(aggregate)
    threshold = comparison.val_day_mae_mean.min() * 1.005
    eligible = comparison.loc[comparison.val_day_mae_mean <= threshold]
    chosen = eligible.sort_values(["parameters", "val_day_mae_mean"]).iloc[0].model
    best = comparison.sort_values("val_day_mae_mean").iloc[0].model
    comparison["within_selection_tolerance"] = comparison.val_day_mae_mean.le(threshold)
    comparison["selected"] = comparison.model.eq(chosen)
    comparison.to_csv(REPORT / "architecture_comparison.csv", index=False)
    inference, daily_frames = [], []
    for name in protocol["architectures"]:
        row = runs.loc[runs.model.eq(name) & runs.seed.eq(42)].iloc[0]
        model = load_model(name, ROOT / row.checkpoint)
        inference.extend(benchmark_model(model, vx, name, scale))
        predictions = predict_batches(model, vx, scale)
        frame = daily_metrics(vy, predictions, days, close)
        frame.insert(0, "model", name)
        daily_frames.append(frame)
    pd.DataFrame(inference).to_csv(REPORT / "inference.csv", index=False)
    zero = np.zeros(len(vy))
    persistence = validation_scores(vy, zero, days, close)
    frame = daily_metrics(vy, zero, days, close)
    frame.insert(0, "model", "persistence")
    daily_frames.append(frame)
    pd.concat(daily_frames).to_csv(REPORT / "daily_validation.csv", index=False)
    diagnostic_names = list(dict.fromkeys([chosen, best]))
    subgroup_rows = []
    masks = {"calls": metadata.option_type.eq("CE").to_numpy(), "puts": metadata.option_type.eq("PE").to_numpy(),
             "close_at_most_1": close <= 1, "close_above_1": close > 1,
             "zero_volume": vx[:, protocol["feature_columns"].index("zero_volume")] == 1,
             "positive_volume": vx[:, protocol["feature_columns"].index("zero_volume")] == 0}
    for name in ["persistence", *diagnostic_names]:
        if name == "persistence":
            predictions = zero
        else:
            row = runs.loc[runs.model.eq(name) & runs.seed.eq(42)].iloc[0]
            checkpoint = ROOT / row.checkpoint
            predictions = np.load(checkpoint.with_name(checkpoint.stem + "_valid_predictions.npy"), mmap_mode="r")
        for group, mask in masks.items():
            if mask.any():
                subgroup_rows.append({"model": name, "group": group, "rows": int(mask.sum()),
                                      **validation_scores(vy[mask], predictions[mask], days[mask], close[mask])})
    pd.DataFrame(subgroup_rows).to_csv(REPORT / "subgroups.csv", index=False)
    selected = runs.loc[runs.model.eq(chosen) & runs.seed.eq(42)].iloc[0]
    summary = {"status": "complete", "finished_at_utc": datetime.now(UTC).isoformat(),
               "protocol_sha256": digest(REPORT / "protocol.json"),
               "selected_architecture": chosen, "best_mean_architecture": best,
               "selected_reference_metrics": {m: float(selected[m]) for m in METRICS},
               "selected_checkpoint": selected.checkpoint, "reference_seed": 42,
               "persistence_metrics": persistence, "target_scale": scale,
               "architectures": len(comparison), "total_runs": len(runs), "new_runs": 15, "reused_runs": 9,
               "test_evaluated": False, "llm_executed": False,
               "limitations": ["Same validation period reused for architecture selection; no untouched test claim",
                               "Earlier EDA inspected all dates; model development opens train/valid only",
                               "Original training times come from a prior session; fresh inference timings cover every model",
                               "Three seed standard deviations describe optimization variability, not market uncertainty",
                               "Models use engineered summaries, not raw sequences or theoretical pricing inputs"]}
    write_json(REPORT / "summary.json", summary)
    print(comparison.to_string(index=False), flush=True)
    print(f"Complete: selected={chosen}; best mean={best}; test not evaluated.", flush=True)


def run(check_only=False):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protocol, data, runs, history = checked_state()
    completed = set(zip(runs.model, runs.seed))
    missing = [(name, seed) for name in CONFIGS for seed in PROTOCOL["seeds"]
               if (name, seed) not in completed]
    print(f"Verified {len(completed)} completed fits; {len(missing)} fits remain: {missing}", flush=True)
    if check_only:
        return
    X = np.load(data / "train/X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(data / "train/y.npy", mmap_mode="r", allow_pickle=False)
    vx = np.load(data / "valid/X.npy", mmap_mode="r", allow_pickle=False)
    vy = np.load(data / "valid/y.npy", mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(data / "valid/metadata.parquet", columns=["timestamp", "close", "option_type"])
    days, close = metadata.timestamp.dt.normalize().to_numpy(), metadata.close.to_numpy()
    scale = float(np.std(y, dtype=np.float64))
    require(scale == protocol["target_scale"], "Training target scale changed")
    require(len(y) == protocol["train_rows"] and len(vy) == protocol["valid_rows"], "Row counts changed")
    resume_path = REPORT / "resume_history.json"
    audit = json.loads(resume_path.read_text()) if resume_path.exists() else {"sessions": []}
    event = {"started_at_utc": datetime.now(UTC).isoformat(), "status": "running",
             "resume_source_sha256": digest(Path(__file__)),
             "protocol_sha256": digest(REPORT / "protocol.json"),
             "completed_at_start": [[name, int(seed)] for name, seed in sorted(completed)],
             "missing_at_start": [[name, int(seed)] for name, seed in missing],
             "runs_sha256_at_start": digest(REPORT / "runs.csv"),
             "history_sha256_at_start": digest(REPORT / "history.csv"),
             "restart_policy": "Incomplete seeds restart with their original seed and unchanged fit_one"}
    audit["sessions"].append(event)
    write_json(resume_path, audit)
    runs, history = runs.to_dict("records"), history.to_dict("records")
    probe = np.sort(np.random.default_rng(9102).choice(len(y), size=min(65536, len(y)), replace=False))
    for name, seed in missing:
        runs.append(fit_one(name, seed, X, y, vx, vy, days, close, scale, probe, history))
        pd.DataFrame(runs).to_csv(REPORT / "runs.csv", index=False)
        pd.DataFrame(history).to_csv(REPORT / "history.csv", index=False)
    finalize(pd.DataFrame(runs), protocol, vx, vy, metadata, days, close, scale)
    event.update(status="complete", finished_at_utc=datetime.now(UTC).isoformat(),
                 total_completed_runs=len(runs))
    write_json(resume_path, audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify resumability without fitting or writing")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(check_only=args.check_only)

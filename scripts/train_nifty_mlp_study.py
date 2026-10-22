"""Extend the verified Nifty baseline study with common MLP families.

The original nine full-data fits are reused with explicit provenance. Five new
families receive the identical full training data, loss, budget and three seeds.
No test arrays or test metadata are read. Existing studies are never overwritten.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nifty_mlp_variants import CONFIGS, build_model
from nifty_models import (
    MODEL_CONFIGS,
    daily_metrics,
    make_model,
    parameter_count,
    predict_batches,
)
from threadpoolctl import threadpool_limits
from torch import nn
from train_nifty_models import (
    PROTOCOL,
    benchmark_model,
    digest,
    validation_scores,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/nifty_mlp_study"
ARTIFACT = ROOT / "data/processed/nifty_mlp_study"
METRICS = ["val_day_mae", "val_mae", "val_rmse", "val_price_mae", "val_price_rmse"]
SOURCES = ["scripts/nifty_models.py", "scripts/train_nifty_models.py",
           "scripts/nifty_mlp_variants.py", "scripts/train_nifty_mlp_study.py"]


def fit_one(name, seed, X, y, valid_X, valid_y, days, close, scale, probe, history):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name, X.shape[1])
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"],
                                  weight_decay=PROTOCOL["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=1, min_lr=1e-5)
    criterion = nn.SmoothL1Loss(beta=1.)
    best, best_epoch, stale, state = float("inf"), 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, PROTOCOL["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(X))
        model.train()
        total_loss = 0.
        for start in range(0, len(order), PROTOCOL["batch_size"]):
            indices = order[start:start + PROTOCOL["batch_size"]]
            bx = torch.from_numpy(np.array(X[indices], dtype=np.float32, copy=True))
            by = torch.from_numpy(np.array(y[indices] / scale, dtype=np.float32, copy=True))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {name}/{seed}/{epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        predictions = predict_batches(model, valid_X, scale)
        metrics = validation_scores(valid_y, predictions, days, close)
        probe_predictions = predict_batches(model, X[probe], scale)
        score = metrics["val_day_mae"]
        history.append({"model": name, "seed": seed, "epoch": epoch,
                        "train_scaled_smooth_l1": total_loss / len(X),
                        "train_probe_mae": float(np.mean(np.abs(probe_predictions - y[probe]))),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": time.perf_counter() - epoch_start, **metrics})
        if score < best - PROTOCOL["early_stopping_min_delta"]:
            best, best_epoch, stale = score, epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        scheduler.step(score)
        print(f"{name} seed={seed} epoch={epoch:02d} day_MAE={score:.8f} "
              f"RMSE={metrics['val_rmse']:.7f} best={best_epoch}", flush=True)
        if stale >= PROTOCOL["early_stopping_patience"]:
            break
    if state is None:
        raise ValueError("No finite checkpoint")
    model.load_state_dict(state)
    predictions = predict_batches(model, valid_X, scale)
    metrics = validation_scores(valid_y, predictions, days, close)
    checkpoint = ARTIFACT / f"{name}_seed{seed}.pt"
    prediction_path = ARTIFACT / f"{name}_seed{seed}_valid_predictions.npy"
    torch.save({"state_dict": model.state_dict(), "architecture": name, "input_dim": X.shape[1],
                "config": CONFIGS[name], "target_scale": scale, "target_center": 0.,
                "seed": seed, "best_epoch": best_epoch, "validation_metrics": metrics}, checkpoint)
    np.save(prediction_path, predictions)
    return {"model": name, "seed": seed, "parameters": parameter_count(model),
            "best_epoch": best_epoch, "epochs_run": epoch,
            "train_seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "origin": "new_run",
            "checkpoint_sha256": digest(checkpoint), "predictions_sha256": digest(prediction_path),
            **metrics}


def load_model(name, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model(name) if name in MODEL_CONFIGS else build_model(name)
    model.load_state_dict(payload["state_dict"])
    return model


def run():
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    if (REPORT / "protocol.json").exists():
        raise ValueError("Study already started; preserve it or explicitly move its outputs before rerunning")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    prior = ROOT / "reports/nifty_models"
    data = ROOT / "data/processed/nifty_training"
    old_protocol = json.loads((prior / "protocol.json").read_text())
    old_verification = json.loads((prior / "verification.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    if old_verification["status"] != "passed" or manifest["status"] != "complete":
        raise ValueError("Prior study and preprocessing must be complete and verified")
    if digest(data / "manifest.json") != old_protocol["preprocessing_manifest_sha256"]:
        raise ValueError("Preprocessing manifest changed")
    inputs = [f"{split}/{file}" for split in ("train", "valid")
              for file in ("X.npy", "y.npy", "metadata.parquet")] + ["feature_spec.json", "scaler.json"]
    hashes = {relative: digest(data / relative) for relative in inputs}
    for relative, checksum in hashes.items():
        if checksum != old_protocol["input_hashes"][relative] or checksum != manifest["outputs"][relative]["sha256"]:
            raise ValueError(f"Changed training/validation input: {relative}")
    for relative, checksum in old_verification["saved_artifact_hashes"].items():
        if digest(ROOT / relative) != checksum:
            raise ValueError(f"Changed verified prior artifact: {relative}")
    for relative, key in [("scripts/nifty_models.py", "model_source_sha256"),
                          ("scripts/train_nifty_models.py", "runner_source_sha256")]:
        if digest(ROOT / relative) != old_protocol[key]:
            raise ValueError(f"Changed prior implementation: {relative}")
    X = np.load(data / "train/X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(data / "train/y.npy", mmap_mode="r", allow_pickle=False)
    vx = np.load(data / "valid/X.npy", mmap_mode="r", allow_pickle=False)
    vy = np.load(data / "valid/y.npy", mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(data / "valid/metadata.parquet", columns=["timestamp", "close", "option_type"])
    days, close = metadata.timestamp.dt.normalize().to_numpy(), metadata.close.to_numpy()
    scale = float(np.std(y, dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training target scale")
    reused = ["summary.json", "protocol.json", "runs.csv", "history.csv", "verification.json"]
    hardware = {"system": platform.system(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                "device": "cpu", "torch_threads": 4,
                "versions": {p: version(p) for p in ["torch", "numpy", "pandas", "scipy", "scikit-learn"]}}
    protocol = {**PROTOCOL, "created_at_utc": datetime.now(UTC).isoformat(),
                "architectures": {**MODEL_CONFIGS, **CONFIGS},
                "train_rows": len(y), "valid_rows": len(vy), "validation_days": int(pd.Series(days).nunique()),
                "feature_columns": old_protocol["feature_columns"], "target_scale": scale,
                "source_hashes": {p: digest(ROOT / p) for p in SOURCES}, "input_hashes": hashes,
                "preprocessing_manifest_sha256": digest(data / "manifest.json"),
                "reused_report_hashes": {str((prior / p).relative_to(ROOT)): digest(prior / p) for p in reused},
                "reused_architectures": list(MODEL_CONFIGS), "new_architectures": list(CONFIGS),
                "hardware": hardware, "reuse_policy": "Nine verified prior full-data fits; no retraining claimed",
                "timing_policy": "New warm CPU inference for all eight; fit times measured in each original fitting session",
                "comparison_scope": "Controlled training recipe, not individually tuned or parameter-matched families",
                "reference_seed": 42, "selection_tolerance": .005}
    write_json(REPORT / "protocol.json", protocol)
    print("Protocol saved; full train/validation data checked. Reusing 9 verified fits; fitting 15 new runs.", flush=True)
    runs = pd.read_csv(prior / "runs.csv").to_dict("records")
    for row in runs:
        checkpoint = ROOT / row["checkpoint"]
        prediction = checkpoint.with_name(checkpoint.stem + "_valid_predictions.npy")
        row.update(origin="prior_study", checkpoint_sha256=digest(checkpoint), predictions_sha256=digest(prediction))
    history = pd.read_csv(prior / "history.csv").to_dict("records")
    probe = np.sort(np.random.default_rng(9102).choice(len(y), size=min(65536, len(y)), replace=False))
    for name in CONFIGS:
        for seed in PROTOCOL["seeds"]:
            runs.append(fit_one(name, seed, X, y, vx, vy, days, close, scale, probe, history))
            pd.DataFrame(runs).to_csv(REPORT / "runs.csv", index=False)
            pd.DataFrame(history).to_csv(REPORT / "history.csv", index=False)
    runs = pd.DataFrame(runs)
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


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()

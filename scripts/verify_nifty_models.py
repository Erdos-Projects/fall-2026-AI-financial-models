"""Independently reload and verify a completed train/validation model experiment.

The verifier does not fit models, call an LLM, or open test arrays/metadata.
It regenerates every MLP candidate's complete validation predictions, checks
the saved metrics and architecture selection, and verifies artifact hashes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nifty_models import (
    MODEL_CONFIGS,
    make_model,
    parameter_count,
    predict_batches,
    predict_ridge,
    regression_metrics,
)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"compact": 1057, "standard": 3137, "wide": 12417}
EXPECTED_SEEDS = [17, 42, 2026]
METRICS = ["val_day_mae", "val_mae", "val_rmse", "val_price_mae", "val_price_rmse"]
EXPECTED_INPUTS = {
    "train/X.npy", "train/y.npy", "train/metadata.parquet",
    "valid/X.npy", "valid/y.npy", "valid/metadata.parquet",
    "feature_spec.json", "scaler.json",
}
BATCH_SIZE = 65536


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def close_number(actual, expected, label, rtol=1e-10, atol=1e-12):
    require(np.isfinite(actual) and np.isfinite(expected), f"Nonfinite {label}")
    require(np.isclose(actual, expected, rtol=rtol, atol=atol),
            f"{label}: {actual!r} != {expected!r}")


def score_predictions(y, predictions, days, close):
    return {f"val_{name}": value for name, value in regression_metrics(y, predictions, days, close).items()}


def compare_metrics(actual, expected, label):
    for metric in METRICS:
        close_number(float(actual[metric]), float(expected[metric]), f"{label}/{metric}")


def target_statistics(y):
    """Recalculate the population standard deviation in bounded-memory passes."""
    total = 0.0
    for start in range(0, len(y), BATCH_SIZE):
        block = np.asarray(y[start:start + BATCH_SIZE], dtype=np.float64)
        require(np.isfinite(block).all(), "Nonfinite training targets")
        total += float(block.sum(dtype=np.float64))
    mean = total / len(y)
    squares = 0.0
    for start in range(0, len(y), BATCH_SIZE):
        residual = np.asarray(y[start:start + BATCH_SIZE], dtype=np.float64) - mean
        squares += float(np.sum(residual * residual, dtype=np.float64))
    return mean, float(np.sqrt(squares / len(y)))


def checkpoint_model(path, name, seed, target_scale, best_epoch):
    # Restrict loading to tensors and primitive metadata, never arbitrary pickle objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["architecture"] == name, f"Wrong architecture in {path.name}")
    require(payload["seed"] == seed, f"Wrong seed in {path.name}")
    require(payload["input_dim"] == 15, f"Wrong input dimension in {path.name}")
    require(payload["hidden"] == MODEL_CONFIGS[name]["hidden"], f"Wrong hidden layers in {path.name}")
    close_number(payload["dropout"], MODEL_CONFIGS[name]["dropout"], f"{path.name}/dropout")
    close_number(payload["target_scale"], target_scale, f"{path.name}/target_scale")
    require(payload["target_center"] == 0.0, f"Unexpected target centering in {path.name}")
    require(payload["best_epoch"] == best_epoch, f"Wrong best epoch in {path.name}")
    require(bool(payload["state_dict"]), f"Missing weights in {path.name}")
    require(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item()
                for value in payload["state_dict"].values()), f"Nonfinite weights in {path.name}")
    model = make_model(name, input_dim=15)
    model.load_state_dict(payload["state_dict"], strict=True)
    require(parameter_count(model) == EXPECTED_COUNTS[name], f"Wrong parameter count in {path.name}")
    return model, payload


def compare_prediction_arrays(actual, expected, label):
    require(actual.shape == expected.shape, f"Wrong prediction shape for {label}")
    for start in range(0, len(actual), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(actual))
        require(np.array_equal(actual[start:stop], expected[start:stop]),
                f"Reloaded predictions differ from saved predictions for {label}, rows {start}:{stop}")


def verify(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, "Verify this repository's finished experiment")
    report = root / "reports/nifty_models"
    artifact = root / "data/processed/nifty_models"
    data = root / "data/processed/nifty_training"
    output = report / "verification.json"
    # A failed new verification must not leave a stale passed status behind.
    output.unlink(missing_ok=True)
    summary = read_json(report / "summary.json")
    protocol = read_json(report / "protocol.json")
    frozen = read_json(report / "frozen_config.json")
    scaler = read_json(artifact / "target_scaler.json")
    manifest = read_json(data / "manifest.json")
    spec = read_json(data / "feature_spec.json")
    require(summary["status"] == "complete", "Training run has not finished")
    for name, mapping in [("summary", summary), ("protocol", protocol), ("frozen", frozen)]:
        require(mapping["test_evaluated"] is False, f"{name} reports test evaluation")
        require(mapping["llm_executed"] is False, f"{name} reports LLM execution")
    require(summary["protocol"] == protocol, "Summary and saved protocol disagree")
    require(summary["selected_model"] == frozen, "Summary and frozen configuration disagree")
    require(summary["target_scaler"] == scaler, "Summary and target scaler disagree")
    require(protocol["architectures"] == MODEL_CONFIGS, "Protocol architectures changed")
    require(protocol["seeds"] == EXPECTED_SEEDS, "The three predeclared seeds changed")
    require(protocol["reference_seed"] == 42 and frozen["seed"] == 42
            and frozen["reference_seed_predeclared"] is True, "Reference seed 42 was not preserved")
    require(frozen["selection_rule"] == summary["selection_rule"] == protocol["selection_rule"],
            "Selection-rule descriptions disagree")
    require(protocol["selection_rule"] ==
            "Smallest parameter count within 0.5% of the lowest three-seed mean val_day_mae",
            "Unexpected architecture selection rule")
    for source, field in [("scripts/nifty_models.py", "model_source_sha256"),
                          ("scripts/train_nifty_models.py", "runner_source_sha256")]:
        require(digest(root / source) == protocol[field], f"Source changed after protocol: {source}")
    manifest_hash = digest(data / "manifest.json")
    require(manifest["status"] == "complete", "Prepared manifest is incomplete")
    require(manifest_hash == protocol["preprocessing_manifest_sha256"] == frozen["preprocessing_manifest_sha256"],
            "Preprocessing manifest hash changed")
    require(set(protocol["input_hashes"]) == EXPECTED_INPUTS, "Unexpected input paths in protocol")
    input_hashes = {}
    # Explicit allowlist: no test data path is opened, even if a report is altered.
    for relative in sorted(EXPECTED_INPUTS):
        checksum = digest(data / relative)
        require(checksum == protocol["input_hashes"][relative]
                == manifest["outputs"][relative]["sha256"], f"Changed input: {relative}")
        input_hashes[relative] = checksum
    require(input_hashes["scaler.json"] == protocol["feature_scaler_sha256"], "Feature scaler changed")
    require(spec["feature_columns"] == protocol["feature_columns"] == frozen["feature_columns"],
            "Feature order changed")
    require(len(spec["feature_columns"]) == 15, "Unexpected feature count")

    train_X = np.load(data / "train/X.npy", mmap_mode="r", allow_pickle=False)
    train_y = np.load(data / "train/y.npy", mmap_mode="r", allow_pickle=False)
    valid_X = np.load(data / "valid/X.npy", mmap_mode="r", allow_pickle=False)
    valid_y = np.load(data / "valid/y.npy", mmap_mode="r", allow_pickle=False)
    require(train_X.shape == (7069922, 15) and train_y.shape == (7069922,), "Unexpected training shapes")
    require(valid_X.shape == (2009818, 15) and valid_y.shape == (2009818,), "Unexpected validation shapes")
    require(protocol["train_rows"] == len(train_y) and protocol["valid_rows"] == len(valid_y), "Wrong protocol row counts")
    train_mean, target_scale = target_statistics(train_y)
    require(target_scale > 0 and np.isfinite(target_scale), "Invalid recomputed training target scale")
    close_number(scaler["scale"], target_scale, "target scaler/scale")
    close_number(scaler["observed_train_mean"], train_mean, "target scaler/mean")
    require(scaler["fit_split"] == "train" and scaler["fit_rows"] == len(train_y), "Target scaler was not fitted on training")
    require(scaler["center"] == 0 and scaler["clipping"] is False, "Unexpected target transformation")
    close_number(frozen["target_scale"], target_scale, "frozen target scale")
    require(frozen["target_center"] == 0.0, "Frozen model centers targets")
    # Preserve the exact saved scalar during inference to allow bitwise reload checks.
    inference_scale = scaler["scale"]
    metadata = pd.read_parquet(data / "valid/metadata.parquet", columns=["timestamp", "close"])
    require(len(metadata) == len(valid_y), "Validation metadata is not aligned")
    days = metadata.timestamp.dt.normalize().to_numpy()
    close = metadata.close.to_numpy()
    require(pd.Series(days).nunique() == summary["validation_days"] == 150, "Unexpected validation date count")
    del train_X, train_y

    runs = pd.read_csv(report / "runs.csv")
    history = pd.read_csv(report / "history.csv")
    comparison = pd.read_csv(report / "architecture_comparison.csv")
    expected_pairs = {(name, seed) for name in EXPECTED_COUNTS for seed in EXPECTED_SEEDS}
    require(len(runs) == 9 and set(zip(runs.model, runs.seed)) == expected_pairs,
            "Expected exactly three architectures by three seeds")
    require(set(zip(history.model, history.seed)) == expected_pairs, "History has missing or extra experiments")
    require(len(comparison) == 3 and set(comparison.model) == set(EXPECTED_COUNTS), "Wrong comparison architecture set")
    checked_metrics = {}
    saved_hashes = {}
    checked_predictions = 0
    for row in runs.itertuples(index=False):
        name, seed = row.model, int(row.seed)
        label = f"{name}_seed{seed}"
        expected_path = artifact / f"{label}.pt"
        require(row.checkpoint == str(expected_path.relative_to(root)), f"Unexpected checkpoint path for {label}")
        require(row.parameters == EXPECTED_COUNTS[name], f"Wrong recorded parameter count for {label}")
        require(np.isfinite(row.train_seconds) and row.train_seconds > 0, f"Invalid training time for {label}")
        epochs = history.loc[history.model.eq(name) & history.seed.eq(seed)].sort_values("epoch")
        require(epochs.epoch.tolist() == list(range(1, int(row.epochs_run) + 1)), f"Incomplete epoch history for {label}")
        require(1 <= row.best_epoch <= row.epochs_run <= protocol["max_epochs"], f"Invalid epoch counts for {label}")
        best_score, best_epoch, bad_epochs = float("inf"), None, 0
        for epoch in epochs.itertuples(index=False):
            require(np.isfinite(epoch.val_day_mae), f"Nonfinite history score for {label}")
            if epoch.val_day_mae < best_score - protocol["early_stopping_min_delta"]:
                best_score, best_epoch, bad_epochs = epoch.val_day_mae, epoch.epoch, 0
            else:
                bad_epochs += 1
            require(bad_epochs < protocol["early_stopping_patience"] or epoch.epoch == row.epochs_run,
                    f"Training continued after the stopping rule for {label}")
        require(row.best_epoch == best_epoch, f"Best checkpoint epoch does not match history for {label}")
        require(row.epochs_run == protocol["max_epochs"] or bad_epochs >= protocol["early_stopping_patience"],
                f"Run ended before its stopping condition for {label}")
        model, payload = checkpoint_model(expected_path, name, seed, target_scale, row.best_epoch)
        predictions = predict_batches(model, valid_X, inference_scale, batch_size=BATCH_SIZE)
        prediction_path = artifact / f"{label}_valid_predictions.npy"
        saved = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        require(saved.shape == (2009818,) and saved.dtype == np.float64, f"Unexpected saved prediction shape/type for {label}")
        compare_prediction_arrays(predictions, saved, label)
        metrics = score_predictions(valid_y, predictions, days, close)
        compare_metrics(metrics, row._asdict(), f"runs/{label}")
        compare_metrics(metrics, payload["validation_metrics"], f"checkpoint/{label}")
        compare_metrics(metrics, epochs.loc[epochs.epoch.eq(best_epoch)].iloc[0], f"history/{label}")
        checked_metrics[(name, seed)] = metrics
        checked_predictions += len(predictions)
        for path in [expected_path, prediction_path]:
            saved_hashes[str(path.relative_to(root))] = digest(path)
        print(f"Verified {label}: {len(predictions):,} exact validation predictions and five metrics", flush=True)
        del predictions, saved, model

    aggregate_scores = {}
    for name, expected_count in EXPECTED_COUNTS.items():
        observed = comparison.loc[comparison.model.eq(name)].iloc[0]
        require(observed["parameters"] == expected_count and observed["runs"] == 3,
                f"Wrong aggregate capacity/run count for {name}")
        require(observed["hidden"] == "-".join(map(str, MODEL_CONFIGS[name]["hidden"])), f"Wrong hidden-layer label for {name}")
        group = runs.loc[runs.model.eq(name)]
        close_number(observed["train_seconds_mean"], group.train_seconds.mean(), f"{name}/train_seconds_mean")
        for metric in METRICS:
            values = np.array([checked_metrics[(name, seed)][metric] for seed in EXPECTED_SEEDS])
            close_number(observed[f"{metric}_mean"], values.mean(), f"{name}/{metric}_mean")
            close_number(observed[f"{metric}_std"], values.std(ddof=1), f"{name}/{metric}_std")
        aggregate_scores[name] = float(np.mean([checked_metrics[(name, seed)]["val_day_mae"] for seed in EXPECTED_SEEDS]))
    threshold = min(aggregate_scores.values()) * 1.005
    eligible = [name for name, score in aggregate_scores.items() if score <= threshold]
    selected = min(eligible, key=lambda name: (EXPECTED_COUNTS[name], aggregate_scores[name]))
    for row in comparison.itertuples(index=False):
        require(bool(row.within_selection_tolerance) == (row.model in eligible), f"Wrong tolerance flag for {row.model}")
        require(bool(row.selected) == (row.model == selected), f"Wrong selected flag for {row.model}")
    require(frozen["architecture"] == selected and frozen["parameters"] == EXPECTED_COUNTS[selected],
            "Frozen architecture does not follow the 0.5% selection rule")
    require(frozen["input_dim"] == 15 and frozen["hidden"] == MODEL_CONFIGS[selected]["hidden"], "Wrong frozen topology")
    close_number(frozen["dropout"], MODEL_CONFIGS[selected]["dropout"], "Frozen dropout")
    require(frozen["activation"] == "ReLU" and frozen["output"] == "unconstrained linear scalar", "Wrong frozen activations")
    frozen_path = artifact / "frozen_mlp.pt"
    require(frozen["checkpoint"] == str(frozen_path.relative_to(root)), "Unexpected frozen checkpoint path")
    reference_path = artifact / f"{selected}_seed42.pt"
    require(digest(frozen_path) == frozen["checkpoint_sha256"] == digest(reference_path),
            "Frozen checkpoint differs from the predeclared reference seed")
    reference_row = runs.loc[runs.model.eq(selected) & runs.seed.eq(42)].iloc[0]
    require(frozen["best_epoch"] == reference_row.best_epoch, "Frozen best epoch differs from seed 42")
    frozen_model, _ = checkpoint_model(frozen_path, selected, 42, target_scale, frozen["best_epoch"])
    frozen_predictions = predict_batches(frozen_model, valid_X, inference_scale, batch_size=BATCH_SIZE)
    reference_predictions = np.load(artifact / f"{selected}_seed42_valid_predictions.npy", mmap_mode="r", allow_pickle=False)
    compare_prediction_arrays(frozen_predictions, reference_predictions, "frozen/reference seed 42")
    compare_metrics(score_predictions(valid_y, frozen_predictions, days, close),
                    summary["selected_validation_metrics"], "summary/frozen")
    require(summary["checkpoint_reload_exact_match"] is True, "Summary reload verification flag is not true")

    # Re-evaluate saved baseline coefficients without refitting them.
    baselines = pd.read_csv(report / "baseline_comparison.csv")
    ridge_names = {f"ridge_{alpha:g}" for alpha in protocol["ridge_alphas"]}
    require(len(baselines) == 4 and set(baselines.model) == ridge_names | {"persistence"}, "Wrong baseline set")
    persistence_metrics = score_predictions(valid_y, np.zeros(len(valid_y)), days, close)
    compare_metrics(persistence_metrics, baselines.loc[baselines.model.eq("persistence")].iloc[0], "baseline/persistence")
    compare_metrics(persistence_metrics, summary["persistence_validation_metrics"], "summary/persistence")
    for name in sorted(ridge_names):
        path = artifact / f"{name}_coefficients.npy"
        coefficients = np.load(path, allow_pickle=False)
        require(coefficients.shape == (16,) and np.isfinite(coefficients).all(), f"Invalid ridge coefficients for {name}")
        predictions = predict_ridge(valid_X, coefficients, batch_size=BATCH_SIZE)
        compare_metrics(score_predictions(valid_y, predictions, days, close),
                        baselines.loc[baselines.model.eq(name)].iloc[0], f"baseline/{name}")
        saved_hashes[str(path.relative_to(root))] = digest(path)
    best_ridge = baselines.loc[baselines.model.ne("persistence")].sort_values("val_day_mae").iloc[0]
    require(summary["best_ridge"]["model"] == best_ridge.model, "Summary chose the wrong ridge baseline")
    compare_metrics(summary["best_ridge"], best_ridge, "summary/best_ridge")
    close_number(summary["architecture_mean_day_mae_improvement_percent"],
                 100 * (1 - aggregate_scores[selected] / persistence_metrics["val_day_mae"]), "Mean architecture improvement")
    close_number(summary["reference_day_mae_improvement_percent"],
                 100 * (1 - checked_metrics[(selected, 42)]["val_day_mae"] / persistence_metrics["val_day_mae"]),
                 "Reference improvement")

    for path in [report / name for name in ["summary.json", "protocol.json", "runs.csv", "history.csv",
                                           "architecture_comparison.csv", "baseline_comparison.csv", "frozen_config.json"]]:
        saved_hashes[str(path.relative_to(root))] = digest(path)
    for path in [frozen_path, artifact / "target_scaler.json"]:
        saved_hashes[str(path.relative_to(root))] = digest(path)
    result = {
        "status": "passed", "verified_at_utc": datetime.now(UTC).isoformat(),
        "mlp_runs_verified": 9, "architectures_verified": 3, "seeds_per_architecture": 3,
        "validation_rows_per_run": len(valid_y), "validation_days": 150,
        "candidate_predictions_exactly_verified": checked_predictions,
        "frozen_predictions_exactly_verified": len(frozen_predictions),
        "metrics_verified_per_run": METRICS, "baselines_verified": 4,
        "all_checkpoint_weights_finite": True, "all_best_epochs_verified": True,
        "all_source_and_input_hashes_verified": True, "architecture_aggregates_verified": True,
        "selection_rule_verified": True, "selected_architecture": selected,
        "predeclared_reference_seed": 42, "frozen_checkpoint_sha256": digest(frozen_path),
        "preprocessing_manifest_sha256": manifest_hash,
        "input_hashes": input_hashes, "saved_artifact_hashes": saved_hashes,
        "verifier_source_sha256": digest(Path(__file__)),
        "test_arrays_or_metadata_opened": False, "test_evaluated": False,
        "llm_executed": False, "models_fitted_by_verifier": False,
        "torch_threads": torch.get_num_threads(),
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"Verification passed; report written to {output.relative_to(root)}", flush=True)
    return result


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        verify()

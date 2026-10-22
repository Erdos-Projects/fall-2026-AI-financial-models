"""Verify the completed eight-family MLP study without fitting or opening test data.

Every candidate is reloaded and its entire validation prediction vector must
match the saved vector exactly. Metrics, model selection and diagnostics are
recomputed; timing observations are checked for consistency, not reproduced.
Only a fully successful verification atomically publishes a passed status.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
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
from train_nifty_models import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 65536
SEEDS = [17, 42, 2026]
METRICS = ["val_day_mae", "val_mae", "val_rmse", "val_price_mae", "val_price_rmse"]
COUNTS = {"compact": 1057, "standard": 3137, "wide": 12417, "shallow": 1089,
          "standard_plain": 3137, "deep": 13569, "batchnorm": 3329, "residual": 17985}
INPUTS = {
    "train/X.npy", "train/y.npy", "train/metadata.parquet",
    "valid/X.npy", "valid/y.npy", "valid/metadata.parquet",
    "feature_spec.json", "scaler.json",
}
SOURCES = {"scripts/nifty_models.py", "scripts/train_nifty_models.py",
           "scripts/nifty_mlp_variants.py", "scripts/train_nifty_mlp_study.py"}
REUSED_REPORTS = {f"reports/nifty_models/{name}" for name in
                  ["summary.json", "protocol.json", "runs.csv", "history.csv", "verification.json"]}
REPORT_FILES = ["protocol.json", "summary.json", "runs.csv", "history.csv",
                "architecture_comparison.csv", "daily_validation.csv", "subgroups.csv", "inference.csv"]
RESUME_SOURCE = "scripts/resume_nifty_mlp_study.py"


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


def compare_metrics(actual, expected, label):
    for metric in METRICS:
        close_number(float(actual[metric]), float(expected[metric]), f"{label}/{metric}")


def metrics_from_daily(daily):
    return {"val_day_mae": float(daily.mae.mean()),
            "val_mae": float(np.average(daily.mae, weights=daily.rows)),
            "val_rmse": float(np.sqrt(np.average(daily.mse, weights=daily.rows))),
            "val_price_mae": float(np.average(daily.price_mae, weights=daily.rows)),
            "val_price_rmse": float(np.sqrt(np.average(daily.price_mse, weights=daily.rows)))}


def target_scale(y):
    """Two bounded-memory float64 passes for the train-only population scale."""
    total = 0.0
    for start in range(0, len(y), BATCH_SIZE):
        block = np.asarray(y[start:start + BATCH_SIZE], dtype=np.float64)
        require(np.isfinite(block).all(), "Nonfinite training target")
        total += float(block.sum(dtype=np.float64))
    mean = total / len(y)
    squares = 0.0
    for start in range(0, len(y), BATCH_SIZE):
        residual = np.asarray(y[start:start + BATCH_SIZE], dtype=np.float64) - mean
        squares += float(np.sum(residual * residual, dtype=np.float64))
    return float(np.sqrt(squares / len(y)))


def expected_prior_artifacts():
    paths = {f"data/processed/nifty_models/{name}_seed{seed}{suffix}"
             for name in MODEL_CONFIGS for seed in SEEDS
             for suffix in [".pt", "_valid_predictions.npy"]}
    paths.update(f"data/processed/nifty_models/{name}" for name in
                 ["ridge_0.0001_coefficients.npy", "ridge_0.01_coefficients.npy",
                  "ridge_1_coefficients.npy", "frozen_mlp.pt", "target_scaler.json"])
    paths.update(f"reports/nifty_models/{name}" for name in
                 ["summary.json", "protocol.json", "runs.csv", "history.csv",
                  "architecture_comparison.csv", "baseline_comparison.csv", "frozen_config.json"])
    return paths


def verify_resume_history(root, report, protocol_hash):
    """Audit optional restarts using fixed local paths, never paths from the audit.

    Historical CSV hashes are retained as audit evidence. Their old contents
    are not available after aggregation, so only the hashes' format is checked.
    Every completed model and the current reports are independently checked by
    the main verifier below.
    """
    audit_path = report / "resume_history.json"
    if not audit_path.exists():
        return {"resume_audit_present": False, "resume_sessions_verified": 0}, {}
    audit = read_json(audit_path)
    sessions = audit.get("sessions")
    require(isinstance(sessions, list) and len(sessions) > 0, "Missing resume sessions")
    # The audit stores only checksums; it cannot choose which source gets opened.
    source_hash = digest(root / RESUME_SOURCE)
    expected = {(name, seed) for name in COUNTS for seed in SEEDS}
    prior = {(name, seed) for name in MODEL_CONFIGS for seed in SEEDS}
    previous_completed, previous_started = prior, None
    for index, session in enumerate(sessions):
        label = f"resume session {index + 1}"
        require(isinstance(session, dict), f"Invalid {label}")
        require(session["resume_source_sha256"] == source_hash, f"Changed resume source: {label}")
        require(session["protocol_sha256"] == protocol_hash, f"Changed resume protocol: {label}")
        final_session = index == len(sessions) - 1
        allowed_statuses = {"complete"} if final_session else {"running", "interrupted"}
        require(session["status"] in allowed_statuses, f"Unexpected/incomplete status: {label}")
        starts = {}
        for field in ["completed_at_start", "missing_at_start"]:
            pairs = session[field]
            require(isinstance(pairs, list) and all(
                isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)
                and type(pair[1]) is int for pair in pairs), f"Invalid {field}: {label}")
            starts[field] = {tuple(pair) for pair in pairs}
            require(len(starts[field]) == len(pairs), f"Duplicate {field}: {label}")
        completed, missing = starts["completed_at_start"], starts["missing_at_start"]
        require(completed.isdisjoint(missing) and completed | missing == expected,
                f"Resume completed/missing sets do not partition the 24 runs: {label}")
        require(prior <= completed and previous_completed <= completed,
                f"Resume lost previously completed runs: {label}")
        require(missing <= expected - prior, f"Resume attempted to refit prior-study runs: {label}")
        require(session["restart_policy"] ==
                "Incomplete seeds restart with their original seed and unchanged fit_one",
                f"Changed restart policy: {label}")
        for field in ["runs_sha256_at_start", "history_sha256_at_start"]:
            checksum = session[field]
            require(isinstance(checksum, str) and len(checksum) == 64
                    and all(character in "0123456789abcdef" for character in checksum),
                    f"Invalid historical CSV digest {field}: {label}")
        started = datetime.fromisoformat(session["started_at_utc"])
        require(started.tzinfo is not None and started.utcoffset().total_seconds() == 0,
                f"Resume start timestamp is not UTC: {label}")
        require(previous_started is None or started >= previous_started, f"Resume session order changed: {label}")
        if final_session:
            require(session["total_completed_runs"] == 24, "Final resume session did not finish all 24 runs")
            finished = datetime.fromisoformat(session["finished_at_utc"])
            require(finished.tzinfo is not None and finished.utcoffset().total_seconds() == 0
                    and finished >= started, "Invalid final resume completion timestamp")
        previous_completed, previous_started = completed, started
    audit_hash = digest(audit_path)
    return ({"resume_audit_present": True, "resume_audit_verified": True,
             "resume_sessions_verified": len(sessions), "resume_source": RESUME_SOURCE,
             "resume_source_sha256": source_hash, "resume_history_sha256": audit_hash,
             "resume_historical_csv_contents_reverified": False},
            {RESUME_SOURCE: source_hash, str(audit_path.relative_to(root)): audit_hash})


def checkpoint_model(path, name, seed, scale, best_epoch):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["architecture"] == name and payload["seed"] == seed,
            f"Wrong checkpoint identity: {path.name}")
    require(payload["input_dim"] == 15, f"Wrong input dimension: {path.name}")
    require(payload["best_epoch"] == best_epoch, f"Wrong checkpoint epoch: {path.name}")
    require(payload["target_center"] == 0.0, f"Unexpected target centering: {path.name}")
    close_number(payload["target_scale"], scale, f"{path.name}/target_scale")
    if name in MODEL_CONFIGS:
        require(payload["hidden"] == MODEL_CONFIGS[name]["hidden"], f"Wrong hidden layers: {path.name}")
        close_number(payload["dropout"], MODEL_CONFIGS[name]["dropout"], f"{path.name}/dropout")
        model = make_model(name, input_dim=15)
    else:
        require(payload["config"] == CONFIGS[name], f"Wrong architecture configuration: {path.name}")
        model = build_model(name, input_dim=15)
    require(bool(payload["state_dict"]), f"Empty state: {path.name}")
    require(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item()
                for value in payload["state_dict"].values()), f"Nonfinite weights/buffers: {path.name}")
    model.load_state_dict(payload["state_dict"], strict=True)
    require(parameter_count(model) == COUNTS[name], f"Wrong parameter count: {path.name}")
    return model, payload


def verify_epochs(row, history, protocol):
    label = f"{row.model}_seed{row.seed}"
    epochs = history.loc[history.model.eq(row.model) & history.seed.eq(row.seed)].sort_values("epoch")
    require(epochs.epoch.tolist() == list(range(1, int(row.epochs_run) + 1)), f"Incomplete history: {label}")
    require(1 <= row.best_epoch <= row.epochs_run <= protocol["max_epochs"], f"Invalid epoch counts: {label}")
    finite_columns = METRICS + ["train_scaled_smooth_l1", "train_probe_mae", "learning_rate", "epoch_seconds"]
    require(np.isfinite(epochs[finite_columns].to_numpy()).all(), f"Nonfinite history values: {label}")
    require((epochs[METRICS + ["train_scaled_smooth_l1", "train_probe_mae"]] >= 0).all().all(),
            f"Negative error/loss in history: {label}")
    require((epochs.epoch_seconds > 0).all(), f"Nonpositive epoch timing: {label}")
    require(epochs.learning_rate.between(1e-5, protocol["learning_rate"]).all(), f"Invalid learning rate: {label}")
    best_score, best_epoch, stale = float("inf"), None, 0
    for epoch in epochs.itertuples(index=False):
        if epoch.val_day_mae < best_score - protocol["early_stopping_min_delta"]:
            best_score, best_epoch, stale = epoch.val_day_mae, epoch.epoch, 0
        else:
            stale += 1
        require(stale < protocol["early_stopping_patience"] or epoch.epoch == row.epochs_run,
                f"Training continued after early stopping: {label}")
    require(row.best_epoch == best_epoch, f"Selected epoch does not match early stopping: {label}")
    require(row.epochs_run == protocol["max_epochs"] or stale >= protocol["early_stopping_patience"],
            f"Training ended before stopping condition: {label}")
    return epochs.loc[epochs.epoch.eq(best_epoch)].iloc[0]


def verify_daily(actual, saved, name):
    observed = saved.loc[saved.model.eq(name)].sort_values("day").reset_index(drop=True)
    actual = actual.sort_values("day").reset_index(drop=True)
    require(len(observed) == len(actual), f"Wrong daily row count: {name}")
    require(np.array_equal(observed.day.to_numpy(), actual.day.to_numpy()), f"Wrong daily dates: {name}")
    require(np.array_equal(observed.rows.to_numpy(), actual.rows.to_numpy()), f"Wrong daily counts: {name}")
    for metric in ["mae", "mse", "price_mae", "price_mse"]:
        require(np.isfinite(observed[metric]).all() and
                np.allclose(observed[metric], actual[metric], rtol=1e-10, atol=1e-12),
                f"Daily {metric} mismatch: {name}")


def verify_subgroups(y, predictions, days, close, masks, saved, name):
    for group, mask in masks.items():
        observed = saved.loc[saved.model.eq(name) & saved.group.eq(group)]
        require(len(observed) == int(mask.any()), f"Wrong subgroup row count: {name}/{group}")
        if mask.any():
            require(observed.iloc[0].rows == int(mask.sum()), f"Wrong subgroup sample count: {name}/{group}")
            actual = metrics_from_daily(daily_metrics(y[mask], predictions[mask], days[mask], close[mask]))
            compare_metrics(actual, observed.iloc[0], f"subgroup/{name}/{group}")


def verify(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, "Verify this repository's experiment")
    report = root / "reports/nifty_mlp_study"
    data = root / "data/processed/nifty_training"
    prior = root / "reports/nifty_models"
    output = report / "verification.json"
    output.unlink(missing_ok=True)  # A failed recheck must not leave a stale passed status.
    summary = read_json(report / "summary.json")
    protocol = read_json(report / "protocol.json")
    old_protocol = read_json(prior / "protocol.json")
    old_verification = read_json(prior / "verification.json")
    manifest = read_json(data / "manifest.json")
    spec = read_json(data / "feature_spec.json")
    require(summary["status"] == "complete", "Study is not complete; wait for training to finish")
    require(old_verification["status"] == "passed" and manifest["status"] == "complete", "Unverified provenance")
    for name, mapping in [("summary", summary), ("protocol", protocol), ("prior protocol", old_protocol)]:
        require(mapping["test_evaluated"] is False and mapping["llm_executed"] is False,
                f"Unexpected test/LLM activity: {name}")
    require(digest(report / "protocol.json") == summary["protocol_sha256"], "Summary protocol hash mismatch")
    resume_audit, resume_hashes = verify_resume_history(root, report, summary["protocol_sha256"])
    for key, value in PROTOCOL.items():
        if key != "architectures":
            require(protocol[key] == old_protocol[key] == value, f"Training recipe changed: {key}")
    require(protocol["architectures"] == {**MODEL_CONFIGS, **CONFIGS}, "Wrong study architectures")
    require(protocol["reused_architectures"] == list(MODEL_CONFIGS), "Wrong reused architecture set/order")
    require(protocol["new_architectures"] == list(CONFIGS), "Wrong new architecture set/order")
    require(protocol["seeds"] == SEEDS and protocol["reference_seed"] == summary["reference_seed"] == 42,
            "Predeclared seeds changed")
    require(protocol["selection_tolerance"] == .005, "Selection tolerance changed")
    require(summary["architectures"] == 8 and summary["total_runs"] == 24
            and summary["new_runs"] == 15 and summary["reused_runs"] == 9, "Wrong experiment counts")
    require(protocol["hardware"]["device"] == "cpu" and protocol["hardware"]["torch_threads"] == 4,
            "Unexpected recorded training hardware")
    require(torch.get_num_threads() == 4, "Exact reload verification requires the recorded four CPU threads")

    # Check keys before opening paths: manifests cannot redirect reads into test data.
    require(set(protocol["source_hashes"]) == SOURCES, "Unexpected source paths")
    for relative in sorted(SOURCES):
        require(digest(root / relative) == protocol["source_hashes"][relative], f"Changed source: {relative}")
    for relative, field in [("scripts/nifty_models.py", "model_source_sha256"),
                            ("scripts/train_nifty_models.py", "runner_source_sha256")]:
        require(protocol["source_hashes"][relative] == old_protocol[field], f"Changed reused implementation: {relative}")
    require(set(protocol["reused_report_hashes"]) == REUSED_REPORTS, "Unexpected reused report paths")
    for relative in sorted(REUSED_REPORTS):
        require(digest(root / relative) == protocol["reused_report_hashes"][relative], f"Changed reused report: {relative}")
    prior_artifacts = expected_prior_artifacts()
    require(set(old_verification["saved_artifact_hashes"]) == prior_artifacts, "Unexpected prior artifact paths")
    for relative in sorted(prior_artifacts):
        require(digest(root / relative) == old_verification["saved_artifact_hashes"][relative],
                f"Changed previously verified artifact: {relative}")
    manifest_hash = digest(data / "manifest.json")
    require(manifest_hash == protocol["preprocessing_manifest_sha256"]
            == old_protocol["preprocessing_manifest_sha256"] == old_verification["preprocessing_manifest_sha256"],
            "Changed preprocessing manifest")
    require(set(protocol["input_hashes"]) == set(old_protocol["input_hashes"]) == INPUTS, "Unexpected input paths")
    input_hashes = {}
    for relative in sorted(INPUTS):
        checksum = digest(data / relative)
        require(checksum == protocol["input_hashes"][relative] == old_protocol["input_hashes"][relative]
                == old_verification["input_hashes"][relative] == manifest["outputs"][relative]["sha256"],
                f"Changed train/validation input: {relative}")
        input_hashes[relative] = checksum
    require(spec["feature_columns"] == protocol["feature_columns"] == old_protocol["feature_columns"]
            and len(spec["feature_columns"]) == 15, "Changed feature order/dimension")

    train_X = np.load(data / "train/X.npy", mmap_mode="r", allow_pickle=False)
    train_y = np.load(data / "train/y.npy", mmap_mode="r", allow_pickle=False)
    valid_X = np.load(data / "valid/X.npy", mmap_mode="r", allow_pickle=False)
    valid_y = np.load(data / "valid/y.npy", mmap_mode="r", allow_pickle=False)
    require(train_X.shape == (7069922, 15) and train_y.shape == (7069922,), "Unexpected training shapes")
    require(valid_X.shape == (2009818, 15) and valid_y.shape == (2009818,), "Unexpected validation shapes")
    require(protocol["train_rows"] == len(train_y) and protocol["valid_rows"] == len(valid_y), "Wrong protocol row counts")
    computed_scale = target_scale(train_y)
    require(computed_scale > 0, "Invalid training target scale")
    close_number(protocol["target_scale"], computed_scale, "protocol/target_scale")
    close_number(summary["target_scale"], computed_scale, "summary/target_scale")
    # Preserve the exact scalar used during fitting for bitwise prediction equality.
    scale = protocol["target_scale"]
    del train_X, train_y
    metadata = pd.read_parquet(data / "valid/metadata.parquet", columns=["timestamp", "close", "option_type"])
    require(len(metadata) == len(valid_y), "Validation metadata length mismatch")
    days = metadata.timestamp.dt.normalize().to_numpy()
    close = metadata.close.to_numpy()
    require(pd.Series(days).nunique() == protocol["validation_days"] == 150, "Wrong validation day count")

    runs = pd.read_csv(report / "runs.csv")
    history = pd.read_csv(report / "history.csv")
    comparison = pd.read_csv(report / "architecture_comparison.csv")
    saved_daily = pd.read_csv(report / "daily_validation.csv", parse_dates=["day"])
    saved_subgroups = pd.read_csv(report / "subgroups.csv")
    pairs = {(name, seed) for name in COUNTS for seed in SEEDS}
    require(len(runs) == 24 and set(zip(runs.model, runs.seed)) == pairs, "Expected exactly eight families by three seeds")
    require(set(zip(history.model, history.seed)) == pairs, "Wrong history experiment set")
    require(len(comparison) == 8 and set(comparison.model) == set(COUNTS), "Wrong comparison architecture set")
    require(len(saved_daily) == 9 * 150 and set(saved_daily.model) == set(COUNTS) | {"persistence"},
            "Wrong daily diagnostics architecture set/row count")
    require(not saved_daily.duplicated(["model", "day"]).any(), "Duplicate daily diagnostics")
    claimed_selected, claimed_best = summary["selected_architecture"], summary["best_mean_architecture"]
    require(claimed_selected in COUNTS and claimed_best in COUNTS, "Unknown summary architecture")
    diagnostic_names = {"persistence", claimed_selected, claimed_best}
    masks = {"calls": metadata.option_type.eq("CE").to_numpy(), "puts": metadata.option_type.eq("PE").to_numpy(),
             "close_at_most_1": close <= 1, "close_above_1": close > 1,
             "zero_volume": valid_X[:, protocol["feature_columns"].index("zero_volume")] == 1,
             "positive_volume": valid_X[:, protocol["feature_columns"].index("zero_volume")] == 0}
    subgroup_pairs = {(name, group) for name in diagnostic_names for group, mask in masks.items() if mask.any()}
    require(len(saved_subgroups) == len(subgroup_pairs)
            and set(zip(saved_subgroups.model, saved_subgroups.group)) == subgroup_pairs, "Wrong subgroup diagnostic set")
    prior_runs = pd.read_csv(prior / "runs.csv")
    prior_history = pd.read_csv(prior / "history.csv")
    reused_runs = runs.loc[runs.model.isin(MODEL_CONFIGS), prior_runs.columns].reset_index(drop=True)
    reused_history = history.loc[history.model.isin(MODEL_CONFIGS), prior_history.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(reused_runs, prior_runs, check_exact=False, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(reused_history, prior_history, check_exact=False, rtol=1e-12, atol=1e-14)

    checked_metrics, saved_hashes = {}, dict(resume_hashes)
    checked_predictions = 0
    for row in runs.itertuples(index=False):
        name, seed = row.model, int(row.seed)
        label = f"{name}_seed{seed}"
        directory = "nifty_models" if name in MODEL_CONFIGS else "nifty_mlp_study"
        checkpoint = root / "data/processed" / directory / f"{label}.pt"
        prediction_path = checkpoint.with_name(f"{label}_valid_predictions.npy")
        require(row.checkpoint == str(checkpoint.relative_to(root)), f"Unexpected checkpoint path: {label}")
        require(row.origin == ("prior_study" if name in MODEL_CONFIGS else "new_run"), f"Wrong provenance: {label}")
        require(row.parameters == COUNTS[name], f"Wrong parameter count: {label}")
        require(np.isfinite(row.train_seconds) and row.train_seconds > 0, f"Invalid fit time: {label}")
        best_history = verify_epochs(row, history, protocol)
        for path, checksum in [(checkpoint, row.checkpoint_sha256), (prediction_path, row.predictions_sha256)]:
            actual = digest(path)
            require(actual == checksum, f"Changed saved artifact: {path.name}")
            saved_hashes[str(path.relative_to(root))] = actual
        model, payload = checkpoint_model(checkpoint, name, seed, scale, row.best_epoch)
        predictions = predict_batches(model, valid_X, scale, batch_size=BATCH_SIZE)
        saved = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        require(saved.shape == (len(valid_y),) and saved.dtype == np.float64, f"Wrong prediction shape/type: {label}")
        for start in range(0, len(predictions), BATCH_SIZE):
            require(np.array_equal(predictions[start:start + BATCH_SIZE], saved[start:start + BATCH_SIZE]),
                    f"Reloaded predictions differ: {label}, starting row {start}")
        daily = daily_metrics(valid_y, predictions, days, close)
        metrics = metrics_from_daily(daily)
        compare_metrics(metrics, row._asdict(), f"runs/{label}")
        compare_metrics(metrics, payload["validation_metrics"], f"checkpoint/{label}")
        compare_metrics(metrics, best_history, f"best history/{label}")
        checked_metrics[(name, seed)] = metrics
        if seed == 42:
            verify_daily(daily, saved_daily, name)
            if name in diagnostic_names:
                verify_subgroups(valid_y, predictions, days, close, masks, saved_subgroups, name)
        checked_predictions += len(predictions)
        print(f"Verified {label}: {len(predictions):,} exact validation predictions and five metrics", flush=True)
        del predictions, saved, model, daily

    scores = {}
    for name, count in COUNTS.items():
        observed = comparison.loc[comparison.model.eq(name)].iloc[0]
        group = runs.loc[runs.model.eq(name)]
        require(observed["parameters"] == count and observed["runs"] == 3, f"Wrong aggregate count: {name}")
        require(observed["origin"] == group.origin.iloc[0], f"Wrong aggregate provenance: {name}")
        close_number(observed["train_seconds_mean"], group.train_seconds.mean(), f"{name}/train_seconds_mean")
        close_number(observed["epochs_run_mean"], group.epochs_run.mean(), f"{name}/epochs_run_mean")
        for metric in METRICS:
            values = np.array([checked_metrics[(name, seed)][metric] for seed in SEEDS])
            close_number(observed[f"{metric}_mean"], values.mean(), f"{name}/{metric}_mean")
            close_number(observed[f"{metric}_std"], values.std(ddof=1), f"{name}/{metric}_std")
        scores[name] = float(np.mean([checked_metrics[(name, seed)]["val_day_mae"] for seed in SEEDS]))
    threshold = min(scores.values()) * (1 + protocol["selection_tolerance"])
    eligible = [name for name in COUNTS if scores[name] <= threshold]
    selected = min(eligible, key=lambda name: (COUNTS[name], scores[name]))
    best = min(scores, key=scores.get)
    require(selected == claimed_selected and best == claimed_best, "Summary violates architecture selection rule")
    for row in comparison.itertuples(index=False):
        require(isinstance(row.selected, (bool, np.bool_)) and row.selected == (row.model == selected),
                f"Invalid selection flag: {row.model}")
        require(isinstance(row.within_selection_tolerance, (bool, np.bool_))
                and row.within_selection_tolerance == (row.model in eligible), f"Invalid tolerance flag: {row.model}")
    reference = runs.loc[runs.model.eq(selected) & runs.seed.eq(42)].iloc[0]
    require(summary["selected_checkpoint"] == reference.checkpoint, "Summary reference checkpoint mismatch")
    compare_metrics(summary["selected_reference_metrics"], checked_metrics[(selected, 42)], "summary/reference")
    zero = np.zeros(len(valid_y))
    persistence_daily = daily_metrics(valid_y, zero, days, close)
    compare_metrics(metrics_from_daily(persistence_daily), summary["persistence_metrics"], "summary/persistence")
    verify_daily(persistence_daily, saved_daily, "persistence")
    verify_subgroups(valid_y, zero, days, close, masks, saved_subgroups, "persistence")

    inference = pd.read_csv(report / "inference.csv")
    require(len(inference) == 16 and set(zip(inference.model, inference.batch_size))
            == {(name, batch) for name in COUNTS for batch in [1, 4096]}, "Wrong inference benchmark set")
    for row in inference.itertuples(index=False):
        require(row.repeats == (500 if row.batch_size == 1 else 50), f"Wrong timing repetitions: {row.model}")
        require(np.isfinite([row.median_batch_ms, row.p95_batch_ms, row.rows_per_second]).all()
                and 0 < row.median_batch_ms <= row.p95_batch_ms, f"Invalid timings: {row.model}")
        close_number(row.rows_per_second, row.batch_size * 1000 / row.median_batch_ms,
                     f"inference/{row.model}/{row.batch_size}/rows_per_second")
    for name in REPORT_FILES:
        path = report / name
        saved_hashes[str(path.relative_to(root))] = digest(path)
    result = {
        "status": "passed", "verified_at_utc": datetime.now(UTC).isoformat(),
        "mlp_runs_verified": 24, "architectures_verified": 8, "seeds_per_architecture": 3,
        "new_runs_verified": 15, "reused_runs_verified": 9,
        "validation_rows_per_run": len(valid_y), "validation_days": 150,
        "candidate_predictions_exactly_verified": checked_predictions, "metrics_verified_per_run": METRICS,
        "all_checkpoint_weights_and_buffers_finite": True, "all_best_epochs_verified": True,
        "all_source_and_input_hashes_verified": True, "prior_artifact_provenance_verified": True,
        "architecture_aggregates_verified": True, "selection_rule_verified": True,
        "selected_architecture": selected, "best_mean_architecture": best, "predeclared_reference_seed": 42,
        "daily_diagnostic_rows_verified": len(saved_daily), "subgroup_rows_verified": len(saved_subgroups),
        "persistence_metrics_verified": True, "inference_timing_rows_consistency_checked": len(inference),
        "inference_timings_reproduced": False, "preprocessing_manifest_sha256": manifest_hash,
        "input_hashes": input_hashes, "saved_artifact_hashes": saved_hashes,
        "source_hashes": protocol["source_hashes"], "reused_report_hashes": protocol["reused_report_hashes"],
        "verifier_source_sha256": digest(Path(__file__)), "test_arrays_or_metadata_opened": False,
        "test_evaluated": False, "llm_executed": False, "models_fitted_by_verifier": False,
        "torch_threads": torch.get_num_threads(),
        **resume_audit,
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

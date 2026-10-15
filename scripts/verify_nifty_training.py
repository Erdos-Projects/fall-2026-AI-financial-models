"""Verify completed Nifty training artifacts without fitting or evaluating a model.

Check every exported tensor and metadata row in bounded-memory batches. By
default, also verify the SHA256 digest of every output named by the completion
manifest. A successful run writes reports/nifty_preprocessing/verification.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
CONTINUOUS_COLUMNS = [
    "log_close", "log_strike", "log_volume", "log_high_low", "log_close_open",
    "return_1m", "return_5m", "return_std_5m", "log_mean_volume_6bars",
]
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + [
    "minute_sin", "minute_cos", "is_call", "zero_volume", "close_at_most_1", "flat_ohlc",
]
FORBIDDEN_FEATURES = {
    "file_id", "source_row", "timestamp", "label_timestamp", "window_start",
    "archive_cohort", "target_close", "target_log_return", "split",
    "extra_field_9", "encoded_expiry_candidate",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify(root: Path = ROOT, batch_size: int = 131_072, hashes: bool = True) -> dict:
    require(batch_size > 0, "Batch size must be positive")
    out = root / "data/processed/nifty_training"
    report = root / "reports/nifty_preprocessing"
    result_path = report / "verification.json"
    # A failed repeat check must not leave a success result from an earlier run.
    result_path.unlink(missing_ok=True)
    manifest_path = out / "manifest.json"
    require(manifest_path.is_file(), "Export is incomplete: manifest.json is absent")
    manifest_digest = digest(manifest_path)
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "complete", "Export manifest is not complete")
    require(manifest.get("raw_hashes_verified") is True, "Export did not verify raw hashes")
    spec = read_json(out / "feature_spec.json")
    scaler = read_json(out / "scaler.json")
    summary = read_json(report / "summary.json")
    require(spec["feature_columns"] == FEATURE_COLUMNS, "Unexpected feature order or contents")
    require(spec["standardized_columns"] == CONTINUOUS_COLUMNS, "Unexpected standardized columns")
    require(not FORBIDDEN_FEATURES.intersection(spec["feature_columns"]), "Forbidden model input")
    require(set(FEATURE_COLUMNS) == set(spec["definitions"]), "Feature definitions are incomplete")
    require(spec["history_bars"] == 6 and not spec["target_standardized"], "Unexpected target/history specification")
    require(spec["target"] == "target_log_return = log(close[t+1] / close[t]); exactly one minute ahead", "Unexpected target")
    require(scaler["continuous_columns"] == CONTINUOUS_COLUMNS, "Scaler column order differs")
    require(scaler["fit_split"] == "train", "Scaler was not marked as fitted on train")
    require(manifest["feature_count"] == len(FEATURE_COLUMNS), "Manifest feature count differs")
    require(manifest["splits"] == summary["splits"], "Manifest and summary splits differ")
    require([row["split"] for row in manifest["splits"]] == list(SPLITS), "Unexpected split order")

    scaler_mean = np.asarray(scaler["mean"], dtype=np.float64)
    scaler_scale = np.asarray(scaler["scale"], dtype=np.float64)
    scaler_variance = np.asarray(scaler["variance"], dtype=np.float64)
    for name, values in [("mean", scaler_mean), ("scale", scaler_scale), ("variance", scaler_variance)]:
        require(values.shape == (len(CONTINUOUS_COLUMNS),), f"Unexpected scaler {name} shape")
        require(np.isfinite(values).all(), f"Nonfinite scaler {name}")
    require((scaler_scale > 0).all() and (scaler_variance >= 0).all(), "Invalid scaler scale/variance")

    dates = pd.read_csv(report / "date_splits.csv", parse_dates=["observation_date"])
    require(dates.observation_date.notna().all(), "Missing split date")
    require(dates.observation_date.is_unique, "Date belongs to more than one split")
    require(dates.observation_date.is_monotonic_increasing, "Split dates are not sorted")
    require(dates.observation_date.eq(dates.observation_date.dt.normalize()).all(), "Split dates include clock times")
    require(set(dates.split) == set(SPLITS), "Unexpected date split labels")
    require(len(dates) == summary["observed_dates"], "Summary date count differs")
    require(int(dates.clean_rows.sum()) == summary["retained_clean_rows"], "Clean row totals differ")
    require(int(dates.examples.sum()) == summary["total_examples"], "Example totals differ")
    lookup = dates.set_index("observation_date").split
    previous_end = None
    split_results = []
    all_day_counts = Counter()
    train_sum = np.zeros(len(CONTINUOUS_COLUMNS), dtype=np.float64)
    train_squares = np.zeros_like(train_sum)
    train_rows = 0

    for expected in manifest["splits"]:
        split = expected["split"]
        count = int(expected["examples"])
        split_dates = dates.loc[dates.split.eq(split)]
        require(count > 0, f"Empty {split} split")
        require(len(split_dates) == expected["dates"], f"{split}: date count differs")
        require(int(split_dates.examples.sum()) == count, f"{split}: daily example count differs")
        require(int(split_dates.clean_rows.sum()) == expected["clean_rows"], f"{split}: clean count differs")
        require(str(split_dates.observation_date.min().date()) == expected["start_date"], f"{split}: start date differs")
        require(str(split_dates.observation_date.max().date()) == expected["end_date"], f"{split}: end date differs")
        if previous_end is not None:
            require(previous_end < split_dates.observation_date.min(), "Split dates overlap or interleave")
        previous_end = split_dates.observation_date.max()
        x = np.load(out / split / "X.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(out / split / "y.npy", mmap_mode="r", allow_pickle=False)
        require(x.shape == (count, len(FEATURE_COLUMNS)), f"{split}: X shape differs")
        require(y.shape == (count,), f"{split}: y shape differs")
        require(x.dtype == np.dtype("float32") and y.dtype == np.dtype("float32"), f"{split}: tensors must be float32")
        metadata = pq.ParquetFile(out / split / "metadata.parquet")
        require(metadata.metadata.num_rows == count, f"{split}: metadata row count differs")
        offset = 0
        max_target_difference = 0.0
        for batch in metadata.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            stop = offset + len(frame)
            chunk = np.asarray(x[offset:stop])
            targets = np.asarray(y[offset:stop])
            require(np.isfinite(chunk).all() and np.isfinite(targets).all(), f"{split}: nonfinite tensors")
            require(frame.split.eq(split).all(), f"{split}: wrong metadata split label")
            current = frame.timestamp
            first = frame.window_start
            label = frame.label_timestamp
            require(current.notna().all() and first.notna().all() and label.notna().all(), f"{split}: invalid time metadata")
            day = current.dt.normalize()
            require(first.dt.normalize().eq(day).all() and label.dt.normalize().eq(day).all(), f"{split}: window/label crosses date")
            require(current.sub(first).eq(pd.Timedelta(minutes=5)).all(), f"{split}: wrong history span")
            require(label.sub(current).eq(pd.Timedelta(minutes=1)).all(), f"{split}: wrong label horizon")
            require(current.eq(current.dt.floor("min")).all(), f"{split}: non-minute timestamp")
            require(day.map(lookup).eq(split).all(), f"{split}: metadata date mapping differs")
            close = frame.close.to_numpy(dtype=np.float64)
            next_close = frame.target_close.to_numpy(dtype=np.float64)
            require(np.isfinite(close).all() and np.isfinite(next_close).all(), f"{split}: nonfinite price metadata")
            require((close > 0).all() and (next_close > 0).all(), f"{split}: nonpositive price metadata")
            direct_target = np.log(next_close / close)
            require(np.allclose(targets, direct_target, rtol=2e-6, atol=1e-7), f"{split}: y does not match metadata prices")
            max_target_difference = max(max_target_difference, float(np.max(np.abs(targets - direct_target))))
            # These independently reconstructable inputs also check array/metadata order.
            scaled_close = (np.log(close) - scaler_mean[0]) / scaler_scale[0]
            require(np.allclose(chunk[:, 0], scaled_close, rtol=2e-6, atol=1e-6), f"{split}: X close feature differs from metadata")
            require(np.array_equal(chunk[:, FEATURE_COLUMNS.index("is_call")], frame.option_type.eq("CE").to_numpy(dtype=np.float32)), f"{split}: call indicator differs")
            require(np.array_equal(chunk[:, FEATURE_COLUMNS.index("close_at_most_1")], (close <= 1).astype(np.float32)), f"{split}: low-price indicator differs")
            for name in ["is_call", "zero_volume", "close_at_most_1", "flat_ohlc"]:
                require(np.isin(chunk[:, FEATURE_COLUMNS.index(name)], [0.0, 1.0]).all(), f"{split}: invalid binary feature {name}")
            if split == "train":
                continuous = chunk[:, :len(CONTINUOUS_COLUMNS)].astype(np.float64)
                train_sum += continuous.sum(axis=0)
                train_squares += np.square(continuous).sum(axis=0)
                train_rows += len(frame)
            all_day_counts.update(day.value_counts().to_dict())
            offset = stop
        require(offset == count, f"{split}: batch row count differs")
        split_results.append({"split": split, "rows_verified": offset, "features": x.shape[1],
                              "dtype": str(x.dtype), "max_target_float32_difference": max_target_difference})
        print(f"Verified {split}: {offset:,} tensor and metadata rows", flush=True)

    actual_counts = dates.observation_date.map(all_day_counts).fillna(0).astype(np.int64)
    require(actual_counts.equals(dates.examples.astype(np.int64)), "Per-date metadata counts differ from date_splits.csv")
    require(train_rows == scaler["n_samples_seen"], "Scaler fitting row count differs from train")
    train_mean = train_sum / train_rows
    train_variance = train_squares / train_rows - np.square(train_mean)
    expected_variance = scaler_variance / np.square(scaler_scale)
    require(np.allclose(train_mean, 0, atol=2e-6, rtol=0), "Train standardized feature means are not zero")
    require(np.allclose(train_variance, expected_variance, atol=2e-5, rtol=2e-5), "Train standardized feature variances differ")

    verified_hashes = 0
    actual_outputs = {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file() and path != manifest_path}
    require(actual_outputs == set(manifest["outputs"]), "Output file set differs from manifest")
    for relative, recorded in manifest["outputs"].items():
        path = out / relative
        require(path.resolve().is_relative_to(out.resolve()), "Manifest output is outside output directory")
        require(path.stat().st_size == recorded["bytes"], f"Output size changed: {relative}")
        if hashes:
            require(digest(path) == recorded["sha256"], f"Output digest changed: {relative}")
            verified_hashes += 1
    require(digest(manifest_path) == manifest_digest, "Manifest changed during verification")
    result = {
        "status": "passed", "verified_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest_digest, "verifier_sha256": digest(Path(__file__)),
        "rows_verified": sum(item["rows_verified"] for item in split_results),
        "splits": split_results, "feature_order": FEATURE_COLUMNS,
        "train_standardized_mean": train_mean.tolist(),
        "train_standardized_variance": train_variance.tolist(),
        "expected_standardized_variance": expected_variance.tolist(),
        "scaler_training_rows": train_rows, "output_files": len(manifest["outputs"]),
        "output_hashes_verified": verified_hashes, "all_output_hashes_verified": hashes,
        "checks": ["completion manifest", "feature specification", "float32 tensor shapes and finiteness",
                   "metadata alignment", "exact date split membership and counts", "same-day five-minute input span",
                   "one-minute label horizon", "targets reconstructed from prices", "training-only scaler row count",
                   "training normalization moments", "output inventory and sizes"],
        "model_training_or_prediction_evaluation": False,
    }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Verification passed: {result['rows_verified']:,} rows; {verified_hashes} output hashes", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--skip-hashes", action="store_true", help="Skip output SHA256 checks; still check sizes and all tensor/metadata rows")
    args = parser.parse_args()
    verify(args.root.resolve(), args.batch_size, hashes=not args.skip_hashes)

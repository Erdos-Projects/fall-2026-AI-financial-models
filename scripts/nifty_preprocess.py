"""Prepare a reproducible, causal Nifty bar-forecasting dataset.

This is an empirical next-minute forecasting baseline, not a theoretical option
pricing dataset. Raw files and the EDA cache are never modified. Run with the
project Python after scripts/nifty_eda.py. Process one archive cohort at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PRICES = ["open", "high", "low", "close"]
ESSENTIAL_NUMERIC = PRICES + ["volume", "strike"]
HISTORY_BARS = 6
CONTINUOUS_COLUMNS = [
    "log_close", "log_strike", "log_volume", "log_high_low", "log_close_open",
    "return_1m", "return_5m", "return_std_5m", "log_mean_volume_6bars",
]
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + [
    "minute_sin", "minute_cos", "is_call", "zero_volume", "close_at_most_1", "flat_ohlc",
]
METADATA_COLUMNS = [
    "file_id", "source_row", "window_start", "timestamp", "label_timestamp",
    "option_type", "strike", "close", "target_close", "archive_cohort", "split",
]
TARGET = "target_log_return = log(close[t+1] / close[t]); exactly one minute ahead"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean_bars(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select positive, consistent bars within the reference clock-time window.

    Exclusions get a single first-applicable reason; marginal rule counts can
    overlap. Weekend/zero-volume/flat/low-price observations are retained. Source
    file identity is a provisional series key, never a verified contract ID.
    """
    frame = frame.copy().reset_index(drop=True)
    reason = pd.Series("", index=frame.index, dtype="object")
    numeric = frame[ESSENTIAL_NUMERIC].to_numpy(dtype=float)
    clock = frame.timestamp.dt.hour * 60 + frame.timestamp.dt.minute
    prices = frame[PRICES]
    exact_key = ["file_id", "timestamp", "option_type", *ESSENTIAL_NUMERIC]
    if "extra_field_9" in frame:
        exact_key.append("extra_field_9")
    if "encoded_expiry_candidate" in frame:
        exact_key.append("encoded_expiry_candidate")
    exact = frame.duplicated(exact_key, keep="first")
    # All conflicting values for the same source/timestamp are quarantined.
    unique = frame.loc[~exact]
    conflicts = unique.duplicated(["file_id", "timestamp"], keep=False)
    conflict_keys = pd.MultiIndex.from_frame(unique.loc[conflicts, ["file_id", "timestamp"]])
    conflict = pd.MultiIndex.from_frame(frame[["file_id", "timestamp"]]).isin(conflict_keys)
    rules = {
        "invalid_timestamp": frame.timestamp.isna(),
        "invalid_symbol_or_strike": ~frame.option_type.isin(["CE", "PE"]) | frame.strike.le(0),
        "missing_or_nonfinite_numeric": ~np.isfinite(numeric).all(axis=1),
        "nonpositive_price": prices.le(0).any(axis=1),
        "negative_volume": frame.volume.lt(0),
        "ohlc_inconsistent": (
            frame.high.lt(prices[["open", "low", "close"]].max(axis=1) - 1e-8)
            | frame.low.gt(prices[["open", "high", "close"]].min(axis=1) + 1e-8)
        ),
        "conflicting_source_timestamp": conflict,
        "exact_duplicate_extra": exact,
        "outside_reference_session": ~clock.ge(555) | ~clock.lt(930),
        "non_minute_timestamp": frame.timestamp.ne(frame.timestamp.dt.floor("min")),
    }
    counts = {"input_rows": len(frame)}
    for name, mask in rules.items():
        mask = pd.Series(mask, index=frame.index).fillna(True)
        counts[f"flag_{name}"] = int(mask.sum())
        chosen = reason.eq("") & mask
        reason.loc[chosen] = name
        counts[f"excluded_{name}"] = int(chosen.sum())
    rejected = frame.loc[reason.ne(""), ["file_id", "source_row", "timestamp"]].copy()
    rejected["reason"] = reason.loc[rejected.index]
    clean = frame.loc[reason.eq("")].copy()
    # Recompute all sequential features after cleaning; never reuse EDA returns.
    clean = clean.drop(columns=["one_min_log_return", "bar_range"], errors="ignore")
    clean = clean.sort_values(["file_id", "timestamp", "source_row"], kind="stable").reset_index(drop=True)
    clean["zero_volume"] = clean.volume.eq(0)
    clean["close_at_most_1"] = clean.close.le(1)
    clean["flat_ohlc"] = clean[PRICES].eq(clean.close, axis=0).all(axis=1)
    clean["weekend_observation"] = clean.timestamp.dt.dayofweek.ge(5)
    counts.update({"retained_rows": len(clean), "excluded_rows": len(rejected)})
    for flag in ["zero_volume", "close_at_most_1", "flat_ohlc", "weekend_observation"]:
        counts[f"retained_{flag}"] = int(clean[flag].sum())
    return clean, rejected, counts


def build_examples(clean: pd.DataFrame) -> pd.DataFrame:
    """Use six consecutive bars ending at t; predict the close at t+1.

    Six ordered unique minute bars spanning five minutes cannot contain a gap.
    Requiring the next bar at t+1 and all seven on one date prevents windows
    crossing gaps, exclusions, files, days, or date-based split boundaries.
    """
    ordered = clean.sort_values(["file_id", "timestamp"], kind="stable").reset_index(drop=True)
    if ordered.groupby("file_id", observed=True)[["option_type", "strike"]].nunique().gt(1).any().any():
        raise ValueError("A source series changes option type or strike; verify contract identity")
    by_file = ordered.groupby("file_id", sort=False, observed=True)
    past_time = by_file.timestamp.shift(HISTORY_BARS - 1)
    next_time = by_file.timestamp.shift(-1)
    same_day = (
        past_time.dt.normalize().eq(ordered.timestamp.dt.normalize())
        & next_time.dt.normalize().eq(ordered.timestamp.dt.normalize())
    )
    eligible = (
        ordered.timestamp.sub(past_time).eq(pd.Timedelta(minutes=HISTORY_BARS - 1))
        & next_time.sub(ordered.timestamp).eq(pd.Timedelta(minutes=1))
        & same_day
    )
    previous = by_file.close.shift(1)
    past_close = by_file.close.shift(HISTORY_BARS - 1)
    next_close = by_file.close.shift(-1)
    returns = np.log(ordered.close / previous)
    # Only eligible rows are used, so each retained rolling window is contiguous.
    ret_std = returns.groupby(ordered.file_id, sort=False).rolling(5, min_periods=5).std(ddof=0).reset_index(level=0, drop=True)
    mean_volume = by_file.volume.rolling(6, min_periods=6).mean().reset_index(level=0, drop=True)
    minute = ordered.timestamp.dt.hour * 60 + ordered.timestamp.dt.minute
    out = ordered.loc[eligible, ["file_id", "source_row", "timestamp", "option_type", "strike", "close"]].copy()
    out["window_start"] = past_time.loc[eligible]
    out["label_timestamp"] = next_time.loc[eligible]
    out["target_close"] = next_close.loc[eligible]
    out["target_log_return"] = np.log(out.target_close / out.close)
    values = {
        "log_close": np.log(ordered.close),
        "log_strike": np.log(ordered.strike),
        "log_volume": np.log1p(ordered.volume),
        "log_high_low": np.log(ordered.high / ordered.low),
        "log_close_open": np.log(ordered.close / ordered.open),
        "return_1m": returns,
        "return_5m": np.log(ordered.close / past_close),
        "return_std_5m": ret_std,
        "log_mean_volume_6bars": np.log1p(mean_volume),
        "minute_sin": np.sin(2 * np.pi * minute / 1440),
        "minute_cos": np.cos(2 * np.pi * minute / 1440),
        "is_call": ordered.option_type.eq("CE").astype(float),
        "zero_volume": ordered.volume.eq(0).astype(float),
        "close_at_most_1": ordered.close.le(1).astype(float),
        "flat_ohlc": ordered[PRICES].eq(ordered.close, axis=0).all(axis=1).astype(float),
    }
    for name in FEATURE_COLUMNS:
        out[name] = values[name].loc[eligible].astype(float)
    if not np.isfinite(out[FEATURE_COLUMNS + ["target_log_return"]].to_numpy()).all():
        raise ValueError("Nonfinite feature/label after constructing eligible windows")
    return out.reset_index(drop=True)


def split_dates(dates) -> pd.DataFrame:
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().drop_duplicates().sort_values()
    if days.hasnans or len(days) < 3:
        raise ValueError("Need at least three valid observed dates")
    n_train, n_valid = max(1, int(.70 * len(days))), max(1, int(.15 * len(days)))
    n_train = min(n_train, len(days) - n_valid - 1)
    labels = np.array(["train"] * n_train + ["valid"] * n_valid + ["test"] * (len(days) - n_train - n_valid))
    return pd.DataFrame({"observation_date": days, "split": labels})


def verify_inputs(root: Path) -> tuple[list[Path], pd.DataFrame, dict]:
    eda_dir = root / "data/processed/nifty_eda"
    eda_summary = json.loads((root / "reports/nifty_eda/summary.json").read_text())
    inventory_path = eda_dir / "file_inventory.csv"
    if digest(inventory_path) != eda_summary["input_inventory_sha256"]:
        raise ValueError("EDA inventory changed; rerun scripts/nifty_eda.py")
    inventory = pd.read_csv(inventory_path)
    selected = inventory.loc[inventory.selected.astype(str).str.lower().eq("true")].sort_values("relative_path").reset_index(drop=True)
    selected["file_id"] = np.arange(len(selected), dtype=np.int32)
    selected["archive_cohort"] = selected.archive_year.astype(str) + "-" + selected.archive_month.astype(str).str.zfill(2)
    # Check every physical representation, including previously excluded copies.
    actual_paths = {str(p.relative_to(root / "Nifty Options Data")) for p in (root / "Nifty Options Data").rglob("*") if p.suffix.lower() in {".csv", ".txt"}}
    if actual_paths != set(inventory.relative_path):
        raise ValueError("Source file list changed; rerun EDA with --refresh-inventory")
    for source in inventory.itertuples(index=False):
        path = root / "Nifty Options Data" / source.relative_path
        if path.stat().st_size != source.size_bytes or digest(path) != source.sha256:
            raise ValueError(f"Raw source changed: {source.relative_path}; refresh EDA first")
    files = sorted((eda_dir / "bars").glob("*.parquet"))
    if {p.stem for p in files} != set(selected.archive_cohort):
        raise ValueError("EDA partitions do not match the selected source inventory")
    if sum(pq.ParquetFile(p).metadata.num_rows for p in files) != eda_summary["selected_rows"]:
        raise ValueError("EDA cached row count does not match its summary")
    return files, selected, eda_summary


def run(root: Path = ROOT) -> None:
    out = root / "data/processed/nifty_training"
    report = root / "reports/nifty_preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    # Consumers require this completion marker. Failed reruns cannot look ready.
    manifest_path = out / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    (report / "summary.json").unlink(missing_ok=True)
    files, selected, eda_summary = verify_inputs(root)
    print(f"Verified all raw hashes; preprocessing {len(files)} audited partitions", flush=True)
    for folder in ["clean_bars", "examples", "rejected", "train", "valid", "test"]:
        (out / folder).mkdir(exist_ok=True)
        # Remove only stale artifacts owned by this pipeline, never source data.
        for old in (out / folder).glob("*.parquet"):
            old.unlink()
    clean_counts, date_counts, checksums = [], Counter(), {}
    for path in files:
        bars = pd.read_parquet(path)
        expected = selected.loc[selected.archive_cohort.eq(path.stem)].set_index("file_id").nonempty_lines.sort_index()
        observed = bars.groupby("file_id").size().sort_index()
        if not observed.equals(expected.astype(observed.dtype).rename(observed.name)):
            raise ValueError(f"EDA source row counts changed in {path.name}")
        if bars.duplicated(["file_id", "source_row"]).any():
            raise ValueError(f"Duplicate source row identifiers in {path.name}")
        checksums[path.name] = digest(path)
        clean, rejected, counts = clean_bars(bars)
        clean["archive_cohort"] = path.stem
        clean.to_parquet(out / "clean_bars" / path.name, index=False)
        rejected.to_parquet(out / "rejected" / path.name, index=False)
        date_counts.update(clean.timestamp.dt.normalize().value_counts().to_dict())
        clean_counts.append({"archive_cohort": path.stem, **counts})
    exclusions = pd.DataFrame(clean_counts)
    exclusions.to_csv(report / "exclusions.csv", index=False)
    date_map = split_dates(list(date_counts))
    date_map["clean_rows"] = date_map.observation_date.map(date_counts)
    split_lookup = date_map.set_index("observation_date").split
    scaler = StandardScaler()
    example_counts, day_examples = Counter(), Counter()
    distribution = []
    for path in files:
        clean = pd.read_parquet(out / "clean_bars" / path.name)
        examples = build_examples(clean)
        examples["archive_cohort"] = path.stem
        examples["split"] = examples.timestamp.dt.normalize().map(split_lookup)
        if examples.split.isna().any():
            raise ValueError("Unassigned split")
        if not examples.label_timestamp.dt.normalize().eq(examples.timestamp.dt.normalize()).all():
            raise ValueError("Label crosses a split date boundary")
        examples.to_parquet(out / "examples" / path.name, index=False)
        train = examples.loc[examples.split.eq("train"), CONTINUOUS_COLUMNS]
        if len(train):
            scaler.partial_fit(train.to_numpy(dtype=np.float64))
        example_counts.update(examples.split.value_counts().to_dict())
        day_examples.update(examples.timestamp.dt.normalize().value_counts().to_dict())
        for (split, side), part in examples.groupby(["split", "option_type"], observed=True):
            distribution.append({"split": split, "option_type": side, "examples": len(part),
                                 "zero_volume": int(part.zero_volume.sum()),
                                 "close_at_most_1": int(part.close_at_most_1.sum())})
        print(f"{path.stem}: {len(clean):,} clean bars; {len(examples):,} eligible examples", flush=True)
    if any(example_counts[split] == 0 for split in ["train", "valid", "test"]):
        raise ValueError("Every split must contain supervised examples")
    date_map["examples"] = date_map.observation_date.map(day_examples).fillna(0).astype(int)
    date_map.to_csv(report / "date_splits.csv", index=False)
    split_rows = []
    for split in ["train", "valid", "test"]:
        part = date_map.loc[date_map.split.eq(split)]
        split_rows.append({"split": split, "start_date": str(part.observation_date.min().date()),
                           "end_date": str(part.observation_date.max().date()), "dates": len(part),
                           "clean_rows": int(part.clean_rows.sum()), "examples": int(part.examples.sum())})
    split_summary = pd.DataFrame(split_rows)
    split_summary["example_percent"] = split_summary.examples / split_summary.examples.sum() * 100
    split_summary.to_csv(report / "split_summary.csv", index=False)
    pd.DataFrame(distribution).groupby(["split", "option_type"]).sum().reset_index().to_csv(report / "split_diagnostics.csv", index=False)
    scaler_data = {"continuous_columns": CONTINUOUS_COLUMNS, "mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist(), "variance": scaler.var_.tolist(),
                   "n_samples_seen": int(scaler.n_samples_seen_), "fit_split": "train"}
    write_json(out / "scaler.json", scaler_data)
    feature_spec = {
        "feature_columns": FEATURE_COLUMNS, "standardized_columns": CONTINUOUS_COLUMNS,
        "target": TARGET, "target_standardized": False, "history_bars": HISTORY_BARS,
        "forecast_origin": "After observing the complete bar labeled t; source start/end bar-label convention unverified",
        "definitions": {
            "log_close": "ln(close[t])", "log_strike": "ln(strike)",
            "log_volume": "ln(1 + reported volume[t]); units unverified",
            "log_high_low": "ln(high[t]/low[t]); bar range, not bid-ask spread",
            "log_close_open": "ln(close[t]/open[t])",
            "return_1m": "ln(close[t]/close[t-1])", "return_5m": "ln(close[t]/close[t-5])",
            "return_std_5m": "Population standard deviation of five one-minute returns from t-4 through t",
            "log_mean_volume_6bars": "ln(1 + mean(reported volume[t-5:t]))",
            "minute_sin": "sin(2*pi*source minute of day/1440)",
            "minute_cos": "cos(2*pi*source minute of day/1440)",
            "is_call": "1 for CE, 0 for PE", "zero_volume": "1 if reported volume[t] = 0",
            "close_at_most_1": "1 if close[t] <= 1", "flat_ohlc": "1 if all OHLC[t] equal",
        },
        "excluded_inputs": ["extra_field_9", "encoded_expiry_candidate", "file_id", "source_row", "archive_cohort",
                            "timestamp", "label_timestamp", "window_start", "target_close", "target_log_return"],
        "price_reconstruction": "predicted_next_close = current_close * exp(predicted_log_return)",
        "naive_baseline": "Predict zero log return, equivalent to next close = current close",
    }
    write_json(out / "feature_spec.json", feature_spec)
    offsets = Counter()
    arrays, writers = {}, {}
    try:
        for split in ["train", "valid", "test"]:
            arrays[split] = (
                np.lib.format.open_memmap(out / split / "X.npy", mode="w+", dtype="float32", shape=(example_counts[split], len(FEATURE_COLUMNS))),
                np.lib.format.open_memmap(out / split / "y.npy", mode="w+", dtype="float32", shape=(example_counts[split],)),
            )
        for path in files:
            examples = pd.read_parquet(out / "examples" / path.name)
            for split, part in examples.groupby("split", sort=False, observed=True):
                x = part[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
                x[:, :len(CONTINUOUS_COLUMNS)] = scaler.transform(x[:, :len(CONTINUOUS_COLUMNS)])
                y = part.target_log_return.to_numpy(dtype=np.float32)
                x = x.astype(np.float32)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite tensor in {split}")
                start, stop = offsets[split], offsets[split] + len(part)
                arrays[split][0][start:stop] = x
                arrays[split][1][start:stop] = y
                metadata = part[METADATA_COLUMNS].copy()
                metadata["option_type"] = metadata.option_type.astype(str)
                table = pa.Table.from_pandas(metadata, preserve_index=False)
                if split not in writers:
                    writers[split] = pq.ParquetWriter(out / split / "metadata.parquet", table.schema)
                writers[split].write_table(table)
                offsets[split] = stop
        for split, (x, y) in arrays.items():
            x.flush()
            y.flush()
            if offsets[split] != example_counts[split]:
                raise ValueError(f"Tensor row count mismatch in {split}")
    finally:
        for writer in writers.values():
            writer.close()
    selected.to_csv(out / "source_manifest.csv", index=False)
    totals = exclusions.drop(columns="archive_cohort").sum().astype(int).to_dict()
    summary = {
        "selected_raw_rows": int(eda_summary["selected_rows"]), "retained_clean_rows": totals["retained_rows"],
        "excluded_rows": totals["excluded_rows"], "total_examples": sum(example_counts.values()),
        "feature_count": len(FEATURE_COLUMNS), "history_bars": HISTORY_BARS, "target": TARGET,
        "observed_dates": len(date_map), "counts": totals, "splits": split_rows,
        "clean_rows_without_eligible_window": totals["retained_rows"] - sum(example_counts.values()),
        "status": "Ready for empirical bar forecasting under documented source semantics assumptions; not theoretical option pricing",
        "split_method": "Chronological 70/15/15 by unique retained source dates; all files on a date share its split",
        "test_status": "Held out from fitting/normalization/tuning; the existing EDA already examined all years",
        "session_scope": "09:15 inclusive to 15:30 exclusive in source clock time; special sessions outside this window excluded by scope, not labeled erroneous",
        "missing_value_policy": "Drop invalid essential fields; omit structurally missing/unverified extra field 9; no imputation or gap filling",
        "identity_caveat": "file_id is only a source-series key; complete exchange contract identity remains unverified",
        "outlier_policy": "Keep valid extreme/low/flat/zero-volume observations; no target-dependent filtering or winsorization",
    }
    write_json(report / "summary.json", summary)
    outputs = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": digest(p)}
               for p in out.rglob("*") if p.is_file() and p != manifest_path}
    manifest = {"status": "complete", "created_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_sha256": digest(Path(__file__)), "input_inventory_sha256": eda_summary["input_inventory_sha256"],
                "input_parquet_sha256": checksums, "raw_hashes_verified": True,
                "versions": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "scikit-learn"]},
                "feature_count": len(FEATURE_COLUMNS), "splits": split_rows, "outputs": outputs}
    write_json(manifest_path, manifest)
    print(split_summary.to_string(index=False), flush=True)
    print(f"Complete: {sum(example_counts.values()):,} examples, {len(FEATURE_COLUMNS)} features; {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root.resolve())

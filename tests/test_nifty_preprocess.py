"""Regression tests for data exclusions, causal features, and temporal splits."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "nifty_preprocess",
    Path(__file__).resolve().parents[1] / "scripts/nifty_preprocess.py",
)
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)


def bars(n=9, start="2024-01-08 09:15", file_id=1):
    close = 10.0 + np.arange(n) * 0.1
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="min"),
        "file_id": file_id,
        "source_row": np.arange(n),
        "option_type": "CE",
        "strike": 10000.0,
        "open": close - 0.02,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.arange(n, dtype=float) + 1,
    })


@pytest.mark.parametrize("column,value", [
    ("timestamp", pd.NaT),
    ("option_type", "UNKNOWN"),
    ("strike", np.nan),
    ("strike", 0.0),
    ("open", np.nan),
    ("high", np.inf),
    ("low", 0.0),
    ("close", -1.0),
    ("volume", np.nan),
    ("volume", -1.0),
    ("open", 100.0),
    ("low", 100.0),
])
def test_corrupt_bars_are_rejected_with_an_auditable_reason(column, value):
    frame = bars(2)
    frame.loc[0, column] = value
    clean, rejected, _ = prep.clean_bars(frame)
    assert clean["source_row"].tolist() == [1]
    assert rejected["source_row"].tolist() == [0]
    assert rejected["reason"].notna().all()
    assert rejected["reason"].astype(str).str.len().gt(0).all()


def test_session_window_is_half_open_and_weekend_low_price_zero_volume_are_kept():
    frame = bars(5, start="2024-01-06 09:14")  # Saturday is not auto-excluded.
    frame["timestamp"] = pd.to_datetime([
        "2024-01-06 09:14", "2024-01-06 09:15", "2024-01-06 15:29",
        "2024-01-06 15:30", "2024-01-06 16:00",
    ])
    frame.loc[1, ["open", "high", "low", "close"]] = [0.05, 0.05, 0.05, 0.05]
    frame.loc[1, "volume"] = 0
    clean, rejected, _ = prep.clean_bars(frame)
    assert clean["source_row"].tolist() == [1, 2]
    assert set(rejected["source_row"]) == {0, 3, 4}
    assert clean["flat_ohlc"].tolist() == [True, False]


@pytest.mark.parametrize("offset", ["1s", "1us", "1ns"])
def test_only_exact_minute_timestamps_are_accepted(offset):
    frame = bars(2)
    frame.loc[0, "timestamp"] += pd.Timedelta(offset)
    clean, rejected, _ = prep.clean_bars(frame)
    assert clean["source_row"].tolist() == [1]
    assert rejected["source_row"].tolist() == [0]
    assert rejected["reason"].tolist() == ["non_minute_timestamp"]


def test_duplicates_are_resolved_only_within_each_source_file():
    base = bars(3)
    exact_duplicate = base.iloc[[0]].copy()
    exact_duplicate["source_row"] = 10
    conflicting_duplicate = base.iloc[[1]].copy()
    conflicting_duplicate["source_row"] = 11
    conflicting_duplicate["volume"] += 1
    other_file = base.iloc[[0]].copy()
    other_file["file_id"] = 2
    frame = pd.concat([base, exact_duplicate, conflicting_duplicate, other_file], ignore_index=True)
    clean, rejected, _ = prep.clean_bars(frame)
    assert list(zip(clean["file_id"], clean["source_row"])) == [(1, 0), (1, 2), (2, 0)]
    assert set(rejected["source_row"]) == {1, 10, 11}


def test_clean_bars_sort_within_source_before_making_windows():
    frame = pd.concat([bars(8, file_id=2), bars(8, file_id=1)], ignore_index=True)
    clean, rejected, _ = prep.clean_bars(frame.sample(frac=1, random_state=9))
    assert rejected.empty
    assert list(zip(clean["file_id"], clean["timestamp"])) == sorted(
        zip(clean["file_id"], clean["timestamp"])
    )


def test_examples_use_six_history_bars_and_the_immediate_next_bar():
    frame, _, _ = prep.clean_bars(bars(9))
    examples = prep.build_examples(frame)
    assert len(examples) == 3
    assert examples["timestamp"].tolist() == frame["timestamp"].iloc[5:8].tolist()
    assert examples["window_start"].tolist() == frame["timestamp"].iloc[:3].tolist()
    assert examples["label_timestamp"].tolist() == frame["timestamp"].iloc[6:9].tolist()
    np.testing.assert_allclose(examples["target_close"], frame["close"].iloc[6:9])
    np.testing.assert_allclose(
        examples["target_log_return"],
        np.log(frame["close"].iloc[6:9].to_numpy() / frame["close"].iloc[5:8].to_numpy()),
    )
    assert np.isfinite(examples[list(prep.FEATURE_COLUMNS)].to_numpy()).all()


@pytest.mark.parametrize("case", ["gap", "day", "file"])
def test_windows_never_bridge_missing_minutes_days_or_source_files(case):
    frame = bars(7)
    if case == "gap":
        frame.loc[3:, "timestamp"] += pd.Timedelta(minutes=1)
    elif case == "day":
        frame.loc[3:, "timestamp"] += pd.Timedelta(days=1)
    else:
        frame.loc[3:, "file_id"] = 2
    clean, _, _ = prep.clean_bars(frame)
    assert prep.build_examples(clean).empty


@pytest.mark.parametrize("column,value", [("option_type", "PE"), ("strike", 11000.0)])
def test_source_series_cannot_change_option_identity(column, value):
    frame = bars(7)
    frame.loc[6, column] = value
    clean, rejected, _ = prep.clean_bars(frame)
    assert rejected.empty
    with pytest.raises(ValueError, match="option type or strike"):
        prep.build_examples(clean)


def test_distinct_sources_can_have_distinct_option_identities():
    first = bars(7, file_id=1)
    second = bars(7, file_id=2)
    second["option_type"] = "PE"
    second["strike"] = 11000.0
    clean, _, _ = prep.clean_bars(pd.concat([first, second], ignore_index=True))
    examples = prep.build_examples(clean)
    assert examples["file_id"].tolist() == [1, 2]
    assert examples["is_call"].tolist() == [1.0, 0.0]
    assert examples["strike"].tolist() == [10000.0, 11000.0]


def test_changing_future_price_changes_label_but_not_features():
    original, _, _ = prep.clean_bars(bars(7))
    changed = original.copy()
    changed.loc[6, ["open", "high", "low", "close"]] *= 2
    changed.loc[6, "volume"] *= 100
    before = prep.build_examples(original)
    after = prep.build_examples(changed)
    pd.testing.assert_frame_equal(
        before[list(prep.FEATURE_COLUMNS)], after[list(prep.FEATURE_COLUMNS)]
    )
    assert before["target_close"].iloc[0] != after["target_close"].iloc[0]
    assert before["target_log_return"].iloc[0] != after["target_log_return"].iloc[0]
    assert not {"target_close", "target_log_return", "label_timestamp", "file_id", "source_row"}.intersection(
        prep.FEATURE_COLUMNS
    )


def test_date_splits_are_sorted_unique_disjoint_and_chronological():
    dates = list(pd.date_range("2024-01-01", periods=10, freq="D"))
    result = prep.split_dates(list(reversed(dates)) + dates[:2])
    assert result["observation_date"].tolist() == dates
    assert result["split"].tolist() == ["train"] * 7 + ["valid"] + ["test"] * 2
    assert result["observation_date"].is_unique


def test_smallest_valid_date_split_keeps_each_partition_nonempty():
    result = prep.split_dates(pd.date_range("2024-01-01", periods=3, freq="D"))
    assert result["split"].tolist() == ["train", "valid", "test"]


@pytest.mark.parametrize("n", [0, 1, 2])
def test_date_splits_require_at_least_three_dates(n):
    with pytest.raises(ValueError):
        prep.split_dates(pd.date_range("2024-01-01", periods=n, freq="D"))

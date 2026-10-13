"""Full-data, read-only-source EDA for the local Nifty archive.

Run from any directory with the project's Python. Only byte-identical format
copies selected by nifty_inventory.py are excluded. Other suspicious records
are counted and retained, not silently repaired or treated as market quotes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_COLUMNS = ["symbol", "date", "time", "open", "high", "low", "close", "volume", "extra_field_9"]
NUMERIC = RAW_COLUMNS[3:]
MONTHS = {name: i for i, name in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
SEED = 20261012
SAMPLE_SIZE = 100_000


def symbol_parts(value: str) -> tuple[str, float, str | None]:
    """Keep encoded dates separate from strike; no expiry calendar inference."""
    value = re.sub(r"\s+", "", str(value)).upper()
    match = re.fullmatch(r"(CE|PE)(\d+(?:\.\d+)?)", value)
    if match:
        return match[1], float(match[2]), None
    match = re.fullmatch(r"NIFTY(\d{2})([A-Z]{3})(\d{2})(\d+(?:\.\d+)?)(CE|PE)", value)
    if match and match[2] in MONTHS:
        encoded = date(2000 + int(match[3]), MONTHS[match[2]], int(match[1])).isoformat()
        return match[5], float(match[4]), encoded
    match = re.fullmatch(r"(?:NIFTY)?(\d+(?:\.\d+)?)(CE|PE)", value)
    if match:
        return match[2], float(match[1]), None
    return "UNKNOWN", float("nan"), None


def weighted_stats(counts: Counter) -> dict:
    """Exact row-weighted quantiles with NumPy's linear interpolation rule."""
    if not counts:
        return {"count": 0}
    pairs = sorted(counts.items())
    values = np.array([x[0] for x in pairs], dtype=float)
    weights = np.array([x[1] for x in pairs], dtype=np.int64)
    cumulative = weights.cumsum()
    n = int(cumulative[-1])
    mean = float(np.dot(values, weights) / n)
    result = {"count": n, "min": float(values[0]), "max": float(values[-1]), "mean": mean,
              "std": float(np.sqrt(np.dot((values - mean) ** 2, weights) / (n - 1))) if n > 1 else 0.0}
    for label, probability in [("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p99", .99)]:
        position = (n - 1) * probability
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        a, b = values[np.searchsorted(cumulative, [lo, hi], side="right")]
        result[label] = float(a + (b - a) * (position - lo))
    return result


def finite_counts(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return pd.Series(values).value_counts().to_dict()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def make_figures(tables: Path, figures: Path, sample: pd.DataFrame) -> list[str]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                        "axes.spines.right": False, "figure.dpi": 130, "axes.titleweight": "bold"})
    colors = {"CE": "#247a96", "PE": "#ce7745"}
    outputs = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs.append(name)

    monthly = pd.read_csv(tables / "monthly_activity.csv")
    monthly["month"] = pd.to_datetime(monthly["month"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly.month, monthly.rows / 1e6, width=24, color="#247a96")
    ax.set(title="Observed bars by actual trading month", ylabel="Rows (millions)", xlabel="Timestamp month")
    ax.text(.01, .97, "Identical CSV/TXT copies removed; changing contract coverage affects counts", transform=ax.transAxes, va="top", fontsize=8)
    finish(fig, "01_monthly_coverage.png")

    calls = pd.read_csv(tables / "call_put_by_year.csv")
    pivot = calls.pivot(index="trade_year", columns="option_type", values="rows").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    (pivot / 1e6).plot.bar(ax=ax, color=[colors.get(c, "grey") for c in pivot.columns], rot=0)
    ax.set(title="Call / put observations", xlabel="Actual trading year", ylabel="Rows (millions)")
    ax.legend(title="Option type")
    finish(fig, "02_call_put_coverage.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for side, color in colors.items():
        part = sample.loc[sample.option_type == side]
        axes[0].hist(np.log10(part.loc[part.close > 0, "close"]), bins=60, histtype="step", linewidth=1.4, color=color, label=side, density=True)
        axes[1].hist(np.log10(part.volume.clip(lower=0) + 1), bins=60, histtype="step", linewidth=1.4, color=color, label=side, density=True)
    axes[0].set(xlabel="log10(close), reported price units", ylabel="Density", title="Positive option close prices")
    axes[1].set(xlabel="log10(volume + 1), source units", ylabel="Density", title="Reported bar volume")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"Uniform priority sample: {len(sample):,} rows; seed {SEED}", fontsize=11)
    finish(fig, "03_price_volume_distributions.png")

    intraday = pd.read_csv(tables / "intraday_activity.csv")
    regular = intraday[(intraday.minute >= 555) & (intraday.minute < 930)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for side, part in regular.groupby("option_type"):
        axes[0].plot(part.minute, part.rows, color=colors.get(side, "grey"), label=side)
        axes[1].plot(part.minute, part.volume_sum / part.rows, color=colors.get(side, "grey"), label=side)
    for ax in axes:
        ax.set_xticks([555, 630, 705, 780, 855, 929], ["09:15", "10:30", "11:45", "13:00", "14:15", "15:29"])
        ax.set_xlabel("Source clock time (timezone not supplied)")
        ax.legend()
    axes[0].set(title="Minute-of-day observation counts", ylabel="Rows")
    axes[1].set(title="Mean reported volume per observed bar", ylabel="Source units")
    finish(fig, "04_intraday_activity.png")

    density = pd.read_csv(tables / "file_density_histogram.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(density.unique_regular_minutes, density.source_days, width=1, color="#247a96")
    ax.set(title="Observed minutes per source file / observed day", xlabel="Unique minutes within 09:15–15:29 (375 possible slots)", ylabel="Source-file days (log scale)", yscale="log")
    ax.text(.02, .96, "Sparse bars do not by themselves establish missing data or no-trade minutes", transform=ax.transAxes, va="top", fontsize=8)
    finish(fig, "05_minute_density.png")

    schema = pd.read_csv(tables / "schema_by_month.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(schema))
    ax.bar(x, schema.rows_8 / 1e6, label="8 fields", color="#247a96")
    ax.bar(x, schema.rows_9 / 1e6, bottom=schema.rows_8 / 1e6, label="9 fields", color="#ce7745")
    ax.set_xticks(x[::3], schema.archive_cohort.iloc[::3], rotation=45, ha="right")
    ax.set(title="Source schema change", xlabel="Archive cohort (not necessarily trading month)", ylabel="Rows (millions)")
    ax.legend()
    finish(fig, "06_schema_change.png")

    gaps = pd.read_csv(tables / "minute_gaps_by_year.csv")
    gap_pivot = gaps.pivot(index="archive_year", columns="gap_bucket", values="count").fillna(0)
    order = [c for c in ["0", "1", "2-5", "6-30", ">30"] if c in gap_pivot]
    proportions = gap_pivot[order].div(gap_pivot[order].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    proportions.plot.bar(stacked=True, ax=ax, rot=0, colormap="viridis")
    ax.set(title="Within-day time gaps in each source file", xlabel="Archive year", ylabel="Share of adjacent observations")
    ax.legend(title="Gap (minutes)", loc="upper left", bbox_to_anchor=(1, 1))
    finish(fig, "07_intraday_gaps.png")
    return outputs


def run(refresh_inventory: bool = False) -> None:
    processed = ROOT / "data/processed/nifty_eda"
    report = ROOT / "reports/nifty_eda"
    tables, figures = report / "tables", report / "figures"
    for directory in [processed, tables, figures, processed / "bars"]:
        directory.mkdir(parents=True, exist_ok=True)
    inventory_path = processed / "file_inventory.csv"
    if refresh_inventory or not inventory_path.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/nifty_inventory.py")], cwd=ROOT, check=True)
    inventory = pd.read_csv(inventory_path)
    selected = inventory.loc[inventory.selected.astype(str).str.lower() == "true"].copy()
    selected = selected.sort_values("relative_path").reset_index(drop=True)
    selected["file_id"] = np.arange(len(selected), dtype=np.int32)
    selected["archive_cohort"] = selected.archive_year.astype(str) + "-" + selected.archive_month.astype(str).str.zfill(2)
    numeric_counts = defaultdict(Counter)
    quality_by_year = defaultdict(Counter)
    gap_counts = Counter()
    intraday_counts, intraday_volume = Counter(), Counter()
    call_put = Counter()
    file_summaries, source_days, coverage, collisions, schema_rows = [], [], [], [], []
    issue_examples = defaultdict(list)
    sample = pd.DataFrame()
    rng = np.random.default_rng(SEED)
    total_loaded = 0
    for cohort, files in selected.groupby("archive_cohort", sort=True):
        frames = []
        cohort_schema = Counter()
        for _, source in files.iterrows():
            path = ROOT / "Nifty Options Data" / source.relative_path
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != source.sha256:
                raise ValueError(f"Inventory hash changed for {path}; rerun with --refresh-inventory")
            raw = pd.read_csv(io.BytesIO(content), header=None, names=RAW_COLUMNS, dtype=str, keep_default_na=False)
            if len(raw) != source.nonempty_lines:
                raise ValueError(f"Inventory row count changed for {path}; rerun with --refresh-inventory")
            if path.stat().st_size != source.size_bytes:
                raise ValueError(f"Inventory file size changed for {path}; rerun with --refresh-inventory")
            for width, count in json.loads(source.field_counts_json).items():
                if int(width) not in [8, 9]:
                    raise ValueError(f"Unsupported source width {width}: {path}")
                cohort_schema[f"rows_{width}"] += int(count)
            parts = {symbol: symbol_parts(symbol) for symbol in raw.symbol.unique()}
            frame = pd.DataFrame(index=raw.index)
            frame["timestamp"] = pd.to_datetime(raw.date.str.strip() + " " + raw.time.str.strip(), format="%Y/%m/%d %H:%M", errors="coerce")
            frame["option_type"] = raw.symbol.map({key: value[0] for key, value in parts.items()})
            frame["strike"] = raw.symbol.map({key: value[1] for key, value in parts.items()})
            frame["encoded_expiry_candidate"] = pd.to_datetime(raw.symbol.map({key: value[2] for key, value in parts.items()}))
            flags = {}
            for column in NUMERIC:
                values = pd.to_numeric(raw[column], errors="coerce").astype(float)
                flags[f"missing_{column}"] = values.isna()
                flags[f"invalid_numeric_{column}"] = raw[column].str.strip().ne("") & values.isna()
                flags[f"nonfinite_{column}"] = values.notna() & ~np.isfinite(values)
                frame[column] = values.replace([np.inf, -np.inf], np.nan)
            frame["file_id"] = int(source.file_id)
            frame["source_row"] = np.arange(1, len(raw) + 1, dtype=np.int32)
            frame["one_min_log_return"] = np.nan
            prices = frame[["open", "high", "low", "close"]]
            clock = frame.timestamp.dt.hour * 60 + frame.timestamp.dt.minute
            regular = clock.ge(555) & clock.lt(930)
            flags.update({
                "invalid_timestamp": frame.timestamp.isna(),
                "unparsed_symbol": frame.option_type.eq("UNKNOWN"),
                "negative_price": prices.lt(0).any(axis=1),
                "zero_close": frame.close.eq(0),
                "nonpositive_close": frame.close.le(0),
                "close_at_most_1": frame.close.le(1),
                "negative_volume": frame.volume.lt(0),
                "zero_volume": frame.volume.eq(0),
                "noninteger_volume": frame.volume.notna() & frame.volume.mod(1).ne(0),
                "negative_extra_field_9": frame.extra_field_9.lt(0),
                "ohlc_inconsistent": frame.high.lt(prices[["open", "close", "low"]].max(axis=1) - 1e-8) | frame.low.gt(prices[["open", "close", "high"]].min(axis=1) + 1e-8),
                "flat_ohlc": prices.nunique(axis=1).eq(1) & prices.notna().all(axis=1),
                "outside_reference_session": frame.timestamp.notna() & ~regular,
                "weekend_observation": frame.timestamp.dt.dayofweek.ge(5),
                "exact_duplicate_extra": raw.duplicated(keep="first"),
                "duplicate_source_timestamp_extra": raw.duplicated(["symbol", "date", "time"], keep="first"),
                "out_of_order_step": frame.timestamp.diff().lt(pd.Timedelta(0)),
                "has_encoded_expiry_candidate": frame.encoded_expiry_candidate.notna(),
            })
            filename = symbol_parts(source.stem)
            flags["filename_symbol_mismatch"] = frame.option_type.ne(filename[0]) | frame.strike.ne(filename[1])
            ordered = frame.sort_values("timestamp", kind="stable")
            elapsed = ordered.timestamp.diff().dt.total_seconds() / 60
            same_day = ordered.timestamp.dt.normalize().eq(ordered.timestamp.shift().dt.normalize())
            for label, mask in [("0", elapsed.eq(0)), ("1", elapsed.eq(1)), ("2-5", elapsed.gt(1) & elapsed.le(5)),
                                ("6-30", elapsed.gt(5) & elapsed.le(30)), (">30", elapsed.gt(30))]:
                gap_counts[(int(source.archive_year), label)] += int((mask & same_day).sum())
            previous = ordered.close.shift()
            valid_return = same_day & elapsed.eq(1) & previous.gt(0) & ordered.close.gt(0)
            log_return = np.log(ordered.loc[valid_return, "close"] / previous.loc[valid_return])
            frame.loc[log_return.index, "one_min_log_return"] = log_return
            flags["valid_one_min_return"] = frame.one_min_log_return.notna()
            flags["unchanged_one_min_close"] = frame.one_min_log_return.eq(0)
            flags["absolute_one_min_log_return_over_log2"] = frame.one_min_log_return.abs().gt(np.log(2))
            stats = {"file_id": int(source.file_id), "relative_path": source.relative_path,
                     "archive_cohort": cohort, "archive_year": int(source.archive_year), "rows": len(frame),
                     "symbol": raw.symbol.iloc[0], "option_type": frame.option_type.iloc[0], "strike": frame.strike.iloc[0],
                     "first_timestamp": frame.timestamp.min(), "last_timestamp": frame.timestamp.max(),
                     "observed_days": frame.timestamp.dt.normalize().nunique(),
                     "encoded_expiry_candidate": next(iter(parts.values()))[2]}
            quality_by_year[int(source.archive_year)]["rows"] += len(frame)
            for name, mask in flags.items():
                count = int(mask.sum())
                stats[name] = count
                quality_by_year[int(source.archive_year)][name] += count
                if count and len(issue_examples[name]) < 5:
                    for index in frame.index[mask][:5 - len(issue_examples[name])]:
                        issue_examples[name].append({"issue": name, "relative_path": source.relative_path,
                                                     "source_row": int(index + 1), **raw.loc[index].to_dict()})
            file_summaries.append(stats)
            total_loaded += len(frame)
            days = frame.assign(observation_date=frame.timestamp.dt.normalize(), minute=clock,
                                regular_minute=clock.where(regular)).groupby("observation_date").agg(
                rows=("timestamp", "size"), unique_minutes=("minute", "nunique"),
                unique_regular_minutes=("regular_minute", "nunique"), first_time=("timestamp", "min"),
                last_time=("timestamp", "max"), volume_sum=("volume", "sum"))
            days = days.reset_index()
            days["file_id"], days["archive_year"], days["archive_cohort"] = int(source.file_id), int(source.archive_year), cohort
            days["option_type"], days["strike"] = frame.option_type.iloc[0], frame.strike.iloc[0]
            source_days.append(days)
            for (minute, side), group in frame.assign(minute=clock).groupby(["minute", "option_type"]):
                intraday_counts[(int(minute), side)] += len(group)
                intraday_volume[(int(minute), side)] += float(group.volume.sum())
            for (year, side), count in frame.groupby([frame.timestamp.dt.year, "option_type"]).size().items():
                call_put[(int(year), side)] += int(count)
            frames.append(frame)
        bars = pd.concat(frames, ignore_index=True)
        bars["file_id"] = bars.file_id.astype("int32")
        bars["option_type"] = pd.Categorical(bars.option_type, categories=["CE", "PE", "UNKNOWN"])
        bars["bar_range"] = bars.high - bars.low
        coverage.append({"archive_cohort": cohort, "files": len(files), "rows": len(bars),
                         "first_timestamp": bars.timestamp.min(), "last_timestamp": bars.timestamp.max(),
                         "observed_days": bars.timestamp.dt.normalize().nunique(),
                         "unique_strikes": bars.strike.nunique(), "min_strike": bars.strike.min(), "max_strike": bars.strike.max()})
        schema_rows.append({"archive_cohort": cohort, "rows_8": cohort_schema["rows_8"], "rows_9": cohort_schema["rows_9"]})
        keys = ["timestamp", "option_type", "strike"]
        valid = bars.timestamp.notna() & bars.strike.notna() & bars.option_type.ne("UNKNOWN")
        candidates = bars.loc[valid & bars.duplicated(keys, keep=False)]
        if len(candidates):
            groups = candidates.groupby(keys, observed=True, dropna=False)
            sizes = groups.size()
            conflicts = groups[NUMERIC].nunique(dropna=False).gt(1).any(axis=1)
            collision = {"candidate_groups": len(sizes), "rows_in_candidates": len(candidates),
                         "excess_rows_if_naively_deduplicated": int((sizes - 1).sum()),
                         "groups_with_different_values": int(conflicts.sum()),
                         "groups_across_source_files": int(groups.file_id.nunique().gt(1).sum())}
        else:
            collision = {name: 0 for name in ["candidate_groups", "rows_in_candidates", "excess_rows_if_naively_deduplicated", "groups_with_different_values", "groups_across_source_files"]}
        collisions.append({"archive_cohort": cohort, **collision})
        for column in ["close", "volume", "extra_field_9", "bar_range", "strike"]:
            numeric_counts[("all", column)].update(finite_counts(bars[column]))
            for year, values in bars.groupby(bars.timestamp.dt.year):
                numeric_counts[(f"trade_year={int(year)}", column)].update(finite_counts(values[column]))
        priorities = rng.random(len(bars))
        positions = np.argpartition(priorities, min(SAMPLE_SIZE, len(bars)) - 1)[:SAMPLE_SIZE]
        chosen = bars.iloc[positions].copy()
        chosen["_priority"] = priorities[positions]
        chosen["archive_cohort"] = cohort
        sample = pd.concat([sample, chosen], ignore_index=True).nsmallest(SAMPLE_SIZE, "_priority")
        bars.to_parquet(processed / "bars" / f"{cohort}.parquet", index=False)
        print(f"{cohort}: {len(files)} files, {len(bars):,} rows; cumulative {total_loaded:,}", flush=True)
    assert total_loaded == int(selected.nonempty_lines.sum())
    source_days_df = pd.concat(source_days, ignore_index=True)
    source_days_df.to_parquet(processed / "source_days.parquet", index=False)
    file_summary = pd.DataFrame(file_summaries)
    save_csv(file_summary, processed / "file_quality.csv")
    save_csv(pd.DataFrame([row for examples in issue_examples.values() for row in examples]), processed / "quality_examples.csv")
    save_csv(pd.DataFrame(coverage), tables / "coverage_by_archive.csv")
    save_csv(pd.DataFrame(schema_rows), tables / "schema_by_month.csv")
    save_csv(pd.DataFrame(collisions), tables / "candidate_key_collisions.csv")
    quality = pd.DataFrame.from_dict(quality_by_year, orient="index").rename_axis("archive_year").reset_index().fillna(0)
    save_csv(quality, tables / "quality_by_year.csv")
    describe = pd.DataFrame([{"group": group, "metric": metric, **weighted_stats(counts)} for (group, metric), counts in numeric_counts.items()])
    save_csv(describe, tables / "describe.csv")
    daily = source_days_df.groupby("observation_date").agg(rows=("rows", "sum"), source_files=("file_id", "nunique"), volume_sum=("volume_sum", "sum")).reset_index()
    save_csv(daily, tables / "daily_activity.csv")
    monthly = daily.assign(month=daily.observation_date.dt.to_period("M").astype(str)).groupby("month").agg(rows=("rows", "sum"), observed_days=("observation_date", "nunique"), volume_sum=("volume_sum", "sum")).reset_index()
    save_csv(monthly, tables / "monthly_activity.csv")
    intraday = pd.DataFrame([{"minute": minute, "option_type": side, "rows": count, "volume_sum": intraday_volume[(minute, side)]} for (minute, side), count in sorted(intraday_counts.items())])
    save_csv(intraday, tables / "intraday_activity.csv")
    save_csv(pd.DataFrame([{"trade_year": year, "option_type": side, "rows": count} for (year, side), count in sorted(call_put.items())]), tables / "call_put_by_year.csv")
    save_csv(pd.DataFrame([{"archive_year": year, "gap_bucket": bucket, "count": count} for (year, bucket), count in gap_counts.items()]), tables / "minute_gaps_by_year.csv")
    density = []
    for year, part in source_days_df.groupby("archive_year"):
        density.append({"archive_year": int(year), **weighted_stats(Counter(part.unique_regular_minutes.value_counts().to_dict())),
                        "at_most_10_minutes": int(part.unique_regular_minutes.le(10).sum()),
                        "all_375_minutes": int(part.unique_regular_minutes.eq(375).sum())})
    save_csv(pd.DataFrame(density), tables / "file_density_summary.csv")
    histogram = source_days_df.unique_regular_minutes.value_counts().sort_index().rename_axis("unique_regular_minutes").rename("source_days").reset_index()
    save_csv(histogram, tables / "file_density_histogram.csv")
    presence = source_days_df.groupby(["observation_date", "option_type", "strike"]).archive_cohort.nunique()
    sample.drop(columns="_priority").to_parquet(processed / "uniform_sample.parquet", index=False)
    inventory_summary = json.loads((report / "inventory_summary.json").read_text())
    totals = quality.drop(columns="archive_year").sum().astype("int64").to_dict()
    figures_created = make_figures(tables, figures, sample)
    summary = {
        "analysis_version": 1, "input_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "selected_files": len(selected), "selected_rows": total_loaded,
        "first_timestamp": str(file_summary.first_timestamp.min()), "last_timestamp": str(file_summary.last_timestamp.max()),
        "observed_dates": len(daily), "source_file_days": len(source_days_df),
        "quality_counts": {key: int(value) for key, value in totals.items()},
        "all_source_days_regular_minutes": weighted_stats(Counter(source_days_df.unique_regular_minutes.value_counts().to_dict())),
        "candidate_key_collisions_within_archive": {key: int(value) for key, value in pd.DataFrame(collisions).drop(columns="archive_cohort").sum().items()},
        "date_type_strike_groups_in_multiple_archive_cohorts": int(presence.gt(1).sum()),
        "date_type_strike_groups_total": len(presence),
        "sample": {"method": "uniform independent random priorities; retain globally smallest priorities", "rows": len(sample), "seed": SEED},
        "reference_session": {"start": "09:15", "end_exclusive": "15:30", "minute_slots": 375, "status": "diagnostic reference only; vendor timestamp convention and special sessions unverified"},
        "numeric_summary_basis": "exact frequencies across all selected rows, including flagged rows; invalid numeric values are missing; physical format copies excluded",
        "expiry_status": "encoded symbol dates preserved only where present; no calendar-derived expiry assigned",
        "source_fields": RAW_COLUMNS, "extra_field_9_status": "unverified meaning; resembles open interest, not confirmed",
        "figures": figures_created, "inventory_summary": inventory_summary,
        "versions": {name: version(name) for name in ["pandas", "numpy", "pyarrow", "matplotlib"]},
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ["inventory_summary", "quality_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-inventory", action="store_true", help="Rehash and rescan raw files before analysis")
    run(parser.parse_args().refresh_inventory)

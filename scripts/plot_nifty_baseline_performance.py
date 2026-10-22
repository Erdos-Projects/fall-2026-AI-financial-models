"""Plot the fixed compact baseline using saved validation-only diagnostics.

No model fitting, prediction, or test-partition access occurs here. The existing
subgroup and daily tables contain the predeclared seed-42 reference checkpoint;
the learning curve displays all three existing compact runs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
MLP_COLOR = "#16665d"
PERSISTENCE_COLOR = "#bd673a"
SECONDARY_COLOR = "#66839b"
STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#fafbf9",
    "axes.facecolor": "#fafbf9",
    "legend.frameon": False,
    "axes.axisbelow": True,
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def _subgroup_figure(rows: list[dict[str, str]], path: Path) -> Path:
    groups = ["calls", "puts", "close_at_most_1", "close_above_1"]
    names = ["Calls", "Puts", "Close ≤ 1", "Close > 1"]
    indexed = {(row["model"], row["group"]): row for row in rows}
    labels = []
    for group, name in zip(groups, names, strict=True):
        count = int(indexed[("compact", group)]["rows"])
        if count != int(indexed[("persistence", group)]["rows"]):
            raise ValueError(f"Mismatched subgroup populations for {group}")
        labels.append(f"{name}\n{count:,} rows")
    values = {
        (model, metric): np.array(
            [float(indexed[(model, group)][metric]) for group in groups]
        )
        for model in ["compact", "persistence"]
        for metric in ["val_rmse", "val_mae"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), layout="constrained")
    positions = np.arange(len(groups))
    height = 0.33
    for offset, model, label, color in [
        (-height / 2, "compact", "Compact MLP · seed 42", MLP_COLOR),
        (height / 2, "persistence", "Persistence", PERSISTENCE_COLOR),
    ]:
        bars = axes[0].barh(
            positions + offset,
            values[(model, "val_rmse")],
            height=height,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.4f", padding=5, fontsize=10)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.285)
    axes[0].set_xlabel("Pooled return RMSE (log-return units; lower is better)")
    axes[0].set_title("Absolute error by subgroup", pad=14)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
    axes[0].grid(axis="x", color="#dce3df", linewidth=0.6)
    for offset, metric, label, color in [
        (-height / 2, "val_rmse", "Return RMSE", MLP_COLOR),
        (height / 2, "val_mae", "Return MAE", SECONDARY_COLOR),
    ]:
        change = 100 * (values[("compact", metric)] / values[("persistence", metric)] - 1)
        bars = axes[1].barh(
            positions + offset, change, height=height, label=label, color=color
        )
        axes[1].bar_label(bars, labels=[f"{v:+.2f}%" for v in change], padding=5, fontsize=10)
    axes[1].set_yticks(positions, names)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#293c36", linewidth=1)
    axes[1].set_xlim(-10.7, 2.7)
    axes[1].set_xlabel("100 × (MLP error / persistence error − 1)\nNegative favors MLP")
    axes[1].set_title("Relative change versus persistence", pad=14)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
    axes[1].grid(axis="x", color="#dce3df", linewidth=0.6)
    fig.suptitle(
        "Compact MLP · validation subgroup diagnostics\n"
        "Call/put and price groups are separate partitions; current close uses source price units",
        fontsize=14,
        fontweight="bold",
    )
    return _save(fig, path)


def _daily_figure(rows: list[dict[str, str]], path: Path) -> Path:
    indexed = {
        model: {row["day"]: row for row in rows if row["model"] == model}
        for model in ["compact", "persistence"]
    }
    days = sorted(indexed["compact"])
    if set(days) != set(indexed["persistence"]) or len(days) != 150:
        raise ValueError("Expected matching populations across 150 validation dates")
    if any(
        indexed["compact"][day]["rows"] != indexed["persistence"][day]["rows"]
        for day in days
    ):
        raise ValueError("Daily model populations do not match")
    dates = [datetime.strptime(day, "%Y-%m-%d") for day in days]
    mlp = np.array([float(indexed["compact"][day]["mae"]) for day in days])
    persistence = np.array([float(indexed["persistence"][day]["mae"]) for day in days])
    difference = mlp - persistence
    better = int(np.count_nonzero(difference < 0))
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1]}, layout="constrained"
    )
    axes[0].plot(
        dates, mlp, color=MLP_COLOR, linewidth=1.8,
        label="Compact MLP · seed 42", zorder=2
    )
    axes[0].plot(
        dates, persistence, color=PERSISTENCE_COLOR, linewidth=1.25,
        linestyle="--", label="Persistence", zorder=3
    )
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Within-date return MAE\n(log-return units)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    axes[0].grid(axis="y", color="#dce3df", linewidth=0.6)
    axes[1].axhline(0, color="#293c36", linewidth=1)
    axes[1].vlines(
        dates, 0, difference,
        colors=np.where(difference < 0, MLP_COLOR, PERSISTENCE_COLOR), linewidth=1.7
    )
    axes[1].scatter(
        dates, difference,
        c=np.where(difference < 0, MLP_COLOR, PERSISTENCE_COLOR), s=11, zorder=3
    )
    axes[1].set_ylabel("MAE difference\nMLP − persistence")
    axes[1].set_xlabel("Validation date · negative differences favor MLP")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.4f}"))
    axes[1].grid(axis="y", color="#dce3df", linewidth=0.6)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[1].margins(x=0.02)
    fig.suptitle(
        f"Compact MLP · performance across {len(days)} validation dates\n"
        f"Lower daily MAE than persistence on {better}/{len(days)} dates; "
        f"mean difference {difference.mean():+.7f}",
        fontsize=14,
        fontweight="bold",
    )
    return _save(fig, path)


def _learning_figure(
    history: list[dict[str, str]],
    runs: list[dict[str, str]],
    daily: list[dict[str, str]],
    path: Path,
) -> Path:
    persistence = np.mean([float(row["mae"]) for row in daily if row["model"] == "persistence"])
    compact_runs = {int(row["seed"]): row for row in runs if row["model"] == "compact"}
    if set(compact_runs) != {17, 42, 2026}:
        raise ValueError("Expected the three established compact seeds")
    fig, ax = plt.subplots(figsize=(13, 6.7), layout="constrained")
    colors = {17: SECONDARY_COLOR, 42: MLP_COLOR, 2026: "#8b729d"}
    for seed in [17, 2026, 42]:
        selected = sorted(
            [row for row in history if row["model"] == "compact" and int(row["seed"]) == seed],
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in selected]
        values = [float(row["val_day_mae"]) for row in selected]
        ax.plot(
            epochs, values, color=colors[seed], marker="o", markersize=4,
            linewidth=2.7 if seed == 42 else 1.5,
            label=f"Seed {seed}" + (" · reference checkpoint" if seed == 42 else ""),
            zorder=4 if seed == 42 else 2,
        )
        best_epoch = int(compact_runs[seed]["best_epoch"])
        best_value = values[epochs.index(best_epoch)]
        if not np.isclose(best_value, float(compact_runs[seed]["val_day_mae"]), rtol=1e-10):
            raise ValueError(f"Checkpoint metric differs from history for seed {seed}")
        ax.scatter(
            [best_epoch], [best_value], color=colors[seed], marker="*", s=230,
            edgecolors="#fafbf9", linewidths=0.7, zorder=5,
        )
        offset = (10, -17) if seed == 42 else ((10, -15) if seed == 17 else (9, -17))
        ax.annotate(
            f"Selected: epoch {best_epoch}", (best_epoch, best_value), xytext=offset,
            textcoords="offset points", color=colors[seed], fontsize=10,
        )
    ax.axhline(
        persistence, color=PERSISTENCE_COLOR, linewidth=1.6,
        linestyle="--", label="Persistence (fixed)", zorder=1
    )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Equal-date validation MAE (log-return units; lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.6f}"))
    ax.set_ylim(persistence - 0.000009, 0.014984)
    ax.margins(x=0.07)
    ax.grid(axis="y", color="#dce3df", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="*", linestyle="none", color="#293c36", markersize=12))
    labels.append("Saved selected checkpoint")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=10)
    fig.suptitle(
        "Compact MLP · learning curves from three existing fits\n"
        "Stars mark the saved best eligible checkpoint; validation controls early stopping",
        fontsize=14,
        fontweight="bold",
    )
    return _save(fig, path)


def make_figures(root: Path, output_dir: Path) -> list[Path]:
    """Write three PNGs to ``output_dir`` using the immutable study CSVs."""
    root = Path(root)
    output_dir = Path(output_dir)
    source = root / "reports/nifty_mlp_study"
    subgroups = _read_rows(source / "subgroups.csv")
    daily = _read_rows(source / "daily_validation.csv")
    history = _read_rows(source / "history.csv")
    runs = _read_rows(source / "runs.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        return [
            _subgroup_figure(subgroups, output_dir / "subgroup_error.png"),
            _daily_figure(daily, output_dir / "daily_error.png"),
            _learning_figure(history, runs, daily, output_dir / "learning_curve.png"),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or arguments.root / "reports/nifty_baseline_performance/figures"
    for path in make_figures(arguments.root, output_dir):
        print(path)


if __name__ == "__main__":
    main()

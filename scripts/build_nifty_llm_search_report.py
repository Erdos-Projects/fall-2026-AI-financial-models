"""Build the LLM-guided search report from saved protocol, trial and result files.

Nothing is trained or predicted here, and no test array is read. Searches that
are still running are reported as in progress and kept out of comparisons.
Rerun after new trials or confirmation results arrive.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

REPORT = ROOT / "reports/nifty_llm_search"
ARMS = ["random", "bayes", "llm_batch", "llm_feedback", "gpt_hypotheses"]
LABELS = {"random": "Random search", "bayes": "Bayesian (GP)", "llm_batch": "LLM, no feedback",
          "llm_feedback": "LLM + feedback", "gpt_hypotheses": "GPT's 3 hypotheses", "baseline": "Baseline config"}
# First five slots of the validated reference categorical order (adjacent pairs pass in light mode).
COLORS = {"random": "#2a78d6", "bayes": "#eb6834", "llm_batch": "#1baf7a", "llm_feedback": "#eda100",
          "gpt_hypotheses": "#e87ba4"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5, "axes.spines.top": False,
         "axes.spines.right": False, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED,
         "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
         "legend.frameon": False}


def pct(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def seconds_between(start: str | None, stop: str | None) -> float | None:
    if not start or not stop:
        return None
    return (datetime.fromisoformat(stop) - datetime.fromisoformat(start)).total_seconds()


def summarize_search(arm: str, repeat: int, trials: list[dict], budget: int, base: float) -> dict:
    best_so_far, current = [], 0.0
    for trial in trials:
        if trial["status"] == "evaluated":
            current = min(current, pct(trial["objective"], base))
        best_so_far.append(current)
    evaluated = [trial for trial in trials if trial["status"] == "evaluated"]
    best = min(evaluated, key=lambda trial: (trial["objective"], trial["trial"])) if evaluated else None
    messages = [m for trial in trials for m in (trial.get("llm") or {}).get("messages", [])]
    latencies = [s for s in (seconds_between(m["prompt_created_at_utc"], m["reply_submitted_at_utc"])
                             for m in messages) if s is not None]
    predicted = [(trial["proposal"]["predicted_change_pct"], pct(trial["objective"], base)) for trial in evaluated
                 if isinstance((trial.get("proposal") or {}).get("predicted_change_pct"), (int, float))]
    statuses = {status: sum(trial["status"] == status for trial in trials)
                for status in ["evaluated", "duplicate", "invalid", "training_failed"]}
    return {
        "arm": arm, "repeat": repeat, "label": LABELS[arm] + (f" r{repeat}" if repeat > 1 else ""),
        "budget": budget, "done": len(trials), "complete": len(trials) == budget, "trials": trials,
        **statuses, "best_so_far": best_so_far,
        "mean_best_so_far": float(np.mean(best_so_far)) if len(trials) == budget else None,
        "best": best, "best_change": pct(best["objective"], base) if best else None,
        "first_best_trial": best["trial"] if best else None,
        "epochs": int(sum(trial["epochs"] for trial in trials)),
        "train_seconds": float(sum(trial["train_seconds"] for trial in trials)),
        "cumulative_epochs": np.cumsum([trial["epochs"] for trial in trials]).tolist(),
        "prompts": len(messages), "repairs": int(sum((trial.get("llm") or {}).get("repairs", 0) for trial in trials)),
        "prompt_chars": int(sum(m["prompt_chars"] for m in messages)),
        "reply_chars": int(sum(m["reply_chars"] for m in messages)),
        "relay_seconds_median": float(np.median(latencies)) if latencies else None,
        "models": sorted({m["model"] for m in messages if m.get("model")}),
        "predicted": predicted,
        "spearman": (float(spearmanr(*zip(*predicted, strict=True)).statistic) if len(predicted) >= 4 else None),
        "direction_agreement": (sum((p < 0) == (a < 0) for p, a in predicted) / len(predicted)
                                if predicted else None),
    }


def describe(config: dict, baseline: dict) -> str:
    changes = []
    for name, value in config.items():
        if value != baseline[name] and not (name == "smooth_l1_beta" and config["loss"] != "smooth_l1"):
            changes.append(f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}")
    return ", ".join(changes) or "baseline"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value, spec: str = "+.3f", suffix: str = "%") -> str:
    return "-" if value is None else f"{value:{spec}}{suffix}"


def end_labels(ax, items: list[tuple[float, float, str, str]], gap: float) -> None:
    """Direct labels at line ends, nudged apart vertically; text stays in ink, a dot carries the color.

    Labels share the rightmost end position so text never crosses a line that ends further right.
    """
    placed = []
    x_text = max(item[0] for item in items)
    for x, y, text, color in sorted(items, key=lambda item: item[1]):
        y_label = max(y, placed[-1] + gap) if placed else y
        placed.append(y_label)
        ax.plot([x], [y], marker="o", markersize=5.5, color=color, markeredgecolor=SURFACE, zorder=4)
        ax.annotate(text, (x_text, y_label), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2, annotation_clip=False)


def progress_figure(searches: list[dict], persistence: float, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, layout="constrained")
        values = [v for s in searches for v in s["best_so_far"]] + [persistence, 0.0]
        span = max(values) - min(values) or 0.1
        for ax, x_key, title, xlabel in [
                (axes[0], "trial", "Best configuration found vs proposals used", "proposals evaluated"),
                (axes[1], "epochs", "Best configuration found vs training compute", ("cumulative training epochs "
                                                                                     "(3 folds)"))]:
            labels = []
            for search in searches:
                x = ([0, *range(1, search["done"] + 1)] if x_key == "trial"
                     else [0, *search["cumulative_epochs"]])
                y = [0.0, *search["best_so_far"]]
                ax.plot(x, y, drawstyle="steps-post", color=COLORS[search["arm"]], linewidth=1.8,
                        label=search["label"], zorder=3)
                labels.append((x[-1], y[-1], search["label"], COLORS[search["arm"]]))
            ax.axhline(0, color=AXIS, linewidth=1, zorder=2)
            ax.axhline(persistence, color=MUTED, linewidth=1, zorder=2)
            ax.annotate("persistence", (0, persistence), xytext=(2, -11), textcoords="offset points",
                        fontsize=8.5, color=MUTED)
            ax.annotate("baseline config", (0, 0), xytext=(2, 4), textcoords="offset points",
                        fontsize=8.5, color=MUTED)
            end_labels(ax, labels, gap=span * 0.045)
            ax.set_title(title, loc="left", color=INK)
            ax.set_xlabel(xlabel)
            ax.margins(x=0.02)
        # Fix the proposal axis to the full budget so in-progress searches keep the same scale.
        axes[0].set_xlim(-0.2, max(s["budget"] for s in searches) + 0.2)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylabel("search objective change vs baseline config (%)\nlower is better")
        axes[0].legend(handles=[Line2D([], [], color=COLORS[s["arm"]], linewidth=1.8, label=s["label"])
                                for s in searches], loc="upper right", fontsize=9)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def trials_figure(searches: list[dict], base: float, persistence: float, path: Path) -> None:
    columns = min(3, len(searches))
    rows = int(np.ceil(len(searches) / columns))
    changes = [pct(t["objective"], base) for s in searches for t in s["trials"] if t["objective"] is not None]
    low = min([*changes, persistence, 0.0])
    high = min(max([*changes, 0.0]), max(0.5, 4 * abs(low)))
    pad = 0.08 * (high - low or 0.1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.5 * rows), sharey=True,
                                 squeeze=False, layout="constrained")
        for ax, search in zip(axes.flat, searches, strict=False):
            ax.axhline(0, color=AXIS, linewidth=1)
            ax.axhline(persistence, color=MUTED, linewidth=1)
            ax.plot(range(search["done"] + 1), [0.0, *search["best_so_far"]], drawstyle="steps-post",
                    color=INK_2, linewidth=1.2)
            for trial in search["trials"]:
                if trial["status"] == "evaluated":
                    change = pct(trial["objective"], base)
                    if change > high + pad:
                        ax.plot(trial["trial"], high + pad * 0.6, marker="^", color=COLORS["random"],
                                markersize=6)
                        ax.annotate(f"{change:+.2f}%", (trial["trial"], high + pad * 0.6), xytext=(0, -12),
                                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
                    else:
                        ax.plot(trial["trial"], change, "o", color=COLORS["random"], markersize=6,
                                markeredgecolor=SURFACE)
                elif trial["status"] == "duplicate":
                    ax.plot(trial["trial"], pct(trial["objective"], base) if trial["objective"] else 0, "o",
                            markerfacecolor="none", markeredgecolor=MUTED, markersize=6)
                else:
                    ax.plot(trial["trial"], low - pad * 0.6, "x", color=MUTED, markersize=6)
            ax.set_title(search["label"] + ("" if search["complete"] else " (in progress)"), loc="left",
                         color=INK)
            ax.set_xlim(0.3, search["budget"] + 0.7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(low - pad, high + pad)
            ax.set_xlabel("proposal")
        for ax in axes.flat[len(searches):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("objective change vs\nbaseline config (%)")
        fig.legend(handles=[
            Line2D([], [], marker="o", linestyle="", color=COLORS["random"], label="evaluated proposal"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED,
                   label="duplicate"),
            Line2D([], [], marker="x", linestyle="", color=MUTED, label="invalid or failed (no value)"),
            Line2D([], [], color=INK_2, linewidth=1.2, label="best so far"),
            Line2D([], [], color=MUTED, linewidth=1, label="persistence")],
            loc="outside lower center", ncols=5, fontsize=9)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def confirmation_figure(confirmation: dict, path: Path, title: str, period: str) -> None:
    rows = confirmation["selections"][::-1]
    base = confirmation["baseline_config_day_mae_mean"]
    persistence = pct(confirmation["persistence"]["day_mae"], base)
    frozen = pct(confirmation["frozen_checkpoint"]["day_mae"], base)
    seeds = {}
    for run in confirmation["runs"]:
        seeds.setdefault(run["config_hash"], []).append(pct(run["day_mae"], base))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8), layout="constrained")
        for y, row in enumerate(rows):
            # The paired interval is already expressed as % of the baseline mean, around this row's mean.
            ax.plot(row["paired_daily_difference_95pct_of_baseline"], [y, y], color=COLORS["random"],
                    linewidth=1.8, solid_capstyle="round")
            centre = row["day_mae_vs_baseline_config_pct"]
            ax.plot(seeds[row["config_hash"]], [y] * len(seeds[row["config_hash"]]), "o", markersize=4,
                    markerfacecolor="none", markeredgecolor=MUTED)
            ax.plot(centre, y, "o", color=COLORS["random"], markersize=7, markeredgecolor=SURFACE)
        ax.axvline(0, color=AXIS, linewidth=1)
        # Reference labels sit on opposite edges so nearby lines cannot collide.
        for value, name, edge in [(persistence, "persistence", 1.0), (frozen, "frozen checkpoint", 0.0)]:
            ax.axvline(value, color=MUTED, linewidth=1)
            ax.annotate(name, (value, edge), xycoords=("data", "axes fraction"), xytext=(3, -3 if edge else 3),
                        textcoords="offset points", fontsize=8.5, color=MUTED, va="top" if edge else "bottom")
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_yticks(range(len(rows)), [LABELS[row["arm"]] + (f" (trial {row['trial']})" if row["trial"] else "")
                                         for row in rows])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK_2)
        ax.set_xlabel(f"{period} equal-date MAE change vs baseline config (%), mean of 3 seeds; lower is better")
        ax.set_title(title, loc="left", color=INK)
        fig.legend(handles=[
            Line2D([], [], marker="o", linestyle="-", color=COLORS["random"], label="seed mean, paired 95% interval"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED,
                   label="individual seeds")], loc="outside lower center", ncols=2, fontsize=9)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def selection_rows(results: dict) -> list[list]:
    rows = []
    for row in results["selections"]:
        low, high = row["paired_daily_difference_95pct_of_baseline"]
        rows.append([LABELS[row["arm"]] + (f" r{row['repeat']}" if row["repeat"] > 1 else ""),
                     row["trial"] or "-", f"{row['day_mae_mean']:.7f}",
                     fmt(row["day_mae_vs_baseline_config_pct"]) + (
                         "" if row["arm"] == "baseline" else f" [{low:+.3f}, {high:+.3f}]"),
                     fmt(row["day_mae_vs_persistence_pct"]), fmt(row["rmse_vs_persistence_pct"], "+.2f"),
                     fmt(row["price_mae_vs_persistence_pct"], "+.2f"),
                     f"{row['dates_better_than_persistence']}/{row['dates']}"])
    return rows


def main(report: Path = REPORT) -> None:
    protocol = json.loads((report / "protocol.json").read_text())
    baseline = json.loads((report / "baseline.json").read_text())
    base_result = json.loads((report / baseline["result_path"]).read_text())
    base, base_config = baseline["objective"], protocol["baseline_config"]
    persistence = pct(base_result["persistence_objective"], base)
    searches = []
    for arm in ARMS:
        for path in sorted((report / "trials").glob(f"{arm}_r*.jsonl")):
            budget = protocol["gpt_hypotheses_trials"] if arm == "gpt_hypotheses" else protocol["budget_per_search"]
            searches.append(summarize_search(arm, int(path.stem.rsplit("_r", 1)[1]), read_jsonl(path), budget, base))
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    budgeted = [s for s in searches if s["arm"] != "gpt_hypotheses" and s["done"]]
    if budgeted:
        progress_figure(budgeted, persistence, figures / "search_progress.png")
    started = [s for s in searches if s["done"]]
    if started:
        trials_figure(started, base, persistence, figures / "search_trials.png")
    confirmation_path = report / "confirmation/results.json"
    confirmation = json.loads(confirmation_path.read_text()) if confirmation_path.exists() else None
    if confirmation:
        confirmation_figure(confirmation, figures / "confirmation.png",
                            "Confirmation on the 150 original validation dates", "validation")
    test_path = report / "final_test/results.json"
    final_test = json.loads(test_path.read_text()) if test_path.exists() else None
    if final_test:
        confirmation_figure(final_test, figures / "final_test.png",
                            "Final test on the 150 reserved dates, June-December 2020", "test")

    complete = [s for s in searches if s["complete"]]
    pending = [s for s in searches if not s["complete"]]
    lines = ["# LLM-guided improvement search for the compact Nifty MLP", "",
             (f"Built {datetime.now().date().isoformat()} from saved files; this builder trains nothing. "
             f"Study protocol written {protocol['created_at_utc'][:10]} before any search training. "
             + ("The frozen selections were evaluated once on the reserved June–December 2020 test dates "
                f"({final_test['finished_at_utc'][:10]})." if final_test else
                "**The reserved June–December 2020 test dates have not been evaluated.**")), ""]
    status = ", ".join(f"{s['label']} {s['done']}/{s['budget']}" for s in searches) or "no search has started"
    lines += [f"**Status:** {status}. Confirmation: {'complete' if confirmation else 'not yet run'}.", ""]
    lines += ["## Question and design", "",
              f"> {protocol['research_question']}", "",
              ("Every search proposes complete settings for the fixed `15→32→16→1` MLP from the same eight-parameter "
              "space (loss, Smooth L1 beta, learning rate, weight decay, dropout, date weighting, low-price "
              "weighting, output calibration). Each proposal is trained on three expanding forward folds inside the "
              "original training dates and scored by the mean equal-date MAE of the folds' score windows. Training "
              "uses only fit dates; tune dates drive early stopping and calibration. Every search has a budget of "
              f"**{protocol['budget_per_search']} proposals**; invalid proposals and duplicates use budget. "
              "The LLM receives a written brief of the task, rules, baseline diagnostics and earlier findings; "
              "it never sees rows, labels or test-period information, and Python computes every number."), "",
              table(["Search", "How proposals are made"],
                    [[LABELS[arm], protocol["arms"][arm]] for arm in ARMS]), "",
              (f"Before any search, the new runner reproduced the frozen checkpoint's weights and validation "
              f"predictions bit for bit "
              f"({'confirmed' if (protocol.get('replication') or {}).get('predictions_bitwise_identical') else 'n/a'}"
              f"). The baseline configuration scored **{base:.7f}** on the search folds; persistence scored "
              f"**{base_result['persistence_objective']:.7f}** ({persistence:+.3f}% vs the baseline config)."), ""]
    if searches:
        lines += ["## Search results", "",
                  table(["Search", "Done", "Best trial", "Best objective change", "Mean best-so-far",
                         "Evaluated / duplicate / invalid / failed", "Training epochs", "Training time"],
                        [[s["label"] + ("" if s["complete"] else " *(in progress)*"), f"{s['done']}/{s['budget']}",
                          s["first_best_trial"] or "-", fmt(s["best_change"]), fmt(s["mean_best_so_far"]),
                          f"{s['evaluated']} / {s['duplicate']} / {s['invalid']} / {s['training_failed']}",
                          f"{s['epochs']:,}", f"{s['train_seconds'] / 60:.1f} min"] for s in searches]), "",
                  ("Objective changes are relative to the baseline configuration on the same folds (negative is "
                  "better). *Mean best-so-far* averages, over proposals 1 to the budget, the best change found so "
                  "far (the baseline counts as already known, so it never exceeds 0%); it rewards finding "
                  "improvements early. Training time is charged per proposal even when an identical configuration "
                  "was cached by another search; epochs are the hardware-independent cost unit because searches "
                  "may have run concurrently."), ""]
        if budgeted:
            lines += ["![Best configuration found by each search](figures/search_progress.png)", ""]
        if started:
            lines += ["![Every proposal by search](figures/search_trials.png)", ""]
        lines += ["### Selected configurations", "",
                  table(["Search", "Trial", "Changes from baseline", "Objective change", "Hypothesis (LLM/GPT)"],
                        [[s["label"], s["best"]["trial"], describe(s["best"]["config"], base_config),
                          fmt(s["best_change"]),
                          ((s["best"].get("proposal") or {}).get("hypothesis") or "-").replace("|", "/")]
                         for s in searches if s["best"]]), ""]
    llm = [s for s in searches if s["arm"] in ("llm_batch", "llm_feedback") and s["done"]]
    if llm:
        lines += ["## LLM behaviour and cost", "",
                  table(["Search", "Model(s)", "Prompts (repairs)", "Prompt / reply characters",
                         "Estimated tokens", "Median relay time", "Predicted vs measured change"],
                        [[s["label"], ", ".join(s["models"]) or "unrecorded", f"{s['prompts']} ({s['repairs']})",
                          f"{s['prompt_chars']:,} / {s['reply_chars']:,}",
                          f"{(s['prompt_chars'] + s['reply_chars']) / 4:,.0f}",
                          "-" if s["relay_seconds_median"] is None else f"{s['relay_seconds_median'] / 60:.1f} min",
                          ("-" if s["spearman"] is None else f"Spearman {s['spearman']:+.2f}; ")
                          + ("-" if s["direction_agreement"] is None
                             else f"direction right {s['direction_agreement']:.0%}")] for s in llm]), "",
                  ("Tokens are estimated as characters / 4. Relay time runs from prompt creation to reply submission "
                  "and includes the person relaying the text, so it is not model latency. *Direction right* is the "
                  "share of evaluated proposals whose predicted and measured changes had the same sign."), ""]
    headers = ["Selection", "Trial", "Equal-date MAE (3-seed mean)", "vs baseline config [paired 95%]",
               "vs persistence", "RMSE vs persistence", "Price MAE vs persistence", "Dates better than persistence"]
    if confirmation:
        rows = selection_rows(confirmation)
        window = protocol["confirmation"]["window"]
        lines += ["## Confirmation on the original validation dates", "",
                  ("Each search's selected configuration and the baseline configuration were refit with seeds "
                   f"{', '.join(map(str, protocol['confirmation']['seeds']))}: fit on {window['fit']}; "
                   f"tuned on {window['tune']}; scored on {window['score']}."), "",
                  table(headers, rows), "",
                  (f"Persistence equal-date MAE: {confirmation['persistence']['day_mae']:.7f}. The unchanged frozen "
                  f"checkpoint scores {confirmation['frozen_checkpoint']['day_mae']:.7f} but was early-stopped on "
                  "these dates, so it is a reference rather than a fair comparator. Intervals come from a circular "
                  "block bootstrap over dates (block 5) of seed-averaged daily differences; they describe "
                  "variation across these dates, not uncertainty over future markets."), "",
                  "![Confirmation](figures/confirmation.png)", ""]
    if final_test:
        lines += ["## Final test on the reserved June–December 2020 dates", "",
                  ("After confirmation, the six frozen selections were refit once more with the same three seeds "
                   "under the predeclared test plan: fit on training dates 0–809 (through the first 110 validation "
                   "dates), tune on dates 810–849, and score once on the 150 reserved test dates. No search, "
                   "selection or prompt used these dates."), "",
                  table(headers, selection_rows(final_test)), "",
                  (f"Persistence equal-date MAE: {final_test['persistence']['day_mae']:.7f}. The unchanged frozen "
                   f"checkpoint scores {final_test['frozen_checkpoint']['day_mae']:.7f}; it never saw these dates, "
                   "but it was trained only on dates before October 2019, whereas the refit selections also learned "
                   "from most validation dates. The baseline configuration refit under the identical plan is the "
                   "like-for-like comparator."), "",
                  "![Final test](figures/final_test.png)", ""]
    lines += ["## Caveats", "", *[f"- {caveat}" for caveat in protocol["caveats"]]]
    if pending:
        lines.append("- Searches still in progress: " + ", ".join(s["label"] for s in pending)
                     + ". Comparisons above should wait for them.")
    lines += ["", "## Reproduce", "", "```bash",
              ".conda/bin/python scripts/run_nifty_llm_search.py --init",
              ("for arm in gpt_hypotheses random bayes; do .conda/bin/python scripts/run_nifty_llm_search.py --arm $arm; "
              "done"),
              ".conda/bin/python scripts/run_nifty_llm_search.py --arm llm_feedback --wait   # relay prompts meanwhile",
              ".conda/bin/python scripts/nifty_llm_handoff.py status",
              ".conda/bin/python scripts/confirm_nifty_llm_search.py",
              ".conda/bin/python scripts/build_nifty_llm_search_report.py", "```", "",
              ("Inputs: [protocol](protocol.json), [brief shown to the LLM](brief.md), per-search trial logs in "
              "`trials/`, per-configuration fold results in `results/`, and every prompt and reply in `handoff/`."),
              ""]
    if (report / "findings.md").exists():
        lines.insert(2, "Interpretation, limitations and next steps: [findings](findings.md).")
        lines.insert(3, "")
    if (report / "llm_relay.md").exists():
        lines.insert(len(lines) - 1, "How the prompts were answered, and deviations from the relay protocol: "
                                     "[llm_relay.md](llm_relay.md).")
        lines.insert(len(lines) - 1, "")
    (report / "report.md").write_text("\n".join(lines))
    print(f"Wrote {(report / 'report.md').relative_to(ROOT)} ({len(complete)} complete, {len(pending)} in progress)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="build the report for the smoke-test study")
    main(ROOT / "data/processed/nifty_llm_search_smoke" if parser.parse_args().smoke else REPORT)

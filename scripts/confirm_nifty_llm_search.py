"""Confirm the configuration each completed search selected, on later dates.

For every requested arm and repeat, the evaluated trial with the lowest search
objective is selected (ties: earliest trial). The selections are saved before
any confirmation training. Each selection and the baseline configuration are
refit with seeds 17, 42 and 2026 on training dates 0-659, tuned on dates
660-699 and scored on the 150 original validation dates. Daily errors are
averaged over seeds and compared with the baseline configuration by a circular
block bootstrap over dates. Confirmation results never enter an LLM prompt.

--final-test evaluates the same frozen selections once on the reserved
June-December 2020 test dates (refit on dates 0-809, tuned on 810-849). It
refuses to run twice and is not part of the default workflow.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from nifty_models import ReturnMLP, predict_batches
from nifty_proposers import now
from nifty_search_runner import (
    PLANS,
    ROOT,
    DataStore,
    evaluate,
    score_part,
    write_json_atomic,
)
from nifty_search_space import BASELINE, SearchConfig
from run_nifty_llm_search import ARMS, CONFIRMATION_SEEDS, Context, Study, read_jsonl
from threadpoolctl import threadpool_limits
from train_nifty_models import digest

METRICS = ["day_mae", "mae", "rmse", "price_mae", "price_rmse"]


def block_interval(difference: np.ndarray, block: int = 5, resamples: int = 2000, seed: int = 8801) -> list[float]:
    """Percentile interval of the mean from circular moving-block resamples (as in the baseline study)."""
    rng = np.random.default_rng(seed)
    count = len(difference)
    starts = rng.integers(0, count, size=(resamples, int(np.ceil(count / block))))
    indices = ((starts[..., None] + np.arange(block)) % count).reshape(resamples, -1)[:, :count]
    return np.quantile(difference[indices].mean(axis=1), [.025, .975]).tolist()


def select(study: Study, arms: list[str], repeats: list[int]) -> list[dict]:
    selections = [{"arm": "baseline", "repeat": 0, "trial": 0, "config": BASELINE.model_dump(),
                   "config_hash": BASELINE.config_hash(), "search_objective": None, "trials_log_sha256": None}]
    baseline = json.loads((study.report / "baseline.json").read_text())
    selections[0]["search_objective"] = baseline["objective"]
    for arm in arms:
        for repeat in [1] if arm == "gpt_hypotheses" else repeats:
            path = study.trials / f"{arm}_r{repeat}.jsonl"
            trials = read_jsonl(path)
            budget = 3 if arm == "gpt_hypotheses" else study.budget
            if len(trials) != budget:
                raise ValueError(f"{arm} r{repeat} is incomplete: {len(trials)}/{budget} trials")
            scored = [trial for trial in trials if trial["status"] == "evaluated"]
            if not scored:
                raise ValueError(f"{arm} r{repeat} has no evaluated trial")
            best = min(scored, key=lambda trial: (trial["objective"], trial["trial"]))
            selections.append({"arm": arm, "repeat": repeat, "trial": best["trial"], "config": best["config"],
                               "config_hash": best["config_hash"], "search_objective": best["objective"],
                               "trials_log_sha256": digest(path)})
    return selections


def frozen_reference(store: DataStore, plan: str) -> dict:
    """The frozen seed-42 checkpoint, unchanged, on the plan's score dates."""
    payload = torch.load(ROOT / "data/processed/nifty_models/frozen_mlp.pt", map_location="cpu", weights_only=True)
    model = ReturnMLP(payload["input_dim"], payload["hidden"], payload["dropout"])
    model.load_state_dict(payload["state_dict"])
    # A bare part keeps the original training-split scaler that the frozen model expects.
    part = store.part(*PLANS[plan][0].score)
    predictions = predict_batches(model, part.X, payload["target_scale"])
    if plan == "validation" and store.row_limit is None:
        saved = np.load(ROOT / "data/processed/nifty_models/compact_seed42_valid_predictions.npy")
        if not np.array_equal(predictions, saved):
            raise ValueError("Frozen checkpoint predictions differ from the saved validation predictions")
    return score_part(predictions, part)


def summarize(selections: list[dict], results: dict, persistence: dict, frozen: dict, plan: str) -> dict:
    rows, summary = [], []
    daily = {}
    for key, runs in results.items():
        for seed, result in runs.items():
            window = result["windows"][0]
            rows.append({"config_hash": key, "seed": seed, "status": result["status"],
                         **{metric: window["metrics"][metric] for metric in METRICS},
                         "best_epoch": window["best_epoch"], "epochs_run": window["epochs_run"],
                         "train_seconds": window["train_seconds"], "calibration": json.dumps(window["calibration"])})
        daily[key] = np.mean([np.asarray(r["windows"][0]["metrics"]["daily_mae"]) for r in runs.values()], axis=0)
    frame = pd.DataFrame(rows)
    base = BASELINE.config_hash()
    base_mean = frame.loc[frame.config_hash.eq(base), "day_mae"].mean()
    for selection in selections:
        key = selection["config_hash"]
        group = frame.loc[frame.config_hash.eq(key)]
        difference = daily[key] - daily[base]
        interval = block_interval(difference) if key != base else [0.0, 0.0]
        groups = {}
        for name in ["close_at_most_1", "close_above_1"]:
            group_runs = [run["windows"][0]["metrics"]["groups"][name] for run in results[key].values()]
            groups[f"{name}_mae_vs_persistence_pct"] = 100 * (np.mean([g["mae"] for g in group_runs])
                                                              / persistence["groups"][name]["mae"] - 1)
            groups[f"{name}_rmse_vs_persistence_pct"] = 100 * (np.mean([g["rmse"] for g in group_runs])
                                                               / persistence["groups"][name]["rmse"] - 1)
        summary.append({
            **{k: selection[k] for k in ["arm", "repeat", "trial", "config_hash", "search_objective"]},
            "config": selection["config"], "seeds": len(group),
            **{f"{metric}_mean": float(group[metric].mean()) for metric in METRICS},
            "day_mae_seed_sd": float(group.day_mae.std(ddof=1)),
            **{f"{metric}_vs_persistence_pct": float(100 * (group[metric].mean() / persistence[metric] - 1))
               for metric in METRICS},
            "day_mae_vs_baseline_config_pct": float(100 * (group.day_mae.mean() / base_mean - 1)),
            "paired_daily_difference_mean": float(difference.mean()),
            "paired_daily_difference_95pct": interval,
            "paired_daily_difference_95pct_of_baseline": [100 * value / base_mean for value in interval],
            "dates_better_than_baseline_config": int((difference < 0).sum()),
            "dates_better_than_persistence": int((daily[key] < np.asarray(persistence["daily_mae"])).sum()),
            "dates": len(difference), **groups})
    return {"plan": plan, "persistence": {metric: persistence[metric] for metric in METRICS},
            "frozen_checkpoint": {metric: frozen[metric] for metric in METRICS},
            "baseline_config_day_mae_mean": float(base_mean), "selections": summary,
            "runs": frame.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--repeats", nargs="+", type=int, default=[1])
    parser.add_argument("--final-test", action="store_true", help="evaluate frozen selections once on test dates")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="confirm the reduced smoke-test study")
    args = parser.parse_args()
    plan = "test" if args.final_test else "validation"
    with threadpool_limits(limits=args.threads):
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        study = Study(args.smoke)
        context = Context.open(study, args.threads)
        output = study.report / ("final_test" if args.final_test else "confirmation")
        if args.final_test:
            if (output / "results.json").exists():
                raise ValueError("The final test has already been evaluated; it runs once")
            confirmed = json.loads((study.report / "confirmation/results.json").read_text())
            selections = [{key: s[key] for key in ["arm", "repeat", "trial", "config", "config_hash",
                                                    "search_objective"]} for s in confirmed["selections"]]
            context.store = DataStore(allow_test=True)
        else:
            selections = select(study, args.arms, args.repeats)
        write_json_atomic(output / "selection.json", {
            "plan": plan, "seeds": CONFIRMATION_SEEDS, "selected_at_utc": now(), "selections": selections,
            "script_sha256": digest(ROOT / "scripts/confirm_nifty_llm_search.py")})
        print(f"Saved {len(selections)} selections to {(output / 'selection.json').relative_to(ROOT)}", flush=True)
        results = {}
        for selection in selections:
            key = selection["config_hash"]
            if key in results:
                continue
            results[key] = {}
            for seed in CONFIRMATION_SEEDS:
                result = evaluate(SearchConfig(**selection["config"]), plan, seed, context.store, study.results,
                                  context.signature, study.max_epochs, study.checkpoints)
                if result["status"] != "evaluated":
                    raise ValueError(f"{selection['arm']} seed {seed} failed: {result['error']}")
                results[key][seed] = result
                print(f"{plan} {selection['arm']} r{selection['repeat']} seed={seed} "
                      f"day_MAE={result['objective']:.7f}", flush=True)
        persistence = context.store.window(PLANS[plan][0]).persistence
        summary = summarize(selections, results, persistence, frozen_reference(context.store, plan), plan)
        summary["finished_at_utc"] = now()
        summary["test_evaluated"] = args.final_test
        write_json_atomic(output / "results.json", summary)
        pd.DataFrame(summary["runs"]).to_csv(output / "runs.csv", index=False)
        for row in summary["selections"]:
            print(f"{row['arm']:>15} r{row['repeat']} trial {row['trial']:>2}: day_MAE {row['day_mae_mean']:.7f} "
                  f"({row['day_mae_vs_baseline_config_pct']:+.3f}% vs baseline config, "
                  f"{row['day_mae_vs_persistence_pct']:+.3f}% vs persistence)", flush=True)


if __name__ == "__main__":
    main()

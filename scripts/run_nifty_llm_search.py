"""Run the Nifty LLM-guided improvement experiment, one search arm at a time.

    --init        verify inputs, replicate the frozen baseline bit for bit, write the
                  predeclared protocol, then evaluate the shared baseline (trial 0)
    --arm NAME    gpt_hypotheses | random | bayes | llm_batch | llm_feedback

All arms share the search plan (three forward folds inside the training dates),
training seed, evaluation code and per-search budget. LLM arms exchange numbered
prompt/reply files under handoff/<arm>_r<repeat>/; relay them with
scripts/nifty_llm_handoff.py. Without --wait an LLM arm exits while a reply is
pending, and rerunning resumes from its trial log and saved replies. No test
array is opened. Result-affecting sources are hashed into the protocol, and
later runs refuse to continue after any of them changes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from nifty_models import predict_batches
from nifty_proposers import (
    BAYES_STARTUP_TRIALS,
    GPT_HYPOTHESES,
    Handoff,
    batch_prompt,
    bayes_proposal,
    brief,
    converse,
    feedback_prompt,
    now,
    random_proposal,
)
from nifty_search_runner import (
    DATA,
    FIXED,
    PLANS,
    ROOT,
    DataStore,
    evaluate,
    train,
    write_json_atomic,
)
from nifty_search_space import BASELINE, PARAMETERS, SearchConfig, changed_parameters
from threadpoolctl import threadpool_limits
from train_nifty_models import digest

ARMS = {
    "gpt_hypotheses": "GPT's three predeclared first hypotheses, one change each from the baseline (3 fixed trials)",
    "random": "Uniform random search over the shared space",
    "bayes": f"Optuna Gaussian-process Bayesian optimization (GPSampler; {BAYES_STARTUP_TRIALS - 1} random starts "
             "after the shared baseline)",
    "llm_batch": "LLM proposes every configuration at once from the diagnostic brief, without feedback",
    "llm_feedback": "LLM proposes one configuration at a time and receives measured diagnostic feedback",
}
LLM_ARMS = ("llm_batch", "llm_feedback")
BUDGET = 12
SEARCH_SEED = 42
CONFIRMATION_SEEDS = [17, 42, 2026]
SOURCES = ["scripts/nifty_models.py", "scripts/nifty_search_space.py", "scripts/nifty_search_runner.py",
           "scripts/nifty_proposers.py", "scripts/run_nifty_llm_search.py"]


class Study:
    """Output locations and budget for the real study or a reduced smoke run."""

    def __init__(self, smoke: bool = False):
        self.smoke = smoke
        self.report = ROOT / ("data/processed/nifty_llm_search_smoke" if smoke else "reports/nifty_llm_search")
        self.artifact = ROOT / ("data/processed/nifty_llm_search_smoke" if smoke else "data/processed/nifty_llm_search")
        self.protocol = self.report / "protocol.json"
        self.results, self.trials, self.handoff = (self.report / name for name in ["results", "trials", "handoff"])
        self.checkpoints = self.artifact / "checkpoints"
        self.budget = 2 if smoke else BUDGET
        self.row_limit = 20000 if smoke else None
        self.max_epochs = 2 if smoke else FIXED["max_epochs"]


def signature(study: Study, threads: int) -> dict:
    return {"sources": {path: digest(ROOT / path) for path in SOURCES}, "max_epochs": study.max_epochs,
            "row_limit": study.row_limit, "threads": threads,
            "versions": {name: version(name) for name in ["torch", "numpy", "pandas", "optuna"]}}


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def append_jsonl(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def check_inputs() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    hashes = {}
    for relative in [f"{split}/{name}" for split in ["train", "valid"] for name in ["X.npy", "y.npy", "metadata.parquet"]
                     ] + ["feature_spec.json", "scaler.json"]:
        hashes[relative] = digest(DATA / relative)
        if hashes[relative] != manifest["outputs"][relative]["sha256"]:
            raise ValueError(f"Prepared input changed: {relative}")
    return {"preprocessing_manifest_sha256": digest(DATA / "manifest.json"), "inputs": hashes}


def replicate_frozen_baseline(store: DataStore) -> dict:
    """The baseline configuration under the original protocol must reproduce the frozen checkpoint exactly."""
    window = PLANS["replication"][0]
    model, training = train(BASELINE, store.window(window), 42, FIXED["max_epochs"])
    data = store.window(window)
    predictions = predict_batches(model, data.score.X, data.target_scale)
    saved = np.load(ROOT / "data/processed/nifty_models/compact_seed42_valid_predictions.npy")
    frozen = torch.load(ROOT / "data/processed/nifty_models/compact_seed42.pt", map_location="cpu", weights_only=True)
    result = {"plan": "replication", "seed": 42, "best_epoch": training["best_epoch"],
              "epochs_run": training["epochs_run"],
              "predictions_bitwise_identical": bool(np.array_equal(predictions, saved)),
              "weights_bitwise_identical": all(torch.equal(frozen["state_dict"][key], value)
                                               for key, value in model.state_dict().items()),
              "reference_checkpoint_sha256": digest(ROOT / "data/processed/nifty_models/compact_seed42.pt")}
    store.release(window)
    if not (result["predictions_bitwise_identical"] and result["weights_bitwise_identical"]):
        raise ValueError(f"Search runner does not reproduce the frozen baseline: {result}")
    return result


def target_profile(part) -> dict:
    y = part.y.astype(np.float64)
    absolute = np.abs(y)
    groups = {name: {"share_rows": float(mask.mean()), "std": float(y[mask].std()),
                     "mean_abs": float(absolute[mask].mean()), "share_exact_zero": float((y[mask] == 0).mean())}
              for name, mask in (("close_at_most_1", part.low_price), ("close_above_1", ~part.low_price))}
    return {**part.describe(), "std": float(y.std()), "mean": float(y.mean()),
            "share_exact_zero": float((y == 0).mean()), "abs_max": float(absolute.max()),
            "abs_quantiles": {str(q): float(np.quantile(absolute, q)) for q in [.5, .75, .9, .95, .99, .999]},
            "groups": groups}


def earlier_findings() -> dict:
    """Relative changes of the frozen checkpoint versus persistence, from the verified validation reports."""
    summary = json.loads((ROOT / "reports/nifty_baseline_performance/summary.json").read_text())
    valid = next(split for split in json.loads((DATA / "manifest.json").read_text())["splits"]
                 if split["split"] == "valid")
    groups = pd.read_csv(ROOT / "reports/nifty_mlp_study/subgroups.csv").set_index(["model", "group"])
    low_m, low_p = groups.loc[("compact", "close_at_most_1")], groups.loc[("persistence", "close_at_most_1")]
    high_m, high_p = groups.loc[("compact", "close_above_1")], groups.loc[("persistence", "close_above_1")]
    rows = summary["validation_rows"]
    total_reduction = rows * (summary["persistence_metrics"]["val_rmse"] ** 2 - summary["metrics"]["val_rmse"] ** 2)
    low_reduction = low_m.rows * (low_p.val_rmse ** 2 - low_m.val_rmse ** 2)
    change = summary["relative_error_change_percent"]
    return {"dates": summary["validation_dates"], "first_date": valid["start_date"], "last_date": valid["end_date"],
            **{key: change[f"val_{key}"] for key in ["day_mae", "mae", "rmse", "price_mae", "price_rmse"]},
            "dates_better": summary["days_better_than_persistence"], "low_price_share": low_m.rows / rows,
            "low_price_mae": 100 * (low_m.val_mae / low_p.val_mae - 1),
            "low_price_rmse": 100 * (low_m.val_rmse / low_p.val_rmse - 1),
            "high_price_mae": 100 * (high_m.val_mae / high_p.val_mae - 1),
            "high_price_rmse": 100 * (high_m.val_rmse / high_p.val_rmse - 1),
            "low_price_share_of_squared_error_reduction": low_reduction / total_reduction,
            "sources": ["reports/nifty_baseline_performance/summary.json", "reports/nifty_mlp_study/subgroups.csv"]}


def build_protocol(study: Study, store: DataStore, threads: int, inputs: dict, replication: dict | None) -> dict:
    windows = [store.window(window) for window in PLANS["search"]]
    train_counts = np.bincount(store.split("train")["day"])
    features = json.loads((DATA / "feature_spec.json").read_text())
    findings = earlier_findings()
    data = {"dates": len(train_counts), "first_date": store.calendar[0], "last_date": store.calendar[len(train_counts) - 1],
            "rows": int(train_counts.sum()), "rows_per_date_min": int(train_counts.min()),
            "rows_per_date_median": int(np.median(train_counts)), "rows_per_date_max": int(train_counts.max())}
    return {
        "study": "nifty_llm_search", "smoke": study.smoke, "created_at_utc": now(),
        "research_question": ("Can an LLM use model diagnostics to find improvements to the fixed compact MLP more "
                              "efficiently than conventional tuning, under the same experiment budget?"),
        "reference_model": json.loads((ROOT / "reports/nifty_models/frozen_config.json").read_text()),
        "arms": ARMS, "budget_per_search": study.budget, "gpt_hypotheses_trials": len(GPT_HYPOTHESES),
        "repeats": ("The pilot runs repeat 1 of every arm. Further repeats reuse this protocol with new random/GP "
                    "seeds and fresh LLM conversations; gpt_hypotheses is fixed and runs once."),
        "search": {"plan": "search", "seed": SEARCH_SEED,
                   "windows": [{key: getattr(w, key).describe() for key in ["fit", "tune", "score"]}
                               for w in windows],
                   "objective": "Mean over the three folds of score-window equal-date MAE in log-return units",
                   "baseline": "Trial 0 evaluates the baseline configuration once; every arm sees or uses it"},
        "fixed_training": FIXED, "search_space": PARAMETERS, "baseline_config": BASELINE.model_dump(),
        "proposers": {
            "random": "numpy default_rng([repeat, trial]); uniform choices, (log-)uniform ranges",
            "bayes": {"library": f"optuna {version('optuna')}", "sampler": "GPSampler",
                      "n_startup_trials": BAYES_STARTUP_TRIALS, "seed": "1000 * repeat + trial",
                      "observations": "baseline plus every evaluated trial of the same search"},
            "gpt_hypotheses": GPT_HYPOTHESES,
            "llm": {"backend": "manual file handoff (no API)", "repair_rounds": 1,
                    "relay_instructions": ("Paste each prompt into a NEW temporary chat with memory/personalization "
                                           "off and no browsing or code tools; return the reply verbatim and record "
                                           "the model name."),
                    "batch": "one prompt asks for all proposals at once; no results are shown",
                    "feedback": "one prompt per proposal containing every earlier result of the same search"},
        },
        "rules": ["Invalid proposals (after one repair round) and duplicates consume budget",
                  "Duplicates of the baseline or of an earlier trial in the same search are not retrained",
                  ("Identical configurations proposed by different arms reuse the cached result but are charged "
                   "its original training time and epochs"),
                  "LLM text never supplies labels or metrics; Python computes every reported number",
                  "No prompt contains test-period data or results; the search opens only training dates"],
        "confirmation": {
            "plan": "validation", "seeds": CONFIRMATION_SEEDS,
            "window": {"fit": f"{store.calendar[0]} to {store.calendar[659]} (training dates 0-659)",
                       "tune": f"{store.calendar[660]} to {store.calendar[699]} (training dates 660-699)",
                       "score": "the 150 original validation dates "
                                f"({findings['first_date']} to {findings['last_date']})"},
            "selection": "Per arm and repeat, the evaluated trial with the lowest search objective; ties -> earliest",
            "comparators": ["baseline configuration under the identical plan and seeds", "persistence",
                            "frozen checkpoint (early-stopped on these dates; reference only)"],
            "uncertainty": ("Paired circular block bootstrap over validation dates (block 5, 2000 resamples, "
                            "seed 8801) of seed-averaged daily MAE differences versus the baseline configuration")},
        "final_test": {"plan": "test", "status": "Not run. Requires explicit approval after confirmation; each "
                                                 "frozen selection is then refit on dates 0-809, tuned on 810-849, "
                                                 "and scored once on the reserved June-December 2020 dates"},
        "analysis": {
            "primary_search": "Best-so-far objective after each proposal, and its mean over proposals 1..budget",
            "primary_confirmation": "Seed-mean validation equal-date MAE change versus the baseline configuration",
            "secondary": ["training epochs and seconds per arm", "valid, invalid and duplicate proposal counts",
                          "LLM prompt/reply characters (tokens estimated as characters / 4)",
                          "Spearman correlation of LLM predicted_change_pct with measured change",
                          "subgroup, price-error and RMSE side effects of the selected configurations"]},
        "caveats": [
            ("The loss choice l1 lies inside the space, so random sampling reaches it with probability 1/3 per draw; "
             "the LLM's value may appear as efficiency rather than discovery."),
            "One search per arm is a pilot: it cannot separate method quality from search luck.",
            ("Search uses one training seed. The compact baseline's seed SD of validation equal-date MAE was "
             "0.0000074 (about 0.05%), so smaller differences are within optimization noise."),
            ("The original validation dates already guided architecture selection and are summarized in the LLM "
             "brief, so confirmation there remains a development estimate."),
            "Manual relay latency includes human time and is not a model-latency measurement."],
        "brief_inputs": {"data": data, "features": features["definitions"],
                         "target_profile": target_profile(windows[0].fit), "earlier_findings": findings},
        "replication": replication, "signature": signature(study, threads), **inputs,
        "hardware": {"system": platform.system(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                     "torch_threads": threads},
        "test_evaluated": False,
    }


class Context:
    def __init__(self, study: Study, protocol: dict, store: DataStore):
        self.study, self.protocol, self.store = study, protocol, store
        self.signature = protocol["signature"]
        path = study.report / "baseline.json"
        self.baseline = json.loads(path.read_text()) if path.exists() else None
        self.baseline_result = self.result(self.baseline) if self.baseline else None

    @classmethod
    def open(cls, study: Study, threads: int) -> Context:
        protocol = json.loads(study.protocol.read_text())
        current = signature(study, threads)
        if current != protocol["signature"]:
            changed = [path for path, value in current["sources"].items()
                       if protocol["signature"]["sources"].get(path) != value]
            raise ValueError(f"Code or settings differ from the protocol (changed sources: {changed or 'none'})")
        return cls(study, protocol, DataStore(row_limit=study.row_limit))

    def result(self, record: dict) -> dict | None:
        path = record.get("result_path")
        return json.loads((self.study.report / path).read_text()) if path else None


def evaluate_trial(context: Context, arm: str, repeat: int, index: int, config: SearchConfig, prior: list[dict],
                   proposal: dict | None = None, proposer: dict | None = None, llm: dict | None = None) -> dict:
    config = config.canonical()
    record = {"arm": arm, "repeat": repeat, "trial": index, "config": config.model_dump(),
              "config_hash": config.config_hash(), "changed_from_baseline": changed_parameters(config),
              "proposal": proposal, "proposer": proposer, "llm": llm}
    duplicate = next((earlier for earlier in prior if earlier.get("config_hash") == config.config_hash()), None)
    if duplicate is not None:
        record.update(status="duplicate", duplicate_of=duplicate["trial"], objective=duplicate["objective"],
                      error=None, train_seconds=0.0, epochs=0, cache_hit=True, evaluation_seconds=0.0,
                      result_path=duplicate.get("result_path"))
    else:
        path = context.study.results / "search" / f"seed{SEARCH_SEED}" / f"{config.config_hash()}.json"
        cache_hit = path.exists()
        started = time.perf_counter()
        result = evaluate(config, "search", SEARCH_SEED, context.store, context.study.results, context.signature,
                          context.study.max_epochs, context.study.checkpoints)
        record.update(status=result["status"], duplicate_of=None, objective=result["objective"],
                      error=result["error"], train_seconds=result["train_seconds"] or 0.0,
                      epochs=result["epochs"] or 0, cache_hit=cache_hit,
                      evaluation_seconds=time.perf_counter() - started,
                      result_path=str(path.relative_to(context.study.report)))
    record["recorded_at_utc"] = now()
    return record


def llm_trial(context: Context, arm: str, repeat: int, index: int, item: dict, prior: list[dict], llm: dict) -> dict:
    proposal = item["proposal"]
    text = {key: value for key, value in proposal.items() if key != "config"} if isinstance(proposal, dict) else None
    if item["config"] is None:
        return {"arm": arm, "repeat": repeat, "trial": index, "status": "invalid", "config": None,
                "config_hash": None, "changed_from_baseline": None, "proposal": text, "proposer": None, "llm": llm,
                "duplicate_of": None, "objective": None, "error": item["error"], "train_seconds": 0.0,
                "epochs": 0, "cache_hit": False, "evaluation_seconds": 0.0, "result_path": None,
                "raw_proposal": proposal, "recorded_at_utc": now()}
    return evaluate_trial(context, arm, repeat, index, item["config"], prior, proposal=text, llm=llm)


def report_pending(study: Study, handoff: Handoff, arm: str, repeat: int) -> bool:
    prompt = handoff.paths(handoff.count)[0]
    command = ".conda/bin/python scripts/nifty_llm_handoff.py" + (" --smoke" if study.smoke else "")
    print(f"{arm} r{repeat}: waiting for a reply to {prompt.relative_to(ROOT)}\n"
          f"  copy prompt:  {command} show {arm} --repeat {repeat}\n"
          f"  save reply:   {command} submit {arm} --repeat {repeat} --model '<model name>'", flush=True)
    return False


def run_arm(context: Context, arm: str, repeat: int, wait: bool) -> bool:
    """Advance one search until its budget is used; return False while an LLM reply is pending."""
    if arm == "gpt_hypotheses" and repeat != 1:
        raise ValueError("The fixed GPT hypotheses run once")
    log = context.study.trials / f"{arm}_r{repeat}.jsonl"
    done = read_jsonl(log)
    budget = len(GPT_HYPOTHESES) if arm == "gpt_hypotheses" else context.study.budget
    handoff = Handoff(context.study.handoff / f"{arm}_r{repeat}", wait) if arm in LLM_ARMS else None
    exchange = None
    if arm == "llm_batch" and len(done) < budget:
        exchange = converse(handoff, batch_prompt(context.protocol, context.baseline_result, budget), budget)
        if exchange is None:
            return report_pending(context.study, handoff, arm, repeat)
    while len(done) < budget:
        index, prior = len(done) + 1, [context.baseline, *done]
        if arm == "random":
            record = evaluate_trial(context, arm, repeat, index, random_proposal(repeat, index), prior,
                                    proposer={"rng": [repeat, index]})
        elif arm == "bayes":
            observed = [(SearchConfig(**r["config"]), r["objective"]) for r in prior if r["status"] == "evaluated"]
            record = evaluate_trial(context, arm, repeat, index, bayes_proposal(repeat, index, observed), prior,
                                    proposer={"seed": 1000 * repeat + index, "observations": len(observed),
                                              "random_start": len(observed) < BAYES_STARTUP_TRIALS})
        elif arm == "gpt_hypotheses":
            item = GPT_HYPOTHESES[index - 1]
            record = evaluate_trial(context, arm, repeat, index, SearchConfig(**item["config"]), prior,
                                    proposal={key: value for key, value in item.items() if key != "config"})
        elif arm == "llm_batch":
            llm = ({"messages": exchange["messages"], "repairs": exchange["repairs"]} if index == 1
                   else {"messages": [], "repairs": 0, "shared_exchange_of_trial": 1})
            record = llm_trial(context, arm, repeat, index, exchange["records"][index - 1], prior, llm)
        else:
            handoff.count = max((m["number"] for r in done for m in r["llm"]["messages"]), default=0)
            history = [{**r, "result": context.result(r) if r["status"] != "duplicate" else None} for r in done]
            exchange = converse(handoff, feedback_prompt(context.protocol, context.baseline_result, history, budget), 1)
            if exchange is None:
                return report_pending(context.study, handoff, arm, repeat)
            record = llm_trial(context, arm, repeat, index, exchange["records"][0], prior,
                               {"messages": exchange["messages"], "repairs": exchange["repairs"]})
        append_jsonl(log, record)
        done.append(record)
        objective = "-" if record["objective"] is None else f"{record['objective']:.7f}"
        print(f"{arm} r{repeat} trial {index}/{budget}: {record['status']} objective={objective} "
              f"changed={record['changed_from_baseline']} train={record['train_seconds']:.0f}s", flush=True)
    print(f"{arm} r{repeat}: complete ({budget} trials)", flush=True)
    return True


def initialize(study: Study, threads: int) -> None:
    if (study.report / "baseline.json").exists():
        raise ValueError(f"{study.report} is already initialized; preserve it or move it before reinitializing")
    if study.protocol.exists():
        # Resume an initialization interrupted after the protocol was saved; the signature must still match.
        context = Context.open(study, threads)
        protocol = context.protocol
    else:
        store = DataStore(row_limit=study.row_limit)
        inputs = check_inputs()
        replication = None if study.smoke else replicate_frozen_baseline(store)
        if replication:
            print("Replication: search runner reproduces the frozen checkpoint bit for bit.", flush=True)
        protocol = build_protocol(study, store, threads, inputs, replication)
        write_json_atomic(study.protocol, protocol)
        print(f"Protocol saved to {study.protocol.relative_to(ROOT)} before any search training.", flush=True)
        context = Context(study, protocol, store)
    baseline = evaluate_trial(context, "baseline", 0, 0, BASELINE, [])
    if baseline["status"] != "evaluated":
        raise ValueError(f"Baseline configuration failed: {baseline['error']}")
    write_json_atomic(study.report / "baseline.json", baseline)
    result = json.loads((study.report / baseline["result_path"]).read_text())
    (study.report / "brief.md").write_text(brief(protocol, result))
    print(f"Baseline objective {baseline['objective']:.7f} (persistence {result['persistence_objective']:.7f}); "
          f"brief saved to {(study.report / 'brief.md').relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", action="store_true", help="write the protocol and evaluate the baseline")
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--wait", action="store_true", help="poll for LLM replies instead of exiting")
    parser.add_argument("--smoke", action="store_true", help="20,000 rows per window, 2 epochs, budget 2")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.init and not args.arm:
        parser.error("Use --init and/or --arm")
    if args.repeat < 1 or args.threads < 1:
        parser.error("--repeat and --threads must be positive")
    with threadpool_limits(limits=args.threads):
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        study = Study(args.smoke)
        if args.init:
            initialize(study, args.threads)
        if args.arm:
            run_arm(Context.open(study, args.threads), args.arm, args.repeat, args.wait)


if __name__ == "__main__":
    main()

"""Search-space, calibration, training, proposer and handoff checks on synthetic data only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / f"scripts/{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_script("nifty_models")
space = _load_script("nifty_search_space")
runner = _load_script("nifty_search_runner")
proposers = _load_script("nifty_proposers")


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _proposal(**config):
    return {"hypothesis": "h", "change": "c", "predicted_change_pct": -0.1, "expected_effect": "e",
            "rejection_criterion": "r", "config": {**space.BASELINE.model_dump(), **config}}


def _part(rng, rows=3000, dates=10, signal=0.02):
    X = rng.normal(size=(rows, 15)).astype(np.float32)
    y = (signal * X[:, 0] + rng.laplace(scale=0.01, size=rows)).astype(np.float32)
    day = np.arange(rows) % dates
    counts = np.bincount(day)
    close = np.where(rng.random(rows) < 0.1, rng.uniform(0.1, 1, rows), rng.uniform(2, 200, rows))
    return runner.Part(X=X, y=y, day=day, counts=counts, date_weight=1 / (dates * counts[day]), close=close,
                       is_call=rng.random(rows) < 0.5, low_price=close <= 1,
                       first_date="2001-01-01", last_date="2001-01-10")


def _window(seed=0):
    rng = np.random.default_rng(seed)
    fit, tune, score = _part(rng), _part(rng, 800), _part(rng, 900)
    return runner.WindowData(runner.PLANS["search"][0], fit, tune, score, float(np.std(fit.y, dtype=np.float64)),
                             {}, runner.score_part(np.zeros(len(score.y)), score),
                             runner.day_mae(np.zeros(len(tune.y)), tune))


def test_config_validation_rounding_and_bounds():
    config = space.SearchConfig(**{**space.BASELINE.model_dump(), "learning_rate": 0.00123456, "dropout": 0.5004})
    assert config.learning_rate == 0.00123 and config.dropout == 0.5
    for update in [{"learning_rate": 0.02}, {"dropout": -0.1}, {"loss": "huber"}, {"low_price_weight": True},
                   {"weight_decay": float("nan")}, {"unknown": 1}]:
        with pytest.raises(ValueError):
            space.SearchConfig(**{**space.BASELINE.model_dump(), **update})
    with pytest.raises(ValueError):
        space.SearchConfig(**{k: v for k, v in space.BASELINE.model_dump().items() if k != "calibration"})


def test_canonical_hash_ignores_only_unused_parameters():
    def make(**update):
        return space.SearchConfig(**{**space.BASELINE.model_dump(), **update})
    assert make(loss="l1", smooth_l1_beta=0.5).config_hash() == make(loss="l1", smooth_l1_beta=2).config_hash()
    assert make(smooth_l1_beta=0.5).config_hash() != make(smooth_l1_beta=2).config_hash()
    assert space.changed_parameters(make(loss="mse", smooth_l1_beta=3)) == ["loss"]
    assert space.changed_parameters(space.BASELINE) == []


def test_random_proposals_are_valid_and_reproducible():
    configs = [proposers.random_proposal(1, index) for index in range(1, 201)]
    assert configs == [proposers.random_proposal(1, index) for index in range(1, 201)]
    assert configs[:12] != [proposers.random_proposal(2, index) for index in range(1, 13)]
    for config in configs:
        assert space.SearchConfig.model_validate(config.model_dump()) == config
    assert {config.loss for config in configs} == {"smooth_l1", "l1", "mse"}


def test_reply_parsing_accepts_prose_and_reports_problems():
    reply = "Here you go:\n```json\n" + json.dumps({"proposals": [_proposal(loss="l1")]}) + "\n```\nThanks"
    records, problems = space.parse_proposals(reply, 1)
    assert problems == [] and records[0]["error"] is None and records[0]["config"].loss == "l1"
    records, problems = space.parse_proposals(json.dumps({"proposals": [_proposal(), _proposal(dropout=0.9)]}), 3)
    assert problems == ["Expected exactly 3 proposal(s); received 2"]
    assert records[0]["error"] is None and records[2]["error"] == "Missing proposal"
    assert records[1]["error"] == "config.dropout: must be within [0, 0.5] (got 0.9)"
    unknown = _proposal()
    del unknown["config"]["calibration"]
    unknown["config"]["clip"] = 3
    error = space.parse_proposals(json.dumps({"proposals": [unknown]}), 1)[0][0]["error"]
    assert error == "config.calibration: Field required; config.clip: Extra inputs are not permitted (got 3)"
    no_number = _proposal()
    no_number["predicted_change_pct"] = "small"
    assert "predicted_change_pct" in space.parse_proposals(json.dumps({"proposals": [no_number]}), 1)[0][0]["error"]
    records, problems = space.parse_proposals("I would try L1 loss.", 2)
    assert problems and all(record["config"] is None for record in records)


def test_optimal_scale_matches_dense_grid_and_bounds():
    rng = np.random.default_rng(3)
    prediction, weight = rng.normal(size=500), rng.uniform(0.1, 2, 500)
    target = 0.4 * prediction + rng.laplace(scale=0.5, size=500)
    alpha = runner.optimal_scale(prediction, target, weight)
    grid = np.linspace(0, 1, 2001)
    losses = [np.sum(weight * np.abs(a * prediction - target)) for a in grid]
    assert np.sum(weight * np.abs(alpha * prediction - target)) <= min(losses) + 1e-9
    assert runner.optimal_scale(prediction, 2 * prediction, weight) == 1.0
    assert runner.optimal_scale(prediction, -prediction, weight) == 0.0
    assert runner.optimal_scale(np.zeros(5), np.ones(5), np.ones(5)) == 1.0


def test_price_group_calibration_fits_each_group_separately():
    part = _part(np.random.default_rng(4))
    prediction = np.random.default_rng(5).normal(size=len(part.y))
    part.y = np.where(part.low_price, 0.3 * prediction, 0.8 * prediction).astype(np.float32)
    config = space.SearchConfig(**{**space.BASELINE.model_dump(), "calibration": "price_group_scale"})
    scales = runner.fit_calibration(config, prediction, part)
    assert scales == pytest.approx({"close_at_most_1": 0.3, "close_above_1": 0.8}, abs=1e-6)
    calibrated = runner.apply_calibration(prediction, scales, part)
    assert runner.day_mae(calibrated, part) < 1e-6


def test_sample_weights_balance_dates_and_scale_low_prices():
    rng = np.random.default_rng(6)
    part = _part(rng)
    part.day = np.sort(rng.integers(0, 10, len(part.y)))
    part.counts = np.bincount(part.day)
    part.date_weight = 1 / (len(part.counts) * part.counts[part.day])
    base = space.BASELINE.model_dump()
    assert runner.sample_weights(space.BASELINE, part) is None
    weights = runner.sample_weights(space.SearchConfig(**{**base, "sample_weighting": "date_balanced"}), part)
    totals = np.bincount(part.day, weights=weights)
    assert weights.mean() == pytest.approx(1) and np.allclose(totals, totals[0], rtol=1e-5)
    weights = runner.sample_weights(space.SearchConfig(**{**base, "low_price_weight": 0.1}), part)
    assert weights[part.low_price][0] / weights[~part.low_price][0] == pytest.approx(0.1)


def test_plans_are_chronological_and_test_dates_stay_reserved(monkeypatch, tmp_path):
    for name in runner.PLANS:
        runner.check_plan(name)
    assert all(window.score[1] <= 700 for window in runner.PLANS["search"])
    assert [w.score for w in runner.PLANS["validation"]] == [(700, 850)]
    assert all(window.score[1] <= 850 for name, plan in runner.PLANS.items() if name != "test" for window in plan)
    monkeypatch.setitem(runner.PLANS, "broken", (runner.Window((0, 10), (12, 20), (20, 30)),))
    with pytest.raises(ValueError):
        runner.check_plan("broken")
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "feature_spec.json").write_text(json.dumps({"feature_columns": []}))
    (tmp_path / "scaler.json").write_text("{}")
    monkeypatch.setattr(runner, "DATA", tmp_path)
    with pytest.raises(PermissionError):
        runner.DataStore().split("test")


@pytest.mark.parametrize("update", [{}, {"loss": "l1", "calibration": "global_scale"},
                                    {"loss": "mse", "sample_weighting": "date_balanced", "low_price_weight": 0.2,
                                     "calibration": "price_group_scale"}])
def test_training_is_finite_and_deterministic(update):
    config = space.SearchConfig(**{**space.BASELINE.model_dump(), **update})
    first = runner.run_window(config, _window(), seed=7, max_epochs=2)
    second = runner.run_window(config, _window(), seed=7, max_epochs=2)
    for key in ["day_mae", "mae", "rmse", "price_mae", "price_rmse"]:
        assert np.isfinite(first["metrics"][key]) and first["metrics"][key] == second["metrics"][key]
    assert first["calibration"] == second["calibration"] and len(first["history"]) == first["epochs_run"]
    assert set(first["calibration"]) == {"none": set(), "global_scale": {"all"},
                                         "price_group_scale": {"close_at_most_1", "close_above_1"}}[config.calibration]
    # A learnable signal exists, so training should beat the zero-return start on the scored rows.
    assert first["metrics"]["day_mae"] < first["persistence"]["day_mae"]


def test_evaluate_caches_results_and_records_training_failures(monkeypatch, tmp_path):
    calls = []

    def failing(*args):
        calls.append(args)
        raise ValueError("Nonfinite training loss at epoch 1")

    monkeypatch.setattr(runner, "run_window", failing)
    monkeypatch.setattr(runner.DataStore, "window", lambda self, window: None)
    store = runner.DataStore.__new__(runner.DataStore)
    result = runner.evaluate(space.BASELINE, "search", 42, store, tmp_path, {"code": 1})
    assert result["status"] == "training_failed" and result["objective"] is None and "Nonfinite" in result["error"]
    assert runner.evaluate(space.BASELINE, "search", 42, store, tmp_path, {"code": 1}) == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="different code"):
        runner.evaluate(space.BASELINE, "search", 42, store, tmp_path, {"code": 2})


def test_bayes_proposals_are_valid_and_reproducible():
    observed = [(proposers.random_proposal(9, index), float(index % 3) + 0.01 * index) for index in range(1, 6)]
    first = proposers.bayes_proposal(1, 6, observed)
    assert first == proposers.bayes_proposal(1, 6, observed)
    assert space.SearchConfig.model_validate(first.model_dump()) == first


def test_handoff_waits_resumes_and_repairs(tmp_path):
    handoff = proposers.Handoff(tmp_path, wait=False)
    assert proposers.converse(handoff, "prompt one", 1) is None
    assert (tmp_path / "01_prompt.md").read_text() == "prompt one"
    (tmp_path / "01_reply.txt").write_text("no json here")
    assert proposers.converse(proposers.Handoff(tmp_path, wait=False), "prompt one", 1) is None
    repair = (tmp_path / "02_prompt.md").read_text()
    assert repair.startswith("# Correction needed") and "prompt one" in repair and "no json here" in repair
    (tmp_path / "02_reply.txt").write_text(json.dumps({"proposals": [_proposal(loss="l1")]}))
    exchange = proposers.converse(proposers.Handoff(tmp_path, wait=False), "prompt one", 1)
    assert exchange["repairs"] == 1 and exchange["records"][0]["config"].loss == "l1"
    assert [message["number"] for message in exchange["messages"]] == [1, 2]
    with pytest.raises(ValueError, match="differs"):
        proposers.Handoff(tmp_path, wait=False).ask("a different prompt")


def test_prompts_render_results_failures_and_duplicates():
    window = _window()
    result = runner.run_window(space.BASELINE, window, seed=7, max_epochs=1)
    baseline = {"status": "evaluated", "error": None, "windows": [result],
                "objective": result["metrics"]["day_mae"], "persistence_objective": result["persistence"]["day_mae"]}
    profile = {"first_date": "2001-01-01", "last_date": "2001-01-10", "dates": 10, "rows": 3000, "std": 0.02,
               "mean": 0.0, "share_exact_zero": 0.1, "abs_max": 0.2, "abs_quantiles": {"0.5": 0.01, "0.99": 0.05},
               "groups": {name: {"share_rows": 0.5, "std": 0.02, "mean_abs": 0.01, "share_exact_zero": 0.1}
                          for name in ["close_at_most_1", "close_above_1"]}}
    findings = {"dates": 150, "first_date": "2001-02-01", "last_date": "2001-08-01", "day_mae": 0.2, "mae": 0.3,
                "rmse": -5.0, "price_mae": 0.6, "price_rmse": 1.2, "dates_better": 41, "low_price_share": 0.02,
                "low_price_mae": 0.1, "low_price_rmse": -8.0, "high_price_mae": 0.3, "high_price_rmse": -0.1,
                "low_price_share_of_squared_error_reduction": 0.99}
    data = {"dates": 10, "first_date": "2001-01-01", "last_date": "2001-01-10", "rows": 3000,
            "rows_per_date_min": 300, "rows_per_date_median": 300, "rows_per_date_max": 300}
    protocol = {"brief_inputs": {"data": data, "features": {"f": "definition"}, "target_profile": profile,
                                 "earlier_findings": findings},
                "search": {"windows": [{key: result[key] for key in ["fit", "tune", "score"]}]}}
    batch = proposers.batch_prompt(protocol, baseline, 12)
    assert "Propose exactly 12 configurations" in batch and all(name in batch for name in space.PARAMETERS)
    history = [
        {"trial": 1, "status": "evaluated", "config": space.BASELINE.model_dump(), "objective": baseline["objective"],
         "proposal": _proposal(), "result": baseline},
        {"trial": 2, "status": "invalid", "config": None, "objective": None, "error": "bad dropout",
         "proposal": None},
        {"trial": 3, "status": "duplicate", "config": space.BASELINE.model_dump(), "duplicate_of": 0,
         "objective": baseline["objective"], "proposal": _proposal()},
        {"trial": 4, "status": "training_failed", "config": space.BASELINE.model_dump(), "objective": None,
         "proposal": _proposal(), "result": {"status": "training_failed", "error": "Nonfinite loss"}}]
    feedback = proposers.feedback_prompt(protocol, baseline, history, 12)
    assert "Evaluations used: 4 of 12" in feedback and "8 remaining" in feedback
    assert "bad dropout" in feedback and "Duplicate of trial 0" in feedback and "Nonfinite loss" in feedback
    assert "Lowest objective so far: trial 1" in feedback

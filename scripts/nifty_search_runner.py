"""Train and score one search configuration on chronological date windows.

Each window fits on earlier dates, tunes on the following dates and scores on
the dates after those. Continuous features are re-standardized with fit-window
statistics, and the target is divided by its fit-window standard deviation.
Only the tune window drives early stopping, learning-rate reduction and output
calibration; score windows only produce reported metrics. Test arrays are
opened only by a store created with ``allow_test=True``.

The training loop mirrors ``train_nifty_models.train_one`` operation for
operation, so the baseline configuration on the "replication" plan reproduces
the frozen checkpoint exactly.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nifty_models import ReturnMLP, predict_batches, regression_metrics
from nifty_search_space import SearchConfig
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/nifty_training"
# Global chronological date-index range of each prepared split.
SPLITS = {"train": (0, 700), "valid": (700, 850), "test": (850, 1000)}
FIXED = {
    "hidden": [32, 16], "activation": "ReLU", "batch_size": 4096, "max_epochs": 12,
    "early_stopping_patience": 3, "early_stopping_min_delta": 1e-6, "optimizer": "AdamW",
    "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-5) on the tune objective",
    "gradient_clip_norm": 1.0, "output_initialization": "zero weights and bias, i.e. persistence before training",
    "train_probe_rows": 65536, "train_probe_seed": 9102,
}


@dataclass(frozen=True)
class Window:
    fit: tuple[int, int]
    tune: tuple[int, int]
    score: tuple[int, int]
    refit_scaler: bool = True


PLANS = {
    # Expanding forward folds inside the original training dates: feedback for every search method.
    "search": (Window((0, 420), (420, 460), (460, 540)),
               Window((0, 500), (500, 540), (540, 620)),
               Window((0, 580), (580, 620), (620, 700))),
    # Confirmation on the original validation dates, which the architecture study already used.
    "validation": (Window((0, 660), (660, 700), (700, 850)),),
    # Reserved June-December 2020 partition: evaluated once, after final selections are frozen.
    "test": (Window((0, 810), (810, 850), (850, 1000)),),
    # Original frozen-baseline protocol, including early stopping on its scored dates.
    "replication": (Window((0, 700), (700, 850), (700, 850), refit_scaler=False),),
}


def check_plan(name: str) -> None:
    for window in PLANS[name]:
        if window.fit[0] != 0 or any(start >= stop for start, stop in (window.fit, window.tune, window.score)):
            raise ValueError(f"Malformed {name} window: {window}")
        if name == "replication":
            if window.fit[1] != window.tune[0] or window.tune != window.score:
                raise ValueError("The replication plan scores exactly its early-stopping dates")
        elif window.fit[1] != window.tune[0] or window.tune[1] != window.score[0]:
            raise ValueError(f"{name} windows must be contiguous fit -> tune -> score: {window}")


@dataclass
class Part:
    """Rows of one date range, standardized for the enclosing window."""

    X: np.ndarray
    y: np.ndarray
    day: np.ndarray
    counts: np.ndarray
    date_weight: np.ndarray
    close: np.ndarray
    is_call: np.ndarray
    low_price: np.ndarray
    first_date: str
    last_date: str

    def describe(self) -> dict:
        return {"first_date": self.first_date, "last_date": self.last_date,
                "dates": len(self.counts), "rows": len(self.y)}


@dataclass
class WindowData:
    window: Window
    fit: Part
    tune: Part
    score: Part
    target_scale: float
    feature_scaler: dict
    persistence: dict
    tune_persistence_day_mae: float


class DataStore:
    """Memory-mapped prepared splits with a chronological global date index per row."""

    def __init__(self, allow_test: bool = False, row_limit: int | None = None):
        manifest = json.loads((DATA / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("Prepared Nifty data must be complete")
        self.allow_test = allow_test
        self.row_limit = row_limit
        self.columns = json.loads((DATA / "feature_spec.json").read_text())["feature_columns"]
        self.scaler = json.loads((DATA / "scaler.json").read_text())
        self.calendar: dict[int, str] = {}
        self._splits: dict[str, dict] = {}
        self._windows: dict[Window, WindowData] = {}

    def split(self, name: str) -> dict:
        if name == "test" and not self.allow_test:
            raise PermissionError("The reserved test partition requires allow_test=True")
        if name not in self._splits:
            meta = pd.read_parquet(DATA / f"{name}/metadata.parquet", columns=["timestamp", "close", "option_type"])
            days = meta.timestamp.dt.normalize().to_numpy()
            unique = np.unique(days)
            start, stop = SPLITS[name]
            if len(unique) != stop - start:
                raise ValueError(f"{name} has {len(unique)} dates; expected {stop - start}")
            for offset, day in enumerate(unique):
                self.calendar[start + offset] = str(day)[:10]
            ordered = [self.calendar[i] for i in sorted(self.calendar)]
            if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
                raise ValueError("Split dates must be chronological and disjoint")
            self._splits[name] = {
                "X": np.load(DATA / f"{name}/X.npy", mmap_mode="r", allow_pickle=False),
                "y": np.load(DATA / f"{name}/y.npy", mmap_mode="r", allow_pickle=False),
                "day": np.searchsorted(unique, days) + start, "close": meta.close.to_numpy(np.float64),
                "is_call": meta.option_type.eq("CE").to_numpy(),
            }
        return self._splits[name]

    def part(self, start: int, stop: int) -> Part:
        pieces = []
        for name, (low, high) in SPLITS.items():
            if low < stop and start < high:
                split = self.split(name)
                index = np.flatnonzero((split["day"] >= start) & (split["day"] < stop))
                if self.row_limit is not None and len(index) > self.row_limit:
                    rng = np.random.default_rng([start, stop])
                    index = np.sort(rng.choice(index, size=self.row_limit, replace=False))
                pieces.append({key: split[key][index] for key in ["X", "y", "day", "close", "is_call"]})
        joined = {key: np.concatenate([np.asarray(piece[key]) for piece in pieces]) for key in pieces[0]}
        unique, day = np.unique(joined["day"], return_inverse=True)
        counts = np.bincount(day)
        return Part(X=np.ascontiguousarray(joined["X"], dtype=np.float32), y=joined["y"], day=day,
                    counts=counts, date_weight=1.0 / (len(counts) * counts[day]), close=joined["close"],
                    is_call=joined["is_call"], low_price=joined["close"] <= 1,
                    first_date=self.calendar[int(unique[0])], last_date=self.calendar[int(unique[-1])])

    def window(self, window: Window) -> WindowData:
        if window not in self._windows:
            fit, tune, score = (self.part(*bounds) for bounds in (window.fit, window.tune, window.score))
            columns = [self.columns.index(name) for name in self.scaler["continuous_columns"]]
            mean0, scale0 = np.asarray(self.scaler["mean"]), np.asarray(self.scaler["scale"])
            feature_scaler = {"source": "prepared training-split scaler", "columns": self.scaler["continuous_columns"]}
            if window.refit_scaler:
                raw = fit.X[:, columns].astype(np.float64) * scale0 + mean0
                mean, scale = raw.mean(axis=0), raw.std(axis=0)
                if not (np.isfinite(scale).all() and (scale > 0).all()):
                    raise ValueError("Fit-window feature scales must be positive")
                for part in (fit, tune, score):
                    restored = part.X[:, columns].astype(np.float64) * scale0 + mean0
                    part.X[:, columns] = ((restored - mean) / scale).astype(np.float32)
                feature_scaler = {"source": "fit window", "columns": self.scaler["continuous_columns"],
                                  "mean": mean.tolist(), "scale": scale.tolist()}
            target_scale = float(np.std(fit.y, dtype=np.float64))
            if not np.isfinite(target_scale) or target_scale <= 0:
                raise ValueError("Fit-window target must have a positive finite standard deviation")
            self._windows[window] = WindowData(window, fit, tune, score, target_scale, feature_scaler,
                                               score_part(np.zeros(len(score.y)), score),
                                               day_mae(np.zeros(len(tune.y)), tune))
        return self._windows[window]

    def release(self, window: Window) -> None:
        self._windows.pop(window, None)


def sample_weights(config: SearchConfig, part: Part) -> np.ndarray | None:
    """Mean-one row weights, or None when every row is weighted equally."""
    if config.sample_weighting == "none" and config.low_price_weight == 1:
        return None
    weight = part.date_weight.copy() if config.sample_weighting == "date_balanced" else np.ones(len(part.y))
    weight[part.low_price] *= config.low_price_weight
    return (weight / weight.mean()).astype(np.float32)


def training_loss(config: SearchConfig, output, target, weight=None):
    reduction = "mean" if weight is None else "none"
    if config.loss == "l1":
        loss = F.l1_loss(output, target, reduction=reduction)
    elif config.loss == "mse":
        loss = F.mse_loss(output, target, reduction=reduction)
    else:
        loss = F.smooth_l1_loss(output, target, reduction=reduction, beta=config.smooth_l1_beta)
    return loss if weight is None else (loss * weight).mean()


def optimal_scale(prediction, target, weight, upper: float = 1.0) -> float:
    """Exact minimizer over [0, upper] of sum(weight * |scale * prediction - target|).

    The objective is sum(weight * |prediction| * |scale - target / prediction|)
    plus terms independent of scale, so a weighted median minimizes it. The
    objective is convex in scale; clipping that median gives the bounded optimum.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    active = prediction != 0
    if not active.any():
        return 1.0
    ratio = np.asarray(target, dtype=np.float64)[active] / prediction[active]
    cost = np.asarray(weight, dtype=np.float64)[active] * np.abs(prediction[active])
    order = np.argsort(ratio, kind="stable")
    cumulative = np.cumsum(cost[order])
    median = ratio[order][np.searchsorted(cumulative, cumulative[-1] / 2)]
    return float(np.clip(median, 0.0, upper))


def fit_calibration(config: SearchConfig, prediction, part: Part) -> dict:
    """Prediction multipliers that minimize equal-date MAE on ``part``."""
    if config.calibration == "none":
        return {}
    if config.calibration == "global_scale":
        return {"all": optimal_scale(prediction, part.y, part.date_weight)}
    # Each group's absolute errors enter equal-date MAE separately, so the optima separate too.
    return {group: optimal_scale(prediction[mask], part.y[mask], part.date_weight[mask]) if mask.any() else 1.0
            for group, mask in (("close_at_most_1", part.low_price), ("close_above_1", ~part.low_price))}


def apply_calibration(prediction, scales: dict, part: Part):
    if not scales:
        return prediction
    if "all" in scales:
        return prediction * scales["all"]
    return np.where(part.low_price, prediction * scales["close_at_most_1"], prediction * scales["close_above_1"])


def day_mae(prediction, part: Part) -> float:
    return regression_metrics(part.y, prediction, part.day)["day_mae"]


def score_part(prediction, part: Part) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = part.y.astype(np.float64)
    error = prediction - target
    groups = {}
    for name, mask in (("close_at_most_1", part.low_price), ("close_above_1", ~part.low_price),
                       ("calls", part.is_call), ("puts", ~part.is_call)):
        if mask.any():
            groups[name] = {"rows": int(mask.sum()), "mae": float(np.mean(np.abs(error[mask]))),
                            "rmse": float(np.sqrt(np.mean(error[mask] ** 2)))}
    spread = np.std(prediction)
    return {**regression_metrics(part.y, prediction, part.day, part.close),
            "daily_mae": (np.bincount(part.day, weights=np.abs(error)) / part.counts).tolist(),
            "groups": groups, "mean_abs_prediction": float(np.mean(np.abs(prediction))),
            "mean_abs_target": float(np.mean(np.abs(target))),
            "correlation": float(np.corrcoef(prediction, target)[0, 1]) if spread > 0 else 0.0}


def train(config: SearchConfig, data: WindowData, seed: int, max_epochs: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ReturnMLP(data.fit.X.shape[1], FIXED["hidden"], config.dropout)
    nn.init.zeros_(model.network[-1].weight)
    nn.init.zeros_(model.network[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=1, min_lr=1e-5)
    X, y = data.fit.X, data.fit.y
    target = y / data.target_scale
    weight = sample_weights(config, data.fit)
    probe = np.sort(np.random.default_rng(FIXED["train_probe_seed"]).choice(
        len(y), size=min(FIXED["train_probe_rows"], len(y)), replace=False))
    best, best_epoch, stale, state, scales, history = float("inf"), 0, 0, None, {}, []
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(len(X))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), FIXED["batch_size"]):
            indices = order[start:start + FIXED["batch_size"]]
            batch_weight = None if weight is None else torch.from_numpy(weight[indices])
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(config, model(torch.from_numpy(X[indices])),
                                 torch.from_numpy(target[indices]), batch_weight)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FIXED["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        raw = predict_batches(model, data.tune.X, data.target_scale)
        epoch_scales = fit_calibration(config, raw, data.tune)
        score = day_mae(apply_calibration(raw, epoch_scales, data.tune), data.tune)
        probe_predictions = predict_batches(model, X[probe], data.target_scale)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(X),
                        "train_probe_mae": float(np.mean(np.abs(probe_predictions - y[probe]))),
                        "learning_rate": optimizer.param_groups[0]["lr"], "tune_day_mae": score,
                        "calibration": epoch_scales, "seconds": time.perf_counter() - started})
        if score < best - FIXED["early_stopping_min_delta"]:
            best, best_epoch, stale = score, epoch, 0
            state, scales = copy.deepcopy(model.state_dict()), epoch_scales
        else:
            stale += 1
        scheduler.step(score)
        if stale >= FIXED["early_stopping_patience"]:
            break
    if state is None:
        raise ValueError("No finite checkpoint")
    model.load_state_dict(state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "tune_day_mae": best,
                   "calibration": scales, "history": history}


def run_window(config: SearchConfig, data: WindowData, seed: int, max_epochs: int, checkpoint: Path | None = None):
    started = time.perf_counter()
    model, training = train(config, data, seed, max_epochs)
    train_seconds = time.perf_counter() - started
    prediction = apply_calibration(predict_batches(model, data.score.X, data.target_scale),
                                   training["calibration"], data.score)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "input_dim": data.fit.X.shape[1], "hidden": FIXED["hidden"],
                    "config": config.model_dump(), "seed": seed, "target_scale": data.target_scale,
                    "feature_scaler": data.feature_scaler, "calibration": training["calibration"]}, checkpoint)
    return {"fit": data.fit.describe(), "tune": data.tune.describe(), "score": data.score.describe(),
            "target_scale": data.target_scale, "train_seconds": train_seconds, **training,
            "tune_persistence_day_mae": data.tune_persistence_day_mae,
            "metrics": score_part(prediction, data.score), "persistence": data.persistence}


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(config: SearchConfig, plan: str, seed: int, store: DataStore, results: Path,
             signature: dict, max_epochs: int = FIXED["max_epochs"], checkpoints: Path | None = None) -> dict:
    """Train/score every window of ``plan``; results are cached by configuration hash."""
    check_plan(plan)
    config = config.canonical()
    path = results / plan / f"seed{seed}" / f"{config.config_hash()}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["signature"] != signature:
            raise ValueError(f"{path} was produced by different code or settings")
        return cached
    result = {"config": config.model_dump(), "config_hash": config.config_hash(), "plan": plan, "seed": seed,
              "signature": signature, "status": "evaluated", "error": None}
    try:
        windows = []
        for index, window in enumerate(PLANS[plan]):
            checkpoint = None if checkpoints is None else (
                checkpoints / plan / f"seed{seed}" / f"{config.config_hash()}_w{index}.pt")
            windows.append(run_window(config, store.window(window), seed, max_epochs, checkpoint))
    except (ValueError, RuntimeError) as error:
        # A diverging configuration is a measured outcome, not a harness failure.
        result.update(status="training_failed", error=f"{type(error).__name__}: {error}", objective=None,
                      persistence_objective=None, windows=[], train_seconds=None, epochs=None)
    else:
        result.update(windows=windows,
                      objective=float(np.mean([w["metrics"]["day_mae"] for w in windows])),
                      persistence_objective=float(np.mean([w["persistence"]["day_mae"] for w in windows])),
                      train_seconds=float(sum(w["train_seconds"] for w in windows)),
                      epochs=int(sum(w["epochs_run"] for w in windows)))
    write_json_atomic(path, result)
    return result

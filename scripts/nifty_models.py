"""Small, conventional regressors for the prepared Nifty return features.

Models consume 15 standardized/encoded features, not a raw sequence tensor.
The MLP head predicts a signed scalar in training-target-standard-deviation
units. Prediction helpers restore the original log-return units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

MODEL_CONFIGS = {
    "compact": {"hidden": [32, 16], "dropout": 0.1},
    "standard": {"hidden": [64, 32], "dropout": 0.1},
    "wide": {"hidden": [128, 64, 32], "dropout": 0.1},
}


class ReturnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        if input_dim < 1 or not hidden or any(width < 1 for width in hidden):
            raise ValueError("Input and hidden dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.input_dim = input_dim
        layers = []
        width = input_dim
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(dropout)])
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected a batch with {self.input_dim} input features")
        return self.network(x).squeeze(-1)


def make_model(name: str, input_dim: int = 15) -> ReturnMLP:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown architecture: {name}")
    return ReturnMLP(input_dim=input_dim, **MODEL_CONFIGS[name])


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _matrix(x) -> np.ndarray:
    x = np.asanyarray(x)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    return x


def _batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def predict_batches(model: nn.Module, X, target_scale: float, batch_size: int = 65536) -> np.ndarray:
    X = _matrix(X)
    _batch_size(batch_size)
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be finite and positive")
    predictions = np.empty(len(X), dtype=np.float64)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            batch = np.array(X[start:stop], dtype=np.float32, copy=True)
            if not np.isfinite(batch).all():
                raise ValueError("Nonfinite input features")
            values = model(torch.from_numpy(batch).to(device)).cpu().numpy()
            predictions[start:stop] = values.astype(np.float64) * target_scale
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite model predictions")
    return predictions


def fit_ridge(X, y, alpha: float, batch_size: int = 65536) -> np.ndarray:
    """Minimize mean squared loss + alpha * squared coefficient norm.

    Accumulate in float64. The final coefficient is an unpenalized intercept.
    No validation observations participate in this fit.
    """
    X = _matrix(X)
    _batch_size(batch_size)
    y = np.asanyarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be a vector aligned with X")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    gram = np.zeros((X.shape[1] + 1, X.shape[1] + 1), dtype=np.float64)
    cross = np.zeros(X.shape[1] + 1, dtype=np.float64)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        block = np.ones((stop - start, X.shape[1] + 1), dtype=np.float64)
        block[:, :-1] = X[start:stop]
        target = np.asarray(y[start:stop], dtype=np.float64)
        if not np.isfinite(block).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite ridge training data")
        gram += block.T @ block
        cross += block.T @ target
    gram /= len(X)
    cross /= len(X)
    penalty = np.full(X.shape[1] + 1, alpha)
    penalty[-1] = 0
    return np.linalg.lstsq(gram + np.diag(penalty), cross, rcond=None)[0]


def predict_ridge(X, coefficients, batch_size: int = 65536) -> np.ndarray:
    X = _matrix(X)
    _batch_size(batch_size)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (X.shape[1] + 1,) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid ridge coefficients")
    predictions = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        block = np.asarray(X[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite input features")
        predictions[start:stop] = block @ coefficients[:-1] + coefficients[-1]
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite ridge predictions")
    return predictions


def daily_metrics(y, predictions, day_index, close=None) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    days = np.asarray(day_index)
    if y.ndim != 1 or predictions.shape != y.shape or days.shape != y.shape or len(y) == 0:
        raise ValueError("Targets, predictions and days must be aligned nonempty vectors")
    if not np.isfinite(y).all() or not np.isfinite(predictions).all() or pd.isna(days).any():
        raise ValueError("Nonfinite targets/predictions or missing day identifiers")
    codes, labels = pd.factorize(days, sort=True)
    counts = np.bincount(codes)
    error = predictions - y
    frame = pd.DataFrame({"day": labels, "rows": counts,
                          "mae": np.bincount(codes, weights=np.abs(error)) / counts,
                          "mse": np.bincount(codes, weights=error ** 2) / counts})
    if close is not None:
        close = np.asarray(close, dtype=np.float64)
        if close.shape != y.shape or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("Current prices must be positive, finite and aligned")
        # No clipping: an overflowing model is invalid, not silently repaired.
        with np.errstate(over="raise", invalid="raise"):
            price_error = close * (np.exp(predictions) - np.exp(y))
        if not np.isfinite(price_error).all():
            raise ValueError("Nonfinite reconstructed price errors")
        frame["price_mae"] = np.bincount(codes, weights=np.abs(price_error)) / counts
        frame["price_mse"] = np.bincount(codes, weights=price_error ** 2) / counts
    return frame


def regression_metrics(y, predictions, day_index, close=None) -> dict:
    daily = daily_metrics(y, predictions, day_index, close)
    result = {"mae": float(np.average(daily.mae, weights=daily.rows)),
              "rmse": float(np.sqrt(np.average(daily.mse, weights=daily.rows))),
              "day_mae": float(daily.mae.mean())}
    if close is not None:
        result.update({"price_mae": float(np.average(daily.price_mae, weights=daily.rows)),
                       "price_rmse": float(np.sqrt(np.average(daily.price_mse, weights=daily.rows)))})
    return result

"""Synthetic checks for next-minute model fitting, prediction, and evaluation.

These tests never load the held-out test partition or the market-data files.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "nifty_models",
    Path(__file__).resolve().parents[1] / "scripts/nifty_models.py",
)
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


@pytest.mark.parametrize("name,expected", [
    ("compact", 1057),
    ("standard", 3137),
    ("wide", 12417),
])
def test_mlp_candidates_have_expected_capacity_and_scalar_output(name, expected):
    torch.manual_seed(19)
    model = models.make_model(name)
    assert models.parameter_count(model) == expected
    output = model(torch.randn(7, 15))
    assert output.shape == (7,)
    assert torch.isfinite(output).all()


def test_regression_head_supports_negative_and_positive_returns():
    model = models.ReturnMLP(input_dim=2, hidden=[3], dropout=0.0)
    output_layer = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        for value in [-2.0, 2.0]:
            output_layer.bias.fill_(value)
            torch.testing.assert_close(model(torch.ones(4, 2)), torch.full((4,), value))


def test_mlp_gradients_reach_hidden_layers_and_are_finite():
    torch.manual_seed(5)
    model = models.make_model("standard")
    X = torch.randn(23, 15)
    y = torch.randn(23)
    loss = torch.mean((model(X) - y) ** 2)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_batch_prediction_disables_dropout_and_converts_target_units():
    torch.manual_seed(13)
    model = models.make_model("compact")
    X = np.random.default_rng(42).normal(size=(19, 15)).astype(np.float32)
    scale = 0.037
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(X)).numpy() * scale
    model.train()
    first = models.predict_batches(model, X, target_scale=scale, batch_size=4)
    second = models.predict_batches(model, X, target_scale=scale, batch_size=100)
    assert first.shape == (19,)
    np.testing.assert_allclose(first, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-8)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_streamed_ridge_matches_mean_loss_closed_form(batch_size):
    rng = np.random.default_rng(20)
    X = rng.normal(size=(37, 4)) + np.array([0.0, 2.0, -3.0, 1.0])
    y = X @ np.array([0.2, -0.5, 1.5, 0.1]) + 2.7 + rng.normal(0, 0.1, len(X))
    alpha = 0.25
    design = np.column_stack([X, np.ones(len(X))])
    penalty = np.diag([alpha] * X.shape[1] + [0.0])
    expected = np.linalg.solve(design.T @ design / len(X) + penalty, design.T @ y / len(X))
    actual = models.fit_ridge(X, y, alpha=alpha, batch_size=batch_size)
    assert actual.shape == (5,)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    predictions = models.predict_ridge(X, actual, batch_size=batch_size)
    np.testing.assert_allclose(predictions, design @ expected, rtol=1e-6, atol=1e-6)


def test_ridge_regularizes_coefficients_without_shrinking_intercept():
    X = np.random.default_rng(1).normal(size=(50, 3)) + 2
    y = np.full(50, 7.3)
    coefficients = models.fit_ridge(X, y, alpha=1e6, batch_size=9)
    np.testing.assert_allclose(coefficients[:-1], 0.0, atol=1e-10)
    np.testing.assert_allclose(coefficients[-1], 7.3, atol=1e-10)


def test_day_mae_gives_equal_weight_to_dates_not_observation_counts():
    y = np.zeros(4)
    prediction = np.array([1.0, 1.0, 1.0, 10.0])
    days = np.array([42, 42, 42, 99])
    metrics = models.regression_metrics(y, prediction, days)
    assert metrics["mae"] == pytest.approx(3.25)
    assert metrics["rmse"] == pytest.approx(np.sqrt(25.75))
    assert metrics["day_mae"] == pytest.approx(5.5)


def test_price_metrics_use_exponential_log_return_reconstruction():
    close = np.array([100.0, 200.0])
    y = np.log(np.array([1.1, 0.5]))
    prediction = np.zeros(2)
    metrics = models.regression_metrics(y, prediction, np.array([0, 1]), close=close)
    assert metrics["price_mae"] == pytest.approx(55.0)
    assert metrics["price_rmse"] == pytest.approx(np.sqrt((10.0**2 + 100.0**2) / 2))


@pytest.mark.parametrize("X,y", [
    (np.ones(5), np.ones(5)),
    (np.ones((5, 2)), np.ones(4)),
    (np.ones((5, 2)), np.ones((5, 1))),
    (np.empty((0, 2)), np.empty(0)),
    (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones(2)),
    (np.ones((2, 2)), np.array([1.0, np.inf])),
])
def test_ridge_rejects_malformed_or_nonfinite_training_data(X, y):
    with pytest.raises(ValueError):
        models.fit_ridge(X, y, alpha=0.1)


@pytest.mark.parametrize("alpha", [-0.1, np.nan, np.inf])
def test_ridge_rejects_invalid_penalties(alpha):
    with pytest.raises(ValueError):
        models.fit_ridge(np.ones((3, 2)), np.ones(3), alpha=alpha)


@pytest.mark.parametrize("scale", [0.0, -0.1, np.nan, np.inf])
def test_neural_prediction_rejects_invalid_target_scale(scale):
    with pytest.raises(ValueError):
        models.predict_batches(models.make_model("compact"), np.ones((3, 15)), target_scale=scale)


def test_neural_prediction_rejects_nonfinite_features():
    X = np.ones((3, 15), dtype=np.float32)
    X[1, 4] = np.nan
    with pytest.raises(ValueError):
        models.predict_batches(models.make_model("compact"), X, target_scale=0.1)


@pytest.mark.parametrize("X,coefficients", [
    (np.ones((3, 2)), np.ones(2)),
    (np.ones((3, 2)), np.array([1.0, np.inf, 0.0])),
    (np.array([[1.0, np.nan]]), np.ones(3)),
])
def test_ridge_prediction_rejects_malformed_or_nonfinite_inputs(X, coefficients):
    with pytest.raises(ValueError):
        models.predict_ridge(X, coefficients)


@pytest.mark.parametrize("y,prediction,days,close", [
    (np.ones(3), np.ones(2), np.arange(3), None),
    (np.ones(3), np.ones(3), np.arange(2), None),
    (np.ones(3), np.array([0.0, np.nan, 1.0]), np.arange(3), None),
    (np.array([0.0, np.inf, 1.0]), np.ones(3), np.arange(3), None),
    (np.ones(3), np.ones(3), np.arange(3), np.ones(2)),
    (np.ones(3), np.ones(3), np.arange(3), np.array([1.0, 0.0, 2.0])),
    (np.empty(0), np.empty(0), np.empty(0), None),
])
def test_metrics_reject_invalid_or_unaligned_observations(y, prediction, days, close):
    with pytest.raises(ValueError):
        models.regression_metrics(y, prediction, days, close=close)


@pytest.mark.parametrize("batch_size", [0, -3])
def test_batch_predictors_and_fitting_reject_nonpositive_batch_sizes(batch_size):
    X = np.ones((3, 15), dtype=np.float32)
    with pytest.raises(ValueError):
        models.fit_ridge(X, np.ones(3), alpha=0.1, batch_size=batch_size)
    with pytest.raises(ValueError):
        models.predict_ridge(X, np.ones(16), batch_size=batch_size)
    with pytest.raises(ValueError):
        models.predict_batches(models.make_model("compact"), X, target_scale=0.1, batch_size=batch_size)

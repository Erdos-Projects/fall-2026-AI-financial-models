"""Architecture and training checks using synthetic data only."""

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


variants = _load_script("nifty_mlp_variants")
baseline = _load_script("nifty_models")


@pytest.fixture(autouse=True)
def single_threaded_torch():
    # Tiny synthetic matrix multiplies should not start a large CPU thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name,expected", [
    ("shallow", 1089), ("standard_plain", 3137), ("deep", 13569),
    ("batchnorm", 3329), ("residual", 17985),
])
def test_capacity_scalar_output_and_checkpoint_reload(name, expected, tmp_path):
    torch.manual_seed(15)
    model = variants.build_model(name)
    assert variants.parameter_count(model) == expected
    X = torch.randn(9, 15)
    # Exercise BatchNorm buffers before serializing a trained-state checkpoint.
    model(X)
    model.eval()
    output = model(X)
    assert output.shape == (9,)
    assert torch.isfinite(output).all()
    checkpoint = tmp_path / f"{name}.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = variants.build_model(name)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    restored.eval()
    torch.testing.assert_close(restored(X), output, rtol=0, atol=0)


@pytest.mark.parametrize("name", list(variants.CONFIGS))
def test_signed_head_zero_initialization_and_custom_input_size(name):
    model = variants.build_model(name, input_dim=3).eval()
    assert isinstance(model.head, torch.nn.Linear)
    with torch.no_grad():
        model.head.weight.zero_()
        for value in (-2.0, 0.0, 2.0):
            model.head.bias.fill_(value)
            torch.testing.assert_close(model(torch.ones(1, 3)), torch.tensor([value]))
    for malformed in (torch.ones(3), torch.ones(4, 15), torch.ones(1, 2, 3)):
        with pytest.raises(ValueError, match="input features"):
            model(malformed)


@pytest.mark.parametrize("name", list(variants.CONFIGS))
def test_prediction_helper_is_deterministic_and_restores_return_units(name):
    torch.manual_seed(3)
    model = variants.build_model(name)
    X = np.random.default_rng(3).normal(size=(19, 15)).astype(np.float32)
    model(torch.from_numpy(X))
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(X)).numpy() * 0.002
    model.train()
    first = baseline.predict_batches(model, X, target_scale=0.002, batch_size=4)
    second = baseline.predict_batches(model, X, target_scale=0.002, batch_size=1)
    np.testing.assert_allclose(first, expected, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(second, first, rtol=1e-5, atol=1e-9)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_batchnorm_uses_fixed_running_statistics_during_evaluation():
    model = variants.build_model("batchnorm")
    norms = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
    assert len(norms) == 2
    model(torch.randn(20, 15) + 3)
    buffers = [(layer.running_mean.clone(), layer.running_var.clone(),
                layer.num_batches_tracked.clone()) for layer in norms]
    model.eval()
    model(torch.randn(7, 15) - 8)
    for layer, (mean, variance, tracked) in zip(norms, buffers):
        torch.testing.assert_close(layer.running_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(layer.running_var, variance, rtol=0, atol=0)
        torch.testing.assert_close(layer.num_batches_tracked, tracked, rtol=0, atol=0)


def test_residual_skip_connection_preserves_gradient_when_branch_is_zero():
    block = variants.ResidualBlock(width=4, dropout=0.1).eval()
    with torch.no_grad():
        block.branch[-1].weight.zero_()
        block.branch[-1].bias.zero_()
    X = torch.tensor([[0.2, 0.4, 1.0, 2.0]], requires_grad=True)
    output = block(X)
    torch.testing.assert_close(output, X, rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(X.grad, torch.ones_like(X), rtol=0, atol=0)


@pytest.mark.parametrize("name", list(variants.CONFIGS))
def test_all_architectures_learn_a_signed_nonlinear_signal(name):
    torch.manual_seed(24)
    model = variants.build_model(name, input_dim=3)
    X = torch.randn(128, 3)
    y = 0.7 * X[:, 0] + 0.5 * torch.sin(X[:, 1]) - 0.3 * X[:, 2].square()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.eval()
    with torch.no_grad():
        initial_loss = torch.mean((model(X) - y).square()).item()
    model.train()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(X) - y).square())
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(X)
        final_loss = torch.mean((prediction - y).square()).item()
    assert final_loss < initial_loss * 0.25
    assert prediction.min() < 0 < prediction.max()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_configuration_serialization_and_invalid_architecture_inputs():
    before = json.dumps(variants.CONFIGS)
    for name in variants.CONFIGS:
        variants.build_model(name)
        with pytest.raises(ValueError):
            variants.build_model(name, input_dim=0)
    assert json.dumps(variants.CONFIGS) == before
    with pytest.raises(ValueError, match="Unknown architecture"):
        variants.build_model("missing")
    with pytest.raises(ValueError, match="normalization"):
        variants.SequentialMLP(3, [4], 0.1, normalization="missing")
    with pytest.raises(ValueError, match="residual blocks"):
        variants.ResidualMLP(3, 4, 0, 0.1)

"""Additional MLP architecture ablations for the existing Nifty features.

Every model consumes a two-dimensional feature batch and predicts one signed
scalar per row. The final linear layer is available as ``model.head`` so the
training runner can initialize each model to the same zero-return prediction.
The existing ``nifty_models.predict_batches`` helper restores target units.

Sequential blocks use Linear -> optional BatchNorm -> ReLU -> Dropout. Residual
blocks use x -> LayerNorm -> Linear -> ReLU -> Dropout -> Linear, followed by
ReLU(x + block(x)). These are small tabular adaptations, not reproductions of a
published benchmark's complete model or hyperparameter search.
"""

from __future__ import annotations

import torch
from torch import nn

# Keep values JSON-serializable: the experiment writes these into its manifest.
# Insertion order is the intended order in the architecture comparison.
CONFIGS = {
    "shallow": {
        "hidden": [64], "dropout": 0.1, "normalization": "none", "residual": False,
    },
    "standard_plain": {
        "hidden": [64, 32], "dropout": 0.0, "normalization": "none", "residual": False,
    },
    "deep": {
        "hidden": [64, 64, 64, 64], "dropout": 0.1,
        "normalization": "none", "residual": False,
    },
    "batchnorm": {
        "hidden": [64, 32], "dropout": 0.1,
        "normalization": "batchnorm", "residual": False,
    },
    "residual": {
        "width": 64, "blocks": 2, "dropout": 0.1,
        "normalization": "layernorm", "residual": True,
    },
}


def _check_input(x: torch.Tensor, input_dim: int) -> None:
    if x.ndim != 2 or x.shape[1] != input_dim:
        raise ValueError(f"Expected a batch with {input_dim} input features")


def _check_dimensions(input_dim: int, hidden: list[int], dropout: float) -> None:
    if not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("Input dimension must be a positive integer")
    if not hidden or any(not isinstance(width, int) or width < 1 for width in hidden):
        raise ValueError("Hidden dimensions must be positive integers")
    if not 0 <= dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")


class SequentialMLP(nn.Module):
    """Feed-forward MLP with optional normalization of hidden activations."""

    def __init__(
        self, input_dim: int, hidden: list[int], dropout: float,
        normalization: str = "none",
    ):
        super().__init__()
        _check_dimensions(input_dim, hidden, dropout)
        if normalization not in {"none", "batchnorm"}:
            raise ValueError(f"Unsupported sequential normalization: {normalization}")
        self.input_dim = input_dim
        layers = []
        width = input_dim
        for next_width in hidden:
            layers.append(nn.Linear(width, next_width))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(next_width))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            width = next_width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.input_dim)
        return self.head(self.features(x)).squeeze(-1)


class ResidualBlock(nn.Module):
    """Two-layer residual transformation with per-observation normalization."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        _check_dimensions(width, [width], dropout)
        self.branch = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.branch(x))


class ResidualMLP(nn.Module):
    """A ReLU stem, constant-width residual blocks, and a signed linear head."""

    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float):
        super().__init__()
        _check_dimensions(input_dim, [width], dropout)
        if not isinstance(blocks, int) or blocks < 1:
            raise ValueError("Number of residual blocks must be a positive integer")
        self.input_dim = input_dim
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(blocks)))
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.input_dim)
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


def build_model(name: str, input_dim: int = 15) -> nn.Module:
    """Construct an unfitted architecture without mutating its configuration."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown architecture: {name}")
    config = CONFIGS[name]
    if config["residual"]:
        return ResidualMLP(
            input_dim=input_dim, width=config["width"],
            blocks=config["blocks"], dropout=config["dropout"],
        )
    return SequentialMLP(
        input_dim=input_dim, hidden=config["hidden"], dropout=config["dropout"],
        normalization=config["normalization"],
    )


def parameter_count(model: nn.Module) -> int:
    """Count learned scalar parameters, excluding normalization buffers."""
    return sum(parameter.numel() for parameter in model.parameters())

"""Shared training/calibration search space for the fixed compact Nifty MLP.

Every search method (random, Bayesian, fixed hypotheses, LLM) proposes values
for the same eight parameters. Architecture, features, batch size, epoch budget
and optimizer family stay fixed. Continuous values are rounded to three
significant digits, so a configuration's canonical JSON fully determines
training and identifies duplicate proposals.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

# Insertion order is the order shown to proposers and stored in records.
PARAMETERS = {
    "loss": {
        "choices": ["smooth_l1", "l1", "mse"], "baseline": "smooth_l1",
        "meaning": "Training loss on the scaled target y / std(y_fit)",
    },
    "smooth_l1_beta": {
        "low": 0.01, "high": 10.0, "log": True, "baseline": 1.0,
        "meaning": "Smooth L1 transition point in scaled-target units; ignored unless loss is smooth_l1",
    },
    "learning_rate": {
        "low": 1e-4, "high": 1e-2, "log": True, "baseline": 1e-3,
        "meaning": "Initial AdamW learning rate",
    },
    "weight_decay": {
        "low": 1e-6, "high": 1e-1, "log": True, "baseline": 1e-4,
        "meaning": "AdamW decoupled weight decay",
    },
    "dropout": {
        "low": 0.0, "high": 0.5, "log": False, "baseline": 0.1,
        "meaning": "Dropout probability after each hidden ReLU layer",
    },
    "sample_weighting": {
        "choices": ["none", "date_balanced"], "baseline": "none",
        "meaning": "none weights fit rows equally; date_balanced gives every fit date equal total weight",
    },
    "low_price_weight": {
        "low": 0.01, "high": 1.0, "log": True, "baseline": 1.0,
        "meaning": "Multiplies the training weight of rows with current close <= 1; weights are renormalized to mean 1",
    },
    "calibration": {
        "choices": ["none", "global_scale", "price_group_scale"], "baseline": "none",
        "meaning": ("Post-training prediction multiplier alpha in [0, 1] fitted on the tune window to minimize "
                    "equal-date MAE; price_group_scale fits separate alphas for close <= 1 and close > 1"),
    },
}
CONTINUOUS = [name for name, spec in PARAMETERS.items() if "choices" not in spec]
PROPOSAL_TEXT_FIELDS = ["hypothesis", "change", "expected_effect", "rejection_criterion"]


def round_significant(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}g}")


class SearchConfig(BaseModel):
    """One complete, validated proposal. Unknown keys and missing keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: Literal["smooth_l1", "l1", "mse"]
    smooth_l1_beta: float
    learning_rate: float
    weight_decay: float
    dropout: float
    sample_weighting: Literal["none", "date_balanced"]
    low_price_weight: float
    calibration: Literal["none", "global_scale", "price_group_scale"]

    @field_validator(*CONTINUOUS, mode="before")
    @classmethod
    def _bounded(cls, value, info):
        spec = PARAMETERS[info.field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("must be a finite number")
        value = round_significant(float(value))
        if not spec["low"] <= value <= spec["high"]:
            raise ValueError(f"must be within [{spec['low']:g}, {spec['high']:g}]")
        return value

    def canonical(self) -> SearchConfig:
        """Reset parameters that cannot affect training, so equivalent proposals share a hash."""
        if self.loss == "smooth_l1":
            return self
        return self.model_copy(update={"smooth_l1_beta": PARAMETERS["smooth_l1_beta"]["baseline"]})

    def config_hash(self) -> str:
        text = json.dumps(self.canonical().model_dump(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


BASELINE = SearchConfig(**{name: spec["baseline"] for name, spec in PARAMETERS.items()})


def changed_parameters(config: SearchConfig, reference: SearchConfig = BASELINE) -> list[str]:
    left, right = config.canonical().model_dump(), reference.canonical().model_dump()
    return [name for name in PARAMETERS if left[name] != right[name]]


def sample_random(rng: np.random.Generator) -> SearchConfig:
    """Uniform over choices, uniform or log-uniform over continuous ranges."""
    values = {}
    for name, spec in PARAMETERS.items():
        if "choices" in spec:
            values[name] = spec["choices"][int(rng.integers(len(spec["choices"])))]
        elif spec["log"]:
            values[name] = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
        else:
            values[name] = float(rng.uniform(spec["low"], spec["high"]))
    return SearchConfig(**values).canonical()


def extract_reply_json(text: str) -> dict:
    """Return the first JSON object with a "proposals" key, ignoring prose or code fences around it."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "proposals" in value:
            return value
    raise ValueError('No JSON object containing a "proposals" list was found')


def _validate_proposal(item) -> dict:
    record = {"proposal": item, "config": None, "error": None}
    try:
        if not isinstance(item, dict):
            raise TypeError("Each proposal must be a JSON object")
        missing = [field for field in [*PROPOSAL_TEXT_FIELDS, "predicted_change_pct", "config"] if field not in item]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        for field in PROPOSAL_TEXT_FIELDS:
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f'"{field}" must be a nonempty string')
        predicted = item["predicted_change_pct"]
        if isinstance(predicted, bool) or not isinstance(predicted, (int, float)) or not math.isfinite(predicted):
            raise ValueError('"predicted_change_pct" must be a finite number')
        record["config"] = SearchConfig.model_validate(item["config"]).canonical()
    except ValidationError as error:
        # Short field-level messages read better in a repair prompt than pydantic's default text.
        messages = []
        for detail in error.errors():
            label = ".".join(["config", *map(str, detail["loc"])])
            message = f"{label}: {detail['msg'].removeprefix('Value error, ')}"
            messages.append(message if detail["type"] == "missing" else f"{message} (got {detail['input']!r})")
        record["error"] = "; ".join(messages)
    except (TypeError, ValueError) as error:
        record["error"] = str(error)
    return record


def parse_proposals(text: str, expected: int) -> tuple[list[dict], list[str]]:
    """Return exactly ``expected`` per-proposal records plus reply-level problems.

    Records carry a validated canonical config or an error. Surplus proposals are
    ignored and missing ones become errors; both are also reported as problems.
    """
    try:
        proposals = extract_reply_json(text)["proposals"]
        if not isinstance(proposals, list):
            raise TypeError('"proposals" must be a JSON list')
    except (TypeError, ValueError) as error:
        return [{"proposal": None, "config": None, "error": f"Unusable reply: {error}"}] * expected, [str(error)]
    problems = [] if len(proposals) == expected else [
        f"Expected exactly {expected} proposal(s); received {len(proposals)}"]
    records = [_validate_proposal(item) for item in proposals[:expected]]
    records += [{"proposal": None, "config": None, "error": "Missing proposal"}] * (expected - len(records))
    return records, problems

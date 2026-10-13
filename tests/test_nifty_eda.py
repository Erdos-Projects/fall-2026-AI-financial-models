"""Small regression checks for the consequential parsing/statistics choices."""

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("nifty_eda", Path(__file__).resolve().parents[1] / "scripts/nifty_eda.py")
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


@pytest.mark.parametrize("symbol, expected", [
    ("CE 8700", ("CE", 8700.0, None)),
    ("PE 11900", ("PE", 11900.0, None)),
    ("NIFTY13000PE", ("PE", 13000.0, None)),
    ("NIFTY25JUN2010000CE", ("CE", 10000.0, "2020-06-25")),
    ("NIFTY25Jun209000PE", ("PE", 9000.0, "2020-06-25")),
])
def test_symbol_dates_do_not_become_strikes(symbol, expected):
    assert eda.symbol_parts(symbol) == expected


def test_unknown_symbol_is_not_assigned_a_contract():
    side, strike, expiry = eda.symbol_parts("unrecognized")
    assert side == "UNKNOWN"
    assert np.isnan(strike)
    assert expiry is None


@pytest.mark.parametrize("values", [[1.0], [0.0, 0.0, 1.0, 10.0], [-3.0, 2.0, 2.0, 2.0, 7.0, 100.0]])
def test_frequency_quantiles_equal_explicit_observation_quantiles(values):
    result = eda.weighted_stats(Counter(values))
    for label, probability in [("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p99", .99)]:
        assert result[label] == pytest.approx(np.quantile(values, probability, method="linear"))
    assert result["mean"] == pytest.approx(np.mean(values))
    if len(values) > 1:
        assert result["std"] == pytest.approx(np.std(values, ddof=1))

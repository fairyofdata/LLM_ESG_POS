"""Tests for the Black-Litterman optimizer with synthetic market data."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.black_litterman import (
    build_portfolio_table,
    optimize_portfolio,
)

TICKERS = ["000001", "000002", "000003", "000004", "000005"]
ESG_WEIGHTS = {"environmental": 0.8, "social": 0.5, "governance": 0.3}


@pytest.fixture()
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(300, len(TICKERS)))
    index = pd.date_range("2023-01-01", periods=300, freq="B")
    return pd.DataFrame(data, index=index, columns=TICKERS)


@pytest.fixture()
def synthetic_esg_scores() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "Company": [f"Company {i}" for i in range(len(TICKERS))],
            "environmental": rng.uniform(1, 5, len(TICKERS)),
            "social": rng.uniform(1, 5, len(TICKERS)),
            "governance": rng.uniform(1, 5, len(TICKERS)),
        },
        index=pd.Index(TICKERS, name="ticker"),
    )


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_weights_are_a_valid_allocation(synthetic_returns, synthetic_esg_scores, tau):
    weights, performance = optimize_portfolio(
        synthetic_returns, synthetic_esg_scores, ESG_WEIGHTS, tau=tau)

    assert set(weights) == set(TICKERS)
    values = np.array(list(weights.values()))
    assert values.sum() == pytest.approx(1.0, abs=1e-6)
    assert (values >= -1e-9).all()
    assert (values <= 1.0 + 1e-9).all()

    assert np.isfinite(performance.expected_return)
    assert performance.volatility > 0
    assert np.isfinite(performance.sharpe_ratio)


def test_missing_overlap_raises(synthetic_returns, synthetic_esg_scores):
    disjoint = synthetic_esg_scores.copy()
    disjoint.index = [f"99999{i}" for i in range(len(disjoint))]
    with pytest.raises(ValueError):
        optimize_portfolio(synthetic_returns, disjoint, ESG_WEIGHTS, tau=1.0)


def test_portfolio_table_structure(synthetic_returns, synthetic_esg_scores):
    weights, _ = optimize_portfolio(
        synthetic_returns, synthetic_esg_scores, ESG_WEIGHTS, tau=1.0)
    table = build_portfolio_table(weights, synthetic_esg_scores)

    assert list(table.columns) == [
        "ticker", "Company", "Weight", "environmental", "social", "governance"]
    assert table["Weight"].sum() == pytest.approx(100.0, abs=1e-4)
    assert table["Weight"].is_monotonic_decreasing

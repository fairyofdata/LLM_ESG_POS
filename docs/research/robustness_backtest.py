"""Robustness checks for the LEPOS backtesting study.

Extends the original benchmark comparison (see RESEARCH.md) with two checks a
reviewer would ask for:

1. **Transaction costs** — annual rebalancing trades are charged realistic
   Korean-market costs (5 bp commission per side + 18 bp securities
   transaction tax on sells), applied to the turnover between the drifted
   previous-year weights and the new target weights.
2. **ESG ablation** — a "financial-only" max-Sharpe portfolio optimized on
   the same universe, the same trailing price history, and the same weight
   cap, but without any ESG views. Comparing it against the ESG-tilted
   portfolio isolates the contribution of the text-derived ESG signal.

Run from the repository root::

    python docs/research/robustness_backtest.py

Inputs (this directory): ``krx_5yr_prices.csv``,
``portfolio_weights_2020..2024.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

RESEARCH_DIR = Path(__file__).resolve().parent
YEARS = (2020, 2021, 2022, 2023, 2024)
ESG_COLUMN = "Optimized ESG (tau=0.5)"

BUY_COST = 0.0005   # 5 bp commission
SELL_COST = 0.0023  # 5 bp commission + 18 bp securities transaction tax
TRADING_DAYS = 252
#: The published ESG weight tables carry no explicit per-asset cap (single
#: positions reach ~60%), so the ablation is run under the same constraint.
WEIGHT_CAP = 1.0


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(RESEARCH_DIR / "krx_5yr_prices.csv",
                         index_col=0, parse_dates=True)
    return prices.sort_index()


def load_weights(year: int) -> pd.DataFrame:
    weights = pd.read_csv(RESEARCH_DIR / f"portfolio_weights_{year}.csv",
                          index_col=0)
    weights.index = weights.index.astype(str).str.zfill(6)
    return weights


def yearly_weight_map(column: str) -> dict[int, pd.Series]:
    """Per-year target weights for one strategy column, normalized to 1."""
    result = {}
    for year in YEARS:
        weights = load_weights(year)[column].fillna(0.0).clip(lower=0.0)
        result[year] = weights / weights.sum()
    return result


def financial_only_weights(prices: pd.DataFrame) -> dict[int, pd.Series]:
    """Max-Sharpe weights per year using only trailing market data.

    Same universe, cap, and long-only constraints as the ESG strategy, but no
    ESG views: expected returns are trailing annualized means and the
    covariance uses Ledoit-Wolf shrinkage.
    """
    result = {}
    returns = prices.pct_change()
    for year in YEARS:
        universe = [t for t in load_weights(year).index if t in prices.columns]
        history = returns.loc[: f"{year - 1}-12-31", universe].dropna()
        mu = history.mean().values * TRADING_DAYS
        cov = LedoitWolf().fit(history).covariance_
        cov += np.eye(len(universe)) * 1e-8

        def negative_sharpe(w: np.ndarray) -> float:
            return -(w @ mu) / np.sqrt(w @ cov @ w)

        n = len(universe)
        solution = minimize(
            negative_sharpe,
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(0.0, WEIGHT_CAP)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"maxiter": 500},
        )
        weights = pd.Series(np.clip(solution.x, 0, None), index=universe)
        result[year] = weights / weights.sum()
    return result


def backtest(
    prices: pd.DataFrame,
    weights_by_year: dict[int, pd.Series],
    with_costs: bool,
) -> pd.Series:
    """Daily strategy returns 2020-2024 with annual rebalancing.

    Within a year the portfolio follows the target weights (constant-mix, as
    in the original study). When ``with_costs`` is set, each rebalance is
    charged trading costs on the turnover versus the drifted previous-year
    weights.
    """
    daily_returns = prices.pct_change()
    parts: list[pd.Series] = []
    previous_drifted: pd.Series | None = None

    for year in YEARS:
        target = weights_by_year[year]
        tickers = [t for t in target.index if t in prices.columns]
        target = target[tickers] / target[tickers].sum()

        year_returns = daily_returns.loc[f"{year}-01-01": f"{year}-12-31",
                                         tickers].fillna(0.0)
        portfolio = year_returns @ target

        if with_costs:
            if previous_drifted is None:
                trades = target  # initial buy-in
            else:
                trades = target.subtract(previous_drifted, fill_value=0.0)
            buys = trades[trades > 0].sum()
            sells = -trades[trades < 0].sum()
            cost = buys * BUY_COST + sells * SELL_COST
            if not portfolio.empty:
                portfolio.iloc[0] -= cost

        # Drift the target weights through the year for next year's turnover.
        gross = (1.0 + year_returns).prod()
        drifted = target * gross
        previous_drifted = drifted / drifted.sum()

        parts.append(portfolio)

    return pd.concat(parts)


def metrics(daily: pd.Series) -> dict[str, float]:
    daily = daily.dropna()
    cumulative = float((1.0 + daily).prod())
    n_years = len(daily) / TRADING_DAYS
    cagr = cumulative ** (1.0 / n_years) - 1.0
    volatility = float(daily.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(daily.mean() * TRADING_DAYS / volatility)
    curve = (1.0 + daily).cumprod()
    drawdown = 1.0 - curve / curve.cummax()
    max_drawdown = float(drawdown.max())
    calmar = cagr / max_drawdown if max_drawdown > 0 else float("nan")
    return {
        "5y Cumulative Return": cumulative,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
    }


def main() -> None:
    prices = load_prices()
    window = prices.loc["2020-01-01":"2024-12-31"]

    rows: dict[str, dict[str, float]] = {}

    for benchmark in ("KOSPI", "ESG_ETF"):
        rows[benchmark.replace("_", " ")] = metrics(window[benchmark].pct_change())

    equal = yearly_weight_map("Equal Weighted")
    esg = yearly_weight_map(ESG_COLUMN)
    financial = financial_only_weights(prices)

    rows["Equal Weighted"] = metrics(backtest(prices, equal, with_costs=False))
    rows["Financial-only (no ESG, w/ costs)"] = metrics(
        backtest(prices, financial, with_costs=True))
    rows["Optimized ESG (gross)"] = metrics(backtest(prices, esg, with_costs=False))
    rows["Optimized ESG (w/ costs)"] = metrics(backtest(prices, esg, with_costs=True))

    table = pd.DataFrame(rows).T
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string())
    print("\nMarkdown:\n")
    print(table.round(4).to_markdown())


if __name__ == "__main__":
    main()

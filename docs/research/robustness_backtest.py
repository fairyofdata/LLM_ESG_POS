"""Reproduction and transaction-cost check for the LEPOS backtesting study.

Rebuilds the portfolio return series from the published yearly weight tables
and the KRX price snapshot in this directory, in three ways:

1. **Published method (validation)** — constant-mix within each year (the
   portfolio is held at its target weights every day) with no costs. This is
   how the original study computed returns; matching its published numbers
   validates the data in this directory.
2. **Buy-and-hold, gross** — weights are set once per year and then drift
   with prices until the next annual rebalance, which is what an annually
   rebalanced portfolio actually experiences.
3. **Buy-and-hold, net of costs** — as (2), charging realistic Korean-market
   costs on every rebalance: 5 bp commission per side plus the 18 bp
   securities transaction tax on sells, applied to the turnover between the
   drifted previous-year weights and the new targets.

Run from the repository root::

    python docs/research/robustness_backtest.py

Inputs (this directory): ``krx_5yr_prices.csv``,
``portfolio_weights_2020..2024.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent
YEARS = (2020, 2021, 2022, 2023, 2024)
ESG_COLUMN = "Optimized ESG (tau=0.5)"

BUY_COST = 0.0005   # 5 bp commission
SELL_COST = 0.0023  # 5 bp commission + 18 bp securities transaction tax
TRADING_DAYS = 252


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(RESEARCH_DIR / "krx_5yr_prices.csv",
                         index_col=0, parse_dates=True)
    return prices.sort_index()


def yearly_weight_map(column: str) -> dict[int, pd.Series]:
    """Per-year target weights for one strategy column, normalized to 1."""
    result = {}
    for year in YEARS:
        table = pd.read_csv(RESEARCH_DIR / f"portfolio_weights_{year}.csv",
                            index_col=0)
        table.index = table.index.astype(str).str.zfill(6)
        weights = table[column].fillna(0.0).clip(lower=0.0)
        result[year] = weights / weights.sum()
    return result


def backtest(
    prices: pd.DataFrame,
    weights_by_year: dict[int, pd.Series],
    method: str,
) -> pd.Series:
    """Daily strategy returns for 2020-2024 with annual rebalancing.

    Args:
        prices: Daily close prices (columns = tickers).
        weights_by_year: Target weights applied at the start of each year.
        method: ``"constant_mix"`` (published method), ``"buy_hold"``, or
            ``"buy_hold_net"`` (buy-and-hold with trading costs).
    """
    daily_returns = prices.pct_change()
    parts: list[pd.Series] = []
    previous_drifted: pd.Series | None = None

    for year in YEARS:
        target = weights_by_year[year]
        tickers = [t for t in target.index if t in prices.columns]
        target = target[tickers] / target[tickers].sum()
        year_returns = daily_returns.loc[f"{year}-01-01":f"{year}-12-31",
                                         tickers].fillna(0.0)
        growth = (1.0 + year_returns).cumprod()

        if method == "constant_mix":
            portfolio = year_returns @ target
        else:
            value = growth @ target
            if method == "buy_hold_net":
                if previous_drifted is None:
                    trades = target
                else:
                    trades = target.subtract(previous_drifted, fill_value=0.0)
                cost = (trades[trades > 0].sum() * BUY_COST
                        + -trades[trades < 0].sum() * SELL_COST)
                value = value * (1.0 - cost)
            portfolio = value.pct_change()
            portfolio.iloc[0] = value.iloc[0] - 1.0

        drifted = target * growth.iloc[-1]
        previous_drifted = drifted / drifted.sum()
        parts.append(portfolio)

    return pd.concat(parts)


def metrics(daily: pd.Series) -> dict[str, float]:
    daily = daily.dropna()
    cumulative = float((1.0 + daily).prod())
    cagr = cumulative ** (TRADING_DAYS / len(daily)) - 1.0
    volatility = float(daily.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(daily.mean() * TRADING_DAYS / volatility)
    curve = (1.0 + daily).cumprod()
    max_drawdown = float((1.0 - curve / curve.cummax()).max())
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
    benchmark_returns = prices[["KOSPI", "ESG_ETF"]].pct_change().loc["2020-01-01":"2024-12-31"]
    equal = yearly_weight_map("Equal Weighted")
    esg = yearly_weight_map(ESG_COLUMN)

    rows = {
        "KOSPI": metrics(benchmark_returns["KOSPI"]),
        "ESG ETF": metrics(benchmark_returns["ESG_ETF"]),
        "Equal-weighted (published method)": metrics(backtest(prices, equal, "constant_mix")),
        "Equal-weighted (buy-and-hold, net)": metrics(backtest(prices, equal, "buy_hold_net")),
        "LEPOS tau=0.5 (published method)": metrics(backtest(prices, esg, "constant_mix")),
        "LEPOS tau=0.5 (buy-and-hold, gross)": metrics(backtest(prices, esg, "buy_hold")),
        "LEPOS tau=0.5 (buy-and-hold, net)": metrics(backtest(prices, esg, "buy_hold_net")),
    }
    table = pd.DataFrame(rows).T
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string())


if __name__ == "__main__":
    main()

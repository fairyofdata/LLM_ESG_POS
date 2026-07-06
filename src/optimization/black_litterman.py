"""Black-Litterman portfolio optimization with ESG views.

The user's ESG preferences enter the model as investor views: the per-company
E/S/G component scores form the pick matrix ``P`` (3 x N), the preference-
weighted scores form the view vector ``Q``, and ``tau`` (derived from the
user's investment style) controls how strongly the views tilt the
market-implied returns. Expected returns are estimated from historical means
and the covariance matrix with Ledoit-Wolf shrinkage for numerical stability.
The final weights maximize the Sharpe ratio under long-only constraints.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

#: Trading days per year, used to annualize mean daily returns.
TRADING_DAYS_PER_YEAR = 252

#: View uncertainty assigned to each ESG pillar view (Omega diagonal).
VIEW_UNCERTAINTY = 0.002

PILLAR_COLUMNS = ("environmental", "social", "governance")


@dataclass(frozen=True)
class PortfolioPerformance:
    """Annualized performance metrics of an optimized portfolio."""

    expected_return: float
    volatility: float
    sharpe_ratio: float


def optimize_portfolio(
    returns: pd.DataFrame,
    esg_scores: pd.DataFrame,
    esg_weights: dict[str, float],
    tau: float,
) -> tuple[dict[str, float], PortfolioPerformance]:
    """Compute Sharpe-optimal weights with ESG-adjusted expected returns.

    Args:
        returns: Daily return frame (columns = tickers).
        esg_scores: Frame indexed by ticker with the pillar component columns
            (``environmental``/``social``/``governance``).
        esg_weights: User preference weight per pillar, e.g.
            ``{"environmental": 0.8, "social": 0.5, "governance": 0.3}``.
        tau: Strength of the ESG views relative to market data.

    Returns:
        ``(weights, performance)`` where ``weights`` maps each usable ticker
        to its optimal allocation (sums to 1).

    Raises:
        ValueError: If ``returns`` and ``esg_scores`` share no tickers.
    """
    valid_tickers = [t for t in esg_scores.index if t in returns.columns]
    if not valid_tickers:
        raise ValueError("no overlapping tickers between ESG data and return data")

    esg_preference = np.array([
        esg_weights["environmental"],
        esg_weights["social"],
        esg_weights["governance"],
    ]).reshape(-1, 1)

    # Investor views: P (3, n) picks each stock's pillar scores; Q holds each
    # stock's preference-blended ESG score (length n), min-max scaled into
    # [0, 0.1] so views live on the same order of magnitude as annual returns
    # (scaling validated in the backtesting study, see docs/research).
    p_matrix = esg_scores.loc[valid_tickers, list(PILLAR_COLUMNS)].values.T
    q_vector = (p_matrix.T @ esg_preference).flatten()
    q_range = q_vector.max() - q_vector.min()
    if q_range > 0:
        q_vector = (q_vector - q_vector.min()) / q_range * 0.1

    valid_returns = returns[valid_tickers].dropna()
    cov_matrix = LedoitWolf().fit(valid_returns).covariance_
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # regularization
    mean_returns = valid_returns.mean() * TRADING_DAYS_PER_YEAR

    omega_inv = np.linalg.inv(np.diag(np.full(p_matrix.shape[0], VIEW_UNCERTAINTY)))

    # Black-Litterman posterior expected returns.
    inv_term = np.linalg.inv(tau * cov_matrix + p_matrix.T @ omega_inv @ p_matrix)
    adjustment = (
        tau * cov_matrix @ mean_returns.values.reshape(-1, 1)
        + p_matrix.T @ (omega_inv @ p_matrix @ q_vector.reshape(-1, 1))
    )
    adjusted_returns = (inv_term @ adjustment).flatten()

    def negative_sharpe(weights: np.ndarray) -> float:
        port_return = weights @ adjusted_returns
        port_volatility = np.sqrt(weights @ cov_matrix @ weights)
        return -port_return / port_volatility

    n_assets = len(valid_tickers)
    result = minimize(
        negative_sharpe,
        np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )

    optimal = np.clip(result.x, 0.0, None)
    optimal /= optimal.sum()
    expected_return = float(optimal @ adjusted_returns)
    # Annualize the daily covariance so volatility matches the annualized
    # expected return shown in the dashboard.
    volatility = float(np.sqrt(optimal @ cov_matrix @ optimal * TRADING_DAYS_PER_YEAR))
    sharpe = expected_return / volatility if volatility > 0 else 0.0

    weights = dict(zip(valid_tickers, optimal.tolist()))
    return weights, PortfolioPerformance(expected_return, volatility, sharpe)


def build_portfolio_table(
    weights: dict[str, float], esg_scores: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the recommendation table shown in the dashboard.

    Args:
        weights: Ticker-to-weight mapping from :func:`optimize_portfolio`.
        esg_scores: Frame indexed by ticker with ``Company`` and pillar
            component columns.

    Returns:
        Frame with ``ticker``/``Company``/``Weight`` (percent) and pillar
        scores, sorted by weight descending.
    """
    rows = []
    for ticker, weight in weights.items():
        if ticker not in esg_scores.index:
            continue
        record = esg_scores.loc[ticker]
        rows.append({
            "ticker": ticker,
            "Company": record.get("Company", "Unknown"),
            "Weight": weight * 100,
            "environmental": record["environmental"],
            "social": record["social"],
            "governance": record["governance"],
        })
    table = pd.DataFrame(
        rows,
        columns=["ticker", "Company", "Weight",
                 "environmental", "social", "governance"],
    )
    return table.sort_values(by="Weight", ascending=False).reset_index(drop=True)

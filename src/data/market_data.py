"""Market data access (prices and index levels) via FinanceDataReader."""

from datetime import date, timedelta

import FinanceDataReader as fdr
import pandas as pd


def load_index_change(index_code: str) -> tuple[float, float] | None:
    """Return the latest level and day-over-day change of a market index.

    Args:
        index_code: FinanceDataReader index code, e.g. ``"KS11"`` (KOSPI)
            or ``"KQ11"`` (KOSDAQ).

    Returns:
        ``(latest_close, change_percent)`` or ``None`` when no data is
        available (e.g. network failure or market holiday with no history).
    """
    today = date.today()
    data = fdr.DataReader(index_code, today - timedelta(days=7), today)
    if data.empty:
        return None
    previous_close = float(data.iloc[0]["Close"])
    latest_close = float(data.iloc[-1]["Close"])
    change_percent = (latest_close - previous_close) / previous_close * 100
    return latest_close, change_percent


def load_close_prices(tickers: list[str], years: int = 5) -> pd.DataFrame:
    """Download daily close prices for ``tickers`` over the last ``years``.

    Tickers that fail to download (delisted, invalid code, ...) are skipped.

    Args:
        tickers: KRX ticker codes (six digits).
        years: Length of the look-back window in years.

    Returns:
        Frame indexed by date with one close-price column per ticker.
    """
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=years * 365)
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            prices[ticker] = fdr.DataReader(ticker, start_date, end_date)["Close"]
        except Exception:
            continue
    return pd.DataFrame(prices)


def load_ohlcv(code: str, ndays: int, frequency: str = "D") -> pd.DataFrame:
    """Load OHLCV history for a single stock.

    Args:
        code: KRX ticker code.
        ndays: Number of calendar days of history.
        frequency: ``"D"`` for daily data or ``"M"`` for monthly candles.

    Returns:
        OHLCV frame suitable for candlestick plotting.
    """
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=ndays)
    data = fdr.DataReader(code, start_date, end_date)
    if frequency == "M":
        data = (
            data.resample("M")
            .agg({"Open": "first", "High": "max", "Low": "min",
                  "Close": "last", "Volume": "sum"})
            .dropna()
        )
    return data

"""Loading and preprocessing of the processed ESG score tables.

The CSV files in ``data/processed`` hold one row per company with the raw
scores published by five rating agencies. This module normalizes the Korean
column names, derives per-pillar (E/S/G) component scores, and builds the
integrated multi-year frame used by the dashboard.
"""

import unicodedata

import pandas as pd

from src.config import (
    COMPANY_PROFILES_FILE,
    ESG_SCORE_YEARS,
    RATING_AGENCIES,
    WORD_FREQUENCIES_FILE,
    esg_scores_file,
)

#: How much of each agency's headline score is attributed to each ESG pillar.
AGENCY_PILLAR_WEIGHTS: dict[str, dict[str, float]] = {
    "MSCI": {"environmental": 0.4, "social": 0.4, "governance": 0.2},
    "S&P": {"environmental": 0.3, "social": 0.4, "governance": 0.3},
    "Sustainalytics": {"environmental": 0.3, "social": 0.3, "governance": 0.4},
    "ISS": {"environmental": 0.35, "social": 0.35, "governance": 0.3},
    "ESG기준원": {"environmental": 0.33, "social": 0.33, "governance": 0.34},
}

_COLUMN_RENAMES = {"연도": "Year", "기업명": "Company"}


def _read_csv_any_encoding(path) -> pd.DataFrame:
    """Read a CSV trying UTF-8 first, then common Korean encodings."""
    for encoding in ("utf-8-sig", "euc-kr", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"could not decode {path}")


def calculate_esg_components(esg_df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-pillar E/S/G component scores from agency scores.

    Each agency's headline score is split across the three pillars using
    :data:`AGENCY_PILLAR_WEIGHTS` and averaged over the agencies present.

    Args:
        esg_df: Frame containing one or more agency score columns.

    Returns:
        Copy of ``esg_df`` with ``environmental``, ``social`` and
        ``governance`` columns added (unchanged if no agency column exists).
    """
    df = esg_df.copy()
    available = [agency for agency in RATING_AGENCIES if agency in df.columns]
    if not available:
        return df

    df["environmental"] = 0.0
    df["social"] = 0.0
    df["governance"] = 0.0
    for agency in available:
        weights = AGENCY_PILLAR_WEIGHTS[agency]
        df["environmental"] += df[agency] * weights["environmental"]
        df["social"] += df[agency] * weights["social"]
        df["governance"] += df[agency] * weights["governance"]

    df[["environmental", "social", "governance"]] /= len(available)
    return df


def load_esg_scores(years: tuple[int, ...] = ESG_SCORE_YEARS) -> dict[int, pd.DataFrame]:
    """Load the per-year ESG score tables keyed by year.

    Tickers are zero-padded to six digits and used as the index; the Korean
    ``연도``/``기업명`` columns are renamed to ``Year``/``Company`` and the
    E/S/G component columns are added.

    Args:
        years: Years to load; missing files yield empty frames.

    Returns:
        Mapping of year to its (possibly empty) score frame.
    """
    esg_data: dict[int, pd.DataFrame] = {}
    for year in years:
        path = esg_scores_file(year)
        if not path.exists():
            esg_data[year] = pd.DataFrame()
            continue
        df = _read_csv_any_encoding(path)
        df = df.rename(columns=_COLUMN_RENAMES)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.zfill(6)
            df = df.set_index("ticker")
        esg_data[year] = calculate_esg_components(df)
    return esg_data


def build_integrated_esg_data(esg_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the per-year frames into one multi-year frame.

    Args:
        esg_data: Output of :func:`load_esg_scores`.

    Returns:
        Single frame with ``Year``/``Company``/component columns and the
        ticker restored as a regular column; empty when no data is loaded.
    """
    frames = []
    for year, df in esg_data.items():
        if df.empty:
            continue
        frame = df.reset_index()
        if "Year" not in frame.columns:
            frame["Year"] = year
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_company_profiles() -> pd.DataFrame:
    """Load company profile data (name, description, sector, ...).

    The source file prefixes tickers with the exchange letter (``A005930``);
    the prefix is stripped so tickers align with the ESG score tables.
    """
    df = _read_csv_any_encoding(COMPANY_PROFILES_FILE)
    df.columns = df.columns.astype(str).str.strip()
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str[1:]
    return df


def normalize_company_name(name: str) -> str:
    """Return ``name`` in NFC form with surrounding whitespace removed."""
    return unicodedata.normalize("NFC", str(name)).strip()


def load_word_frequencies() -> pd.DataFrame:
    """Load per-company word frequencies used for the word cloud.

    Company names are unicode-normalized so lookups match the score tables.
    """
    if not WORD_FREQUENCIES_FILE.exists():
        return pd.DataFrame(columns=["word", "frequency", "company"])
    df = pd.read_csv(WORD_FREQUENCIES_FILE, encoding="utf-8-sig")
    df["company"] = df["company"].map(normalize_company_name)
    return df

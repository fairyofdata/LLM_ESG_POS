"""Central path and constant configuration for LEPOS.

Every module resolves data files through this module so the project can be
run from any working directory.
"""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
DUMMY_DATA_DIR: Final[Path] = DATA_DIR / "dummy"
USER_DATA_DIR: Final[Path] = DATA_DIR / "user"

COMPANY_PROFILES_FILE: Final[Path] = PROCESSED_DATA_DIR / "company_profiles.csv"
WORD_FREQUENCIES_FILE: Final[Path] = PROCESSED_DATA_DIR / "company_word_frequencies.csv"

#: Years for which agency ESG score tables exist in ``data/processed``.
ESG_SCORE_YEARS: Final[tuple[int, ...]] = (2019, 2020, 2021, 2022, 2023)

#: ESG rating agencies whose scores feed the scoring and optimization pipeline.
RATING_AGENCIES: Final[tuple[str, ...]] = (
    "MSCI",
    "S&P",
    "Sustainalytics",
    "ISS",
    "ESG기준원",
)


def esg_scores_file(year: int) -> Path:
    """Return the path of the processed ESG score table for ``year``."""
    return PROCESSED_DATA_DIR / f"esg_scores_{year}.csv"

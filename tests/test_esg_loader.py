"""Tests for ESG data loading against the committed processed data."""

from src.config import ESG_SCORE_YEARS
from src.data.esg_loader import (
    build_integrated_esg_data,
    load_company_profiles,
    load_esg_scores,
    load_word_frequencies,
)

PILLAR_COLUMNS = ["environmental", "social", "governance"]


def test_scores_load_for_every_year():
    esg_data = load_esg_scores()
    assert set(esg_data) == set(ESG_SCORE_YEARS)
    for year, df in esg_data.items():
        assert not df.empty, f"no data loaded for {year}"
        assert {"Year", "Company", *PILLAR_COLUMNS} <= set(df.columns)
        # Tickers are zero-padded six-digit strings usable with KRX APIs.
        assert df.index.str.fullmatch(r"\d{6}").all()


def test_pillar_components_are_positive():
    esg_data = load_esg_scores()
    for df in esg_data.values():
        assert (df[PILLAR_COLUMNS] > 0).all().all()


def test_integrated_frame_covers_all_years():
    integrated = build_integrated_esg_data(load_esg_scores())
    assert sorted(integrated["Year"].unique()) == sorted(ESG_SCORE_YEARS)
    assert len(integrated) > 300  # 5 years x ~64 companies


def test_company_profiles_ticker_format():
    profiles = load_company_profiles()
    assert "종목명" in profiles.columns
    assert "종목설명" in profiles.columns
    assert profiles["ticker"].str.fullmatch(r"\d{6}").all()


def test_word_frequencies_schema():
    frequencies = load_word_frequencies()
    assert {"word", "frequency", "company"} <= set(frequencies.columns)
    assert len(frequencies) > 1000

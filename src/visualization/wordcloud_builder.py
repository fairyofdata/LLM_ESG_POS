"""Portfolio-weighted word cloud generation.

Word frequencies were precomputed per company from five years of news
articles (see ``notebooks/04_data_engineering``). At runtime the per-company
frequencies are blended with the portfolio weights so dominant holdings
contribute more words.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from wordcloud import WordCloud

from src.data.esg_loader import normalize_company_name

#: Candidate fonts with Korean glyph coverage, in priority order.
_KOREAN_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/malgun.ttf"),                              # Windows
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),          # Debian/Ubuntu
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),               # macOS
)


def find_korean_font() -> str | None:
    """Return the first available Korean font path, or ``None``."""
    for candidate in _KOREAN_FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def build_blended_frequencies(
    portfolio: pd.DataFrame, word_frequencies: pd.DataFrame
) -> Counter:
    """Blend per-company word frequencies by portfolio weight.

    Args:
        portfolio: Frame with ``Company`` and ``Weight`` columns.
        word_frequencies: Frame with ``company``/``word``/``frequency``.

    Returns:
        Counter mapping words to weight-scaled frequencies.
    """
    blended: Counter = Counter()
    for _, row in portfolio.iterrows():
        company_name = normalize_company_name(row["Company"])
        company_words = word_frequencies[word_frequencies["company"] == company_name]
        for _, word_row in company_words.iterrows():
            blended[word_row["word"]] += word_row["frequency"] * row["Weight"]
    return blended


def generate_wordcloud(frequencies: Counter) -> WordCloud | None:
    """Render a word cloud image from blended frequencies.

    Returns:
        The generated :class:`WordCloud`, or ``None`` when there is nothing
        to draw or no Korean-capable font is installed.
    """
    if not frequencies:
        return None
    font_path = find_korean_font()
    if font_path is None:
        return None
    return WordCloud(
        font_path=font_path,
        background_color="white",
        width=800,
        height=600,
    ).generate_from_frequencies(frequencies)

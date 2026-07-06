"""LEPOS Streamlit entrypoint.

Run with::

    streamlit run app/main.py

Defines the multipage navigation; each page renders UI only and delegates all
data processing to the ``src`` package.
"""

import sys
from pathlib import Path

# Make the project root importable so pages can use `src.*` and `app.*`.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="ESG 정보 제공 플랫폼",
    page_icon=":earth_africa:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

navigation = st.navigation([
    st.Page("pages/home.py", title="홈", icon="🎯", default=True),
    st.Page("pages/survey.py", title="설문", icon="📋"),
    st.Page("pages/portfolio_result.py", title="설문 결과", icon="📊"),
    st.Page("pages/recent_news.py", title="최신 뉴스", icon="🆕"),
    st.Page("pages/esg_introduction.py", title="ESG 소개 / 투자 방법", icon="🧩"),
])
navigation.run()

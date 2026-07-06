"""ESG preference survey page (15 preference questions + investment style).

Questions and the scoring matrix live in :mod:`src.scoring.survey`; this page
only renders the form and stores the scored result in the session state.
"""

import streamlit as st

from app.styles import inject_centered_radios, inject_global_font
from src.scoring.survey import (
    CARE_LEVEL_OPTIONS,
    INVESTMENT_STYLE_QUESTION,
    INVESTMENT_STYLES,
    SURVEY_QUESTIONS,
    esg_interest_percent,
    score_survey,
)

inject_global_font()
inject_centered_radios()

_FORM_CSS = """
    <style>
        div.row-widget.stRadio > div{display: flex; justify-content: center;align-items: center;border-radius: 10px;}
        div[class="question"]{
            margin: auto;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        button[data-testid="baseButton-secondaryFormSubmit"]{
            background-color: #AAAAAA;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        p { font-family: Pretendard; }
    </style>
    """

_QUESTION_STYLE = (
    '<div class="question" style="font-size:20px;text-align:center;'
    'font-family: Pretendard;font-weight: bold;">{}</div>'
)


def _spacer(lines: int = 6) -> None:
    for _ in range(lines):
        st.write("")


with st.form("usersurvey", clear_on_submit=False):
    st.markdown(_FORM_CSS, unsafe_allow_html=True)

    answers: list[str] = []
    for index, question in enumerate(SURVEY_QUESTIONS):
        st.markdown(_QUESTION_STYLE.format(question.text), unsafe_allow_html=True)
        # Streamlit requires unique labels; keep them visually empty.
        answers.append(st.radio(" " * (index + 1), options=CARE_LEVEL_OPTIONS))
        _spacer()

    st.markdown(_QUESTION_STYLE.format(INVESTMENT_STYLE_QUESTION), unsafe_allow_html=True)
    investment_style = st.radio(" " * 17, options=INVESTMENT_STYLES)

    _, submit_column, _ = st.columns([3, 1, 3])
    with submit_column:
        submitted = st.form_submit_button("설문 완료")

    if submitted:
        st.session_state["survey_result"] = score_survey(answers)
        st.session_state["investment_style"] = investment_style
        st.session_state["esg_interest"] = esg_interest_percent(investment_style)
        st.switch_page("pages/portfolio_result.py")

"""Landing page: welcome text and user name input before the survey."""

import streamlit as st

from app.styles import inject_global_font

inject_global_font()

st.markdown(
    '''
    <div>
        <h2 style="font-size:40px; text-align:center; color:#666666; font-family: Pretendard;">ESG 선호도 설문</h2>
    </div>
    ''',
    unsafe_allow_html=True,
)

_, start_page, _ = st.columns([1, 2, 1])
with start_page:
    st.markdown(
        """
        <style>
            header[data-testid="stHeader"]::after {
                content: "\\00a0\\00a0Kwargs";
                font-family: Pretendard;
                display: block;
                font-size: 30px;
                word-spacing: 30px;
                font-weight: bold;
                color: black;
                padding: 10px;
            }
            button[data-testid="baseButton-secondary"]{
                background-color:#BBBBBB;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            div[data-testid="stVerticalBlock"]{
                text-align : center;
                font-family: Pretendard;
            }
            .container {
                max-width: 800px;
                margin: auto;
                padding: 20px;
                background-color: #BBBBBB;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            p { font-size: 18px; font-family: Pretendard; }
        </style>
        <div class="container">
            <p style="text-align:center; font-family: Pretendard;">환영합니다!</p>
            <p style="font-family: Pretendard;">해당 설문은 귀하의 <strong>ESG(환경, 사회, 지배구조)</strong> 투자 관점과 가치에 대한 이해를 돕기 위해 마련되었습니다. 귀하의 선호도를 반영하여 보다 개인화된 투자 분석과 포트폴리오 제안을 제공하기 위해, 간단한 질문에 응답해 주세요.&ensp;설문 결과를 반영하여 보다 신뢰할 수 있는 투자 정보를 제공하며, 사회적 책임과 환경적 가치를 고려한 맞춤형 포트폴리오를 설계합니다.</p>
            <h2 style="font-size:22px; text-align:center; font-family: Pretendard;">소요 시간</h2>
            <p style="text-align:center; font-family: Pretendard;">약 <strong>3분</strong>정도 소요됩니다.</p>
            <p style="text-align:center; font-family: Pretendard; font-size:15px;">여러분의 소중한 의견은 지속 가능한 투자의 중요한 지침이 됩니다. 지금 바로 설문을 시작해 주세요!</p>
            <h3 style="font-size:20px; font-family: Pretendard; text-align:center;">아래 입력창에 이름을 입력해 주세요</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_name = st.text_input(" ", key="user_name_input")
    _, start_button, _ = st.columns(3)
    with start_button:
        if st.button("설문 시작하기"):
            st.session_state["user_name"] = f"{user_name}님" if user_name else "당신"
            st.switch_page("pages/survey.py")

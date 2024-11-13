import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import os.path
from streamlit_js_eval import streamlit_js_eval
from passlib.context import CryptContext
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
        page_title="ESG 정보 제공 플랫폼",
        page_icon=":earth_africa:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과',icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스',icon="🆕")
    st.page_link('pages/esg_introduce.py', label='ESG 소개 / 투자 방법', icon="🧩")

font_css = """
    <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: Pretendard;
        }}
    </style>
    """
# Streamlit에 CSS 적용
st.markdown(font_css, unsafe_allow_html=True)

st.markdown('''
            <div>
                <h2 style="font-size:40px; text-align:center; font-family: Pretendard;">ESG 선호도 설문</h2>
            </div>
            ''',unsafe_allow_html=True)
_,start_page,_ = st.columns([1,2,1])

with start_page:
    st.markdown("""
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        div[data-testid="stHeadingWithActionElements"]{
                            font-size: 40px;
                            font-family: Pretendard;
                        }
                        div[data-testid="stApp"]{
                            background-image: linear-gradient(rgb(178,221,247),rgb(231,246,255))
                        }
                        header[data-testid="stHeader"]{
                            background-color: #b2ddf7;
                            padding-left:80px;
                        }
                        header[data-testid="stHeader"]::after {
                            content: "Kwargs";
                            font-family: Pretendard;
                            display: block;
                            font-size: 30px;
                            word-spacing: 30px;
                            font-weight: bold;
                            color: black;
                            padding: 10px;
                        }
                        button[data-testid="baseButton-secondary"]{
                            background-color: #e7f6ff;
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
                            background-color: #e7f6ff;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                        h1 {
                            text-align: center;
                            font-family: Pretendard;
                        }
                        p {
                            font-size: 18px;
                            font-family: Pretendard;
                        }
                        .btn-start {
                            display: block;
                            width: 100%;
                            background-color: #4CAF50;
                            color: white;
                            padding: 15px;
                            text-align: center;
                            border: none;
                            font-family: Pretendard;
                            border-radius: 5px;
                            font-size: 18px;
                            cursor: pointer;
                            margin-top: 20px;
                        }
                        .btn-start:hover {
                            background-color: #45a049;
                        }
                    </style>
                </head>
                <body>
                <div class="container">
                    <p style="text-align:center; text-color:#0000; font-family: Pretendard;">환영합니다</p>
                    <p style="font-family: Pretendard;">해당 설문은 귀하의 <strong>ESG(환경, 사회, 지배구조)</strong> 투자 관점과 가치에 대한 이해를 돕기 위해 마련되었습니다. 귀하의 선호도를 반영하여 보다 개인화된 투자 분석과 포트폴리오 제안을 제공하기 위해, 간단한 질문에 응답해 주세요.&ensp;설문 결과를 반영하여 보다 신뢰할 수 있는 투자 정보를 제공하며, 사회적 책임과 환경적 가치를 고려한 맞춤형 포트폴리오를 설계합니다.</p>
                    <h2 style="font-size:22px; text-align:center;text-color:#0000;font-family: Pretendard;">소요 시간</h2>
                    <p style="text-align:center;text-color:#0000;font-family: Pretendard;">약 <strong>3분</strong>정도 소요됩니다.</p>
                    <p style="text-align:center;text-color:#0000;font-family: Pretendard;font-size:15px;">여러분의 소중한 의견은 지속 가능한 투자의 중요한 지침이 됩니다. 지금 바로 설문을 시작해 주세요!</p>
                    <h3 style="font-size:20px;font-family: Pretendard;text-align:center;">아래 입력창에 이름을 입력해 주세요</h3>
                </div>
                </body>
                </html>
                """,unsafe_allow_html=True)
    user_name = st.text_input(" ",key="user_name")
    _,start_button,_ = st.columns(3)
    with start_button:
        switch_page = st.button("설문 시작하기")
        if switch_page:
            if user_name:
                with open(r"C:\esgpage\LLM.ESG.POS\interface\user_name.txt", 'w', encoding='utf-8') as f:
                    f.write(user_name + '님')
            else:
                with open(r"C:\esgpage\LLM.ESG.POS\interface\user_name.txt", 'w', encoding='utf-8') as f:
                    f.write('당신')
            selected = '설문 페이지'
            st.switch_page('pages/survey_page.py')
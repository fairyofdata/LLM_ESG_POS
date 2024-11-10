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
import os
import streamlit as st


import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 필요한 패키지 리스트
packages = ["streamlit", "bs4", "FinanceDataReader", "mplfinance", "pypfopt", "cvxopt", "streamlit_plotly_events"]

for package in packages:
    install_and_import(package)


st.set_page_config(
        page_title="ESG 중심 포트폴리오 최적화",
        page_icon=":earth_africa:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

# 세션 상태를 초기화 
if 'ndays' not in st.session_state: 
    st.session_state['ndays'] = 100
    
if 'code_index' not in st.session_state:
    st.session_state['code_index'] = 0
    
if 'chart_style' not in st.session_state:
    # 차트의 유형은 디폴트로 지정
    st.session_state['chart_style'] = 'default'

if 'volume' not in st.session_state:
    # 거래량 출력 여부는 true 값으로 초기화
    st.session_state['volume'] = True

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
    
if 'user_name' not in st.session_state:
    st.session_state['username'] = None

if 'clicked_points' not in st.session_state:
    st.session_state['clicked_points'] = None
    
if 'sliders' not in st.session_state:
    st.session_state['sliders'] = {}

for key in ['environmental', 'social', 'governance']:
    if key not in st.session_state['sliders']:
        st.session_state['sliders'][key] = 0

with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과',icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스',icon="🆕")
    st.page_link('pages/esg_introduce.py', label='ESG 소개 / 투자 방법', icon="🧩")


st.markdown('''
            <div>
                <h2 style="font-size:40px; text-align:center;">ESG 영역별 관심 조사</h2>
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
                        /* 전체 페이지 배경을 밝은 색으로 설정하고 텍스트 색상을 어두운 색으로 강제 적용 */
                        body {
                            background-color: #ffffff; /* 밝은 배경 */
                            color: #000000; /* 어두운 텍스트 */
                            margin: 0;
                            padding: 0;
                            font-family: Arial, sans-serif;
                        }
                
                        /* Streamlit App의 배경색을 밝게 설정 */
                        div[data-testid="stApp"] {
                            background-color: #ffffff !important; /* 배경을 강제로 밝은 색으로 설정 */
                        }
                
                        div[data-testid="stHeadingWithActionElements"] {
                            font-size: 40px;
                            color: #000000; /* 제목 텍스트 색상 */
                        }
                
                        header[data-testid="stHeader"] {
                            background-color: #b2ddf7; /* 헤더 색상 */
                            padding-left: 80px;
                        }
                
                        header[data-testid="stHeader"]::after {
                            content: "Kwargs";
                            display: block;
                            font-size: 30px;
                            word-spacing: 30px;
                            font-weight: bold;
                            color: black;
                            padding: 10px;
                        }
                
                        button[data-testid="baseButton-secondary"] {
                            background-color: #e7f6ff;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                            color: #000000;
                        }
                
                        div[data-testid="stVerticalBlock"] {
                            text-align: center;
                        }
                
                        .container {
                            max-width: 800px;
                            margin: auto;
                            padding: 20px;
                            background-color: #e7f6ff; /* 밝은 색 배경 */
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                
                        h1, p {
                            color: #000000; /* 텍스트 색상 통일 */
                        }
                
                        p {
                            font-size: 18px;
                        }
                
                        .btn-start {
                            display: block;
                            width: 100%;
                            background-color: #4CAF50;
                            color: white;
                            padding: 15px;
                            text-align: center;
                            border: none;
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
                    <p style="text-align:center;">환영합니다</p>
                    <p>해당 설문은 귀하의 <strong>ESG(환경, 사회, 지배구조)</strong> 투자 관점과 가치에 대한 이해를 돕기 위해 마련되었습니다. 귀하의 선호도를 반영하여 보다 개인화된 투자 분석과 포트폴리오 제안을 제공하기 위해, 간단한 질문에 응답해 주세요.&ensp;설문 결과를 반영하여 보다 신뢰할 수 있는 투자 정보를 제공하며, 사회적 책임과 환경적 가치를 고려한 맞춤형 포트폴리오를 설계합니다.</p>
                    <h2 style="font-size:22px; text-align:center;">소요 시간</h2>
                    <p style="text-align:center;">약 <strong>3분</strong>정도 소요됩니다.</p>
                    <p style="text-align:center; font-size:15px;">여러분의 소중한 의견은 지속 가능한 투자의 중요한 지침이 됩니다. 지금 바로 설문을 시작해 주세요!</p>
                    <h3 style="font-size:20px;text-align:center;">아래 입력창에 이름을 입력해 주세요</h3>
                </div>
                </body>
                </html>
                """, unsafe_allow_html=True)



    user_name = st.text_input(" ",key="user_name")
    _,start_button,_ = st.columns(3)
    # 현재 스크립트의 디렉토리 경로를 기준으로 상대 경로 설정
    current_directory = os.path.dirname(__file__)  # 현재 스크립트의 경로
    name_path = os.path.join(current_directory, 'user_name.txt')  # 상대 경로로 user_name.txt 경로 설정
    with start_button:
        switch_page = st.button("설문 시작하기")
        if switch_page:
            if user_name:
                with open(name_path, 'w', encoding='utf-8') as f:
                    f.write(user_name + '님')  # 유저 이름을 파일에 저장
            else:
                with open(name_path, 'w', encoding='utf-8') as f:
                    f.write('당신')  # 기본값 저장
            selected = '설문 페이지'
            st.switch_page('pages/survey_page.py')


import streamlit as st
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service  # Service 클래스 임포트
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import FinanceDataReader as fdr
import mplfinance as mpf
from datetime import datetime, timedelta
import json
import yaml
import streamlit_authenticator as stauth
import numpy as np
import requests as rq
from streamlit_authenticator.utilities.hasher import Hasher
import os.path
import pickle as pkle
from streamlit_js_eval import streamlit_js_eval
from passlib.context import CryptContext
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
from cvxopt import matrix, solvers
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)
from streamlit_extras.switch_page_button import switch_page
from pymongo import MongoClient
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pypfopt import risk_models, BlackLittermanModel, expected_returns

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
    
_,col1,_ = st.columns([1,2,1])

with col1:
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://media.istockphoto.com/id/1447057524/ko/%EC%82%AC%EC%A7%84/%ED%99%98%EA%B2%BD-%EB%B0%8F-%EB%B3%B4%EC%A0%84%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B2%BD%EC%98%81-esg-%EC%A7%80%EC%86%8D-%EA%B0%80%EB%8A%A5%EC%84%B1-%EC%83%9D%ED%83%9C-%EB%B0%8F-%EC%9E%AC%EC%83%9D-%EC%97%90%EB%84%88%EC%A7%80%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9E%90%EC%97%B0%EC%9D%98-%EA%B0%9C%EB%85%90%EC%9C%BC%EB%A1%9C-%EB%85%B9%EC%83%89-%EC%A7%80%EA%B5%AC%EB%B3%B8%EC%9D%84-%EB%93%A4%EA%B3%A0-%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.jpg?s=612x612&w=0&k=20&c=ghQnfLcD5dDfGd2_sQ6sLWctG0xI0ouVaISs-WYQzGA="><br>
            <h1 style="font-size:20px;font-weight:bold;"><br><strong>ESG</strong>란 환경(Environment), 사회(Social), 그리고 지배구조(Governance)의 약자로, 기업이 책임감 있고 지속 가능하게 경영하기 위해 고려해야 할 세 가지 핵심 요소를 의미합니다.</h1> 
            <p style="font-size:25px;font-weight:bold;"> 1. 환경 (Environment)</p> 
            기업이 환경에 미치는 영향에서 탄소 배출량, 자원 소모, 에너지 효율 등을 고려해 지속 가능한 경영을 추구하는 것을 뜻합니다. 환경 보호를 위한 기업의 노력은 기후 변화 대응과 자연 자원 보호에 크게 기여합니다. 
            <p style="font-size:25px;font-weight:bold;">2. 사회 (Social)</p> 
            사회적 책임을 다하는 경영을 지향합니다. 근로자 권리 보장, 지역사회 기여, 고객과의 신뢰 구축 등 기업이 사회와 맺는 관계를 평가하는 요소입니다. 이를 통해 사회에 미치는 긍정적 영향도 함께 고려됩니다. 
            <p style="font-size:25px;font-weight:bold;">3. 지배구조 (Governance)</p> 
            투명하고 공정한 경영을 통해 주주와 임직원 등 이해관계자의 신뢰를 유지하려는 기업의 의지를 나타냅니다. 의사결정 구조, 경영진의 윤리, 주주와의 관계 등이 중요한 평가 항목입니다. 
            <p style="font-size:25px;font-weight:bold;"><br>ESG를 고려한 투자의 중요성</p> 
            ESG를 고려한 투자는 윤리적 접근을 넘어 기업의 장기적인 안정성과 지속 가능성 확보에 중요한 역할을 합니다. 특히 환경 책임과 사회적 역할은 투자자뿐 아니라 고객, 정부, 전 세계 사회 구성원들에게도 큰 관심사입니다. ESG 투자 전략을 통해 투자자들은 장기적으로 안정적이고 지속 가능한 수익을 기대할 수 있으며, 이는 기업의 재무 성과와 평판 향상에도 기여합니다. 기업이 ESG를 잘 준수할 경우 장기적인 성공과 지속 가능성에 긍정적 영향을 미치며, 많은 투자자들이 이를 기준으로 기업의 사회적 책임과 미래 가치를 평가하고 투자 결정을 내리고 있습니다.
        </div>
    """, unsafe_allow_html=True)

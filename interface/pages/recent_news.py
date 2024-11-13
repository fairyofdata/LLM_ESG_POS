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
import unicodedata
import matplotlib.pyplot as plt
from pypfopt import risk_models, BlackLittermanModel, expected_returns

st.set_page_config(
        page_title="ESG 정보 제공 플랫폼",
        page_icon=":earth_africa:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

# 크롤링  필요한 함수 정의
def setup_webdriver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--headless')  # UI 없이 실행하기 위한 headless 모드
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 서비스 객체를 사용하여 드라이버 초기화
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3)
    
    return driver

# URL 생성 함수 정의
def makePgNum(num):
    return 1 + 10 * (num - 1)

def makeUrl(search, page):
    page_num = makePgNum(page)
    url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&start={page_num}"
    return url

# 뉴스 크롤링 함수 추가
def crawl_naver_news(search):
    driver = setup_webdriver()

    target_article_count = 10
    collected_article_count = 0
    current_page = 1
    naver_urls = []

    while collected_article_count < target_article_count:
        search_url = makeUrl(search, current_page)
        driver.get(search_url)
        time.sleep(1)  # 대기시간 변경 가능

        a_tags = driver.find_elements(By.CSS_SELECTOR, 'a.info')

        for a_tag in a_tags:
            if collected_article_count >= target_article_count:
                break
            a_tag.click()
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(3)

            url = driver.current_url
            if "news.naver.com" in url:
                naver_urls.append(url)
                collected_article_count += 1

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        current_page += 1

    driver.quit()
    return fetch_news_contents(naver_urls)

def fetch_news_contents(naver_urls):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    news_list = []

    for url in naver_urls:
        original_html = requests.get(url, headers=headers)
        html = BeautifulSoup(original_html.text, "html.parser")
        title_element = html.select_one("div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        title = title_element.get_text(strip=True) if title_element else "No title found"
        news_list.append((title, url))

    return news_list


with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과',icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스',icon="🆕")
    st.page_link('pages/esg_introduce.py', label='ESG 소개 / 투자 방법', icon="🧩")


st.markdown('''
            <h1 style="font-size:35px;text-align:center;font-weight:bold;">최신 뉴스</h1>
            ''',unsafe_allow_html=True)
st.markdown('''
            <h1 style="font-size:15px;text-align:center;">궁금한 키워드를 검색해보세요.</h1>
            ''', unsafe_allow_html=True)
# 버튼 클릭 시 크롤링 시작

search = st.text_input(" ")
_,col,_ = st.columns([5,1,5])
with col:
    search_button = st.button("검색")
if search_button:
    if search:
        st.markdown(f'''<p style="text-align:center;font-size:17px;">{search}관련 기사를 검색 중입니다...</p>''',unsafe_allow_html=True)
        news_list = crawl_naver_news(search)
        if news_list:
            # st.write(f"수집된 기사 수: {len(news_list)}개")
            for title, link in news_list:
                st.markdown(f"- [{title}]({link})")
        else:
            st.write("기사를 찾을 수 없습니다.")
    else:
        st.write("검색어를 입력해주세요.")
# Streamlit 및 웹 관련 라이브러리
import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# 데이터 처리 및 분석 관련 라이브러리
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
import os
import pickle as pkle

# 인증 및 보안 관련 라이브러리
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from streamlit_authenticator.utilities import (
    CredentialsError, ForgotError, Hasher, LoginError, RegisterError,
    ResetError, UpdateError
)
from passlib.context import CryptContext
from dotenv import load_dotenv

# 시각화 및 플로팅 관련 라이브러리
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from wordcloud import WordCloud
from collections import Counter

# 금융 및 최적화 관련 라이브러리
import FinanceDataReader as fdr
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel
from cvxopt import matrix, solvers

# 기타 유틸리티 라이브러리
from PIL import Image
import base64
import tempfile
import pdfkit
from pdfkit.api import configuration
import pyautogui
from fpdf import FPDF
import pyscreenshot as ImageGrab
from tqdm import tqdm
import unicodedata

# 한글 텍스트 분석
from konlpy.tag import Okt

# Streamlit용 확장 기능
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from streamlit_vertical_slider import vertical_slider
from streamlit_plotly_events import plotly_events
from streamlit_js_eval import streamlit_js_eval

# 현재 스크립트 파일의 위치를 기준으로 상대 경로 설정
current_directory = os.path.dirname(__file__)

# 경로 변수 정의
survey_result_file = os.path.join(current_directory, "survey_result.csv")
user_investment_style_file = os.path.join(current_directory, "user_investment_style.txt")
user_interest_file = os.path.join(current_directory, "user_interest.txt")
user_name_file = os.path.join(current_directory, "user_name.txt")
company_colletion_file = os.path.join(current_directory, 'company_collection.csv')
word_freq_file = os.path.join(current_directory, "company_word_frequencies.csv")
survey_result_page = 'pages/survey_result.py'

# 파일이 존재하는지 확인 후 불러오기
if os.path.exists(survey_result_file):
    survey_result = pd.read_csv(survey_result_file, encoding='utf-8', index_col=0)
else:
    # 파일이 없으면 기본값으로 빈 데이터프레임 생성
    survey_result = pd.DataFrame()

company_colletion = pd.read_csv(company_colletion_file, encoding='utf-8', index_col=0)
company_colletion.columns = company_colletion.columns.astype(str).str.strip()
company_colletion.reset_index(inplace=True)

if os.path.exists(user_investment_style_file):
    with open(user_investment_style_file, 'r', encoding='utf-8') as f:
        user_investment_style = f.read().strip()
else:
    user_investment_style = ''

if os.path.exists(user_interest_file):
    with open(user_interest_file, 'r', encoding='utf-8') as f:
        user_interest = f.read().strip()
else:
    user_interest = ''

if os.path.exists(user_name_file):
    with open(user_name_file, 'r', encoding='utf-8') as f:
        user_name = f.read().strip()
else:
    user_name = '당신'

if os.path.exists(word_freq_file):
    word_freq_df = pd.read_csv(word_freq_file)
else:
    word_freq_df = pd.DataFrame()

st.set_page_config(
    page_title = "설문 조사 결과",
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

os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk-23'

if 'ndays' not in st.session_state: 
    st.session_state['ndays'] = 100
    
if 'code_index' not in st.session_state:
    st.session_state['code_index'] = 0
    
if 'chart_style' not in st.session_state:
    st.session_state['chart_style'] = 'default'

if 'volume' not in st.session_state:
    st.session_state['volume'] = True

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
    
if 'user_name' not in st.session_state:
    st.session_state['username'] = None

if 'clicked_points' not in st.session_state:
    st.session_state['clicked_points'] = None
    
if 'sliders' not in st.session_state:
    st.session_state['sliders'] = {}
    
if 'selected_companies' not in st.session_state:
    st.session_state['selected_companies'] = []

for key in ['environmental', 'social', 'governance']:
    if key not in st.session_state['sliders']:
        st.session_state['sliders'][key] = 0
        
# MongoDB 연결 설정 (8월 해리)
# load_dotenv()
# client = MongoClient(os.getenv("mongodb_url"))
# db = client['kwargs']
# collection = db['kwargs']

# # MongoDB 연결 (11월 지헌)
# connection_string = "mongodb+srv://kwargs:57qBBuXYQel4W6oV@kwargsai.5yhiymt.mongodb.net/?retryWrites=true&w=majority&appName=kwargsai" #mongodb_url  # MongoDB 연결 문자열을 입력하세요
# client = MongoClient(connection_string)
# db = client['kwargsai']
# collection = db['test_collection']

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
values = {'msci': 0, 'iss': 0, 'sustain': 0, 'sandp': 0, 'esg1': 0}
st.markdown("""
    <style>
        .element-container st-emotion-cache-1c12lws e1f1d6gn4{
            margin: 0;
            padding: 0;
        }
        .slicetext{
            font-family: Pretendard;
        }
    </style>
    """,unsafe_allow_html=True)

# 전처리 함수 정의

def preprocess_data(df):
    # 기존 컬럼명을 사용할 수 있도록 유효성을 확인
    df = df.copy()
    if 'environmental' in df.columns and 'social' in df.columns and 'governance' in df.columns:
        # ESG 영역 비중을 백분율로 환산
        df['env_percent'] = df['environmental'] / (df['environmental'] + df['social'] + df['governance'])
        df['soc_percent'] = df['social'] / (df['environmental'] + df['social'] + df['governance'])
        df['gov_percent'] = df['governance'] / (df['environmental'] + df['social'] + df['governance'])

        # 각 영역별 최종 점수 계산 (average_label 필요)
        df['env_score'] = df['average_label'] * df['env_percent']
        df['soc_score'] = df['average_label'] * df['soc_percent']
        df['gov_score'] = df['average_label'] * df['gov_percent']

        # 연도별 가중치 설정
        latest_year = df['Year'].max()
        year_weights = {
            latest_year: 0.5,
            latest_year - 1: 0.25,
            latest_year - 2: 0.125,
            latest_year - 3: 0.0625,
            latest_year - 4: 0.0625
        }

        # 가중치를 반영한 각 영역별 점수 합산
        df['environmental'] = df.apply(lambda x: x['env_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['social'] = df.apply(lambda x: x['soc_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['governance'] = df.apply(lambda x: x['gov_score'] * year_weights.get(x['Year'], 0), axis=1)

        # 동일 기업의 연도별 점수를 합산하여 최종 점수 도출
        final_df = df.groupby(['Company', 'industry', 'ticker']).agg({
            'environmental': 'sum',
            'social': 'sum',
            'governance': 'sum'
        }).reset_index()

        return final_df
    else:
        raise KeyError("The expected columns 'environmental', 'social', and 'governance' are not present in the dataframe.")


# 현재 스크립트 파일의 부모 디렉터리로 이동하여 경로 설정
current_directory = os.path.dirname(os.path.dirname(__file__))

# 경로 변수 정의
file_path = "241007_dummy_noharim.csv"  # 올바른 파일 경로로 설정
dummy_file_path = os.path.join(current_directory, file_path)

# 필요한 파일을 읽어오기
if os.path.exists(dummy_file_path):
    try:
        # 시도 순서대로 다른 인코딩을 적용하여 파일을 읽기
        try:
            dummy = pd.read_csv(dummy_file_path, encoding='euc-kr')
        except UnicodeDecodeError:
            try:
                dummy = pd.read_csv(dummy_file_path, encoding='cp949')
            except UnicodeDecodeError:
                dummy = pd.read_csv(dummy_file_path, encoding='utf-8')

        # 파일이 제대로 읽혔는지 확인
        print("데이터프레임 미리보기:")
        print(dummy.head())
        print(f"데이터프레임의 컬럼 목록: {dummy.columns.tolist()}")

    except Exception as e:
        print(f"파일 읽기 오류 발생: {e}")
        dummy = pd.DataFrame()  # 오류 발생 시 기본값으로 빈 데이터프레임 사용
else:
    # 파일이 없을 경우 빈 데이터프레임 사용
    print(f"파일이 존재하지 않습니다: {dummy_file_path}")
    dummy = pd.DataFrame()

# 데이터 전처리 단계 실행 (빈 데이터프레임이어도 오류 없이 실행될 수 있도록 설정)
df_new = preprocess_data(dummy) if not dummy.empty else pd.DataFrame()

# 데이터프레임 정보 출력 후 'industry' 열 확인
if not df_new.empty:
    print("전처리 후 데이터프레임 미리보기:")
    print(df_new.head())
    if 'industry' in df_new.columns:
        industries = df_new['industry'].unique().tolist()
        print(f"산업 목록: {industries}")
    else:
        print("전처리된 데이터프레임에 'industry' 열이 존재하지 않습니다.")
        industries = []  # 기본값으로 빈 리스트 사용
else:
    print("전처리된 데이터프레임이 비어 있습니다.")
    industries = []


# 한국거래소 코스피 인덱스에 해당하는 종목 가져오기
@st.cache_data
def getSymbols(market='KOSPI',sort='Marcap'): # 정렬하는 기준을 시장가치(Marcap)으로 함
    df = fdr.StockListing(market)
    # 정렬 설정 (= 시가총액 기준으로는 역정렬)
    ascending = False if sort == 'Marcap' else True
    df.sort_values(by=[sort],ascending=ascending, inplace=True)
    return df[['Code','Name','Market']]

@st.cache_data
def load_stock_data(code, ndays, frequency='D'):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=ndays)
    data = fdr.DataReader(code, start_date, end_date)

    if frequency == 'M':  # 월봉 설정
        data = data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()  # 월봉 리샘플링, 결측값 제거

    return data

# 캔들차트 출력 함수
def plotChart(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    chart_style = st.session_state['chart_style']
    marketcolors = mpf.make_marketcolors(up='red', down='blue')
    mpf_style = mpf.make_mpf_style(base_mpf_style=chart_style, marketcolors=marketcolors)

    fig, ax = mpf.plot(
        data=data,
        volume=st.session_state['volume'],
        type='candle',
        style=mpf_style,
        figsize=(10, 7),
        fontscale=1.1,
        mav=(5, 10, 30),
        mavcolors=('red', 'green', 'blue'),
        returnfig=True
    )
    st.pyplot(fig)


# 상위 기업 선정 (esg 기반)
def recommend_companies(esg_weights, df):
    # 전처리된 데이터에서 사용자의 ESG 선호도 가중치를 반영하여 최종 점수 계산
    df['final_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    # 상위 10개 기업 선정
    top_companies = df.sort_values(by='final_score', ascending=False).head(10)

    return top_companies


st.markdown("""
            <style>
            .st-emotion-cache-10hsuxw e1f1d6gn2{
                margin:3px;
            }
            </style>
            """,unsafe_allow_html=True)

header = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            header[data-testid="stHeader"]::after {{
                content: "\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0{user_name}을 위한 ESG 투자 최적화 포트폴리오";
                display: block;
                font-size: 30px;
                word-spacing: 3px;
                font-weight: bold;
                color: #999999;
                padding: 10px;
                font-family: Pretendard;
            }}
            a{{
                font-family: Pretendard;
            }}
        </style>
    </head>
    </html>
    """
st.markdown(header, unsafe_allow_html=True)

#--- 최적화 알고리즘 ---
import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import yfinance as yf
from pypfopt import BlackLittermanModel, expected_returns, risk_models, CovarianceShrinkage

# 수정된 포트폴리오 비중 계산 함수 with Black-Litterman 및 공분산 행렬 축소
# 기존 방식: 사용자의 ESG 선호도가 시장 수익률과 별개로 반영되어 최적화 과정에서 영향력이 미비함
# 개선 방식: ESG 선호도를 반영하여 시장 균형 수익률 자체를 조정하고, 이를 최적화에 반영
# 블랙리터만 모델 적용 함수
def calculate_portfolio_weights(df, esg_weights, user_investment_style):
    # 데이터 수집 및 전처리
    tickers = df['ticker'].tolist()
    price_data = yf.download(tickers, start="2019-01-01", end="2023-01-01")['Adj Close']
    price_data = price_data.dropna(axis=1)
    if price_data.isnull().values.any():
        return "일부 데이터가 누락되었습니다. 다른 기업을 선택해 주세요.", None

    # 평균 수익률과 공분산 행렬 계산
    mu_market = expected_returns.capm_return(price_data)  # CAPM을 통한 시장 균형 수익률 계산
    Sigma = risk_models.sample_cov(price_data)  # 샘플 공분산 행렬

    # 공분산 행렬 정규화: 비가역성을 방지하기 위해 작은 값 추가
    Sigma += np.eye(Sigma.shape[0]) * 1e-6

    # ESG 가중치 스케일링 (비율 조정)
    esg_weights = {key: value / 30000 for key, value in esg_weights.items()}

    # 사용자 선호도와 ESG 가중치를 반영한 최종 ESG 점수 계산
    df['final_esg_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    # 사용자 투자 스타일에 따른 ESG 가중치 설정
    if user_investment_style == "재무적인 요소를 중심적으로 고려한다.":
        esg_weight_factor = 10.0
    elif user_investment_style == "ESG와 재무적인 요소를 모두 고려한다.":
        esg_weight_factor = 20.0
    elif user_investment_style == "ESG 요소를 중심적으로 고려한다.":
        # esg_weight_factor = 2.5
        esg_weight_factor = 100.0
    else:
        esg_weight_factor = 1.0  # 기본값 설정

    # 최종 ESG 점수에 투자 스타일 반영
    df['adjusted_esg_score'] = df['final_esg_score'] * esg_weight_factor

    # Black-Litterman 모델의 투자자의 의견으로 반영할 데이터 준비
    valid_tickers = price_data.columns.tolist()
    df_valid = df[df['ticker'].isin(valid_tickers)]

    # 개선된 P 매트릭스 설정: 자산별로 더욱 다양하게 반영하여 상관관계 고려
    P = np.zeros((len(valid_tickers), len(valid_tickers)))
    np.fill_diagonal(P, [1.0 / len(valid_tickers)] * len(valid_tickers))

    # Q 벡터 설정: ESG 점수를 반영한 투자자의 의견
    Q = df_valid['adjusted_esg_score'].values

    # Black-Litterman 모델 적용
    tau = 0.1  # tau 값을 적절히 조정하여 모델 안정성 확보
    bl = BlackLittermanModel(Sigma, pi=mu_market, P=P, Q=Q, tau=tau)
    adjusted_returns = bl.bl_returns()

    # 최적화 문제 설정 및 최적 가중치 계산
    n = len(mu_market)
    P_opt = matrix(Sigma.values)
    q_opt = matrix(-adjusted_returns.values)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    # 쿼드라틱 프로그래밍 솔버 실행
    sol = solvers.qp(P_opt, q_opt, G, h, A, b)

    # 최적 가중치 추출
    weights = np.array(sol['x']).flatten()

    # 포트폴리오 성과 지표 계산
    expected_return = np.dot(weights, mu_market)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma.values, weights)))
    sharpe_ratio = expected_return / expected_volatility

    # 가중치 정리
    cleaned_weights = dict(zip(valid_tickers, weights))

    return cleaned_weights, (expected_return, expected_volatility, sharpe_ratio)

# 최종 가중치를 optimized_weights로 적용
def calculate_adjusted_weights(df, optimized_weights, esg_weights,performance_metrics):
    environmental_scores = df['environmental']
    social_scores = df['social']
    governance_scores = df['governance']

    # Calculate ESG-based adjustment
    esg_adjustment = (
        (environmental_scores * esg_weights['environmental']) +
        (social_scores * esg_weights['social']) +
        (governance_scores * esg_weights['governance'])
    ) / 3

    esg_adjustment_normalized = esg_adjustment / esg_adjustment.sum()
    if isinstance(optimized_weights, dict):
        adjusted_weights = {ticker: 0.5 * optimized_weights[ticker] + 0.5 * esg_adjustment_normalized[i]
                            for i, ticker in enumerate(optimized_weights.keys())}
    else:
        adjusted_weights = 0.2 * adjusted_weights + 0.8 * esg_adjustment_normalized

    # Normalize adjusted weights to sum to 1
    if isinstance(adjusted_weights, dict):
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {ticker: weight / total_weight for ticker, weight in adjusted_weights.items()}
    else:
        adjusted_weights /= adjusted_weights.sum()

    return adjusted_weights, performance_metrics
    # # Normalize ESG adjustment to have the same range as optimized_weights
    # esg_adjustment_normalized = esg_adjustment / esg_adjustment.sum()

    # # Adjust the weights: 50% original weight + 50% ESG-adjusted weight
    # adjusted_weights = 0.5 * optimized_weights + 0.5 * esg_adjustment_normalized

    # # Normalize adjusted weights to sum to 1
    # adjusted_weights /= adjusted_weights.sum()




    # 최적화 후 가중치 조정: 각 영역별 점수를 반영하여 가중치 수정
    # for ticker in cleaned_weights:
    #     company_data = df_valid[df_valid['ticker'] == ticker]
    #     if not company_data.empty:
    #         environmental_score = company_data['environmental'].values[0]
    #         social_score = company_data['social'].values[0]
    #         governance_score = company_data['governance'].values[0]
    #         cleaned_weights[ticker] = (cleaned_weights[ticker] * 0.5) + (
    #             (environmental_score * esg_weights['environmental'] +
    #              social_score * esg_weights['social'] +
    #              governance_score * esg_weights['governance']) * 0.5
    #         )

    # return cleaned_weights, (expected_return, expected_volatility, sharpe_ratio)


# 결과 출력
# 개선된 코드에서는 사용자의 ESG 선호도가 시장 균형 수익률에 직접 반영되므로,
# 최적화 과정에서 사용자의 ESG 선호가 명확히 드러나도록 합니다.
def display_text_on_hover(hover_text, i, origin_text):
    # 각 텍스트 호버 영역에 고유한 클래스 이름을 생성
    hover_class = f'hoverable_{i}'
    tooltip_class = f'tooltip_{i}'
    text_popup_class = f'text-popup_{i}'

    # 각 호버 텍스트에 대한 고유한 CSS 정의
    hover_css = f'''
        .{hover_class} {{
            position: relative;
            display: block;
            cursor: pointer;
            text-align: center;
            font-family: Pretendard;
        }}
        .{hover_class} .{tooltip_class} {{
            display: none; /* Hover to see text를 숨김 */
        }}
        .{hover_class}:hover .{tooltip_class} {{
            opacity: 1;
        }}
        .{text_popup_class} {{
            display: none;
            position: absolute;
            background-color: #f1f1f1;
            padding: 8px;
            border-radius: 4px;
            width: 80%; /* 화면 너비의 80%로 설정 */
            left: 50%;  /* 중앙 정렬을 위해 left를 50%로 설정 */
            transform: translateX(-50%);
            max-width: 200px;
            font-family: Pretendard;
            color: #333;
            font-size: 14px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .{hover_class}:hover .{text_popup_class} {{
            display: block;
            z-index: 999;
        }}
    '''
    tooltip_css = f"<style>{hover_css}</style>"

    # origin_text의 스타일을 수정하여 HTML 정의
    text_hover = f'''
        <div class="{hover_class}">
            <a href="#hover_text" style="color: #999999; font-family: Pretendard; font-size: 20px; text-align: center; text-decoration: none;font-weight:bold;">{origin_text}&ensp;&ensp;</a>
            <div class="{tooltip_class}"></div>
            <div class="{text_popup_class}">{hover_text}</div>
        </div>
    '''
    
    # 동적 HTML 및 CSS를 콘텐츠 컨테이너에 작성
    st.markdown(f'<p>{text_hover}{tooltip_css}</p>', unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,1,3])
with col1:
    if user_investment_style == "재무적인 요소를 중심적으로 고려한다.":
        esg_weight_factor = 0.5
    elif user_investment_style == "ESG와 재무적인 요소를 모두 고려한다.":
        esg_weight_factor = 1.0
    elif user_investment_style == "ESG 요소를 중심적으로 고려한다.":
        esg_weight_factor = 2.0

    st.markdown("""
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
                    <style>
                    .stSlider{
                        padding:16px;
                    }
                    .element-container st-emotion-cache-1yvhuls e1f1d6gn4{
                        padding:16px;
                    }
                    .st-emotion-cache-uzeiqp e1nzilvr4{
                        height: 50px;
                        width : 100%
                    }
                    .st-dt st-d4 st-d3 st-cb st-af st-c2 st-du{
                        padding:10px;
                    }
                    .label{
                        font-family: Pretendard;
                    }
                    p{
                        text-color: #999999;
                        font-family: Pretendard;
                    }
                </style>
                </head>
    """, unsafe_allow_html=True)

    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    kospi, kosdaq = st.columns(2)
    kospi_data = fdr.DataReader('KS11', yesterday, today)
    kosdaq_data = fdr.DataReader('KQ11', yesterday, today)
    with kospi:
        if not kospi_data.empty:
            yesterday_kospi = kospi_data.iloc[0]['Close']
            today_kospi = kospi_data.iloc[-1]['Close']

            # 등락률 계산
            change = today_kospi - yesterday_kospi
            change_percent = (change / yesterday_kospi) * 100

            # Streamlit metric으로 출력
            st.metric(label="오늘의 코스피 지수", value=round(today_kospi, 2), delta=f"{round(change_percent, 2)}%",delta_color="inverse")

    with kosdaq:
        if not kosdaq_data.empty:
            yesterday_kosdaq = kosdaq_data.iloc[0]['Close']
            today_kosdaq = kosdaq_data.iloc[-1]['Close']

            # 등락률 계산
            change = today_kosdaq - yesterday_kosdaq
            change_percent = (change / yesterday_kosdaq) * 100

            # Streamlit metric으로 출력
            st.metric(label="오늘의 코스닥 지수", value=round(today_kosdaq, 2), delta=f"{round(change_percent, 2)}%",delta_color="inverse")

    sl1, sl2, sl3= st.columns(3)
    with sl1:
        origin_e = survey_result.loc['E'].sum() * 10 / 4.99
        display_text_on_hover('-탄소 관리<br>-폐기물 관리<br>-기후 변화 전략',1,'&emsp;E')
        e_value = vertical_slider(
            label = "환경",
            key = "environmental" ,
            height = 195,
            step = 0.1,
            default_value=survey_result.loc['E'].sum() * 1/ 4.99,#Optional - Defaults to 0
            min_value= 0.01, # Defaults to 0
            max_value= 1.0, # Defaults to 10
            track_color = "#f0f0f0", #Optional - Defaults to #D3D3D3
            slider_color = '#006699', #Optional - Defaults to #29B5E8
            thumb_color = "#FF9933",
            value_always_visible = True ,#Optional - Defaults to False
        )
    with sl2:
        display_text_on_hover('-사회적 기회<br>-지역사회 관계<br>-인적 자원',1,'&emsp;S')
        s_value = vertical_slider(
            label = "사회",  #Optional
            key = "social" ,
            height = 195, #Optional - Defaults to 300
            step = 0.1, #Optional - Defaults to 1
            default_value=survey_result.loc['S'].sum() *1/4.79,#Optional - Defaults to 0
            min_value= 0.01, # Defaults to 0
            max_value= 1.0, # Defaults to 10
            track_color = "#f0f0f0", #Optional - Defaults to #D3D3D3
            slider_color = '#006699', #Optional - Defaults to #29B5E8
            thumb_color = "#FF9933",
            value_always_visible = True ,#Optional - Defaults to False
        )
    with sl3:
        display_text_on_hover('-주주권 보호<br>-기업이사회운영<br>',1,'&emsp;G')
        g_value = vertical_slider(
            label = "지배구조",  #Optional
            key = "governance" ,
            height = 195, #Optional - Defaults to 300
            step = 0.1, #Optional - Defaults to 1
            default_value=survey_result.loc['G'].sum()*1/4.16,
            min_value= 0.01, # Defaults to 0
            max_value= 1.0, # Defaults to 10
            track_color = "#f0f0f0", #Optional - Defaults to #D3D3D3
            slider_color = '#006699', #Optional - Defaults to #29B5E8
            thumb_color = "#FF9933",
            value_always_visible = True ,#Optional - Defaults to False
        )
    # 사용자의 ESG 선호도
    esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}            
    # 블랙리터만 적용 버전
    industries = df_new['industry'].unique().tolist()
    processed_df = df_new[df_new['industry'].isin(industries)].copy()
    portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights, user_investment_style)
    # portfolio_weights, portfolio_performance = calculate_adjusted_weights(processed_df, portfolio_weights, esg_weights,portfolio_performance)
    # portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights,user_investment_style) # cleaned_weights:각 자산에 할당된 최적의 투자 비율, performance:최적화된 포트폴리오의 성과 지표
    top_companies = df_new[df_new['ticker'].isin(portfolio_weights)].copy()
    # ticker 열과 portfolio_weights를 매핑하여 새로운 top_companies 데이터프레임 생성_ 블랙리터만 모델 버전
    # portfolio_weights의 값을 'Weight' 컬럼으로 추가
    total_weight = sum(portfolio_weights.values())
    # total_weight =  sum(portfolio_weights.values)
    top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
    top_companies['Weight'] = top_companies['Weight'] * 100
    # Weight를 기준으로 내림차순 정렬
    top_companies = top_companies.sort_values(by='Weight', ascending=False)
    selected_companies = st.multiselect(
        "",
        top_companies['Company'],
        placeholder="제외하고 싶은 기업을 선택"
    )

    if selected_companies:
        top_companies = top_companies[~top_companies['Company'].isin(selected_companies)]
        
# 사용자의 ESG 선호도
esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}       
st.write('')
    
# 포트폴리오 비중 계산
# top_companies = recommend_companies(esg_weights,df_new)

# 블랙리터만 적용 버전
# industries = df_new['industry'].unique().tolist()
# processed_df = df_new[df_new['industry'].isin(industries)].copy()
# portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights, user_investment_style)
# # portfolio_weights, portfolio_performance = calculate_adjusted_weights(processed_df, portfolio_weights, esg_weights,portfolio_performance)
# # portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights,user_investment_style) # cleaned_weights:각 자산에 할당된 최적의 투자 비율, performance:최적화된 포트폴리오의 성과 지표
# top_companies = df_new[df_new['ticker'].isin(portfolio_weights)].copy()
# # ticker 열과 portfolio_weights를 매핑하여 새로운 top_companies 데이터프레임 생성_ 블랙리터만 모델 버전
# # portfolio_weights의 값을 'Weight' 컬럼으로 추가
# total_weight = sum(portfolio_weights.values())
# # total_weight =  sum(portfolio_weights.values)
# top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
# top_companies['Weight'] = top_companies['Weight'] * 100

# cvxopt 적용 버전
# portfolio_weights, portfolio_performance = calculate_portfolio_weights(top_companies)
# industries = df_new['industry'].unique().tolist()
    # processed_df = df_new[df_new['industry'].isin(industries)].copy()

# top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
    
with col2:

    if selected_companies:
        top_companies = top_companies[~top_companies['Company'].isin(selected_companies)]
    st.markdown(f"""<div>
                        <h2 style="font-family: Pretendard;font-size: 13px; text-align:center; text-decoration: none;">차트에서 여러분의 관심 회사 이름을 클릭하여<br>더 다양한 정보를 경험해 보세요.</h2>
                    </div>
            """, unsafe_allow_html=True)
    
    # 전체 Weight 합계 계산
    total_weight = top_companies['Weight'].sum()
    # Weight 기준으로 최소 비율 이하의 회사 필터링
    # top_companies = top_companies[top_companies['Weight'] / total_weight * 100 >= 5.0]
    
    
    # 파이 차트 생성
    fig = px.pie(
        top_companies, 
        names='Company', 
        values='Weight', 
        color_discrete_sequence=px.colors.qualitative.G10,
        custom_data=top_companies[['environmental', 'social', 'governance']]
    )

    # customdata로 ESG 정보 표시
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        hovertemplate=(
            '추천 포트폴리오 비중 : %{percent}<br>' +  # Weight 정보
            'Environmental 점수 : '+' ' +'%{customdata[0][0]:.2f}<br>' +  # Environmental 점수
            'Social 점수  :  %{customdata[0][1]:.2f}<br>' +  # Social 점수
            'Governance : %{customdata[0][2]:.2f}<br>'  # Governance 점수
        ),
        texttemplate='%{label}',
    )

    # 차트 레이아웃 설정
    fig.update_layout(
        font=dict(size=16, color='black'),
        showlegend=False,
        margin=dict(t=40, b=40, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=250,
        height=400,
    )

    clicked_points = plotly_events(fig, click_event=True,key="company_click")
    

with col3:
    company_colletion['ticker'] = company_colletion['ticker'].str[1:]
    top_companies['ticker'] = top_companies['ticker'].str.replace('.KS', '')

    expected_return = portfolio_performance[0]
    expected_volatility = portfolio_performance[1]
    sharpe_ratio = portfolio_performance[2]
    for company in top_companies['Company']:
        condition = (dummy['Year'] == 2023) & (dummy['Company'] == company)
        if condition.any():
            top_companies.loc[top_companies['Company'] == company, ['environmental', 'social', 'governance']] = dummy.loc[condition, ['environmental', 'social', 'governance']].values
    top5_companies = top_companies.nlargest(5, 'Weight')
    filtered_companies = pd.merge(company_colletion, top5_companies, left_on='ticker', right_on='ticker')
    filtered_companies = filtered_companies[['Company','Weight','environmental','social','governance','종목설명']]
    filtered_companies = filtered_companies.rename(columns={
        'Company': '종목명',
        'Weight': '제안 비중',
        'environmental': 'E',
        'social': 'S',
        'governance': 'G',
        '종목설명' :'종목 소개'
    })
    # 상단에 기대수익률, 변동성, 샤프비율 표시
    # _,col1, col2, col3,_ = st.columns([2,3,3,3,2])
    col1, col2, col3 = st.columns(3)
    with col1:
        display_text_on_hover("해당 지표는 포트폴리오가 1년 동안 벌어들일 것으로 예상되는 수익률입니다.",1,f"연간 기대 수익률 &emsp; {expected_return * 100:.2f} %")
        st.markdown('')
    with col2:
        display_text_on_hover("해당 지표는 수익률이 얼마나 변동할 수 있는지를 나타내는 위험 지표입니다.",1,f"연간 변동성 &emsp; {expected_volatility * 100:.2f} %")
    with col3:
        display_text_on_hover("해당 지표는 포트폴리오가 위험 대비 얼마나 효과적으로 수익을 내는지를 나타내는 성과 지표입니다.",1,f"샤프 비율 &emsp;{sharpe_ratio * 100:.2f}")

    # HTML 코드에 툴팁 추가 및 두 행 구조로 변환
    html_code = f"""
    <div style="font-family: Arial, sans-serif; text-align:center;">
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
            }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            font-size:15px;
            font-family: Pretendard;
        }}
        th {{
            text-color:#666666;
        }}
        </style>
    </style>
    <table>
            <thead>
            <tr>
                <th rowspan='2'>종목</th>
                <th rowspan='2'>제안<br>비중</th>
                <th colspan="3">ESG Score<br>(2023)</th>
                <th rowspan='2'>종목 소개</th>
            </tr>
            <tr>
                <th>E</th>
                <th>S</th>
                <th>G</th>
            </tr>
            </thead>
            <tbody>
        """

    filtered_companies = filtered_companies.sort_values(by='제안 비중', ascending=False)
    for _, row in filtered_companies.iterrows():
        html_code += f"""<tr>
        <td style="font-size:13px;">{row['종목명']}</td>
        <td>{row['제안 비중']:.2f}%</td>
        <td>{int(row['E'])}</td>
        <td>{int(row['S'])}</td>
        <td>{int(row['G'])}</td>
        <td style="text-align: left;">{row['종목 소개']}</td>
        </tr>"""


    html_code += """
    </tbody>
    </table>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    
    _,_,bt1,bt2 = st.columns(4)
    with bt1:
        check = st.button(label="포트폴리오 확인  ➡️")
        if check:
            screenshot = ImageGrab.grab(bbox=(400, 430, 790, 840))
            screenshot.save("pie_chart_capture.png")

    # 현재 스크립트 파일의 디렉토리 경로를 기준으로 상대 경로 설정
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(current_directory, "pie_chart_capture.png")


    # HTML 생성 함수
    def generate_html():
        # 데이터프레임 필터링 및 컬럼 이름 변경
        filtered_companies = pd.merge(company_colletion, top_companies, left_on='ticker', right_on='ticker')
        filtered_companies = filtered_companies[['Company', 'Weight', 'environmental', 'social', 'governance', '종목설명']]
        filtered_companies = filtered_companies.rename(columns={
            'Company': '종목명',
            'Weight': '제안 비중',
            'environmental': 'E',
            'social': 'S',
            'governance': 'G',
            '종목설명': '종목 소개'
        })
        filtered_companies = filtered_companies.sort_values(by='제안 비중', ascending=False)

        with open("pie_chart_capture.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        # HTML 콘텐츠 생성
        html_content = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>ESG 포트폴리오 제안서</title>
            <style>
                body {{
                    text-align: center;
                    font-family: Pretendard;
                }}
                .block {{
                    display: table;
                    width: 100%;
                    margin: 20px auto;
                }}
                .box {{
                    display: table-cell;
                    vertical-align: middle;
                    padding: 10px;
                }}
                .img {{
                    width: 100%;
                    max-width: 300px;
                }}
                table{{
                    margin: auto;
                }}
                th, td {{
                    text-align: center;
                    padding: 10px;
                    border: 1px solid #ddd;
                }}
                th {{
                    font-size:15px;
                    background-color: #e3edfa;
                }}
                .detail-table-container {{
                    width: 100%;
                    margin-top: 40px;
                }}
            </style>
        </head>
        <body>
            <h1 style="color: #666666;">{user_name}을 위한 ESG 중심 포트폴리오 제안서</h1>
            <p>다음은 {user_name}의 ESG 선호도를 바탕으로 최적화된 포트폴리오 비중입니다.</p>
            <div class="block">
                <div class="box">
                    <img src="data:image/png;base64,{encoded_string}" alt="ESG 포트폴리오 파이차트" class="img">
                </div>
                <div class="box">
                    <br>
                    <h2 style="font-family: Pretendard;font-size:20px;">ESG 관심도</h2>
                    <table style="width: 90%;">
                        <tr>
                            <th>환경</th>
                            <td>{e_value}</td>
                        </tr>
                        <tr>
                            <th>사회</th>
                            <td>{s_value}</td>
                        </tr>
                        <tr>
                            <th>거버넌스</th>
                            <td>{g_value}</td>
                        </tr>
                    </table>
                    <h2 style="font-family: Pretendard;font-size:20px;">포트폴리오 정보</h2>
                    <table style="width: 90%;">
                        <tr>
                            <th>예상 수익률</th>
                            <td>{expected_return:.2%}</td>
                        </tr>
                        <tr>
                            <th>예상 변동성</th>
                            <td>{expected_volatility:.2%}</td>
                        </tr>
                        <tr>
                            <th>샤프 비율</th>
                            <td>{sharpe_ratio:.2f}</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div class="detail-table-container">
                <table class="detail-table">
                    <thead>
                    <tr>
                        <th rowspan='2'>종목</th>
                        <th rowspan='2'>제안 비중</th>
                        <th colspan="3">ESG Score<br>(2023)</th>
                        <th rowspan='2'>종목 소개</th>
                    </tr>
                    <tr>
                        <th>E</th>
                        <th>S</th>
                        <th>G</th>
                    </tr>
                    </thead>
        """
        percent = 0
        for _, row in filtered_companies.iterrows():
            if float(f"{row['제안 비중']:.2f}") == 0.00:
                percent = 100 - percent
                html_content += f"""<tr>
                    <td>{row['종목명']}</td>
                    <td>{percent:.2f}%</td>
                    <td>{int(row['E'])}</td>
                    <td>{int(row['S'])}</td>
                    <td>{int(row['G'])}</td>
                    <td style="text-align: left;">{row['종목 소개']}</td>
                    </tr>
                    """
                break
                
            html_content += f"""<tr>
                <td>{row['종목명']}</td>
                <td>{row['제안 비중']:.2f}%</td>
                <td>{int(row['E'])}</td>
                <td>{int(row['S'])}</td>
                <td>{int(row['G'])}</td>
                <td style="text-align: left;">{row['종목 소개']}</td>
                </tr>
                """
            percent += float(f"{row['제안 비중']:.2f}")
            
        html_content += """
            <tfoot>
            <tr>
                <td colspan="6" style="font-size:15px; text-align: left;font-family:Pretendard;">
                    <p>해당 차트의 환경(E), 사회(S), 거버넌스(G)의 점수는 2023년 기준 점수입니다.</p>
                </td>
            </tr>
            </tfoot>
        """

        html_content += f"""
                </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        return html_content

    # HTML 저장 및 PDF 변환 함수
    def save_as_pdf(html_content):
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        options = {
            'enable-local-file-access': None,  # 로컬 파일 접근 허용
            'encoding': "UTF-8",  # UTF-8 인코딩 설정
            'no-pdf-compression': ''  # 폰트 압축 방지
        }
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
            # HTML 파일 저장
            tmp_html.write(html_content.encode('utf-8'))
            tmp_html_path = tmp_html.name

        # PDF 변환 파일 경로 설정
        pdf_path = tmp_html_path.replace(".html", ".pdf")

        # PDF 변환
        pdfkit.from_file(tmp_html_path, pdf_path, configuration=config)

        # Streamlit 다운로드 버튼 생성
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="💾 pdf 다운",
                data=pdf_file,
                file_name="esg_report.pdf",
                mime="application/pdf"
            )
    
    if check:
        with bt2:
            html_content = generate_html()
            save_as_pdf(html_content)


            
# col_1, col_2,col_3,col_4 = st.columns(4)
col_1, col_2, col_3 = st.columns(3)

with col_1:
    if clicked_points:
        clicked_point = clicked_points[0]
        if 'pointNumber' in clicked_point:
            company_index = clicked_point['pointNumber']
            if company_index < len(top_companies):
                company_info = top_companies.iloc[company_index]
                clicked_company = company_info['Company']
                st.markdown(f"""<div>
                            <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">{clicked_company} ESG 스코어</h2>
                            </div>
                """, unsafe_allow_html=True)
                clicked_df = dummy[dummy['Company'] == clicked_company]
                clicked_df['Year'] = clicked_df['Year'].astype(int)
                clicked_df = clicked_df[['Year', 'environmental', 'social', 'governance']]
                clicked_df = clicked_df.melt(id_vars='Year',
                         value_vars=['environmental', 'social', 'governance'],
                         var_name='Category',
                         value_name='Score')

                fig = px.line(clicked_df, x='Year', y='Score', color='Category')
                fig.update_layout(showlegend=True,
                    legend=dict(
                        orientation='h',  # 가로 방향으로 배치
                        yanchor='bottom',  # 범례의 y축 앵커를 하단에 맞추기
                        y=-0.6,  # 범례를 그래프 아래로 이동, 적절한 값으로 수정
                        xanchor='center',  # 범례의 x축 앵커를 중앙에 맞추기
                        x=0.5
                    ), width=750,height=350)
                # fig.update_xaxes(showticklabels=False, title='')
                # fig.update_yaxes(showticklabels=False, title='')

                # 그래프 출력
                st.plotly_chart(fig)

    else:
        st.write(' ')
        
with col_2:
    if clicked_points:
        st.markdown(f"""<div>
                            <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">&emsp;&ensp;{clicked_company} &ensp;주가 그래프</h2>
                            </div>
            """, unsafe_allow_html=True)
                
        company_choices = top_companies['Company'].tolist()
        ticker_choices = top_companies['ticker'].tolist()
        ticker_choices = [ticker.replace('.KS', '') for ticker in ticker_choices]

        if st.session_state['code_index'] >= len(company_choices):
            st.session_state['code_index'] = 0

        choice = clicked_company
        code_index = company_choices.index(choice)
        code = ticker_choices[code_index] 

        chart_style = 'default'
    
        # 세션 상태 업데이트
        st.session_state['ndays'] = 1825
        st.session_state['code_index'] = code_index
        st.session_state['chart_style'] = chart_style
        st.session_state['volume'] = True
        
        # 선택된 종목의 주가 데이터 로드
        data = load_stock_data(code, 1825)
        
        # 주가 차트 시각화 함수 호출
        plotChart(data)
        
    else:
        st.write('')




# 회사 이름 정규화 함수
def normalize_company_name(name):
    return unicodedata.normalize('NFC', name).strip()


# word_freq_df의 'company' 컬럼 정규화
word_freq_df['company'] = word_freq_df['company'].apply(normalize_company_name)


# 가중 평균 워드 클라우드 생성 함수
def generate_blended_word_cloud(top_companies, word_freq_df):
    blended_word_freq = Counter()

    # top_companies에서도 회사 이름 정규화
    top_companies['Company'] = top_companies['Company'].apply(normalize_company_name)

    for _, row in tqdm(top_companies.iterrows(), total=top_companies.shape[0], desc="Generating Blended Word Cloud"):
        company_name = row['Company']
        weight = row['Weight']

        # 해당 회사의 단어 빈도 필터링
        company_word_freq = word_freq_df[word_freq_df['company'] == company_name]

        if company_word_freq.empty:
        #     st.warning(f"{company_name}의 빈도 데이터가 없습니다.")
            continue

        # 각 단어에 대해 가중치를 곱한 빈도 계산
        for _, word_row in company_word_freq.iterrows():
            word = word_row['word']
            freq = word_row['frequency']
            blended_word_freq[word] += freq * weight

    # 워드 클라우드 생성 및 반환
    if not blended_word_freq:
        st.warning("워드 클라우드를 생성할 데이터가 없습니다.")
        return None

    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/malgun.ttf',  # 한글 폰트 설정
        background_color='white',
        width=800,
        height=600
    ).generate_from_frequencies(blended_word_freq)

    return wordcloud


# Streamlit column for Word Cloud display
with col_3:
    if clicked_points:
        st.markdown(f"""<div>
                                <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">포트폴리오 기반 워드 클라우드</h2>
                                </div>
                """, unsafe_allow_html=True)
        # 미리 선언된 top_companies를 기반으로 워드 클라우드 생성
        wordcloud = generate_blended_word_cloud(top_companies, word_freq_df)

        # 워드 클라우드 출력
        if wordcloud:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("생성할 데이터가 충분하지 않아 워드 클라우드를 표시할 수 없습니다.")

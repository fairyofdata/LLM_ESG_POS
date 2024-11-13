import streamlit as st
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service 
import pandas as pd
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
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
from cvxopt import matrix, solvers
from PIL import Image
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
from pymongo import MongoClient
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import unicodedata
import matplotlib.pyplot as plt
from pypfopt import risk_models, BlackLittermanModel, expected_returns
import os
import pdfkit
from pdfkit.api import configuration
import tempfile
from  streamlit_vertical_slider import vertical_slider
import base64
from dotenv import load_dotenv
import pyautogui
from fpdf import FPDF
import pyscreenshot as ImageGrab

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

for key in ['environmental', 'social', 'governance']:
    if key not in st.session_state['sliders']:
        st.session_state['sliders'][key] = 0
        
# MongoDB 연결 설정
# load_dotenv()
# client = MongoClient(os.getenv("mongodb_url"))
# db = client['kwargs']
# collection = db['kwargs']
# connection_string = mongodb_url  # MongoDB 연결 문자열을 입력하세요
# client = MongoClient(os.getenv("mongodb_url"))
# db = client['kwargsai']
# collection = db['test_collection']
# connection_string = "mongodb+srv://kwargs:57qBBuXYQel4W6oV@kwargsai.5yhiymt.mongodb.net/?retryWrites=true&w=majority&appName=kwargsai" # MongoDB 연결 문자열을 입력하세요
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

survey_result = pd.read_csv(r"C:\esgpage\LLM.ESG.POS\interface\survey_result.csv", encoding='utf-8', index_col=0)
with open(r"C:\esgpage\LLM.ESG.POS\interface\user_investment_style.txt", 'r', encoding='utf-8') as f:
    user_investment_style = f.read().strip()

with open(r"C:\esgpage\LLM.ESG.POS\interface\user_interest.txt", 'r', encoding='utf-8') as f:
    user_interest = f.read().strip()

with open(r"C:\esgpage\LLM.ESG.POS\interface\user_name.txt", 'r', encoding='utf-8') as f:
    user_name = f.read().strip()
    
company_list = pd.read_excel(r"C:\esgpage\LLM.ESG.POS\interface\텍스트 데이터 수집 현황 + 평가기관 점수 수집 + 기업 정보 요약.xlsx")

# 전처리 함수 정의
def preprocess_data(df):
    # 기존 컬럼명을 사용할 수 있도록 유효성을 확인
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

# step 1 : load the provided dataset
file_path = r"C:\esgpage\LLM.ESG.POS\interface\241113_dummy_sample.csv"
# file_path = r"interface/241007_dummy_update.csv"
dummy = pd.read_csv(file_path, encoding='euc-kr')
# dummy = pd.read_csv(file_path, encoding='cp949')
# dummy = pd.read_csv(file_path, encoding='utf-8')
# dummy = pd.read_csv(file_path)
df_new = preprocess_data(dummy)        

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

# 포트폴리오 비중 계산 함수 with CVXOPT
# def calculate_portfolio_weights(top_companies):
#     tickers = top_companies['ticker'].tolist()
#     price_data = yf.download(tickers, start="2019-01-01", end="2023-01-01")['Adj Close']
#     price_data = price_data.dropna(axis=1, how='any')
#     if price_data.isnull().values.any():
#         return "일부 데이터가 누락되었습니다. 다른 기업을 선택해 주세요.", None

#     # 일별 수익률 계산
#     returns = price_data.pct_change().dropna()

#     # 평균 수익률과 공분산 행렬
#     mu = returns.mean().values
#     Sigma = returns.cov().values

#     # `cvxopt`에서 사용할 행렬 형식으로 변환
#     n = len(mu)
#     P = matrix(Sigma)
#     q = matrix(np.zeros(n))
#     G = matrix(-np.eye(n))
#     h = matrix(np.zeros(n))
#     A = matrix(1.0, (1, n))
#     b = matrix(1.0)

#     # 쿼드라틱 프로그래밍 솔버 실행
#     sol = solvers.qp(P, q, G, h, A, b)

#     # 최적 가중치 추출
#     weights = np.array(sol['x']).flatten()

#     # 포트폴리오 성과 지표 계산
#     expected_return = np.dot(weights, mu)
#     expected_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
#     sharpe_ratio = expected_return / expected_volatility

#     # 가중치 정리
#     cleaned_weights = dict(zip(tickers, weights))

#     return cleaned_weights, (expected_return, expected_volatility, sharpe_ratio)


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
                color: black;
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

# 블랙리터만 모델 적용 함수
def calculate_portfolio_weights(df, esg_weights, user_investment_style):
    tickers = df['ticker'].tolist()
    price_data = yf.download(tickers, start="2019-01-01", end="2023-01-01")['Adj Close']
    price_data = price_data.dropna(axis=1)
    if price_data.isnull().values.any():
        return "일부 데이터가 누락되었습니다. 다른 기업을 선택해 주세요.", None
    
    # 평균 수익률과 공분산 행렬
    mu_market = expected_returns.capm_return(price_data)  # CAPM을 통한 시장 균형 수익률 계산
    Sigma = risk_models.sample_cov(price_data)  # 샘플 공분산 행렬
    
    esg_weights['environmental'] *= 1/700
    esg_weights['social'] *= 1/700
    esg_weights['governance'] *= 1/700
    
    # 사용자 선호도를 반영한 ESG 점수 조정
    df['final_esg_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    # 사용자 투자시 고려하는 부분에 따른 가중치 설정
    if user_investment_style == "재무적인 요소를 중심적으로 고려한다.":
        esg_weight_factor = 0.5
    elif user_investment_style == "ESG와 재무적인 요소를 모두 고려한다.":
        esg_weight_factor = 1.0
    elif user_investment_style == "ESG 요소를 중심적으로 고려한다.":
        esg_weight_factor = 2.0

    # 최종 ESG 점수와 성향에 따른 조정
    df['adjusted_esg_score'] = df['final_esg_score'] * esg_weight_factor

    valid_tickers = price_data.columns.tolist()
    df_valid = df[df['ticker'].isin(valid_tickers)]
    # 사용자 ESG 점수를 투자자의 견해로 반영 (Q: 주관적 수익률 벡터)
    P = np.eye(len(valid_tickers))
    Q = df_valid['adjusted_esg_score'].values  # Q 벡터: 각 자산에 대한 투자자의 의견 (ESG 점수 반영)
    # Black-Litterman 모델 적용
    bl = BlackLittermanModel(Sigma, pi=mu_market, P=P, Q=Q)
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
    cleaned_weights = dict(zip(tickers, weights))

    return cleaned_weights, (expected_return, expected_volatility, sharpe_ratio)


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
            <a href="#hover_text" style="color: black; font-family: Pretendard; font-size: 20px; text-align: center; text-decoration: none;font-weight:bold;">{origin_text}&ensp;&ensp;</a>
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
                        font-family: Pretendard;
                    }
                </style>
                </head>
    """, unsafe_allow_html=True)

    sl1, sl2, sl3= st.columns(3)
    with sl1:
        origin_e = survey_result.loc['E'].sum() * 10 / 4.99
        display_text_on_hover('탄소 관리, 오염물질 및 폐기물 관리, 기후 변화 전략 등과 관련된 정책',1,'&emsp;E')
        e_value = vertical_slider(
            label = "환경", 
            key = "environmental" ,
            height = 350, 
            step = 0.1,
            default_value=survey_result.loc['E'].sum() * 10 / 4.99,#Optional - Defaults to 0
            min_value= 0.0, # Defaults to 0
            max_value= 10.0, # Defaults to 10
            track_color = "#f0f0f0", #Optional - Defaults to #D3D3D3
            slider_color = '#006699', #Optional - Defaults to #29B5E8
            thumb_color = "#FF9933",
            value_always_visible = True ,#Optional - Defaults to False
        )
    with sl2:
        display_text_on_hover('탄소 관리, 오염물질 및 폐기물 관리, 기후 변화 전략 등과 관련된 정책',1,'&emsp;S')
        s_value = vertical_slider(
            label = "사회",  #Optional
            key = "social" ,
            height = 350, #Optional - Defaults to 300
            step = 0.1, #Optional - Defaults to 1
            default_value=survey_result.loc['S'].sum() *10/4.79,#Optional - Defaults to 0
            min_value= 0.0, # Defaults to 0
            max_value= 10.0, # Defaults to 10
            track_color = "#f0f0f0", #Optional - Defaults to #D3D3D3
            slider_color = '#006699', #Optional - Defaults to #29B5E8
            thumb_color = "#FF9933",
            value_always_visible = True ,#Optional - Defaults to False
        )
    with sl3:
        display_text_on_hover('탄소 관리, 오염물질 및 폐기물 관리, 기후 변화 전략 등과 관련된 정책',1,'&emsp;G')
        g_value = vertical_slider(
            label = "지배구조",  #Optional
            key = "governance" ,
            height = 350, #Optional - Defaults to 300
            step = 0.1, #Optional - Defaults to 1
            default_value=survey_result.loc['G'].sum()*10/4.16,
            min_value= 0.0, # Defaults to 0
            max_value= 10.0, # Defaults to 10
            track_color = "#f0f0f0", #Optional - Defaults to #D3D3D3
            slider_color = '#006699', #Optional - Defaults to #29B5E8
            thumb_color = "#FF9933",
            value_always_visible = True ,#Optional - Defaults to False
        )

# 사용자의 ESG 선호도
esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}       
st.write('')
    
# 포트폴리오 비중 계산
# top_companies = recommend_companies(esg_weights,df_new)

# 블랙리터만 적용 버전
industries = df_new['industry'].unique().tolist()
processed_df = df_new[df_new['industry'].isin(industries)].copy()
portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights,user_investment_style) # cleaned_weights:각 자산에 할당된 최적의 투자 비율, performance:최적화된 포트폴리오의 성과 지표
top_companies = df_new[df_new['ticker'].isin(portfolio_weights)].copy()
# ticker 열과 portfolio_weights를 매핑하여 새로운 top_companies 데이터프레임 생성_ 블랙리터만 모델 버전
# portfolio_weights의 값을 'Weight' 컬럼으로 추가
total_weight = sum(portfolio_weights.values())
top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
top_companies['Weight'] = top_companies['Weight'] * 100

# cvxopt 적용 버전
# portfolio_weights, portfolio_performance = calculate_portfolio_weights(top_companies)
# industries = df_new['industry'].unique().tolist()
    # processed_df = df_new[df_new['industry'].isin(industries)].copy()

# top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
    # top_companies['Weight'] = top_companies['ticker'].map(cleaned_weights)
    
with col2:
    st.markdown(f"""<div>
                        <h2 style="font-family: Pretendard;font-size: 13px; text-align:center; text-decoration: none;">차트에서 여러분의 관심 회사 이름을 클릭하여<br>더 다양한 정보를 경험해 보세요.</h2>
                    </div>
            """, unsafe_allow_html=True)
    
    # 전체 Weight 합계 계산
    total_weight = top_companies['Weight'].sum()
    # Weight 기준으로 최소 비율 이하의 회사 필터링
    # top_companies = top_companies[top_companies['Weight'] / total_weight * 100 >= 5.0]
    
    # Weight를 기준으로 내림차순 정렬
    top_companies = top_companies.sort_values(by='Weight', ascending=False)
    
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
    company_list['종목코드'] = company_list['종목코드'].str[1:]
    top_companies['ticker'] = top_companies['ticker'].str.replace('.KS', '')

    expected_return = portfolio_performance[0]
    expected_volatility = portfolio_performance[1]
    sharpe_ratio = portfolio_performance[2]   

    top5_companies = top_companies.nlargest(5, 'Weight')
    filtered_companies = pd.merge(company_list, top5_companies, left_on='종목코드', right_on='ticker')
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
            font-family: Pretendard;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        </style>
    </style>
    <table>
        <thead>
        <tr>
            <th>종목명</th>
            <th>제안 비중</th>
            <th>E</th>
            <th>S</th>
            <th>G</th>
            <th>종목 소개</th>
        </tr>
        </thead>
        <tbody>
    """
    
    filtered_companies = filtered_companies.sort_values(by='제안 비중', ascending=False)
    for _, row in filtered_companies.iterrows():
        html_code += f"""<tr>
        <td style="font-size=13px;">{row['종목명']}</td>
        <td>{row['제안 비중']:.2f}%</td>
        <td>{row['E']:.1f}</td>
        <td>{row['S']:.1f}</td>
        <td>{row['G']:.1f}</td>
        <td style="text-align: left;">{row['종목 소개']}</td>
        </tr>
        """

    html_code += """
    </tbody>
    </table>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    
    _,_,bt1,bt2 = st.columns(4)
    with bt1: 
        if st.button(label="포트폴리오 확인  ➡️"):
            screenshot = ImageGrab.grab(bbox=(400,420,790,830))
            screenshot.save("pie_chart_capture.png") 
        
    def generate_html():
        filtered_companies = pd.merge(company_list, top_companies, left_on='종목코드', right_on='ticker')
        filtered_companies = filtered_companies[['Company','Weight','environmental','social','governance','종목설명']]
        filtered_companies = filtered_companies.rename(columns={
            'Company': '종목명',
            'Weight': '제안 비중',
            'environmental': 'E',
            'social': 'S',
            'governance': 'G',
            '종목설명' :'종목 소개'
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
                    background-color: #e3edfa;
                }}
                .detail-table-container {{
                    width: 100%;
                    margin-top: 40px;
                }}
            </style>
        </head>
        <body>
            <h1>{user_name}을 위한 ESG 중심 포트폴리오 제안서</h1>
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
                    <tr>
                        <th>종목</th>
                        <th>제안<br>비중</th>
                        <th>환경</th>
                        <th>사회</th>
                        <th>거버넌스</th>
                        <th>종목 소개</th>
                    </tr>
        """

        for _, row in filtered_companies.iterrows():
            html_content += f"""<tr>
            <td>{row['종목명']}</td>
            <td>{row['제안 비중']:.3f}%</td>
            <td>{row['E']:.1f}</td>
            <td>{row['S']:.1f}</td>
            <td>{row['G']:.1f}</td>
            <td>{row['종목 소개']}</td>
            </tr>
            """
        html_content += f"""
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
    
    with bt2:
        html_content = generate_html()
        save_as_pdf(html_content)

   
            
# col_1, col_2,col_3,col_4 = st.columns(4)
col_1, col_2, col_3 = st.columns(3)

# 
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
                
with col_3:
    if clicked_points:
        st.markdown(f"""<div>
                            <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">{clicked_company}&ensp;워드 클라우드</h2>
                            </div>
                """, unsafe_allow_html=True)
        # MongoDB에서 Company 필드의 고유 값들을 불러오기
        company_list = collection.distinct('Company')
            
        # 유니코드 정규화를 사용해 clicked_company와 company_list 값을 동일한 형식으로 변환
        clicked_company_normalized = unicodedata.normalize('NFC', clicked_company)

        # 리스트 내의 각 값을 정규화 후 비교
        clicked_company = next((company for company in company_list if unicodedata.normalize('NFC', company) == clicked_company_normalized), None)
        titles = collection.find({'Company': clicked_company}, {'_id': 0, 'title': 1})

# 불러온 데이터 리스트로 저장
        title_list = [document['title'] for document in titles if 'title' in document]

# title_list가 비어 있는지 확인
        if not title_list:
            st.warning("데이터가 없습니다. 다른 기업을 선택해 주세요.")
        else:
    # 형태소 분석기 설정
            okt = Okt()
            nouns_adj_verbs = []

    # 명사, 형용사만 추출
            for title in title_list:
                tokens = okt.pos(title, stem=True)
                for word, pos in tokens:
                    if pos in ['Noun', 'Adjective']:
                        nouns_adj_verbs.append(word)

    # 빈도수 계산
            word_counts = Counter(nouns_adj_verbs)
            data = word_counts.most_common(500)
            tmp_data = dict(data)

    # 워드 클라우드 생성 - 폰트 경로 확인 후 설정
            try:
                wordcloud = WordCloud(
                    font_path='C:/Windows/Fonts/malgun.ttf',  # Windows 시스템에서 사용할 기본 폰트 설정
                    background_color='white',
                    width=800,
                    height=600
                        ).generate_from_frequencies(tmp_data)
            except OSError:
                st.error("폰트 파일을 불러올 수 없습니다. 폰트 경로를 확인하거나 설치해 주세요.")
                st.stop()

    # 워드 클라우드 시각화 및 출력
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

    # Streamlit에 워드 클라우드 출력
            st.pyplot(fig)
            
            

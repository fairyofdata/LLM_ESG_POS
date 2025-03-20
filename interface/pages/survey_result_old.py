#Streamlit and web related library
import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Library related to data processing and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
import os
import pickle as pkle

# Library related to certification and security
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from streamlit_authenticator.utilities import (
    CredentialsError, ForgotError, Hasher, LoginError, RegisterError,
    ResetError, UpdateError
)
from passlib.context import CryptContext
from dotenv import load_dotenv

# Library related to visualization and floating
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from wordcloud import WordCloud
from collections import Counter

# Library related to finance and optimization
import FinanceDataReader as fdr
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel
from cvxopt import matrix, solvers

# Other Utility Library
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

# Hangul text analysis
from konlpy.tag import Okt

#Streamlit extension function
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from streamlit_vertical_slider import vertical_slider
from streamlit_plotly_events import plotly_events
from streamlit_js_eval import streamlit_js_eval

# Set the relative path based on the location of the current script file
current_directory = os.path.dirname(__file__)

#Referral variable definition
survey_result_file = os.path.join(current_directory, "survey_result.csv")
user_investment_style_file = os.path.join(current_directory, "user_investment_style.txt")
user_interest_file = os.path.join(current_directory, "user_interest.txt")
user_name_file = os.path.join(current_directory, "user_name.txt")
company_colletion_file = os.path.join(current_directory, 'company_collection.csv')
word_freq_file = os.path.join(current_directory, "company_word_frequencies.csv")
survey_result_page = 'pages/survey_result.py'

# Import after confirming that the file exists
if os.path.exists(survey_result_file):
    survey_result = pd.read_csv(survey_result_file, encoding='utf-8', index_col=0)
else:
    # If there is no file, create an empty data frame as the default value
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
        
# MongoDB Connection Settings (August Harry)
#Load_dotenv ()
# Client = Mongoclient (OS.GETENV ("mongodb_url"))
# DB = Client ['Kwargs']
# Collection = DB ['Kwargs']

# # MongoDB Connection (November Jiheon)
#Connection_string = "MongoDB+srv: // kwargs: 57QBBBUXYQEL4W6OV@v@vsai.5yhiymt.mongodb.net/? Retrywrites = True & W = Majority & Appname = kwargsai" " #MongoDB_URL #MongoDB Connection String
# Client = mongoclient (connection_string)
# DB = Client ['Kwargsai']
# Collection = DB ['test_collection']

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

#Destructive function definition

def preprocess_data(df):
    # Confirm the effectiveness of using the existing column name
    df = df.copy()
    if 'environmental' in df.columns and 'social' in df.columns and 'governance' in df.columns:
        # Convert the proportion of the ESG area into a percentage
        df['env_percent'] = df['environmental'] / (df['environmental'] + df['social'] + df['governance'])
        df['soc_percent'] = df['social'] / (df['environmental'] + df['social'] + df['governance'])
        df['gov_percent'] = df['governance'] / (df['environmental'] + df['social'] + df['governance'])

        # Calculate the final score for each area (Average_label required)
        df['env_score'] = df['average_label'] * df['env_percent']
        df['soc_score'] = df['average_label'] * df['soc_percent']
        df['gov_score'] = df['average_label'] * df['gov_percent']

        #Setting by year
        latest_year = df['Year'].max()
        year_weights = {
            latest_year: 0.5,
            latest_year - 1: 0.25,
            latest_year - 2: 0.125,
            latest_year - 3: 0.0625,
            latest_year - 4: 0.0625
        }

        # In total scores for each area reflecting the weight
        df['environmental'] = df.apply(lambda x: x['env_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['social'] = df.apply(lambda x: x['soc_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['governance'] = df.apply(lambda x: x['gov_score'] * year_weights.get(x['Year'], 0), axis=1)

        # The final score is derived by adding the score by year of the same company
        final_df = df.groupby(['Company', 'industry', 'ticker']).agg({
            'environmental': 'sum',
            'social': 'sum',
            'governance': 'sum'
        }).reset_index()

        return final_df
    else:
        raise KeyError("The expected columns 'environmental', 'social', and 'governance' are not present in the dataframe.")


# Go to the parent directory of the current script file and set the path
current_directory = os.path.dirname(os.path.dirname(__file__))

#Referral variable definition
file_path = "241113_dummy_update.csv"  # 올바른 파일 경로로 설정
dummy_file_path = os.path.join(current_directory, file_path)

# Read the necessary files
if os.path.exists(dummy_file_path):
    try:
        # Read the file by applying other encoding in the order of attempts
        try:
            dummy = pd.read_csv(dummy_file_path, encoding='euc-kr')
        except UnicodeDecodeError:
            try:
                dummy = pd.read_csv(dummy_file_path, encoding='cp949')
            except UnicodeDecodeError:
                dummy = pd.read_csv(dummy_file_path, encoding='utf-8')

        # Check if the file is read properly
        print("데이터프레임 미리보기:")
        print(dummy.head())
        print(f"데이터프레임의 컬럼 목록: {dummy.columns.tolist()}")

    except Exception as e:
        print(f"파일 읽기 오류 발생: {e}")
        dummy = pd.DataFrame()  # 오류 발생 시 기본값으로 빈 데이터프레임 사용
else:
    # Use empty data frame if there is no file
    print(f"파일이 존재하지 않습니다: {dummy_file_path}")
    dummy = pd.DataFrame()

# Data pretreatment step execution (set to run without an error even in the empty data frame)
df_new = preprocess_data(dummy) if not dummy.empty else pd.DataFrame()

# Output of data frame information and confirm'INDUSTRY 'column
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


# Bringing stocks corresponding to the Korea Exchange KOSPI index
@st.cache_data
def getSymbols(market='KOSPI',sort='Marcap'): # 정렬하는 기준을 시장가치(Marcap)으로 함
    df = fdr.StockListing(market)
    # Sort setting (= reverse row based on market cap)
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

# Candle chart output function
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


#Selection of top companies (based on ESG)
def recommend_companies(esg_weights, df):
    # Calculate the final score by reflecting the user's ESG preference weight in the pretreated data reflects the weight of ESG preference
    df['final_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    #Selection of the top 10 companies
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

#--- optimization algorithm ---
import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import yfinance as yf
from pypfopt import BlackLittermanModel, expected_returns, risk_models, CovarianceShrinkage

# Modified portfolio weight calculation function with black-litterman and reduced covariance matrix
# Existing method: The user's preference for ESG is reflected separately from the market return, which is insufficient in the optimization process.
# Improvement: Reflecting ESG preference, adjusting the market balance yield itself and reflecting this in optimization
# Black literal model function
def calculate_portfolio_weights(df, esg_weights, user_investment_style):
    # Data collection and pretreatment
    tickers = df['ticker'].tolist()
    price_data = yf.download(tickers, start="2019-01-01", end="2023-01-01")['Adj Close']
    price_data = price_data.dropna(axis=1)
    if price_data.isnull().values.any():
        return "일부 데이터가 누락되었습니다. 다른 기업을 선택해 주세요.", None

    # Calculation of average returns and covenant matrix
    mu_market = expected_returns.capm_return(price_data)  # CAPM을 통한 시장 균형 수익률 계산
    Sigma = risk_models.sample_cov(price_data)  # 샘플 공분산 행렬

    # Normalization of covenant matrix: Add a small value to prevent non -politicality
    Sigma += np.eye(Sigma.shape[0]) * 1e-6

    # ESG weight scaling (ratio adjustment)
    esg_weights = {key: value / 30000 for key, value in esg_weights.items()}

    # Calculation of the final ESG score reflecting the user preference and ESG weight
    df['final_esg_score'] = (
        esg_weights['environmental'] * df['environmental'] +
        esg_weights['social'] * df['social'] +
        esg_weights['governance'] * df['governance']
    )

    # ESG weight setting according to user investment style
    if user_investment_style == "재무적인 요소를 중심적으로 고려한다.":
        esg_weight_factor = 10.0
    elif user_investment_style == "ESG와 재무적인 요소를 모두 고려한다.":
        esg_weight_factor = 20.0
    elif user_investment_style == "ESG 요소를 중심적으로 고려한다.":
        #ESG_WEIGHT_FACTOR = 2.5
        esg_weight_factor = 100.0
    else:
        esg_weight_factor = 1.0  # 기본값 설정

    # Reflecting investment style in the final ESG score
    df['adjusted_esg_score'] = df['final_esg_score'] * esg_weight_factor

    Data to be reflected in the opinion of investors in #Black-Litterman model
    valid_tickers = price_data.columns.tolist()
    df_valid = df[df['ticker'].isin(valid_tickers)]

    # Improved P Matrix settings: Considering correlation by reflecting more diverse for each asset
    P = np.zeros((len(valid_tickers), len(valid_tickers)))
    np.fill_diagonal(P, [1.0 / len(valid_tickers)] * len(valid_tickers))

    # Q Vector Settings: Investors' opinion reflecting ESG scores
    Q = df_valid['adjusted_esg_score'].values

    # Black-Litterman model application
    tau = 0.1  # tau 값을 적절히 조정하여 모델 안정성 확보
    bl = BlackLittermanModel(Sigma, pi=mu_market, P=P, Q=Q, tau=tau)
    adjusted_returns = bl.bl_returns()

    # Optimization Problems Set and Optimal weight calculation
    n = len(mu_market)
    P_opt = matrix(Sigma.values)
    q_opt = matrix(-adjusted_returns.values)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    #Washing Quadrantic Programming Solver Run
    sol = solvers.qp(P_opt, q_opt, G, h, A, b)

    # Optimal weight extraction
    weights = np.array(sol['x']).flatten()

    # Calculation of portfolio performance indicators
    expected_return = np.dot(weights, mu_market)
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma.values, weights)))
    sharpe_ratio = expected_return / expected_volatility

    #Brubilion
    cleaned_weights = dict(zip(valid_tickers, weights))

    return cleaned_weights, (expected_return, expected_volatility, sharpe_ratio)

# Apply the final weight to optimized_weights
def calculate_adjusted_weights(df, optimized_weights, esg_weights,performance_metrics):
    environmental_scores = df['environmental']
    social_scores = df['social']
    governance_scores = df['governance']

    # Calcule esg-based Adjustment
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

    # Normalize Adjusted Weights to Sum to 1
    if isinstance(adjusted_weights, dict):
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {ticker: weight / total_weight for ticker, weight in adjusted_weights.items()}
    else:
        adjusted_weights /= adjusted_weights.sum()

    return adjusted_weights, performance_metrics
    # # Normalize esg Adjustment to have the Same Range as optimized_weights
    #ESG_ADJUSTMENT_NONORMALIZED = ESG_ADJUSTMENT / ESG_ADJUST.SUM ()

    # # Adjust the weights: 50% Original weight + 50% ESG-Adjusted weight
    # Adjusted_weights = 0.5 * optimized_weights + 0.5

    # # Normalize Adjusted Weights to Sum to 1
    # Adjusted_weights /= admin_weights.sum ()




    # Adjustment after optimization: weight modification by reflecting scores for each area
    #FOR TICKER in Cleaned_Weights:
    # Company_data = DF_VALID [DF_VALID ['Ticker'] == Ticker]
    #IF Not Company_data.empty:
    # Environmental_score = Company_data ['Environmental']. Values ​​[0]
    # SOCIAL_SCORE = Company_Data ['Social']. Values ​​[0]
    # GOVERNance_SCORE = Company_Data ['GOVERNANCE']. Values ​​[0]
    # Cleaned_weights [Ticker] = (Cleaned_weights [Ticker] * 0.5) + (
    # (Environmental_score * esg_weights ['Environmental'] +
    # Social_score * esg_weights ['Social'] +
    #GOVERNance_SCORE * esg_weights ['GOVERNance']) * 0.5
    #)

    # Return Cleaned_Weights, (Expected_Return, Expected_volatility, Sharpe_ratio)


# Output
# In the improved code, the user's ESG preference is directly reflected in the market balanced return,
# In the process of optimization, the user's preference is clearly revealed.
def display_text_on_hover(hover_text, i, origin_text):
    # Create a unique class name in each text hover area
    hover_class = f'hoverable_{i}'
    tooltip_class = f'tooltip_{i}'
    text_popup_class = f'text-popup_{i}'

    #Definition of unique CSS for each hover text
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

    #Defining HTML by modifying the style of Origin_text
    text_hover = f'''
        <div class="{hover_class}">
            <a href="#hover_text" style="color: #999999; font-family: Pretendard; font-size: 20px; text-align: center; text-decoration: none;font-weight:bold;">{origin_text}&ensp;&ensp;</a>
            <div class="{tooltip_class}"></div>
            <div class="{text_popup_class}">{hover_text}</div>
        </div>
    '''
    
    # Write dynamic HTML and CSS in content containers
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

            #Calculation of fluctuations
            change = today_kospi - yesterday_kospi
            change_percent = (change / yesterday_kospi) * 100

            # Output to Streamlit Metric
            st.metric(label="오늘의 코스피 지수", value=round(today_kospi, 2), delta=f"{round(change_percent, 2)}%",delta_color="inverse")

    with kosdaq:
        if not kosdaq_data.empty:
            yesterday_kosdaq = kosdaq_data.iloc[0]['Close']
            today_kosdaq = kosdaq_data.iloc[-1]['Close']

            #Calculation of fluctuations
            change = today_kosdaq - yesterday_kosdaq
            change_percent = (change / yesterday_kosdaq) * 100

            # Output to Streamlit Metric
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
    # User's ESG preference
    esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}            
    #Bay only Black Lighter
    industries = df_new['industry'].unique().tolist()
    processed_df = df_new[df_new['industry'].isin(industries)].copy()
    portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights, user_investment_style)
    # Portfolio_Weights, Portfolio_Performance = Calcule_Adjusted_Weights (Processed_df, Portfolio_Weights, ESG_WEIGHTS, PORTFOLIO_PERFORMANCE)
    # Portfolio_Weights, Portfolio_Performance = Calcule_Portfolio_Weights (Processed_df, ESG_WEIGHTS Cleaned_Weights: Optimal investment rate allocated to each asset, performance: Performance indicators of optimized portfolio
    top_companies = df_new[df_new['ticker'].isin(portfolio_weights)].copy()
    # Ticker column and portfolio_weights to map the new top_companies data frame _ Black literal model version
    Add the value of # Portfolio_Weights as 'Weight' column
    total_weight = sum(portfolio_weights.values())
    # Total_Weight = SUM (Portfolio_Weights.values)
    top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
    top_companies['Weight'] = top_companies['Weight'] * 100
    # Sorting the difference between the weight based on the weight
    top_companies = top_companies.sort_values(by='Weight', ascending=False)
    selected_companies = st.multiselect(
        "",
        top_companies['Company'],
        placeholder="제외하고 싶은 기업을 선택"
    )

    if selected_companies:
        top_companies = top_companies[~top_companies['Company'].isin(selected_companies)]
        
# User's ESG preference
esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}       
st.write('')
    
# Calculation of portfolio ratio
# TOP_COMPANIES = Recommend_Companies (ESG_WEIGHTS, DF_NEW)

#Bay only Black Lighter
#INDUSTRIES = df_new ['industrial']. Unique (). ToList ()
# Processed_df = df_new [df_new ['industrial']. Isin (industries)]. Copy ()
# Portfolio_Weights, Portfolio_Performance = Calcule_Portfolio_Weights (Processed_df, ESG_WEIGHTS
# Portfolio_Weights, Portfolio_PerFormance = Calcule_Adjusted_Weights (Processed_df, Portfolio_Weights ESG_WEIGHTS, PORTFOLIO_PERFORMANCE)
# Portfolio_weights, Portfolio_PerFormance = Calculate_Portfolio_Weights (Processed_df, ESG_WEIGHTS Cleaned_Weights: Optimal investment rate allocated to each asset, performance: Performance indicators of optimized portfolio
# TOP_COMPANIES = DF_NEW [DF_NEW ['Ticker']. Isin (Portfolio_Weights)]. Copy ()
# # Ticker column and portfolio_weights to map the new top_companies data frame _ Black Lighter only model version
# # PORTFOLIO_WEIGHTS adds the value of 'weight' column
# Total_Weight = SUM (Portfolio_Weights.values ​​())
# # Total_Weight = SUM (Portfolio_Weights.values)
# TOP_COMPANIES ['Weight'] = TOP_COMPANIES ['Ticker']. MAP (Portfolio_Weights)
# TOP_COMPANIES ['Weight'] = TOP_COMPANIES ['Weight'] * 100

# CVXOPT application version
# Portfolio_Weights, Portfolio_Performance = Calcule_Portfolio_Weights (TOP_COMPANES)
#INDUSTRIES = df_new ['industrial']. Unique (). ToList ()
    # Processed_df = df_new [df_new ['industrial']. Isin (industries)]. Copy ()

# TOP_COMPANIES ['Weight'] = TOP_COMPANIES ['Ticker']. MAP (Portfolio_Weights)
    
with col2:

    if selected_companies:
        top_companies = top_companies[~top_companies['Company'].isin(selected_companies)]
    st.markdown(f"""<div>
                        <h2 style="font-family: Pretendard;font-size: 13px; text-align:center; text-decoration: none;">차트에서 여러분의 관심 회사 이름을 클릭하여<br>더 다양한 정보를 경험해 보세요.</h2>
                    </div>
            """, unsafe_allow_html=True)
    
    # Total Weight Calculation
    total_weight = top_companies['Weight'].sum()
    # Filtering company with a minimum ratio based on # weight
    # TOP_COMPANIES = TOP_COMPANIES [TOP_COMPANIES [['Weight'] / Total_Weight * 100> = 5.0]
    
    
    # Create a pie chart
    fig = px.pie(
        top_companies, 
        names='Company', 
        values='Weight', 
        color_discrete_sequence=px.colors.qualitative.G10,
        custom_data=top_companies[['environmental', 'social', 'governance']]
    )

    #ESG information as CustomData
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

    # Chart layout setting
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
    # Expected return, volatility, and sharp ratio at the top
    # _, COL1, COL2, COL3, _ = st.columns ([[2,3,3,3,2)))
    col1, col2, col3 = st.columns(3)
    with col1:
        display_text_on_hover("해당 지표는 포트폴리오가 1년 동안 벌어들일 것으로 예상되는 수익률입니다.",1,f"연간 기대 수익률 &emsp; {expected_return * 100:.2f} %")
        st.markdown('')
    with col2:
        display_text_on_hover("해당 지표는 수익률이 얼마나 변동할 수 있는지를 나타내는 위험 지표입니다.",1,f"연간 변동성 &emsp; {expected_volatility * 100:.2f} %")
    with col3:
        display_text_on_hover("해당 지표는 포트폴리오가 위험 대비 얼마나 효과적으로 수익을 내는지를 나타내는 성과 지표입니다.",1,f"샤프 비율 &emsp;{sharpe_ratio * 100:.2f}")

    # Add tooltip to HTML code and convert to two row structures
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

    # Set your relative path based on the current script file directory path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(current_directory, "pie_chart_capture.png")


    # HTML creation function
    def generate_html():
        # Change data frame filtering and column name
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

        # Create html content
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

    # HTML storage and PDF conversion function
    def save_as_pdf(html_content):
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        options = {
            'enable-local-file-access': None,  # 로컬 파일 접근 허용
            'encoding': "UTF-8",  # UTF-8 인코딩 설정
            'no-pdf-compression': ''  # 폰트 압축 방지
        }
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
            #Save html file
            tmp_html.write(html_content.encode('utf-8'))
            tmp_html_path = tmp_html.name

        # PDF conversion file path setting
        pdf_path = tmp_html_path.replace(".html", ".pdf")

        # PDF conversion
        pdfkit.from_file(tmp_html_path, pdf_path, configuration=config)

        #Streamlit download button creation
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


            
# COL_1, COL_2, COL_3, COL_4 = St.columns (4)
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
                # Fig.update_xaxes (showticklabels = false, title = '')
                # Fig.update_yaxes (showticklabels = false, title = '')

                # Graph output
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
    
        # Session status update
        st.session_state['ndays'] = 1825
        st.session_state['code_index'] = code_index
        st.session_state['chart_style'] = chart_style
        st.session_state['volume'] = True
        
        # Load the stock price of the selected stock
        data = load_stock_data(code, 1825)
        
        # Call the stock chart visualization function
        plotChart(data)
        
    else:
        st.write('')




# Company name normalization function
def normalize_company_name(name):
    return unicodedata.normalize('NFC', name).strip()


# Word_FREQ_DF's'COMPANY 'column normalization
word_freq_df['company'] = word_freq_df['company'].apply(normalize_company_name)


# Weighted average word cloud generation function
def generate_blended_word_cloud(top_companies, word_freq_df):
    blended_word_freq = Counter()

    # Normalization of the company name in TOP_COMPANIES
    top_companies['Company'] = top_companies['Company'].apply(normalize_company_name)

    for _, row in tqdm(top_companies.iterrows(), total=top_companies.shape[0], desc="Generating Blended Word Cloud"):
        company_name = row['Company']
        weight = row['Weight']

        # Filtering the word frequency of the company
        company_word_freq = word_freq_df[word_freq_df['company'] == company_name]

        if company_word_freq.empty:
        # St.warning (F "{Company_Name} is not the frequency data.")
            continue

        # Calculation of frequency multiplied by each word
        for _, word_row in company_word_freq.iterrows():
            word = word_row['word']
            freq = word_row['frequency']
            blended_word_freq[word] += freq * weight

    #Wordcloud creation and return
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


# Streamlit Column for Word Cloud Display
with col_3:
    if clicked_points:
        st.markdown(f"""<div>
                                <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">포트폴리오 기반 워드 클라우드</h2>
                                </div>
                """, unsafe_allow_html=True)
        # Create a Word Cloud based on the pre -declared TOP_COMPANIES
        wordcloud = generate_blended_word_cloud(top_companies, word_freq_df)

        #Wordcloud output
        if wordcloud:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("생성할 데이터가 충분하지 않아 워드 클라우드를 표시할 수 없습니다.")

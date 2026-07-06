# Streamlit and web related libraries
import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Data processing and analysis related libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
import os
import pickle as pkle

# Authentication and security related libraries
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from streamlit_authenticator.utilities import (
    CredentialsError, ForgotError, Hasher, LoginError, RegisterError,
    ResetError, UpdateError
)
from passlib.context import CryptContext
from dotenv import load_dotenv

# Visualization and plotting related libraries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from wordcloud import WordCloud
from collections import Counter

# Finance and optimization related libraries
import FinanceDataReader as fdr
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel
from cvxopt import matrix, solvers
from sklearn.covariance import LedoitWolf  # Added: Ledoit-Wolf method for covariance estimation

# Other utility libraries
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

# Korean text analysis
from konlpy.tag import Okt

# Streamlit extension features
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from streamlit_vertical_slider import vertical_slider
from streamlit_plotly_events import plotly_events
from streamlit_js_eval import streamlit_js_eval

# Set relative path based on current script file location
current_directory = os.path.dirname(__file__)

# Define path variables
survey_result_file = os.path.join(current_directory, "survey_result.csv")
user_investment_style_file = os.path.join(current_directory, "user_investment_style.txt")
user_interest_file = os.path.join(current_directory, "user_interest.txt")
user_name_file = os.path.join(current_directory, "user_name.txt")
company_colletion_file = os.path.join(current_directory, 'company_collection.csv')
word_freq_file = os.path.join(current_directory, "company_word_frequencies.csv")
survey_result_page = 'pages/survey_result.py'

# Check if files exist and load them
if os.path.exists(survey_result_file):
    survey_result = pd.read_csv(survey_result_file, encoding='utf-8', index_col=0)
else:
    # Create empty dataframe if file doesn't exist
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
    page_title="설문 조사 결과",
    page_icon=":earth_africa:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과', icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스', icon="🆕")
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

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
         unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
         unsafe_allow_html=True)
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
    """, unsafe_allow_html=True)


# Define preprocessing function

def preprocess_data(df):
    # Check validity to use existing column names
    df = df.copy()
    if 'environmental' in df.columns and 'social' in df.columns and 'governance' in df.columns:
        # Convert ESG area weights to percentages
        df['env_percent'] = df['environmental'] / (df['environmental'] + df['social'] + df['governance'])
        df['soc_percent'] = df['social'] / (df['environmental'] + df['social'] + df['governance'])
        df['gov_percent'] = df['governance'] / (df['environmental'] + df['social'] + df['governance'])

        # Calculate final score for each area (requires average_label)
        df['env_score'] = df['average_label'] * df['env_percent']
        df['soc_score'] = df['average_label'] * df['soc_percent']
        df['gov_score'] = df['average_label'] * df['gov_percent']

        # Set year weights
        latest_year = df['Year'].max()
        year_weights = {
            latest_year: 0.5,
            latest_year - 1: 0.25,
            latest_year - 2: 0.125,
            latest_year - 3: 0.0625,
            latest_year - 4: 0.0625
        }

        # Apply weights to each area score
        df['environmental'] = df.apply(lambda x: x['env_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['social'] = df.apply(lambda x: x['soc_score'] * year_weights.get(x['Year'], 0), axis=1)
        df['governance'] = df.apply(lambda x: x['gov_score'] * year_weights.get(x['Year'], 0), axis=1)

        # Sum scores by company to get final scores
        final_df = df.groupby(['Company', 'industry', 'ticker']).agg({
            'environmental': 'sum',
            'social': 'sum',
            'governance': 'sum'
        }).reset_index()

        return final_df
    else:
        raise KeyError(
            "The expected columns 'environmental', 'social', and 'governance' are not present in the dataframe.")


# Set path to parent directory of current script file
current_directory = os.path.dirname(os.path.dirname(__file__))

# Setup ESG data file paths
esg_files = {
    2019: os.path.join(current_directory, "esg_data_2019.csv"),
    2020: os.path.join(current_directory, "esg_data_2020.csv"),
    2021: os.path.join(current_directory, "esg_data_2021.csv"),
    2022: os.path.join(current_directory, "esg_data_2022.csv"),
    2023: os.path.join(current_directory, "esg_data_2023.csv"),
}


# ESG data loading function
def load_esg_data():
    """
    Load and process ESG data files by year
    Returns:
        dict: Dictionary containing ESG data by year
    """
    esg_data = {}

    for year, file_path in esg_files.items():
        try:
            # Try loading CSV file with different encodings
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='euc-kr')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')

            # Convert ticker to string and pad to 6 digits
            if 'ticker' in df.columns:
                df["ticker"] = df["ticker"].astype(str).str.zfill(6)

            # Set index
            if 'ticker' in df.columns:
                df.set_index("ticker", inplace=True)

            # Save dataframe
            esg_data[year] = df

            # Print log message
            print(f"Loaded ESG data for {year}: {len(df)} companies")

        except Exception as e:
            print(f"Failed to load ESG data for {year}: {str(e)}")
            esg_data[year] = pd.DataFrame()  # Initialize with empty dataframe

    return esg_data


# Load ESG data
esg_data = load_esg_data()


# Convert yearly ESG data into a single integrated dataframe (optional)
def create_integrated_esg_data(esg_data_dict):
    """
    Convert yearly ESG data into a single integrated dataframe

    Parameters:
        esg_data_dict (dict): Dictionary of ESG data by year

    Returns:
        DataFrame: Integrated ESG dataframe
    """
    all_data = []

    for year, df in esg_data_dict.items():
        if not df.empty:
            # Reset index to columns
            df_copy = df.reset_index()

            # Add 'Year' column if it doesn't exist
            if '연도' not in df_copy.columns:
                df_copy['연도'] = year

            all_data.append(df_copy)

    if all_data:
        # Merge all data
        integrated_df = pd.concat(all_data, ignore_index=True)

        # Standardize column names if needed
        if 'ticker' in integrated_df.columns and '기업명' in integrated_df.columns:
            integrated_df = integrated_df.rename(columns={
                'ticker': 'ticker',
                '기업명': 'Company',
                '연도': 'Year'
            })

        return integrated_df
    else:
        return pd.DataFrame()


# Create integrated ESG data (can be used in Streamlit app)
integrated_esg_df = create_integrated_esg_data(esg_data)


# Function to calculate environmental (E), social (S), governance (G) scores from ESG data
def calculate_esg_components(esg_df):
    """
    Convert rating agency scores to environmental (E), social (S), governance (G) components

    Parameters:
        esg_df (DataFrame): Dataframe containing ESG rating agency scores

    Returns:
        DataFrame: Dataframe with added E, S, G components
    """
    df = esg_df.copy()

    # Check original rating agency columns
    agency_columns = ['MSCI', 'S&P', 'Sustainalytics', 'ISS', 'ESG기준원']
    available_columns = [col for col in agency_columns if col in df.columns]

    if not available_columns:
        print("Rating agency columns not found.")
        return df

    # Set E, S, G weights for each rating agency (example - adjust based on actual data)
    agency_weights = {
        'MSCI': {'environmental': 0.4, 'social': 0.4, 'governance': 0.2},
        'S&P': {'environmental': 0.3, 'social': 0.4, 'governance': 0.3},
        'Sustainalytics': {'environmental': 0.3, 'social': 0.3, 'governance': 0.4},
        'ISS': {'environmental': 0.35, 'social': 0.35, 'governance': 0.3},
        'ESG기준원': {'environmental': 0.33, 'social': 0.33, 'governance': 0.34}
    }

    # Initialize E, S, G components
    df['environmental'] = 0
    df['social'] = 0
    df['governance'] = 0

    # Distribute each rating agency's score to E, S, G components
    for agency in available_columns:
        if agency in agency_weights:
            df['environmental'] += df[agency] * agency_weights[agency]['environmental']
            df['social'] += df[agency] * agency_weights[agency]['social']
            df['governance'] += df[agency] * agency_weights[agency]['governance']

    # Normalize each component by number of available agencies
    num_agencies = len(available_columns)
    if num_agencies > 0:
        df['environmental'] /= num_agencies
        df['social'] /= num_agencies
        df['governance'] /= num_agencies

    return df


# Apply ESG component calculation
for year in esg_data:
    if not esg_data[year].empty:
        esg_data[year] = calculate_esg_components(esg_data[year])

# Get the most recent year's ESG data (immediately usable in UI)
latest_year = max(esg_data.keys()) if esg_data else None
latest_esg_data = esg_data[latest_year] if latest_year and not esg_data[latest_year].empty else pd.DataFrame()

# Check data
if not latest_esg_data.empty:
    print(f"Latest year ({latest_year}) ESG data sample:")
    print(latest_esg_data.head())
    print(f"Column list: {latest_esg_data.columns.tolist()}")
else:
    print("Could not load ESG data.")


# Get stocks from Korea Stock Exchange KOSPI index
@st.cache_data
def getSymbols(market='KOSPI', sort='Marcap'):  # Sort by market value (Marcap)
    df = fdr.StockListing(market)
    # Set sorting (reverse sort by market cap)
    ascending = False if sort == 'Marcap' else True
    df.sort_values(by=[sort], ascending=ascending, inplace=True)
    return df[['Code', 'Name', 'Market']]


@st.cache_data
def load_stock_data(code, ndays, frequency='D'):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=ndays)
    data = fdr.DataReader(code, start_date, end_date)

    if frequency == 'M':  # Monthly candle setting
        data = data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()  # Monthly resampling, remove missing values

    return data


# Candlestick chart output function
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


# Select top companies (ESG-based)
def recommend_companies(esg_weights, df):
    # Calculate final score reflecting user's ESG preference weights
    df['final_score'] = (
            esg_weights['environmental'] * df['environmental'] +
            esg_weights['social'] * df['social'] +
            esg_weights['governance'] * df['governance']
    )

    # Select top 10 companies
    top_companies = df.sort_values(by='final_score', ascending=False).head(10)

    return top_companies


st.markdown("""
            <style>
            .st-emotion-cache-10hsuxw e1f1d6gn2{
                margin:3px;
            }
            </style>
            """, unsafe_allow_html=True)

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

# --- Optimization Algorithm ---
import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import yfinance as yf
from pypfopt import BlackLittermanModel, expected_returns, risk_models, CovarianceShrinkage


# Modified portfolio optimization function
def optimize_portfolio(returns, esg_data, esg_weights, user_investment_style, year=None):
    """
    Portfolio optimization using Black-Litterman model reflecting user's ESG preferences

    Parameters:
    returns (DataFrame): Stock return data
    esg_data (dict): Dictionary of ESG data by year
    esg_weights (dict): User-defined E, S, G weights
    user_investment_style (str): User investment style
    year (int): Year of ESG data to use (default: most recent year)

    Returns:
    dict: Optimal investment weights by ticker
    tuple: Portfolio performance metrics (expected return, volatility, Sharpe ratio)
    """
    try:
        # Determine which year's ESG data to use
        if year is None:
            year = max(esg_data.keys())

        if year not in esg_data or esg_data[year].empty:
            st.error(f"No ESG data available for {year}.")
            return {}, (0, 0, 0)

        # Load ESG data
        year_esg_data = esg_data[year]

        # Select valid tickers (only stocks with ESG data)
        available_tickers = year_esg_data.index.tolist()

        # Filter valid tickers from return data
        valid_tickers = [t for t in available_tickers if t in returns.columns]

        if len(valid_tickers) == 0:
            st.error("No matching stocks between ESG data and return data.")
            return {}, (0, 0, 0)

        # Set tau value based on user investment style
        if user_investment_style == "재무적인 요소를 중심적으로 고려한다.":
            tau = 0.5  # Emphasize market data
        elif user_investment_style == "ESG와 재무적인 요소를 모두 고려한다.":
            tau = 1.0  # Balanced
        elif user_investment_style == "ESG 요소를 중심적으로 고려한다.":
            tau = 2.0  # Emphasize ESG
        else:
            tau = 1.0  # Default

        # User ESG preference vector
        esg_preference = np.array([
            esg_weights['environmental'],
            esg_weights['social'],
            esg_weights['governance']
        ]).reshape(-1, 1)

        # Prepare ESG score matrix for valid stocks
        esg_columns = ['environmental', 'social', 'governance']
        P_matrix = year_esg_data.loc[valid_tickers, esg_columns].values.T  # (3, n) shape matrix

        # Calculate Q vector: adjusted scores reflecting ESG preferences
        Q_vector = (P_matrix @ esg_preference).flatten()

        # Extract return data for valid stocks
        valid_returns = returns[valid_tickers].dropna()

        # Estimate covariance matrix using Ledoit-Wolf method
        cov_estimator = LedoitWolf()
        cov_matrix = cov_estimator.fit(valid_returns).covariance_

        # Calculate annual average returns (annualized)
        mean_returns = valid_returns.mean() * 252

        # Omega matrix: ESG data uncertainty (diagonal matrix)
        omega_diagonal = np.full(P_matrix.shape[0], 0.002)  # Uncertainty for each ESG component
        Omega = np.diag(omega_diagonal)

        # Regularize covariance matrix for numerical stability
        cov_matrix_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6

        # Black-Litterman model calculation
        try:
            # Calculate inverse matrix
            inv_term = np.linalg.inv(tau * cov_matrix_reg + P_matrix.T @ np.linalg.inv(Omega) @ P_matrix)

            # Calculate adjustment term
            adjustment_term = (
                    tau * cov_matrix_reg @ mean_returns.values.reshape(-1, 1) +
                    P_matrix.T @ (np.linalg.inv(Omega) @ P_matrix @ Q_vector.reshape(-1, 1))
            )

            # Adjusted expected returns
            adjusted_returns = inv_term @ adjustment_term
            adjusted_returns = adjusted_returns.flatten()

            # Sharpe ratio maximization objective function
            def negative_sharpe(weights):
                port_return = np.dot(weights, adjusted_returns)
                port_volatility = np.sqrt(weights.T @ cov_matrix_reg @ weights)
                return -port_return / port_volatility

            # Optimization constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(valid_tickers))]

            # Initial weights: equal distribution
            initial_weights = np.ones(len(valid_tickers)) / len(valid_tickers)

            # Run optimization
            optimized_result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            # Optimal weights and performance metrics
            optimized_weights = optimized_result.x
            expected_return = np.dot(optimized_weights, adjusted_returns)
            expected_volatility = np.sqrt(optimized_weights.T @ cov_matrix_reg @ optimized_weights)
            sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0

            # Convert results to dictionary
            weights_dict = dict(zip(valid_tickers, optimized_weights))

            return weights_dict, (expected_return, expected_volatility, sharpe_ratio)

        except np.linalg.LinAlgError as e:
            st.error(f"Optimization calculation error: {str(e)}")
            # Fallback to equal weights
            weights_dict = {ticker: 1.0 / len(valid_tickers) for ticker in valid_tickers}
            return weights_dict, (0, 0, 0)

    except Exception as e:
        st.error(f"Portfolio optimization error: {str(e)}")
        return {}, (0, 0, 0)


# Portfolio weight calculation and adjustment function (top-level function called from UI)
def calculate_optimal_portfolio(esg_weights, user_investment_style):
    """
    Calculate optimal portfolio reflecting user's ESG preferences and investment style

    Parameters:
    esg_weights (dict): User-defined E, S, G weights
    user_investment_style (str): User investment style

    Returns:
    DataFrame: Recommended stock information (ticker, company name, weight, ESG scores, etc.)
    tuple: Portfolio performance metrics
    """
    try:
        # Load latest stock data (caching if needed)
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.Timedelta(days=5 * 365)  # 5 years of data

        # Load KRX stock data and calculate returns
        # (Use FinanceDataReader or yfinance in actual implementation)
        tickers = latest_esg_data.index.tolist()  # Stocks from latest ESG data

        @st.cache_data
        def load_price_data(tickers, start_date, end_date):
            price_data = {}
            for ticker in tickers:
                try:
                    df = fdr.DataReader(ticker, start_date, end_date)['Close']
                    price_data[ticker] = df
                except Exception:
                    pass
            return pd.DataFrame(price_data)

        # Load price data
        price_data = load_price_data(tickers, start_date, end_date)

        # Calculate returns
        returns = price_data.pct_change().dropna()

        # Run portfolio optimization
        portfolio_weights, performance_metrics = optimize_portfolio(
            returns=returns,
            esg_data=esg_data,
            esg_weights=esg_weights,
            user_investment_style=user_investment_style
        )

        if not portfolio_weights:
            st.error("Portfolio optimization failed.")
            return pd.DataFrame(), (0, 0, 0)

        # Convert optimization results to dataframe
        result_df = pd.DataFrame(columns=['ticker', 'Company', 'Weight', 'environmental', 'social', 'governance'])

        # Add company information from latest ESG data
        for ticker, weight in portfolio_weights.items():
            if ticker in latest_esg_data.index:
                company_name = latest_esg_data.loc[ticker, '기업명'] if '기업명' in latest_esg_data.columns else "Unknown"
                env_score = latest_esg_data.loc[ticker, 'environmental']
                soc_score = latest_esg_data.loc[ticker, 'social']
                gov_score = latest_esg_data.loc[ticker, 'governance']

                # Add to result dataframe
                result_df = result_df.append({
                    'ticker': ticker,
                    'Company': company_name,
                    'Weight': weight * 100,  # Convert to percentage
                    'environmental': env_score,
                    'social': soc_score,
                    'governance': gov_score
                }, ignore_index=True)

        # Sort by weight
        result_df = result_df.sort_values(by='Weight', ascending=False)

        return result_df, performance_metrics

    except Exception as e:
        st.error(f"Portfolio calculation error: {str(e)}")
        return pd.DataFrame(), (0, 0, 0)


# Display results
# In the improved code, user's ESG preferences are directly reflected in market equilibrium returns,
# so user's ESG preferences are clearly reflected in the optimization process.
def display_text_on_hover(hover_text, i, origin_text):
    # Create unique class names for each text hover area
    hover_class = f'hoverable_{i}'
    tooltip_class = f'tooltip_{i}'
    text_popup_class = f'text-popup_{i}'

    # Define unique CSS for each hover text
    hover_css = f'''
        .{hover_class} {{
            position: relative;
            display: block;
            cursor: pointer;
            text-align: center;
            font-family: Pretendard;
        }}
        .{hover_class} .{tooltip_class} {{
            display: none; /* Hide the "Hover to see text" */
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
            width: 80%; /* Set to 80% of screen width */
            left: 50%;  /* Center alignment by setting left to 50% */
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

    # Define HTML with modified style for origin_text
    text_hover = f'''
        <div class="{hover_class}">
            <a href="#hover_text" style="color: #999999; font-family: Pretendard; font-size: 20px; text-align: center; text-decoration: none;font-weight:bold;">{origin_text}&ensp;&ensp;</a>
            <div class="{tooltip_class}"></div>
            <div class="{text_popup_class}">{hover_text}</div>
        </div>
    '''

    # Write dynamic HTML and CSS to content container
    st.markdown(f'<p>{text_hover}{tooltip_css}</p>', unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 1, 3])
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

            # Calculate change rate
            change = today_kospi - yesterday_kospi
            change_percent = (change / yesterday_kospi) * 100

            # Output with Streamlit metric
            st.metric(label="오늘의 코스피 지수", value=round(today_kospi, 2), delta=f"{round(change_percent, 2)}%",
                      delta_color="inverse")

    with kosdaq:
        if not kosdaq_data.empty:
            yesterday_kosdaq = kosdaq_data.iloc[0]['Close']
            today_kosdaq = kosdaq_data.iloc[-1]['Close']

            # Calculate change rate
            change = today_kosdaq - yesterday_kosdaq
            change_percent = (change / yesterday_kosdaq) * 100

            # Output with Streamlit metric
            st.metric(label="오늘의 코스닥 지수", value=round(today_kosdaq, 2), delta=f"{round(change_percent, 2)}%",
                      delta_color="inverse")

    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        origin_e = survey_result.loc['E'].sum() * 10 / 4.99
        display_text_on_hover('-탄소 관리<br>-폐기물 관리<br>-기후 변화 전략', 1, '&emsp;E')
        e_value = vertical_slider(
            label="환경",
            key="environmental",
            height=195,
            step=0.1,
            default_value=survey_result.loc['E'].sum() * 1 / 4.99,  # Optional - Defaults to 0
            min_value=0.01,  # Defaults to 0
            max_value=1.0,  # Defaults to 10
            track_color="#f0f0f0",  # Optional - Defaults to #D3D3D3
            slider_color='#006699',  # Optional - Defaults to #29B5E8
            thumb_color="#FF9933",
            value_always_visible=True,  # Optional - Defaults to False
        )
    with sl2:
        display_text_on_hover('-사회적 기회<br>-지역사회 관계<br>-인적 자원', 1, '&emsp;S')
        s_value = vertical_slider(
            label="사회",  # Optional
            key="social",
            height=195,  # Optional - Defaults to 300
            step=0.1,  # Optional - Defaults to 1
            default_value=survey_result.loc['S'].sum() * 1 / 4.79,  # Optional - Defaults to 0
            min_value=0.01,  # Defaults to 0
            max_value=1.0,  # Defaults to 10
            track_color="#f0f0f0",  # Optional - Defaults to #D3D3D3
            slider_color='#006699',  # Optional - Defaults to #29B5E8
            thumb_color="#FF9933",
            value_always_visible=True,  # Optional - Defaults to False
        )
    with sl3:
        display_text_on_hover('-주주권 보호<br>-기업이사회운영<br>', 1, '&emsp;G')
        g_value = vertical_slider(
            label="지배구조",  # Optional
            key="governance",
            height=195,  # Optional - Defaults to 300
            step=0.1,  # Optional - Defaults to 1
            default_value=survey_result.loc['G'].sum() * 1 / 4.16,
            min_value=0.01,  # Defaults to 0
            max_value=1.0,  # Defaults to 10
            track_color="#f0f0f0",  # Optional - Defaults to #D3D3D3
            slider_color='#006699',  # Optional - Defaults to #29B5E8
            thumb_color="#FF9933",
            value_always_visible=True,  # Optional - Defaults to False
        )
    # User's ESG preferences
    esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}
    # Black-Litterman model version
    try:
        # Call the optimal portfolio calculation function
        top_companies, portfolio_performance = calculate_optimal_portfolio(esg_weights, user_investment_style)

        # Handle empty results
        if top_companies.empty:
            st.error("Portfolio optimization failed. Try different ESG weights.")
        else:
            # Weight column is already included, no additional processing needed
            # Sort by Weight (descending order)
            top_companies = top_companies.sort_values(by='Weight', ascending=False)
    except Exception as e:
        st.error(f"Error occurred during portfolio optimization: {str(e)}")
        # Initialize with empty dataframe in case of error
        top_companies = pd.DataFrame(columns=['ticker', 'Company', 'Weight', 'environmental', 'social', 'governance'])
        portfolio_performance = (0, 0, 0)  # Initialize expected return, volatility, Sharpe ratio
    # Sort by Weight in descending order
    top_companies = top_companies.sort_values(by='Weight', ascending=False)
    selected_companies = st.multiselect(
        "",
        top_companies['Company'],
        placeholder="제외하고 싶은 기업을 선택"
    )

    if selected_companies:
        top_companies = top_companies[~top_companies['Company'].isin(selected_companies)]

# User's ESG preferences
esg_weights = {'environmental': e_value, 'social': s_value, 'governance': g_value}
st.write('')

# Calculate portfolio weights
# top_companies = recommend_companies(esg_weights,integrated_esg_df)

# Black-Litterman model version
# industries = integrated_esg_df['industry'].unique().tolist()
# processed_df = integrated_esg_df[integrated_esg_df['industry'].isin(industries)].copy()
# portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights, user_investment_style)
# # portfolio_weights, portfolio_performance = calculate_adjusted_weights(processed_df, portfolio_weights, esg_weights,portfolio_performance)
# # portfolio_weights, portfolio_performance = calculate_portfolio_weights(processed_df, esg_weights,user_investment_style) # cleaned_weights: optimal investment ratio for each asset, performance: performance metrics of optimized portfolio
# top_companies = integrated_esg_df[integrated_esg_df['ticker'].isin(portfolio_weights)].copy()
# # Create new top_companies dataframe by mapping ticker column and portfolio_weights _ Black-Litterman model version
# # Add portfolio_weights values as 'Weight' column
# total_weight = sum(portfolio_weights.values())
# # total_weight =  sum(portfolio_weights.values)
# top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)
# top_companies['Weight'] = top_companies['Weight'] * 100

# cvxopt version
# portfolio_weights, portfolio_performance = calculate_portfolio_weights(top_companies)
# industries = integrated_esg_df['industry'].unique().tolist()
# processed_df = integrated_esg_df[integrated_esg_df['industry'].isin(industries)].copy()

# top_companies['Weight'] = top_companies['ticker'].map(portfolio_weights)

with col2:
    if selected_companies:
        top_companies = top_companies[~top_companies['Company'].isin(selected_companies)]
    st.markdown(f"""<div>
                        <h2 style="font-family: Pretendard;font-size: 13px; text-align:center; text-decoration: none;">차트에서 여러분의 관심 회사 이름을 클릭하여<br>더 다양한 정보를 경험해 보세요.</h2>
                    </div>
            """, unsafe_allow_html=True)

    # Calculate total Weight sum
    total_weight = top_companies['Weight'].sum()
    # Filter companies below minimum ratio by Weight
    # top_companies = top_companies[top_companies['Weight'] / total_weight * 100 >= 5.0]

    # Create pie chart
    fig = px.pie(
        top_companies,
        names='Company',
        values='Weight',
        color_discrete_sequence=px.colors.qualitative.G10,
        custom_data=top_companies[['environmental', 'social', 'governance']]
    )

    # Display ESG information with customdata
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        hovertemplate=(
                '추천 포트폴리오 비중 : %{percent}<br>' +  # Weight information
                'Environmental 점수 : ' + ' ' + '%{customdata[0][0]:.2f}<br>' +  # Environmental score
                'Social 점수  :  %{customdata[0][1]:.2f}<br>' +  # Social score
                'Governance : %{customdata[0][2]:.2f}<br>'  # Governance score
        ),
        texttemplate='%{label}',
    )

    # Chart layout settings
    fig.update_layout(
        font=dict(size=16, color='black'),
        showlegend=False,
        margin=dict(t=40, b=40, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=250,
        height=400,
    )

    clicked_points = plotly_events(fig, click_event=True, key="company_click")

with col3:
    company_colletion['ticker'] = company_colletion['ticker'].str[1:]
    top_companies['ticker'] = top_companies['ticker'].str.replace('.KS', '')

    expected_return = portfolio_performance[0]
    expected_volatility = portfolio_performance[1]
    sharpe_ratio = portfolio_performance[2]
    for company in top_companies['Company']:
        condition = (esg_data[year]['Year'] == 2023) & (esg_data[year]['Company'] == company)
        if condition.any():
            top_companies.loc[top_companies['Company'] == company, ['environmental', 'social', 'governance']] = \
            esg_data[year].loc[condition, ['environmental', 'social', 'governance']].values
    top5_companies = top_companies.nlargest(5, 'Weight')
    filtered_companies = pd.merge(company_colletion, top5_companies, left_on='ticker', right_on='ticker')
    filtered_companies = filtered_companies[['Company', 'Weight', 'environmental', 'social', 'governance', '종목설명']]
    filtered_companies = filtered_companies.rename(columns={
        'Company': '종목명',
        'Weight': '제안 비중',
        'environmental': 'E',
        'social': 'S',
        'governance': 'G',
        '종목설명': '종목 소개'
    })
    # Display expected return, volatility, and Sharpe ratio at the top
    # _,col1, col2, col3,_ = st.columns([2,3,3,3,2])
    col1, col2, col3 = st.columns(3)
    with col1:
        display_text_on_hover("This indicator represents the expected annual return of the portfolio.", 1,
                              f"연간 기대 수익률 &emsp; {expected_return * 100:.2f} %")
        st.markdown('')
    with col2:
        display_text_on_hover("This indicator is a risk metric showing how much the return may fluctuate.", 1,
                              f"연간 변동성 &emsp; {expected_volatility * 100:.2f} %")
    with col3:
        display_text_on_hover(
            "This indicator is a performance metric showing how effectively the portfolio generates returns relative to risk.",
            1, f"샤프 비율 &emsp;{sharpe_ratio * 100:.2f}")

    # Add tooltips to HTML code and convert to two-row structure
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

    _, _, bt1, bt2 = st.columns(4)
    with bt1:
        check = st.button(label="포트폴리오 확인  ➡️")
        if check:
            screenshot = ImageGrab.grab(bbox=(400, 430, 790, 840))
            screenshot.save("pie_chart_capture.png")

    # Set relative path based on current script file's directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(current_directory, "pie_chart_capture.png")


    # HTML generation function
    def generate_html():
        # Filter dataframe and rename columns
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

        # Create HTML content
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


    # HTML save and PDF conversion function
    def save_as_pdf(html_content):
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        options = {
            'enable-local-file-access': None,  # Allow local file access
            'encoding': "UTF-8",  # UTF-8 encoding setting
            'no-pdf-compression': ''  # Prevent font compression
        }
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
            # Save HTML file
            tmp_html.write(html_content.encode('utf-8'))
            tmp_html_path = tmp_html.name

        # Set PDF conversion file path
        pdf_path = tmp_html_path.replace(".html", ".pdf")

        # Convert to PDF
        pdfkit.from_file(tmp_html_path, pdf_path, configuration=config)

        # Create Streamlit download button
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
                clicked_df = esg_data[year][esg_data[year]['Company'] == clicked_company]
                clicked_df['Year'] = clicked_df['Year'].astype(int)
                clicked_df = clicked_df[['Year', 'environmental', 'social', 'governance']]
                clicked_df = clicked_df.melt(id_vars='Year',
                                             value_vars=['environmental', 'social', 'governance'],
                                             var_name='Category',
                                             value_name='Score')

                fig = px.line(clicked_df, x='Year', y='Score', color='Category')
                fig.update_layout(showlegend=True,
                                  legend=dict(
                                      orientation='h',  # Horizontal layout
                                      yanchor='bottom',  # Anchor the y-axis of the legend to the bottom
                                      y=-0.6,  # Move the legend below the graph, adjust as needed
                                      xanchor='center',  # Anchor the x-axis of the legend to the center
                                      x=0.5
                                  ), width=750, height=350)
                # fig.update_xaxes(showticklabels=False, title='')
                # fig.update_yaxes(showticklabels=False, title='')

                # Output graph
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

        # Update session state
        st.session_state['ndays'] = 1825
        st.session_state['code_index'] = code_index
        st.session_state['chart_style'] = chart_style
        st.session_state['volume'] = True

        # Load stock price data for selected stock
        data = load_stock_data(code, 1825)

        # Call stock chart visualization function
        plotChart(data)

    else:
        st.write('')


# Company name normalization function
def normalize_company_name(name):
    return unicodedata.normalize('NFC', name).strip()


# Normalize 'company' column in word_freq_df
word_freq_df['company'] = word_freq_df['company'].apply(normalize_company_name)


# Weighted average word cloud generation function
def generate_blended_word_cloud(top_companies, word_freq_df):
    blended_word_freq = Counter()

    # Normalize company names in top_companies as well
    top_companies['Company'] = top_companies['Company'].apply(normalize_company_name)

    for _, row in tqdm(top_companies.iterrows(), total=top_companies.shape[0], desc="Generating Blended Word Cloud"):
        company_name = row['Company']
        weight = row['Weight']

        # Filter word frequency for the company
        company_word_freq = word_freq_df[word_freq_df['company'] == company_name]

        if company_word_freq.empty:
            #     st.warning(f"{company_name}의 빈도 데이터가 없습니다.")
            continue

        # Calculate frequency multiplied by weight for each word
        for _, word_row in company_word_freq.iterrows():
            word = word_row['word']
            freq = word_row['frequency']
            blended_word_freq[word] += freq * weight

    # Create and return word cloud
    if not blended_word_freq:
        st.warning("워드 클라우드를 생성할 데이터가 없습니다.")
        return None

    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/malgun.ttf',  # Korean font setting
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
        # Generate word cloud based on pre-declared top_companies
        wordcloud = generate_blended_word_cloud(top_companies, word_freq_df)

        # Display word cloud
        if wordcloud:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("생성할 데이터가 충분하지 않아 워드 클라우드를 표시할 수 없습니다.")
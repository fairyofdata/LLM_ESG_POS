import streamlit as st
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import streamlit_authenticator as stauth
import numpy as np
from streamlit_authenticator.utilities.hasher import Hasher
import os.path
import pickle as pkle
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

# Set the relative path based on the location of the current script file
current_directory = os.path.dirname(__file__)

#Referral variable definition
survey_result_file = os.path.join(current_directory, "survey_result.csv")
user_investment_style_file = os.path.join(current_directory, "user_investment_style.txt")
user_interest_file = os.path.join(current_directory, "user_interest.txt")
user_name_file = os.path.join(current_directory, "user_name.txt")
company_list_file = os.path.join(current_directory, 'company_collection.csv')
word_freq_file = os.path.join(current_directory, "company_word_frequencies.csv")
survey_result_page = 'pages/survey_result.py'

# Import after confirming that the file exists
if os.path.exists(survey_result_file):
    survey_result = pd.read_csv(survey_result_file, encoding='utf-8', index_col=0)
else:
    # If there is no file, create an empty data frame as the default value
    survey_result = pd.DataFrame()

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
    user_name = ''

if os.path.exists(company_list_file):
    company_list = pd.read_csv(company_list_file)
else:
    company_list = pd.DataFrame()

if os.path.exists(word_freq_file):
    word_freq_df = pd.read_csv(word_freq_file)
else:
    word_freq_df = pd.DataFrame()


st.set_page_config(
    page_title = "설문 조사",
    page_icon=":earth_africa:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
font_css = """
    <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: Pretendard;
        }}
    </style>
    """
# Apply CSS to Streamlit
st.markdown(font_css, unsafe_allow_html=True)
with st.sidebar:
    st.page_link('main_survey_introduce.py', label='홈', icon="🎯")
    st.page_link('pages/survey_page.py', label='설문', icon="📋")
    st.page_link('pages/survey_result.py', label='설문 결과',icon="📊")
    st.page_link('pages/recent_news.py', label='최신 뉴스',icon="🆕")
    st.page_link('pages/esg_introduce.py', label='ESG 소개 / 투자 방법', icon="🧩")
    
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
# St.markdown ('' ''
# <style>
# .st-AF ST-BZ ST-C0 ST-C0 ST-C2 ST-C3 ST-C3 ST-C4 ST-C5 {{{{{
# Flex-Direction: ROW;
# Justify-Content: Center;
# </style>
#}}
            # '' '', unsafe_allow_html = true)

#St.markdown ('<STYLE> DIV.ROW-WIDGET.STRADIO> DIV {Display: Flex; Justify-Content: Center; COLOR: #55FF00; Align -items: CENTEMS: CENTEMS: CENTEMS </Style> ', unsafe_allow_html = true)
st.markdown('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-right:2px;}</style>', unsafe_allow_html=True)

values = {'msci': 0, 'iss': 0, 'sustain': 0, 'sandp': 0, 'esg1': 0}

def evaluate_care_level(response):
    if response == "신경 쓴다.":
        return 1
    elif response == "보통이다.":
        return 0.5
    elif response == "신경 쓰지 않는다.":
        return 0
    
with st.form('usersurvey',clear_on_submit=False):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    # E sector question
    st.markdown('''
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
                    <style>
                        div.row-widget.stRadio > div{display: flex; justify-content: center;align-items: center;border-radius: 10px;}
                        div[class="question"]{
                            margin: auto; 
                            padding: 40px; 
                            border-radius: 10px; 
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                        div[class="st-ae st-af st-ag st-ah st-ai st-aj st-ak st-al"]{
                            margin:auto;
                            padding:10px;
                        }
                        div[class="st-ay st-az st-b0 st-b1 st-b2 st-b3 st-b4 st-ae st-b5 st-b6 st-b7 st-b8 st-b9 st-ba st-bb st-bc st-bd st-be st-bf st-bg"] {
                            transform: scale(2.5);
                            margin-right: 10px;
                        }
                        div[class="st-ay st-c1 st-b0 st-b1 st-b2 st-b3 st-b4 st-ae st-b5 st-b6 st-b7 st-b8 st-b9 st-ba st-bb st-bc st-bd st-be st-bf st-bg"]{
                            transform: scale(1.5);
                            margin-right: 10px;
                        }
                        button[data-testid="baseButton-secondaryFormSubmit"]{
                            background-color: #AAAAAA;
                            border-radius: 10px; 
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                        p{
                            font-family: Pretendard;
                        }
                    </style>
                </head>
                ''',unsafe_allow_html=True)

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-family: Pretendard;font-weight: bold;">1. 투자할 때 기업이 탄소 배출이나 오염물질 관리 등 자연을 보호하는 데 신경 쓰는지 고려하시나요?</div>', unsafe_allow_html=True)
    q1 = st.radio('', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-family: Pretendard;font-weight: bold;">2. 투자할 때 기업이 환경 관리 시스템을 구축하는 등 기후 변화에 적극 대응하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q2 = st.radio(' ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">3. 투자할 때 기업이 생산 과정에서 친환경적으로 제품과 서비스를 제공하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q3 = st.radio('  ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    #Sustainalytics ESG standard question
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">4. 투자할 때 기업이 자원을 효율적으로 사용하고 배출량을 줄이는지 고려 하시나요?</div>', unsafe_allow_html=True)
    q4 = st.radio('   ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">5. 투자할 때 기업이 신재생에너지를 활용하는 등 친환경적으로 활동하는지  고려하시나요?</div>', unsafe_allow_html=True)
    q5 = st.radio('    ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">6. 투자할 때 기업이 직원의 안전을 보장하고 소비자의 권리를 보호하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q6 = st.radio('     ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    #MSCI ESG standard question
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">7. 투자할 때 기업이 지역사회와의 관계를 잘 유지하고 공정하게 운영하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q7 = st.radio('      ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">8. 투자할 때 기업이 건강과 사회에 미치는 부정적인 영향을 줄이는지 고려하시나요?</div>', unsafe_allow_html=True)
    q8 = st.radio('       ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">9. 투자할 때 기업이 직원에게 차별 없이 워라벨을 지켜주고, 역량 개발을 지원하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q9 = st.radio('        ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    #Korea ESG Standard ESG Standard Question
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">10. 투자할 때 기업이 환경 보호, 직원 복지, 공정 거래 등 사회적 책임을 다하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q10 = st.radio('         ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">11. 투자할 때 기업이 개인정보 보호 등 사이버 보안을 잘 관리하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q11 = st.radio('          ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">12. 투자할 때 기업이 경영 구조를 유지하기 위해 이사회의 독립성과 전문성을 높이려는 것을 고려하시나요?</div>', unsafe_allow_html=True)
    q12 = st.radio('           ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    # ISS ESG Standard Question
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">13. 투자할 때 기업이 감사팀을 운영하고 회계 규정을 잘 지키는지 고려하시나요?</div>', unsafe_allow_html=True)
    q13 = st.radio('            ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">14. 투자할 때 기업이 주주의 권리를 보호하고 이익을 돌려주는지 고려하시나요?</div>', unsafe_allow_html=True)
    q14 = st.radio('             ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">15. 투자할 때 기업이 나라에 미치는 영향을 잘 관리하고, 새로운 경영 방식을 도입하는 것을 고려하시나요?</div>', unsafe_allow_html=True)
    q15 = st.radio('              ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    # Questions about investment tendency ()
    st.markdown('<div class="question" style="font-family: Pretendard;font-size:20px;text-align:center;font-weight: bold;">16. 귀하는 투자시 무엇을 고려하시나요?</div>', unsafe_allow_html=True)
    q16 = st.radio('               ', options=('ESG 요소를 중심적으로 고려한다.','ESG와 재무적인 요소를 모두 고려한다.','재무적인 요소를 중심적으로 고려한다.'))
    st.markdown('</div>',unsafe_allow_html=True)
    
    care_levels = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
    esg_interest = 0
    financial_interest = 0
    results = [evaluate_care_level(level) for level in care_levels]
    for i in range(1, 16):
        exec(f'q{i} = evaluate_care_level(q{i})')
    _,survey_submitted, _ = st.columns([3,1,3])
    with survey_submitted:
        submitted = st.form_submit_button('설문 완료')


    
    if submitted:
        try:
            survey_result = pd.DataFrame(index=['E', 'S', 'G'], columns=['esg1', 'sandp', 'sustain', 'iss', 'msci'])
            survey_result.loc[:, :] = 0
            yes_interest = 1
            no_esg_interest = 1
            if q1 == 1:
                survey_result.at['E', 'sustain'] += (1 * q1)
                survey_result.at['E', 'msci'] += (0.5 * q1)
            elif q1 == 0.5: 
                survey_result.at['E', 'sustain'] += (0.5 * q1)
                survey_result.at['E', 'msci'] += (0.25 * q1)

                
            if q2 == 1:
                survey_result.at['E', 'iss'] += (0.33 * q2)
                survey_result.at['E', 'sandp'] += (1 * q2)

            elif q2 == 0.5:
                survey_result.at['E', 'iss'] += (0.165 * q2)
                survey_result.at['E', 'sandp'] += (0.5 * q2)
                
            if q3 == 1:
                survey_result.at['E', 'iss'] += (0.33 * q3)
                survey_result.at['E', 'esg1'] += (1 * q3)

            elif q3 == 0.5:
                survey_result.at['E', 'iss'] += (0.165 * q3)
                survey_result.at['E', 'esg1'] += (0.5 * q3)
                
            if q4 == 1:
                survey_result.at['E', 'iss'] += (0.33 * q4)
            elif q4 == 0.5:
                survey_result.at['S', 'iss'] += (0.165 * q4)

            if q5 == 1:
                survey_result.at['E', 'msci'] += (0.5 * q5)
            elif q5 == 0.5:
                survey_result.at['E', 'msci'] += (0.25 * q5)
                
            if q6 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q6)
                survey_result.at['S', 'msci'] += (0.2 * q6)
            elif q6 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q6)
                survey_result.at['S', 'msci'] += (0.1 * q6)

            if q7 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q7)
                survey_result.at['S', 'msci'] += (0.2 * q7)
                survey_result.at['S', 'iss'] += (0.33 * q7)
            elif q7 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q7)
                survey_result.at['S', 'msci'] += (0.1 * q7)
                survey_result.at['S', 'iss'] += (0.165 * q7)
                
            if q8 == 1:
                survey_result.at['S', 'msci'] += (0.2 * q8)
            elif q8 == 0.5:
                survey_result.at['S', 'msci'] += (0.1 * q8)
                
            if q9 == 1:
                survey_result.at['S', 'iss'] += (0.33 * q9)
                survey_result.at['S', 'esg1'] += (1 * q9)
            elif q9 == 0.5:
                survey_result.at['S', 'iss'] += (0.165 * q9)
                survey_result.at['S', 'esg1'] += (0.5 * q9)
                
            if q10 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q10)
                survey_result.at['S', 'iss'] += (0.33 * q10)
            elif q10 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q10)
                survey_result.at['S', 'iss'] += (0.165 * q10)
                
            if q11 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q11)
                survey_result.at['S', 'msci'] += (0.2 * q11)
                survey_result.at['S', 'sandp'] += (1 * q11)
            elif q11 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q11)
                survey_result.at['S', 'msci'] += (0.1 * q11)
                survey_result.at['S', 'sandp'] += (0.5 * q11)
                
            if q12 == 1:
                survey_result.at['G', 'sustain'] += (0.25 * q12)
                survey_result.at['G', 'msci'] += (0.2 * q12)
                survey_result.at['G', 'iss'] += (0.2 * q12)
                survey_result.at['G', 'sandp'] += (1 * q12)
                survey_result.at['G', 'esg1'] += (0.2 * q12)
            elif q12 == 0.5:
                survey_result.at['G', 'sustain'] += (0.5 * q12)
                survey_result.at['G', 'msci'] += (0.5 * q12)
                survey_result.at['G', 'iss'] += (0.165 * q12)
                survey_result.at['G', 'sandp'] += (0.165 * q12)
                survey_result.at['G', 'esg1'] += (0.165 * q12)
                
            if q13 == 1:
                survey_result.at['G', 'iss'] += (0.33 * q13)
                survey_result.at['G', 'sandp'] += (0.33 * q13)
                survey_result.at['G', 'esg1'] += (0.33 * q13)
            elif q13 == 0.5:
                survey_result.at['G', 'iss'] += (0.165 * q13)
                survey_result.at['G', 'sandp'] += (0.165 * q13)
                survey_result.at['G', 'esg1'] += (0.165 * q13)
                
            if q14 == 1:
                survey_result.at['G', 'iss'] += (0.33 * q14)
                survey_result.at['G', 'esg1'] += (0.33 * q14)
            elif q14 == 0.5:
                survey_result.at['G', 'iss'] += (0.165 * q14)
                survey_result.at['G', 'esg1'] += (0.165 * q14)
                
            if q15 == 1:
                survey_result.at['G', 'sandp'] += (0.33 * q15)
                survey_result.at['G', 'esg1'] += (0.33 * q15)
            elif q15 == 0.5:
                survey_result.at['G', 'sandp'] += (0.33 * q15)
                survey_result.at['G', 'esg1'] += (0.33 * q15)

        finally:
            #Save the file with the opponent path
            survey_result.to_csv(survey_result_file, encoding='utf-8', index=True)
            with open(user_investment_style_file, 'w', encoding='utf-8') as f:
                f.write(q16)
            if q16 == "재무적인 요소를 중심적으로 고려한다.":
                q16 = 0.5
            elif q16 == "ESG와 재무적인 요소를 모두 고려한다.":
                q16 = 1
            elif q16 == "ESG 요소를 중심적으로 고려한다.":
                q16 = 1
            user_interest = yes_interest / (q16 + no_esg_interest + yes_interest) * 100
            with open(user_interest_file, 'w', encoding='utf-8') as f:
                f.write(str(user_interest))
            st.switch_page(survey_result_page)

#ELIF SELECTED == 'ESG Introduction':
# COL1, _, _ = st.columns ([[1,2,1]])
# with Col1:
# St.subHeader ('** esg introduction **')
# St.image ('https://media.istockphoto.com/id/1447057524/ko 7%84/%ED%99%98%EA%B2%BD-%BD-%EB%B3%B4%B4%B4%EC%A0%84%EC%9D%84-%EC%9C%84% ED%95%9C-%EA%B2%BD%EC%98%81-EC%A7%80%86%86%8D-%EA%80%80%EB%8A%A5%EC %84%B1-%EC%83%9D%ED%83%9C-%EB%B0%8F-%EC%9E%9E%EC%83%9D-%97%97%90%EB%84%84% 8%EC%A7%80%97%97%90%8C%8C%80%ED%9C-%9C-%9C-%9E%90%97%97%B0%98-%EA% B0%9C%EB%85%9C%9C%EB%A1%A1%A1%A1%EB%85%B9%EC%83%89-%EC%EC%A7%80%EA%B5% B%B3%B8%EC%9D%84-%EB%93%A4%A4%EA%B3%A0-%EC%9E%88%EC%8A%B5%EB%88%88%EB%8B%A4 .jpg? S = 612x612 & W = 0 & K = 20 & C = GHQNFLCD5DDFGD2_SQ6SLWCTG0OUVAISS-WYQZGA = ' WIDTH = 600)
# St.write ("" "" ""
# ESG stands for Environment, Social, and Governance, which means three key elements that companies must consider for sustainable and responsible management. ESG plays an important role in securing the success and sustainability of long -term companies beyond just ethical concepts.

### Environment
# Environmental factors focus on measuring and improving the impact of the company on the environment. This includes problems such as responding to climate change, resource efficiency, pollution prevention, and biodiversity preservation. Strengthening environmental sustainability increases the company's reputation, reduces legal risks, and enables cost savings in the long run.

### Society (SOCIAL)
# Social elements evaluate the effects of companies on society. This includes human rights protection, improving working conditions, contributions to community, and promoting diversity and inclusiveness. Companies that have a positive social impact can improve their morale and productivity and gain trust in customers and communities.

### Governance (GOVERNANCE)
# Governance elements cover the company's management method and decision -making process. This includes transparent accounting practices, the composition of the board of directors, the ethical behavior of the management, and the protection of shareholder rights. Sound governance structure ensures the stability and sustainability of the company and increases the trust of investors.

# # Why is ESG important?
# ### 1. Risk management
# Companies considering ESG can better manage environmental, social and legal risks. This promotes the stability and growth of long -term companies.

# ### 2. Investment attraction
# Many investors decide to invest in consideration of ESG factors. Companies that faithfully implement ESGs can get more investment opportunities.

# ### 3. Reputation improvement
# Companies that are responsible for ESG gain higher trust and positive reputation from customers and communities. This increases the brand value and contributes to business success in the long run.

# ### 4. Legal compliance
# ESG -related regulations are being tightened worldwide. Companies that comply with ESG standards can minimize legal risks and flexibly respond to regulations.

#         ## conclusion
# ESG is not just a trend, but an essential element for the sustainability of the company and the long -term success. We want to make responsible management based on the ESG principles and create a better future through environmental protection, social contribution and transparent governance structure. Please ask for your continued interest and support.
# "" ")

# Elif SELECTED == 'Methodology':
# St.write ("" "" ""
#         hello 
# Thank you for visiting our stock recommendation site. We have a comprehensive evaluation of the company's environment, social, and governance, providing services that recommend the optimal shares for users. Our methodology includes the following main factors:

#         ## 1. ESG 스코어 정의 및 평가 기준
# ESG score is an indicator of the company's sustainability and responsible management. It includes three major areas:

#### Environment
# Evaluate the efforts and achievements of companies to protect the environment. This is measured by greenhouse gas emissions, energy efficiency, resource management, and renewable energy use.

#### Society (SOCIAL)
# Evaluate the corporate social responsibility. It includes factors such as employee welfare, contributions to the community, human rights protection, and supply chain management.

#### Governance
# Evaluate transparency and responsibility for the management and operation of the company. The rescue of the board of directors, the ethics of the management, the anti -corruption policy, and the protection of shareholder rights are considered.

# ## 2. Data collection and analysis
# We use a variety of reliable data sources to calculate the ESG score. Major data sources include annual reports of companies, sustainability reports, news and media articles, and reports from professional ESG evaluation agencies. Based on this data, we proceed with the following analysis process:

##### quantitative analysis
# Enforcement analysis of environmental, social, and governance based on numerical data and KPI (core performance indicators).

#### qualitative analysis
# Evaluate the company's policies, initiatives, and industry reputation to analyze the quality of ESG -related activities.

# ## 3. ESG score calculation and weight application
# Calculate the comprehensive score based on the ESG performance of each company, and calculate the entire ESG score by applying weights to each item of environment, society, and governance. The weight is adjusted to meet the characteristics of each industry and region. In this process, more accurate evaluation is made by reflecting the company's industry and characteristics.

# ## 4. Stock recommendation algorithm
# ESG scores operate user customized stock recommendation algorithm. Considering the user's investment goals, risk acceptance, and interest industries, companies with high ESG scores are recommended. The algorithm reflects the following factors:

#### ESG score
# Priority for companies with high ESG scores.
#### Financial performance
# Considering the financial soundness and growth potential of the company.
##### Market trend
# It is recommended to reflect the current market trend and the characteristics of each industry.
    
# ## 5. Continuous monitoring and update
# ESG information is continuously updated, and the company's ESG score is regularly reevaluated. Through this, we provide accurate recommendations to the user based on the latest information and respond quickly to the company's ESG performance changes.

# ## 6. Provide transparent information
# We open the ESG score calculation process and data source transparently to provide reliable information. The user can check the details of the ESG performance of each company, which can make a better investment decision.
        
# Our ESG score -based stock recommendation service aims for responsible investment and sustainable growth. I hope it will help your investment decision. "")

#ELIF SELECTED == 'Recent News':
# St.write ('')
# St.write ('')
# St.subHeader ('Latest Economic News')

# # Enter search term
# Search = st.text_input ("Enter the keyword to search:")

# # Start crawling when clicking the button
# If St.Button ("News Search"):
# If Search:
# St.write (f "" '{search}' is searching for articles ... ")
# News_list = crawl_naver_news (search)

# If news_list:
# # St.write (F "The number of articles collected: {LEN (news_list)} dog")
#                 for title, link in news_list:
# St.markdown (f "- [{title}] ({link})")
# Else:
# St.write ("I can't find an article.")
# Else:
# St.write ("Please enter the search term.")
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

st.set_page_config(
    page_title = "설문 조사",
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
    
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
values = {'msci': 0, 'iss': 0, 'sustain': 0, 'sandp': 0, 'esg1': 0}

with st.form('usersurvey',clear_on_submit=False):
# 설문지 제목
    # st.markdown('<h1 style="font-size:30px;text-align:center;">ESG 관심 테스트</h1>', unsafe_allow_html=True)
    # st.markdown('<p style="font-size:15px;text-align:center;"><strong>지속 가능한 투자</strong>에 관심이 있으신가요?</p>', unsafe_allow_html=True)
    # st.markdown('<p style="font-size:15px;text-align:center;">여러분의 ESG 우선순위를 파악하여 <strong>맞춤형 포트폴리오</strong>를 만들어 드립니다.</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    # E 섹터 질문
    st.markdown('''
                <!DOCTYPE html>
                <html lang="ko">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        div[data-testid="stForm"]{
                                background-color:#FFFF;
                            }
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
                            background-color:#070B19;
                        }
                        div[class="st-ay st-c1 st-b0 st-b1 st-b2 st-b3 st-b4 st-ae st-b5 st-b6 st-b7 st-b8 st-b9 st-ba st-bb st-bc st-bd st-be st-bf st-bg"]{
                            transform: scale(1.5);
                            margin-right: 10px;
                            background-color:#070B19;
                        }
                        button[data-testid="baseButton-secondaryFormSubmit"]{
                            border-radius: 10px; 
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                    </style>
                </head>
                ''',unsafe_allow_html=True)
    

    
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">1. 투자할 때 기업이 탄소 배출이나 오염물질 관리 등 자연을 보호하는 데 신경 쓰는지 고려하시나요?</div>', unsafe_allow_html=True)
    q1 = st.radio('', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">2. 투자할 때 기업이 환경 관리 시스템을 구축하는 등 기후 변화에 적극 대응하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q2 = st.radio(' ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">3. 투자할 때 기업이 생산 과정에서 친환경적으로 제품과 서비스를 제공하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q3 = st.radio('  ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    # Sustainalytics ESG 기준 질문
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">4. 투자할 때 기업이 자원을 효율적으로 사용하고 배출량을 줄이는지 고려 하시나요?</div>', unsafe_allow_html=True)
    q4 = st.radio('   ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">5. 투자할 때 기업이 신재생에너지를 활용하는 등 친환경적으로 활동하는지  고려하시나요?</div>', unsafe_allow_html=True)
    q5 = st.radio('    ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">6. 투자할 때 기업이 직원의 안전을 보장하고 소비자의 권리를 보호하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q6 = st.radio('     ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    # MSCI ESG 기준 질문
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">7. 투자할 때 기업이 지역사회와의 관계를 잘 유지하고 공정하게 운영하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q7 = st.radio('      ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">8. 투자할 때 기업이 건강과 사회에 미치는 부정적인 영향을 줄이는지 고려하시나요?</div>', unsafe_allow_html=True)
    q8 = st.radio('       ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">9. 투자할 때 기업이 직원에게 차별 없이 워라벨을 지켜주고, 역량 개발을 지원하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q9 = st.radio('        ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    # 한국ESG기준원 ESG 기준 질문
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">10. 투자할 때 기업이 환경 보호, 직원 복지, 공정 거래 등 사회적 책임을 다하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q10 = st.radio('         ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">11. 투자할 때 기업이 개인정보 보호 등 사이버 보안을 잘 관리하는지 고려하시나요?</div>', unsafe_allow_html=True)
    q11 = st.radio('          ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">12. 투자할 때 기업이 경영 구조를 유지하기 위해 이사회의 독립성과 전문성을 높이려는 것을 고려하시나요?</div>', unsafe_allow_html=True)
    q12 = st.radio('           ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    # ISS ESG 기준 질문
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">13. 투자할 때 기업이 감사팀을 운영하고 회계 규정을 잘 지키는지 고려하시나요?</div>', unsafe_allow_html=True)
    q13 = st.radio('            ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">14. 투자할 때 기업이 주주의 권리를 보호하고 이익을 돌려주는지 고려하시나요?</div>', unsafe_allow_html=True)
    q14 = st.radio('             ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">15. 투자할 때 기업이 나라에 미치는 영향을 잘 관리하고, 새로운 경영 방식을 도입하는 것을 고려하시나요?</div>', unsafe_allow_html=True)
    q15 = st.radio('              ', options=('신경 쓴다.','보통이다.','신경 쓰지 않는다.'))
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    # 투자 성향에 대한 질문 ()
    st.markdown('<div class="question" style="font-size:20px;text-align:center;font-weight: bold;">16. 귀하는 투자시 무엇을 고려하시나요?</div>', unsafe_allow_html=True)
    q16 = st.radio('               ', options=('ESG 요소를 중심적으로 고려한다.','ESG와 재무적인 요소를 모두 고려한다.','재무적인 요소를 중심적으로 고려한다.'))
    st.markdown('</div>',unsafe_allow_html=True)
    
    
    care_levels = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
    esg_interest = 0
    financial_interest = 0
    results = [evaluate_care_level(level) for level in care_levels]
    for i in range(1, 16):
        exec(f'q{i} = evaluate_care_level(q{i})')
    for result in results:
        if result == 1:  # 신경 쓴다 -> ESG 관심도 +1
            esg_interest += 1
        elif result == 0.5:  # 보통이다 -> ESG +1, 재무적 관심도 +1
            esg_interest += 1
            financial_interest += 1
        elif result == 0:  # 신경 쓰지 않는다 -> 재무적 관심도 +1
            financial_interest += 1
    financial_interest = financial_interest/(esg_interest + financial_interest) * 100
    esg_interest = esg_interest/(esg_interest + financial_interest) * 100
    _,survey_submitted, _ = st.columns([3,1,3])
    with survey_submitted:
        submitted = st.form_submit_button('설문 완료')
    
    if submitted:
        try:
            survey_result = pd.DataFrame(index=['E', 'S', 'G'], columns=['esg1', 'sandp', 'sustain', 'iss', 'msci'])
            survey_result.loc[:, :] = 0
            no_esg_interest = 0
            yes_interest = 0
            if q1 == 1:
                survey_result.at['E', 'sustain'] += (1 * q1)
                survey_result.at['E', 'msci'] += (0.5 * q1)
                yes_interest += 1
            elif q1 == 0.5: 
                survey_result.at['E', 'sustain'] += (0.5 * q1)
                survey_result.at['E', 'msci'] += (0.25 * q1)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q2 == 1:
                survey_result.at['E', 'iss'] += (0.33 * q2)
                survey_result.at['E', 'sandp'] += (1 * q2)
                yes_interest += 1
            elif q2 == 0.5:
                survey_result.at['E', 'iss'] += (0.165 * q2)
                survey_result.at['E', 'sandp'] += (0.5 * q2)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q3 == 1:
                survey_result.at['E', 'iss'] += (0.33 * q3)
                survey_result.at['E', 'esg1'] += (1 * q3)
                yes_interest += 1
            elif q3 == 0.5:
                survey_result.at['E', 'iss'] += (0.165 * q3)
                survey_result.at['E', 'esg1'] += (0.5 * q3)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q4 == 1:
                survey_result.at['E', 'iss'] += (0.33 * q4)
                yes_interest += 1
            elif q4 == 0.5:
                survey_result.at['S', 'iss'] += (0.165 * q4)
                yes_interest += 1
            else:
                no_esg_interest += 1

            if q5 == 1:
                survey_result.at['E', 'msci'] += (0.5 * q5)
                yes_interest += 1
            elif q5 == 0.5:
                survey_result.at['E', 'msci'] += (0.25 * q5)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q6 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q6)
                survey_result.at['S', 'msci'] += (0.2 * q6)
                yes_interest += 1
            elif q6 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q6)
                survey_result.at['S', 'msci'] += (0.1 * q6)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q7 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q7)
                survey_result.at['S', 'msci'] += (0.2 * q7)
                survey_result.at['S', 'iss'] += (0.33 * q7)
                yes_interest += 1
            elif q7 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q7)
                survey_result.at['S', 'msci'] += (0.1 * q7)
                survey_result.at['S', 'iss'] += (0.165 * q7)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q8 == 1:
                survey_result.at['S', 'msci'] += (0.2 * q8)
                yes_interest += 1
            elif q8 == 0.5:
                survey_result.at['S', 'msci'] += (0.1 * q8)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q9 == 1:
                survey_result.at['S', 'iss'] += (0.33 * q9)
                survey_result.at['S', 'esg1'] += (1 * q9)
                yes_interest += 1
            elif q9 == 0.5:
                survey_result.at['S', 'iss'] += (0.165 * q9)
                survey_result.at['S', 'esg1'] += (0.5 * q9)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q10 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q10)
                survey_result.at['S', 'iss'] += (0.33 * q10)
                yes_interest += 1
            elif q10 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q10)
                survey_result.at['S', 'iss'] += (0.165 * q10)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q11 == 1:
                survey_result.at['S', 'sustain'] += (0.25 * q11)
                survey_result.at['S', 'msci'] += (0.2 * q11)
                survey_result.at['S', 'sandp'] += (1 * q11)
                yes_interest += 1
            elif q11 == 0.5:
                survey_result.at['S', 'sustain'] += (0.125 * q11)
                survey_result.at['S', 'msci'] += (0.1 * q11)
                survey_result.at['S', 'sandp'] += (0.5 * q11)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q12 == 1:
                survey_result.at['G', 'sustain'] += (0.25 * q12)
                survey_result.at['G', 'msci'] += (0.2 * q12)
                survey_result.at['G', 'iss'] += (0.2 * q12)
                survey_result.at['G', 'sandp'] += (1 * q12)
                survey_result.at['G', 'esg1'] += (0.2 * q12)
                yes_interest += 1
            elif q12 == 0.5:
                survey_result.at['G', 'sustain'] += (0.5 * q12)
                survey_result.at['G', 'msci'] += (0.5 * q12)
                survey_result.at['G', 'iss'] += (0.165 * q12)
                survey_result.at['G', 'sandp'] += (0.165 * q12)
                survey_result.at['G', 'esg1'] += (0.165 * q12)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q13 == 1:
                survey_result.at['G', 'iss'] += (0.33 * q13)
                survey_result.at['G', 'sandp'] += (0.33 * q13)
                survey_result.at['G', 'esg1'] += (0.33 * q13)
                yes_interest += 1
            elif q13 == 0.5:
                survey_result.at['G', 'iss'] += (0.165 * q13)
                survey_result.at['G', 'sandp'] += (0.165 * q13)
                survey_result.at['G', 'esg1'] += (0.165 * q13)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q14 == 1:
                survey_result.at['G', 'iss'] += (0.33 * q14)
                survey_result.at['G', 'esg1'] += (0.33 * q14)
                yes_interest += 1
            elif q14 == 0.5:
                survey_result.at['G', 'iss'] += (0.165 * q14)
                survey_result.at['G', 'esg1'] += (0.165 * q14)
                yes_interest += 1
            else:
                no_esg_interest += 1
                
            if q15 == 1:
                survey_result.at['G', 'sandp'] += (0.33 * q15)
                survey_result.at['G', 'esg1'] += (0.33 * q15)
                yes_interest += 1
            elif q15 == 0.5:
                survey_result.at['G', 'sandp'] += (0.33 * q15)
                survey_result.at['G', 'esg1'] += (0.33 * q15)
                yes_interest += 1
            else:
                no_esg_interest += 1
            
            
                
        finally:
            survey_result.to_csv(r"C:\esgpage\LLM-ESG-POS\interface\survey_result.csv", encoding='utf-8', index=True)

            with open(r"C:\esgpage\LLM-ESG-POS\interface\user_investment_style.txt", 'w', encoding='utf-8') as f:
                f.write(q16)
                
            if q16 == "재무적인 요소를 중심적으로 고려한다.":
                q16 = 0.5
            elif q16 == "ESG와 재무적인 요소를 모두 고려한다.":
                q16 = 1 
            elif q16 == "ESG 요소를 중심적으로 고려한다.":
                q16 = 1
                
            user_interest = yes_interest / (q16 + no_esg_interest + yes_interest) * 100
            with open(r"C:\esgpage\LLM-ESG-POS\interface\user_interest.txt", 'w', encoding='utf-8') as f:
                f.write(str(user_interest))
                
            st.switch_page('pages/survey_result.py')
            

# elif selected == 'ESG 소개':
#     col1,_,_ = st.columns([1,2,1])
#     with col1:
#         st.subheader('**ESG 소개**')
#         st.image('https://media.istockphoto.com/id/1447057524/ko/%EC%82%AC%EC%A7%84/%ED%99%98%EA%B2%BD-%EB%B0%8F-%EB%B3%B4%EC%A0%84%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B2%BD%EC%98%81-esg-%EC%A7%80%EC%86%8D-%EA%B0%80%EB%8A%A5%EC%84%B1-%EC%83%9D%ED%83%9C-%EB%B0%8F-%EC%9E%AC%EC%83%9D-%EC%97%90%EB%84%88%EC%A7%80%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9E%90%EC%97%B0%EC%9D%98-%EA%B0%9C%EB%85%90%EC%9C%BC%EB%A1%9C-%EB%85%B9%EC%83%89-%EC%A7%80%EA%B5%AC%EB%B3%B8%EC%9D%84-%EB%93%A4%EA%B3%A0-%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.jpg?s=612x612&w=0&k=20&c=ghQnfLcD5dDfGd2_sQ6sLWctG0xI0ouVaISs-WYQzGA=', width=600)
#     st.write("""
#     ESG는 환경(Environment), 사회(Social), 지배구조(Governance)의 약자로, 기업이 지속 가능하고 책임 있는 경영을 위해 고려해야 하는 세 가지 핵심 요소를 의미합니다. ESG는 단순한 윤리적 개념을 넘어, 장기적인 기업의 성공과 지속 가능성을 확보하기 위해 중요한 역할을 합니다.

#         ### 환경 (Environment)
#         환경 요소는 기업이 환경에 미치는 영향을 측정하고 개선하는 데 중점을 둡니다. 이는 기후 변화 대응, 자원 효율성, 오염 방지, 생물 다양성 보전 등의 문제를 포함합니다. 환경 지속 가능성을 강화하는 것은 기업의 평판을 높이고, 법적 리스크를 줄이며, 장기적으로 비용 절감을 가능하게 합니다.

#         ### 사회 (Social)
#         사회 요소는 기업이 사회에 미치는 영향을 평가합니다. 이는 인권 보호, 노동 조건 개선, 지역 사회 기여, 다양성과 포용성 증진 등을 포함합니다. 긍정적인 사회적 영향을 미치는 기업은 직원의 사기와 생산성을 높이고, 고객과 지역 사회의 신뢰를 얻을 수 있습니다.

#         ### 지배구조 (Governance)
#         지배구조 요소는 기업의 경영 방식과 의사 결정 과정을 다룹니다. 이는 투명한 회계 관행, 이사회 구성, 경영진의 윤리적 행동, 주주 권리 보호 등을 포함합니다. 건전한 지배구조는 기업의 안정성과 지속 가능성을 보장하고, 투자자들의 신뢰를 증대시킵니다.

#         ## 왜 ESG가 중요한가요?
#         ### 1. 위험 관리
#         ESG를 고려하는 기업은 환경적, 사회적, 법적 리스크를 더 잘 관리할 수 있습니다. 이는 장기적인 기업의 안정성과 성장을 도모합니다.

#         ### 2. 투자 유치
#         많은 투자자들이 ESG 요인을 고려하여 투자를 결정합니다. ESG를 충실히 이행하는 기업은 더 많은 투자 기회를 얻을 수 있습니다.

#         ### 3. 평판 향상
#         ESG에 대한 책임을 다하는 기업은 고객과 지역 사회로부터 더 높은 신뢰와 긍정적인 평판을 얻습니다. 이는 브랜드 가치를 높이고, 장기적으로 비즈니스 성공에 기여합니다.

#         ### 4. 법적 준수
#         전 세계적으로 ESG 관련 규제가 강화되고 있습니다. ESG 기준을 준수하는 기업은 법적 리스크를 최소화하고, 규제 변경에 유연하게 대응할 수 있습니다.

#         ## 결론
#         ESG는 단순한 트렌드가 아니라, 기업의 지속 가능성과 장기적인 성공을 위한 필수적인 요소입니다. 우리는 ESG 원칙을 바탕으로 책임 있는 경영을 실천하며, 환경 보호, 사회적 기여, 투명한 지배구조를 통해 더 나은 미래를 만들어 나가고자 합니다. 여러분의 지속적인 관심과 지지를 부탁드립니다.
#         """)

# elif selected == '방법론':
#     st.write("""
#         안녕하십니까 
#         당사의 주식 추천 사이트에 방문해 주셔서 감사합니다. 저희는 기업의 환경(Environment), 사회(Social), 지배구조(Governance) 측면을 종합적으로 평가하여 사용자에게 최적의 주식을 추천하는 서비스를 제공합니다. 당사의 방법론은 다음과 같은 주요 요소를 포함합니다.

#         ## 1. ESG 스코어 정의 및 평가 기준
#         ESG 스코어는 기업의 지속 가능성과 책임 있는 경영을 측정하는 지표로, 다음과 같은 세 가지 주요 분야를 포함합니다:

#         #### 환경(Environment)
#         기업이 환경 보호를 위해 수행하는 노력과 성과를 평가합니다. 이는 온실가스 배출량, 에너지 효율성, 자원 관리, 재생 가능 에너지 사용 등으로 측정됩니다.

#         #### 사회(Social)
#         기업의 사회적 책임을 평가합니다. 직원 복지, 지역 사회에 대한 기여, 인권 보호, 공급망 관리 등과 같은 요소가 포함됩니다.

#         #### 지배구조(Governance)
#         기업의 관리 및 운영 방식에 대한 투명성과 책임성을 평가합니다. 이사회 구조, 경영진의 윤리, 부패 방지 정책, 주주 권리 보호 등이 고려됩니다.

#         ## 2. 데이터 수집 및 분석
#         저희는 ESG 스코어를 산출하기 위해 신뢰할 수 있는 다양한 데이터 소스를 활용합니다. 주요 데이터 소스에는 기업의 연례 보고서, 지속 가능성 보고서, 뉴스 및 미디어 기사, 그리고 전문 ESG 평가 기관의 리포트가 포함됩니다. 이 데이터를 바탕으로 저희는 다음과 같은 분석 과정을 진행합니다:

#         #### 정량적 분석
#         수치 데이터 및 KPI(핵심 성과 지표)를 기반으로 한 환경적, 사회적, 지배구조적 성과 분석을 수행합니다.

#         #### 정성적 분석
#         기업의 정책 및 이니셔티브, 업계 평판 등을 평가하여 ESG 관련 활동의 질적 측면을 분석합니다.

#         ## 3. ESG 스코어 산출 및 가중치 적용
#         각 기업의 ESG 성과를 기반으로 종합 스코어를 산출하며, 환경, 사회, 지배구조 각 항목에 대해 가중치를 적용하여 전체 ESG 스코어를 계산합니다. 가중치는 산업별, 지역별 특성에 맞추어 조정됩니다. 이 과정에서 기업의 업종과 특성을 반영하여 보다 정확한 평가가 이루어집니다.

#         ## 4. 주식 추천 알고리즘
#         ESG 스코어를 바탕으로 사용자 맞춤형 주식 추천 알고리즘을 운영합니다. 사용자의 투자 목표, 리스크 수용 범위, 관심 산업 등을 고려하여 ESG 점수가 높은 기업을 추천합니다. 알고리즘은 다음과 같은 요소를 반영합니다:

#         #### ESG 스코어
#         높은 ESG 스코어를 가진 기업을 우선 추천합니다.
#         #### 재무 성과
#         기업의 재무 건전성과 성장 잠재력도 함께 고려합니다.
#         #### 시장 동향
#         현재 시장 동향 및 산업별 특성을 반영하여 추천합니다.
    
#         ## 5. 지속적인 모니터링 및 업데이트
#         ESG 관련 정보는 지속적으로 업데이트되며, 기업의 ESG 스코어는 정기적으로 재평가됩니다. 이를 통해 최신 정보를 바탕으로 사용자에게 정확한 추천을 제공하며, 기업의 ESG 성과 변화에 신속하게 대응합니다.

#         ## 6. 투명한 정보 제공
#         저희는 사용자가 신뢰할 수 있는 정보를 제공하기 위해 ESG 스코어 산출 과정과 데이터 출처를 투명하게 공개합니다. 사용자는 각 기업의 ESG 성과에 대한 자세한 정보를 확인할 수 있으며, 이를 바탕으로 보다 나은 투자 결정을 내릴 수 있습니다.
        
#         저희의 ESG 스코어 기반 주식 추천 서비스는 책임 있는 투자와 지속 가능한 성장을 지향합니다. 여러분의 투자 결정에 도움이 되기를 바랍니다.""")

# elif selected == '최근 뉴스':
#     st.write(' ')
#     st.write(' ')
#     st.subheader('최근 경제 뉴스')

#     # 검색어 입력
#     search = st.text_input("검색할 키워드를 입력하세요:")

#     # 버튼 클릭 시 크롤링 시작
#     if st.button("뉴스 검색"):
#         if search:
#             st.write(f"'{search}' 관련 기사를 검색 중입니다...")
#             news_list = crawl_naver_news(search)

#             if news_list:
#                 # st.write(f"수집된 기사 수: {len(news_list)}개")
#                 for title, link in news_list:
#                     st.markdown(f"- [{title}]({link})")
#             else:
#                 st.write("기사를 찾을 수 없습니다.")
#         else:
#             st.write("검색어를 입력해주세요.")
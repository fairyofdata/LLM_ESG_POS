"""Recent news page: keyword search over Naver news articles."""

import streamlit as st

from app.styles import inject_global_font
from src.collection.news_crawler import crawl_naver_news

inject_global_font()

st.markdown(
    '<h1 style="font-size:35px;text-align:center;font-weight:bold;">최신 뉴스</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 style="font-size:15px;text-align:center;">궁금한 키워드를 검색해보세요.</h1>',
    unsafe_allow_html=True,
)

search = st.text_input(" ")
_, col, _ = st.columns([5, 1, 5])
with col:
    search_button = st.button("검색")

if search_button:
    if search:
        st.markdown(
            f'<p style="text-align:center;font-size:17px;">{search}관련 기사를 검색 중입니다...</p>',
            unsafe_allow_html=True,
        )
        try:
            news_list = crawl_naver_news(search)
        except Exception as error:
            st.error(f"뉴스 수집 중 오류가 발생했습니다: {error}")
            news_list = []
        if news_list:
            for title, link in news_list:
                st.markdown(f"- [{title}]({link})")
        else:
            st.write("기사를 찾을 수 없습니다.")
    else:
        st.write("검색어를 입력해주세요.")

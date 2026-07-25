"""Portfolio dashboard: optimization results personalized by the survey.

All heavy lifting (data loading, Black-Litterman optimization, word cloud
blending, report rendering) is delegated to the ``src`` package; this page
only wires the widgets together.
"""

import base64

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_vertical_slider import vertical_slider

from app.styles import (
    display_text_on_hover,
    inject_centered_radios,
    inject_global_font,
    page_header,
)
from src.data.esg_loader import (
    build_integrated_esg_data,
    load_company_profiles,
    load_esg_scores,
    load_word_frequencies,
)
from src.data.market_data import load_close_prices, load_index_change, load_ohlcv
from src.optimization.black_litterman import (
    PortfolioPerformance,
    build_portfolio_table,
    optimize_portfolio,
)
from src.reporting.portfolio_report import html_to_pdf, render_report_html
from src.scoring.survey import normalized_pillar_preferences, tau_for_style
from src.visualization.wordcloud_builder import (
    build_blended_frequencies,
    generate_wordcloud,
)

inject_global_font()
inject_centered_radios()


# --- Cached data access -----------------------------------------------------

@st.cache_data
def get_esg_data() -> dict[int, pd.DataFrame]:
    return load_esg_scores()


@st.cache_data
def get_integrated_esg_data() -> pd.DataFrame:
    return build_integrated_esg_data(load_esg_scores())


@st.cache_data
def get_company_profiles() -> pd.DataFrame:
    return load_company_profiles()


@st.cache_data
def get_word_frequencies() -> pd.DataFrame:
    return load_word_frequencies()


@st.cache_data
def get_close_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return load_close_prices(list(tickers))


@st.cache_data
def get_ohlcv(code: str, ndays: int) -> pd.DataFrame:
    return load_ohlcv(code, ndays)


@st.cache_data
def get_index_change(index_code: str):
    return load_index_change(index_code)


# --- Session guard ----------------------------------------------------------

if "survey_result" not in st.session_state:
    st.info("설문을 먼저 완료해 주세요. 설문 결과를 바탕으로 맞춤형 포트폴리오를 제안해 드립니다.")
    st.page_link("pages/survey.py", label="설문 하러 가기", icon="📋")
    st.stop()

survey_result: pd.DataFrame = st.session_state["survey_result"]
investment_style: str = st.session_state.get("investment_style", "ESG와 재무적인 요소를 모두 고려한다.")
user_name: str = st.session_state.get("user_name", "당신")

page_header(f"{user_name}을 위한 ESG 투자 최적화 포트폴리오")

slider_defaults = normalized_pillar_preferences(survey_result)


# --- Optimization helper ----------------------------------------------------

def calculate_optimal_portfolio(
    esg_weights: dict[str, float],
) -> tuple[pd.DataFrame, PortfolioPerformance]:
    """Run the full optimization for the current slider values."""
    esg_data = get_esg_data()
    latest_year = max(year for year, df in esg_data.items() if not df.empty)
    latest_esg = esg_data[latest_year]

    prices = get_close_prices(tuple(latest_esg.index))
    returns = prices.pct_change().dropna()

    weights, performance = optimize_portfolio(
        returns=returns,
        esg_scores=latest_esg,
        esg_weights=esg_weights,
        tau=tau_for_style(investment_style),
    )
    return build_portfolio_table(weights, latest_esg), performance


# --- Layout: controls | pie chart | summary ---------------------------------

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    kospi, kosdaq = st.columns(2)
    with kospi:
        snapshot = get_index_change("KS11")
        if snapshot:
            level, change = snapshot
            st.metric(label="오늘의 코스피 지수", value=round(level, 2),
                      delta=f"{round(change, 2)}%", delta_color="inverse")
    with kosdaq:
        snapshot = get_index_change("KQ11")
        if snapshot:
            level, change = snapshot
            st.metric(label="오늘의 코스닥 지수", value=round(level, 2),
                      delta=f"{round(change, 2)}%", delta_color="inverse")

    sl1, sl2, sl3 = st.columns(3)
    slider_config = dict(
        height=195, step=0.1, min_value=0.01, max_value=1.0,
        track_color="#f0f0f0", slider_color="#006699", thumb_color="#FF9933",
        value_always_visible=True,
    )
    with sl1:
        display_text_on_hover("-탄소 관리<br>-폐기물 관리<br>-기후 변화 전략", 1, "&emsp;E")
        e_value = vertical_slider(
            label="환경", key="environmental",
            default_value=slider_defaults["environmental"], **slider_config)
    with sl2:
        display_text_on_hover("-사회적 기회<br>-지역사회 관계<br>-인적 자원", 2, "&emsp;S")
        s_value = vertical_slider(
            label="사회", key="social",
            default_value=slider_defaults["social"], **slider_config)
    with sl3:
        display_text_on_hover("-주주권 보호<br>-기업이사회운영<br>", 3, "&emsp;G")
        g_value = vertical_slider(
            label="지배구조", key="governance",
            default_value=slider_defaults["governance"], **slider_config)

    esg_weights = {"environmental": e_value, "social": s_value, "governance": g_value}

    try:
        top_companies, portfolio_performance = calculate_optimal_portfolio(esg_weights)
    except Exception as error:
        st.error(f"포트폴리오 최적화 중 오류가 발생했습니다: {error}")
        top_companies = pd.DataFrame(
            columns=["ticker", "Company", "Weight",
                     "environmental", "social", "governance"])
        portfolio_performance = PortfolioPerformance(0.0, 0.0, 0.0)

    excluded_companies = st.multiselect(
        "", top_companies["Company"], placeholder="제외하고 싶은 기업을 선택")
    if excluded_companies:
        top_companies = top_companies[~top_companies["Company"].isin(excluded_companies)]

with col2:
    st.markdown(
        """<div>
            <h2 style="font-family: Pretendard;font-size: 13px; text-align:center; text-decoration: none;">차트에서 여러분의 관심 회사 이름을 클릭하여<br>더 다양한 정보를 경험해 보세요.</h2>
        </div>""",
        unsafe_allow_html=True,
    )

    pie_fig = px.pie(
        top_companies,
        names="Company",
        values="Weight",
        color_discrete_sequence=px.colors.qualitative.G10,
        custom_data=top_companies[["environmental", "social", "governance"]],
    )
    pie_fig.update_traces(
        textposition="inside",
        textinfo="percent+label+value",
        hovertemplate=(
            "추천 포트폴리오 비중 : %{percent}<br>"
            "Environmental 점수 :  %{customdata[0][0]:.2f}<br>"
            "Social 점수  :  %{customdata[0][1]:.2f}<br>"
            "Governance : %{customdata[0][2]:.2f}<br>"
        ),
        texttemplate="%{label}",
    )
    pie_fig.update_layout(
        font=dict(size=16, color="black"),
        showlegend=False,
        margin=dict(t=40, b=40, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=250,
        height=400,
    )
    clicked_points = plotly_events(pie_fig, click_event=True, key="company_click")

with col3:
    company_profiles = get_company_profiles()

    performance = portfolio_performance
    top5_companies = top_companies.nlargest(5, "Weight")
    detail_table = pd.merge(company_profiles, top5_companies, on="ticker")
    detail_table = detail_table[
        ["Company", "Weight", "environmental", "social", "governance", "종목설명"]]
    detail_table = detail_table.rename(columns={
        "Company": "종목명", "Weight": "제안 비중",
        "environmental": "E", "social": "S", "governance": "G",
        "종목설명": "종목 소개",
    })

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        display_text_on_hover(
            "포트폴리오의 연간 기대 수익률을 나타내는 지표입니다.", 4,
            f"연간 기대 수익률 &emsp; {performance.expected_return * 100:.2f} %")
        st.markdown("")
    with metric2:
        display_text_on_hover(
            "수익률이 얼마나 변동할 수 있는지 나타내는 위험 지표입니다.", 5,
            f"연간 변동성 &emsp; {performance.volatility * 100:.2f} %")
    with metric3:
        display_text_on_hover(
            "위험 대비 수익을 얼마나 효율적으로 창출하는지 나타내는 성과 지표입니다.", 6,
            f"샤프 비율 &emsp;{performance.sharpe_ratio:.2f}")

    table_html = """
    <div style="font-family: Arial, sans-serif; text-align:center;">
    <style>
        table { width: 100%; border-collapse: collapse; table-layout: auto; }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            font-size:15px;
            font-family: Pretendard;
        }
    </style>
    <table>
        <thead>
        <tr>
            <th rowspan='2'>종목</th>
            <th rowspan='2'>제안<br>비중</th>
            <th colspan="3">ESG Score<br>(2023)</th>
            <th rowspan='2'>종목 소개</th>
        </tr>
        <tr><th>E</th><th>S</th><th>G</th></tr>
        </thead>
        <tbody>
    """
    for _, row in detail_table.sort_values(by="제안 비중", ascending=False).iterrows():
        table_html += f"""<tr>
        <td style="font-size:13px;">{row['종목명']}</td>
        <td>{row['제안 비중']:.2f}%</td>
        <td>{int(row['E'])}</td>
        <td>{int(row['S'])}</td>
        <td>{int(row['G'])}</td>
        <td style="text-align: left;">{row['종목 소개']}</td>
        </tr>"""
    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

    _, _, bt1, bt2 = st.columns(4)
    with bt1:
        export_requested = st.button(label="포트폴리오 확인  ➡️")
    if export_requested:
        with bt2:
            full_table = pd.merge(company_profiles, top_companies, on="ticker")
            full_table = full_table[
                ["Company", "Weight", "environmental", "social", "governance", "종목설명"]]
            full_table = full_table.rename(columns={
                "Company": "종목명", "Weight": "제안 비중",
                "environmental": "E", "social": "S", "governance": "G",
                "종목설명": "종목 소개",
            })

            try:
                chart_png = base64.b64encode(
                    pie_fig.to_image(format="png")).decode("utf-8")
            except Exception:
                chart_png = None  # static image export unavailable

            report_html = render_report_html(
                user_name=user_name,
                portfolio=full_table,
                esg_weights=esg_weights,
                performance=performance,
                chart_png_base64=chart_png,
            )
            pdf_bytes = html_to_pdf(report_html)
            if pdf_bytes:
                st.download_button(
                    label="💾 pdf 다운", data=pdf_bytes,
                    file_name="esg_report.pdf", mime="application/pdf")
            else:
                st.download_button(
                    label="💾 리포트 다운(HTML)", data=report_html.encode("utf-8"),
                    file_name="esg_report.html", mime="text/html")


# --- Detail row: ESG history | price chart | word cloud ---------------------

detail1, detail2, detail3 = st.columns(3)

clicked_company: str | None = None
if clicked_points:
    point = clicked_points[0]
    if "pointNumber" in point and point["pointNumber"] < len(top_companies):
        clicked_company = top_companies.iloc[point["pointNumber"]]["Company"]

with detail1:
    if clicked_company:
        st.markdown(
            f"""<div>
                <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">{clicked_company} ESG 스코어</h2>
            </div>""",
            unsafe_allow_html=True,
        )
        history = get_integrated_esg_data()
        history = history[history["Company"] == clicked_company]
        history = history[["Year", "environmental", "social", "governance"]].copy()
        history["Year"] = history["Year"].astype(int)
        history = history.melt(
            id_vars="Year",
            value_vars=["environmental", "social", "governance"],
            var_name="Category",
            value_name="Score",
        )
        line_fig = px.line(history, x="Year", y="Score", color="Category")
        line_fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.6,
                        xanchor="center", x=0.5),
            width=750, height=350,
        )
        st.plotly_chart(line_fig)

with detail2:
    if clicked_company:
        st.markdown(
            f"""<div>
                <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">&emsp;&ensp;{clicked_company} &ensp;주가 그래프</h2>
            </div>""",
            unsafe_allow_html=True,
        )
        ticker = top_companies.loc[
            top_companies["Company"] == clicked_company, "ticker"].iloc[0]
        ohlcv = get_ohlcv(ticker, 1825)
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv.index = pd.to_datetime(ohlcv.index)
        market_colors = mpf.make_marketcolors(up="red", down="blue")
        mpf_style = mpf.make_mpf_style(base_mpf_style="default",
                                       marketcolors=market_colors)
        candle_fig, _ = mpf.plot(
            data=ohlcv, volume=True, type="candle", style=mpf_style,
            figsize=(10, 7), fontscale=1.1,
            mav=(5, 10, 30), mavcolors=("red", "green", "blue"),
            returnfig=True,
        )
        st.pyplot(candle_fig)

with detail3:
    if clicked_company:
        st.markdown(
            """<div>
                <h2 style="font-family: Pretendard;font-size: 20px; text-align:center;">포트폴리오 기반 워드 클라우드</h2>
            </div>""",
            unsafe_allow_html=True,
        )
        frequencies = build_blended_frequencies(top_companies, get_word_frequencies())
        wordcloud = generate_wordcloud(frequencies)
        if wordcloud:
            cloud_fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(cloud_fig)
        else:
            st.info("생성할 데이터가 충분하지 않아 워드 클라우드를 표시할 수 없습니다.")

**🌐 Available Versions:** [🇰🇷 한국어 (Korean)](/README_KR.md) | [🇯🇵 日本語 (Japanese)](/README_JP.md)

---

# LEPOS: LLM-based ESG-Focused Portfolio Optimization Service 📊🌱

> 🏆🥈 Excellence Award (2nd) — 8th Industry-Academia SW Project Exhibition, Kwangwoon University
> 🏆🥈 Excellence Award (2nd) — 2024 Graduation Exhibition, Kwangwoon University College of SW Convergence

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

LEPOS evaluates the ESG (Environmental / Social / Governance) profile of Korean listed
companies **directly from news text using LLMs and fine-tuned Korean language models**,
then builds a **personalized portfolio** with a Black-Litterman optimizer that treats
the user's ESG preferences as investor views.

Traditional ESG ratings are opaque, disagree across agencies, and cover only large
companies. LEPOS addresses all three: the scoring pipeline is fully transparent, it
learns the rating behaviour of five agencies (MSCI, S&P, Sustainalytics, ISS, KCGS),
and — because it needs only text — it can score companies that no agency covers.

## 🎬 Demo

[![UI/UX Demo Video](https://img.youtube.com/vi/kHAtgLC4PJY/0.jpg)](https://www.youtube.com/watch?v=kHAtgLC4PJY)

*(Click the thumbnail to watch the UI/UX walkthrough on YouTube.)*

## 🔬 Research Highlights

After the graduation exhibition, the optimization model was refined and validated in a
follow-up backtesting study (2020–2024, annual rebalancing with year *t−1* scores —
no look-ahead). The LLM-ESG optimized portfolio beat every benchmark **on every
metric**:

| Portfolio | 5y Cumulative Return | CAGR | Volatility | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|---|
| KOSPI | 1.092 | 0.018 | 0.202 | 0.190 | 0.357 | 0.051 |
| ESG ETF | 1.159 | 0.031 | 0.203 | 0.251 | 0.352 | 0.087 |
| Equal-weighted | 1.247 | 0.046 | 0.204 | 0.323 | 0.362 | 0.128 |
| **LEPOS (τ = 1.3)** | **1.377** | **0.068** | **0.154** | **0.503** | **0.187** | **0.362** |

➡️ Full methodology, charts, and reproducible data: [docs/research/RESEARCH.md](docs/research/RESEARCH.md)

## 🏗️ How It Works

![Architecture](docs/architecture.png)

### 1. Text data pipeline (research, in `notebooks/`)

News articles for 68 KOSPI companies (2019–2023, **1.38M articles**) were crawled from
Naver News, cleaned (company-name normalization & anonymization, metadata removal),
and passed through a cascade of fine-tuned **KoELECTRA** classifiers whose labels were
bootstrapped with the OpenAI API (GPT-3.5-turbo):

| Model | Task | Output | Accuracy / R² |
|---|---|---|---|
| A0 | Filter irrelevant articles | keep / drop | 0.69 |
| A1 | Company relevance | relevant / not | — |
| A2 | ESG relevance | relevant / not | 0.76 |
| A3 | ESG sentiment | −1 / 0 / +1 | 0.73 |
| B1 | Agency-style ESG score regression | 0.0–7.0 per agency | R² 0.69 |
| C | E/S/G sector classification | per-pillar flags | 0.83 |

Model B1 is trained per agency (5 models) on article text plus quantitative domain
data (emissions, board composition, financials), so it reproduces each agency's rating
tendencies — and can score companies the agencies don't cover.

### 2. Personalization & optimization (serving, in `src/`)

1. A 15-question survey maps the user's values onto the five agencies' methodologies,
   yielding a pillar-by-agency preference matrix (`src/scoring/survey.py`).
2. Company E/S/G component scores + user preferences form the views (P, Q) of a
   **Black-Litterman** model; the investment style (financial ↔ ESG centered) sets τ.
   Covariance uses **Ledoit-Wolf shrinkage** (`src/optimization/black_litterman.py`).
3. Weights maximize the Sharpe ratio (long-only, fully invested) and are visualized in
   the Streamlit dashboard with per-company drill-downs (5-year ESG history, price
   candlesticks, news word cloud) and PDF/HTML report export.

## 📁 Project Structure

```text
LLM_ESG_POS/
├── app/                     # Streamlit UI (st.navigation multipage)
│   ├── main.py              #   entry point
│   └── pages/               #   home, survey, portfolio dashboard, news, ESG intro
├── src/                     # Business logic (typed, documented, UI-free)
│   ├── config.py            #   pathlib-based paths & constants
│   ├── data/                #   ESG table loading, market data (FinanceDataReader)
│   ├── scoring/             #   survey scoring matrix
│   ├── optimization/        #   Black-Litterman + max-Sharpe solver
│   ├── collection/          #   Naver news crawler
│   ├── visualization/       #   portfolio-weighted word cloud
│   └── reporting/           #   HTML/PDF report export
├── notebooks/               # Research pipeline (01 collection → 06 optimization)
├── data/
│   ├── processed/           # ESG score tables (2019–2023), company profiles
│   ├── dummy/               # sample data for experimentation
│   └── user/                # runtime state (gitignored)
├── docs/                    # architecture, final report, presentation
│   └── research/            # backtesting study (RESEARCH.md + data + charts)
├── tests/                   # pytest suite (scoring / loading / optimization)
└── requirements.txt
```

## 🚀 Getting Started

```bash
git clone https://github.com/fairyofdata/LLM_ESG_POS.git
cd LLM_ESG_POS
pip install -r requirements.txt
streamlit run app/main.py
```

Then complete the survey; the dashboard downloads 5 years of KRX prices on first run
(takes a minute) and renders your personalized portfolio.

Run the tests with:

```bash
pytest
```

**Optional system dependencies** (the app degrades gracefully without them):

| Dependency | Feature |
|---|---|
| [wkhtmltopdf](https://wkhtmltopdf.org/) | PDF report export (HTML export always available) |
| Chrome / Chromium | "Recent news" live crawling page |
| Korean font (Malgun Gothic / NanumGothic) | Word cloud rendering |

## 📚 Documentation

- [Final project report (KR, PDF)](docs/final_report_kr.pdf) — full 28-page write-up
- [Exhibition presentation (KR, PPTX)](docs/presentation_kr.pptx)
- [Backtesting research](docs/research/RESEARCH.md)
- System diagram: [docs/system_diagram_kr.png](docs/system_diagram_kr.png)

## 🔭 Future Extensions

1. **Coverage expansion**: extend scoring to ~1,000 companies (B2 model) including
   startups and unlisted companies where agency ratings don't exist.
2. **Real-time scoring**: refresh ESG scores from live news streams.
3. **Optimization constraints**: sector caps and turnover limits.
4. **Per-agency trust weighting**: expose the research-grade P/Q design (agency-level
   views with user trust vector) in the app.

## 👥 About

**KWU 8th Industry-Academic Cooperation SW Project & College of SW Convergence Graduation Project**

- **Team KWargs**: Baek Ji-heon (PM · preprocessing, NLP models A0/C, optimization),
  Kim Na-yeon (FE · crawling pipeline, Streamlit UI), Jang Han-jae (BE · labeling
  pipeline, NLP models A2/A3/B1)
- **Adviser**: Prof. Cho Min-soo (Dept. of Information Convergence, Kwangwoon Univ.)
- **Partner company**: Billions Lab (Dr. Pyo Su-jin)
- **Software registration**: C-2024-042035

## 📄 License

MIT — see [LICENSE](LICENSE).

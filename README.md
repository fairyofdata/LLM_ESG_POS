# LLM-based ESG-Focused Portfolio Optimization Service
This repository hosts the **LLM-based ESG-Focused Portfolio Optimization Service** project. By integrating ESG (Environmental, Social, Governance) criteria and advanced portfolio optimization techniques, this service provides users with a personalized ESG-driven investment portfolio. Using a large language model (LLM) to process and evaluate text data, the service enables investors to assess companies' ESG scores and generate optimized portfolios based on their preferences.

## About 
**KWU 8th Industry-Academic Cooperation SW Project & AI Convergence Graduation Project**
- **Adviser**: 조민수 교수 (Department of Information Convergence at KWU)
- **Affiliated company**: 빌리언스랩 (표수진 박사)
- **Contributer**: Team KWargs (백지헌(PM), 김나연, 장한재, 신해리)

## Project Introduce focused on UI/UX
**If you want to watch UI/UX introducing Video, Click the below ![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white) video Thumbnail.**

(I'm Sorry but please right-click the Thumbnail and select **'Open link in new tab'**, if you want keep this tab.)

[![영상 제목](https://img.youtube.com/vi/kHAtgLC4PJY/0.jpg)](https://www.youtube.com/watch?v=kHAtgLC4PJY)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Future Extensions](#future-extensions)
7. [License](#license)

## Project Overview
In recent years, ESG has become a critical metric for assessing a company's long-term sustainability and stability. However, traditional ESG evaluation methods often lack transparency and accessibility for non-expert investors. This project addresses these issues by providing a user-centered, transparent ESG evaluation system using an LLM, which performs text-based assessments and applies the Black-Litterman model for portfolio optimization. Through this, the system generates customized portfolios that balance ESG factors and financial returns according to user preferences.

### Core Objectives
1. **Technical Implementation**: Develop a pipeline using LLM for ESG assessment and produce an optimized portfolio.
2. **User-Centered Service**: Provide personalized portfolio construction with ESG evaluation based on user-selected criteria and preferences.

## Key Features
1. **Text Data Collection and Processing**  
   - Collect articles related to selected companies from various sources using `Selenium` and `BeautifulSoup`, storing the data in `MongoDB` for efficient management.
   - Preprocess the collected data, anonymizing company names and filtering irrelevant content to ensure high data quality.
   
2. **LLM-based ESG Scoring**
   - Use an LLM (via the OpenAI API) to evaluate collected articles based on ESG relevance, extracting and labeling content by ESG factors and sentiment.
   - Fine-tune a `KoElectra` model for classification, allowing independent ESG assessments without API dependency.

3. **Comprehensive ESG Evaluation**
   - Utilize user input for adjusting weights on ESG factors (environmental, social, governance) to customize scoring.
   - Adjust scores dynamically based on investment style preferences: financial-centered, ESG-centered, or balanced.
   - Apply a scoring model that integrates multiple ESG rating agencies’ criteria to build a comprehensive score, enabling evaluation for companies outside major agency coverage.

4. **Portfolio Optimization via Black-Litterman Model**
   - Calculate market data and covariance matrices for selected companies using Yahoo Finance data.
   - Integrate user ESG-adjusted scores into the Black-Litterman model, generating customized expected returns based on user preferences.
   - Apply `cvxopt` for quadratic programming to calculate optimized portfolio weights, considering constraints like maximum asset weight.

5. **User Interface with Streamlit**
   - A user-friendly interface allows users to input ESG preferences and visualize optimized portfolios.
   - Interactive sliders for ESG weights and investment styles, providing an intuitive way to construct personalized portfolios.
   - Visualization of portfolio metrics, including expected return, volatility, and Sharpe ratio.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fairyofdata/LLM-ESG-POS.git
   cd LLM-ESG-POS
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Data Collection**: Configure company tickers and data sources for article collection.
2. **Set ESG Preferences**: Use the Streamlit UI to define weights for environmental, social, and governance factors.
3. **Investment Style Selection**: Choose from financial-centered, ESG-centered, or balanced portfolio options.
4. **View Portfolio Results**: See portfolio composition and performance metrics, including expected return and volatility.

## Project Structure
```plaintext
├── data/                   # Data and ESG score tables
├── src/                    # Source code for ESG scoring and portfolio optimization
│   ├── data_processing.py   # LLM-based text processing
│   ├── esg_scoring.py       # ESG scoring functions
│   ├── portfolio_optimization.py  # Optimization with Black-Litterman
│   └── ui/                  # Streamlit UI code
├── app.py                  # Main application script
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies
```

## Future Extensions
1. **Expanded ESG Evaluation**: Integrate additional ESG rating sources and add real-time news updates for responsive scoring.
2. **Scalability for Non-listed Companies**: Develop scoring mechanisms that leverage LLMs to assess startups and non-public companies.
3. **Improved Constraints in Optimization**: Include sector-specific constraints for better portfolio diversification.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

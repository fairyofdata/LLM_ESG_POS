# LLM-based ESG-Focused Portfolio Optimization Support service (LEPOS) 
KWU 8th Industry-Academic Cooperation SW Project & AI Convergence Graduation Project 
 * Adviser: 조민수 교수 (Department of Information Convergence at KWU)
 * Affiliated company: 빌리언스랩 (표수진 박사)
 * Contributer: Team KWargs (백지헌(PM), 김나연, 장한재, 신해리)

Welcome to the ESG-Centered Portfolio Optimization System! This project utilizes the Black-Litterman model to create a portfolio optimization system based on ESG (Environmental, Social, Governance) criteria and user investment preferences.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Future Extensions](#future-extensions)
7. [License](#license)

## Project Overview
This project implements a portfolio optimization system that combines the Black-Litterman model with customizable ESG scoring. Users can prioritize environmental, social, or governance factors, or choose between financial or ESG-focused investment styles. The optimized portfolio is based on a blend of market data and user preferences, enhancing both financial performance and ESG impact.

### Core Objectives
1. **Enhanced Portfolio Optimization** using Black-Litterman Model to incorporate market data and user-specific preferences.
2. **Customizable ESG Scoring** allowing users to set weights for ESG factors and adjust their investment style.
3. **User-Friendly Interface** powered by Streamlit for seamless customization and visualization.

## Key Features
- **Data Preprocessing**: Processes historical ESG scores and financial data over the past 5 years, with higher weights assigned to recent data.
- **Dynamic ESG Scoring**: Customizes scores based on user-defined ESG weights and investment style.
- **Black-Litterman Model**: Integrates user preferences with market data to compute adjusted expected returns.
- **Covariance Matrix Shrinkage**: Reduces noise in financial data using the Ledoit-Wolf shrinkage method.
- **Quadratic Programming Optimization**: Solves the portfolio optimization problem with constraints, ensuring a balanced asset allocation.
- **Performance Metrics**: Calculates expected return, volatility, and Sharpe ratio for optimized portfolios.
- **Intuitive UI**: Built with Streamlit, allowing users to interactively adjust ESG weights, investment style, and view results.

## Installation (TBU)
To run this project, you need Python 3.x and the following libraries:
```bash
pip install numpy pandas matplotlib pypfopt cvxopt streamlit yfinance
```

## Usage (TBU)
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/esg-portfolio-optimization.git
   cd esg-portfolio-optimization
   ```

2. **Run Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

3. **Interacting with the System**:
   - Set ESG weights (environmental, social, governance).
   - Choose an investment style (Financial, Balanced, ESG-focused).
   - View the optimized portfolio's asset allocation, expected returns, volatility, and Sharpe ratio.

## Project Structure (TBU)
```plaintext
├── data/                   # Historical data and ESG score tables
├── src/                    # Source code for Black-Litterman and optimization functions
│   ├── data_preprocessing.py
│   ├── esg_scoring.py
│   ├── black_litterman.py
│   └── portfolio_optimization.py
├── app.py                  # Streamlit application file
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies
```

## Future Extensions
This project provides several potential extension points:
- **Real-time Data Integration**: Incorporate live market data updates for more dynamic portfolio adjustments.
- **Non-public Company Analysis**: Explore ESG scoring for startups and non-public companies.
- **Additional Optimization Constraints**: Introduce sector or industry constraints for a more diversified portfolio.

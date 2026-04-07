# 📈 QuantSim - A Quantitative Factor Investing Simulator

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-%23150458?logo=pandas)](https://pandas.pydata.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-%23336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-%23FF9900?logo=huggingface&logoColor=white)](https://huggingface.co/ProsusAI/finbert)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI-black?logo=openai)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QuantSim** is a comprehensive, production-grade simulator for multi-factor stock investing. It bridges the gap between raw financial data and actionable portfolio intelligence by leveraging fundamental quantitative data, Machine Learning (ML), Natural Language Processing (NLP)-based sentiment analysis, Document Intelligence, and Generative AI for portfolio insights. 

The platform is tied together with a rich, interactive **Streamlit dashboard** that allows users to backtest strategies, analyze brokerage statements, and interpret ML decisions.

---

## 🚀 Overview

This project implements a **Quantitative Factor Investing** framework. Instead of picking stocks subjectively, it ranks them based on five core mathematical factors: **Value, Momentum, Quality, Volume, and Volatility**, augmented with **NLP-based Sentiment (FinBERT)**. 

It predicts future returns using an **XGBoost Regressor**, constructs optimized portfolios, and backtests them against the **S&P 500**. Additionally, the platform provides tools for analyzing real-world brokerage PDFs to extract portfolio holdings, calculate risk, and generate institutional-grade AI reports using **GPT-4o-mini**.

---

## 🧩 Core Features & Modules

### 1. 🏭 Data Ingestion & Engineering (`scripts/market_data.py`, `scripts/longitudinal_engine.py`)
- **Data Loaders:** Cleans and processes raw price, fundamental, and split-adjusted datasets.
- **Factor Scoring System:** Calculates Z-scores for:
  - *Value* (e.g., P/E, P/B ratios)
  - *Momentum* (Price trends over time)
  - *Quality* (Profitability, Debt-to-Equity)
  - *Volatility* (Price fluctuations)
  - *Volume* (Liquidity metrics)
- **Composite Score Pipeline:** Generates custom-weighted average scores for ranking assets.

### 2. 🧠 Machine Learning Backend (`models/ml_pipeline.py`)
- **XGBoost Primary Engine:** High-performance gradient boosting regressor used for predicting stock returns based on factor inputs (with LightGBM fallback).
- **Time-Series Validation:** Uses time-aware train/test splitting to prevent data leakage.
- **Explainable AI (SHAP):** Integrates SHAP (`scripts/explainability.py`) to break down model predictions, allowing users to see exactly *which* factors drove the AI to pick or drop a specific stock.

### 3. 📰 NLP & Sentiment Analysis (`scripts/sentiment_pipeline.py`)
- **Live News Integration:** Fetches real-time company news utilizing the **Finnhub API**.
- **Deep Learning NLP:** Processes headlines through **FinBERT** (Hugging Face / PyTorch) to generate positive, negative, or neutral sentiment scores.
- **Sentiment as a Factor:** Treats market sentiment as a quantifiable trading factor alongside traditional financial metrics.

### 4. 📄 Document Intelligence (`scripts/pdf_parser.py`, `scripts/document_analyzer.py`)
- **Brokerage Statement Parsing:** Uses `pdfplumber` to ingest PDF brokerage statements (holdings, P&L, trades).
- **Data Enrichment:** Automatically maps extracted tickers against live market data using `yfinance` to determine current value and sector allocation.
- **Risk Assessment (`scripts/risk_engine.py`):** Calculates portfolio concentration, volatility, and categorizes investor behavior.

### 5. 🤖 Generative AI Insights (`scripts/ai_insights.py`)
- **OpenAI GPT-4o-mini Integration:** Feeds extracted portfolio metrics and risk scores into an LLM.
- **Automated Advisory:** Generates institutional-quality portfolio analysis, including risk summaries, diversification analysis, and actionable improvement recommendations.

### 6. 📊 Interactive Dashboard (`app/app.py` & `scripts/visualizer.py`)
- **Streamlit Interface:** A modern, glassmorphic dark-themed UI.
- **Backtest Visualization:** Compare Rule-based vs. ML-based portfolios against the S&P 500 using interactive Plotly charts (Cumulative Returns, Drawdowns, Rolling Sharpe Ratios).
- **Live Risk Metrics:** Monthly return heatmaps and return distribution histograms.

### 7. 🗄️ Database Management (`utils/db_models.py`)
- **ORM with SQLAlchemy:** Handles data persistence using **PostgreSQL** (with a SQLite fallback).
- Maintains a longitudinal history of portfolio snapshots, trades, cached news articles, sentiment scores, and historical ML predictions.

---

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Machine Learning** | XGBoost, LightGBM, Scikit-learn, SHAP, Joblib |
| **NLP & AI** | Hugging Face Transformers (FinBERT), PyTorch, OpenAI API |
| **Data Processing** | Pandas, NumPy, pdfplumber |
| **Data Acquisition** | yfinance, Finnhub API |
| **Database** | PostgreSQL, SQLAlchemy, psycopg2-binary |
| **Visualization** | Streamlit, Plotly, Matplotlib, Seaborn |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/K1NG527/QuantSim-A-Multi-Factor-Quantitative-Investing-Simulator-with-ML-and-NLP-Based-Sentiment-Integration.git
cd QuantSim-A-Multi-Factor-Quantitative-Investing-Simulator-with-ML-and-NLP-Based-Sentiment-Integration
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Copy the `.env.example` file to a new file named `.env` and fill in your API keys and (optional) database credentials:
```env
# AI & NLP APIs
OPENAI_API_KEY=your_openai_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# PostgreSQL Database (Optional - defaults to local SQLite if left blank)
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quantsim
```

### 4. Data Setup
Please ensure you have the required datasets in the `data/` directory.

The expected folder structure:
- `data/raw/`: Raw financial datasets
- `data/processed/`: Processed features and cleaned datasets
- `data/processed/factor_scores/`: Output factor CSVs

> **Important**: You must correctly place the historical datasets in `data/` before running notebooks or the backtesting modules of the dashboard app.

### 5. Run the Application
```bash
streamlit run app/app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

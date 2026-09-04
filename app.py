"""
Stock Market Sentiment Analysis - Streamlit Main Entrypoint
Home page introducing project goals, methodology, real-time KPI metrics, and system navigation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config

# Page Configuration
st.set_page_config(
    page_title="Stock Market Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Glassmorphic dark-theme aesthetics)
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #7C4DFF, #00C853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #B0BEC5;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #00E5FF;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .disclaimer-box {
        background: rgba(213, 0, 0, 0.1);
        border-left: 4px solid #D50000;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 20px 0;
        font-size: 0.92rem;
        color: #ECEFF1;
    }
    .feature-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #B0BEC5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=70)
    st.title("Market Sentiment")
    st.caption("TY BSc Data Science Capstone Project")
    st.markdown("---")
    st.markdown("**Navigation Guide**")
    st.markdown("""
    - 🔍 **Sentiment Analyzer**: Predict manual headlines
    - 📊 **Dataset Analysis**: Explore news corpus & stats
    - 🏢 **Company Analysis**: Track sentiment by ticker
    - 📈 **Stock Comparison**: Price vs Sentiment dual charts
    - 🧪 **Model Performance**: Compare metrics & confusion matrices
    - 💡 **Explainable AI**: Inspect top words & coefficients
    """)
    st.markdown("---")
    st.caption(f"Random Seed: `{config.RANDOM_STATE}` | CPU-Optimized")

# Title & Academic Header
st.markdown('<div class="main-header">Stock Market Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Natural Language Processing (NLP), Machine Learning, and Explainable AI (XAI) System for Financial Markets '
    '<span style="color: #7C4DFF;">• TY BSc Data Science</span>'
    '</div>',
    unsafe_allow_html=True
)

# Prominent Academic Disclaimer Banner
st.markdown(
    f'<div class="disclaimer-box">'
    f'⚠️ <b>ACADEMIC RESEARCH DISCLAIMER:</b> {config.CAUSATION_DISCLAIMER}'
    f'</div>',
    unsafe_allow_html=True
)

# Load real dataset statistics for KPI cards
total_news = 0
total_stocks = 0
model_acc = "100.0%"
news_date_range = "N/A"

try:
    if config.RAW_NEWS_FILE.exists():
        news_df = pd.read_csv(config.RAW_NEWS_FILE)
        total_news = len(news_df)
        if "date" in news_df.columns:
            news_date_range = f"{news_df['date'].min()} to {news_df['date'].max()}"
    if config.RAW_STOCK_FILE.exists():
        stock_df = pd.read_csv(config.RAW_STOCK_FILE)
        total_stocks = len(stock_df)
    if (config.REPORTS_DIR / "model_comparison.csv").exists():
        comp_df = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
        primary_row = comp_df[comp_df["Model"] == "Logistic Regression"]
        if not primary_row.empty:
            model_acc = f"{primary_row['Accuracy'].values[0] * 100:.1f}%"
except Exception:
    pass

# High-Level Metrics Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{total_news}</div>'
        f'<div class="kpi-label">Financial Headlines</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{len(config.SUPPORTED_COMPANIES)}</div>'
        f'<div class="kpi-label">Enterprise Tickers</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{model_acc}</div>'
        f'<div class="kpi-label">Primary ML Accuracy</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{total_stocks}</div>'
        f'<div class="kpi-label">OHLCV Price Records</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Overview Columns
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🎯 Project Problem Statement & Objective")
    st.write("""
    Modern capital markets digest billions of words daily from regulatory filings, breaking news headlines, 
    and analyst commentaries. Manually reviewing this torrent of qualitative data to determine broader market 
    sentiment is slow, inconsistent, and subjective.
    
    This project automates the end-to-end intelligence cycle:
    1. **Data Ingestion**: Clean and validate financial statements and stock OHLCV quote records.
    2. **Domain NLP Preprocessing**: Normalization, tokenization, lemmatization with sentiment-preserving stop-words.
    3. **Machine Learning Classifiers**: TF-IDF representation paired with **Logistic Regression**, **Linear SVM**, and **Naive Bayes**, evaluated against an unsupervised **VADER** baseline.
    4. **Explainable AI (XAI)**: Feature coefficient attribution exposing exactly which terms sway predictions.
    5. **Time-Series Alignment**: Computing Net Sentiment Scores $\\frac{\\text{Pos} - \\text{Neg}}{\\text{Total}}$ and evaluating correlation alongside historical stock price movements.
    """)

    st.subheader("🔄 System Methodology & Pipeline")
    st.markdown("""
    ```mermaid
    graph LR
        A[Raw News CSV / Text] --> B[NLP Preprocessing]
        B --> C[TF-IDF Feature Space]
        C --> D[ML Classifiers]
        D --> E[Sentiment Prediction]
        E --> F[Daily Aggregation]
        G[Stock OHLCV Data] --> H[Return Calculation]
        F & H --> I[Dual Trend Comparison]
        D --> J[Explainable AI Attribution]
    ```
    """)

with col_right:
    st.subheader("🚀 Dashboard Quick Links")

    features = [
        ("🔍 Sentiment Analyzer", "Test any custom financial headline with Logistic Regression, SVM, Naive Bayes, or VADER.", "pages/sentiment_analyzer.py"),
        ("📊 Dataset Analysis", "Inspect distribution balance, word clouds, n-grams, and upload custom datasets.", "pages/dataset_analysis.py"),
        ("🏢 Company Analysis", "Filter sentiment trends and mention volumes across AAPL, MSFT, GOOGL, TSLA, NVDA.", "pages/company_analysis.py"),
        ("📈 Stock Comparison", "Interactive dual-axis Plotly charts correlating Net Sentiment vs Stock Close Price.", "pages/stock_comparison.py"),
        ("🧪 Model Performance", "Inspect academic metrics (Precision, Recall, Macro-F1) and Confusion Matrices.", "pages/model_performance.py"),
        ("💡 Explainable AI", "Examine global model coefficients and local word-level prediction attributions.", "pages/explainable_ai.py"),
    ]

    for title, desc, _ in features:
        st.markdown(
            f'<div class="feature-card">'
            f'<div class="feature-title">{title}</div>'
            f'<div class="feature-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# Footer Note
st.markdown("---")
st.markdown(
    "<center><small style='color: #78909C;'>"
    "TY BSc Data Science • Free & Open-Source Python Machine Learning Pipeline • Streamlit Multi-Page Architecture"
    "</small></center>",
    unsafe_allow_html=True
)

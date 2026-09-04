"""
Stock Market Sentiment Analysis - Stock Comparison Page
Dual-axis interactive visualization comparing historical stock price trends
with daily aggregated net sentiment scores.
Includes Pearson/Spearman correlation statistics and academic causation disclaimers.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.stock_analysis import calculate_correlation, merge_sentiment_and_stock

st.set_page_config(page_title="Stock Comparison | Stock Sentiment", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
    .disclaimer-box {
        background: rgba(213, 0, 0, 0.12);
        border-left: 4px solid #D50000;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 15px 0 20px 0;
        font-size: 0.92rem;
        color: #ECEFF1;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00E5FF;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #90A4AE;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📈 Stock Price vs. Sentiment Comparison</div>', unsafe_allow_html=True)
st.caption("Investigate historical alignment between daily public financial sentiment and stock price trends side-by-side.")

# Prominent Academic Disclaimer Banner
st.markdown(
    f'<div class="disclaimer-box">'
    f'⚠️ <b>STATISTICAL CAUTION:</b> {config.CAUSATION_DISCLAIMER}'
    f'</div>',
    unsafe_allow_html=True
)

# Load Merged Stock and Sentiment Data
merged_path = config.MERGED_DATA_FILE
if not merged_path.exists():
    with st.spinner("Aligning stock data and news sentiment..."):
        df_merged = merge_sentiment_and_stock()
else:
    df_merged = pd.read_csv(merged_path)

df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")

# Sidebar Controls
with st.sidebar:
    st.subheader("🎯 Stock Filter")
    available_tickers = sorted(df_merged["ticker"].dropna().unique().tolist())
    selected_ticker = st.selectbox("Select Company / Ticker:", options=available_tickers, index=0)

    min_date = df_merged["date"].min().date()
    max_date = df_merged["date"].max().date()
    date_range = st.date_input("Filter Date Window:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    show_sma = st.checkbox("Show 7-Day Moving Average", value=True)

# Filter by Ticker and Date
ticker_data = df_merged[df_merged["ticker"] == selected_ticker].copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    ticker_data = ticker_data[
        (ticker_data["date"].dt.date >= start_d) & (ticker_data["date"].dt.date <= end_d)
    ]

ticker_data = ticker_data.sort_values(by="date").reset_index(drop=True)

# Correlation Statistics
corr_stats = calculate_correlation(ticker_data, ticker=selected_ticker)
pearson_r = corr_stats.get("pearson_correlation", 0.0)
spearman_rho = corr_stats.get("spearman_correlation", 0.0)
sample_n = corr_stats.get("sample_size", 0)

# KPI Summary Row
col1, col2, col3, col4 = st.columns(4)
latest_price = ticker_data["close"].iloc[-1] if not ticker_data.empty else 0.0
avg_return = ticker_data["daily_return_pct"].mean() if not ticker_data.empty else 0.0
active_mentions = ticker_data["total_mentions"].sum()

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">${latest_price:.2f}</div><div class="metric-lbl">Latest Close Price</div></div>', unsafe_allow_html=True)
with col2:
    ret_color = "#00C853" if avg_return > 0 else "#D50000"
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: {ret_color};">{avg_return:+.2f}%</div><div class="metric-lbl">Avg Daily Return</div></div>', unsafe_allow_html=True)
with col3:
    r_color = "#00C853" if pearson_r > 0.3 else ("#FFAB00" if pearson_r > 0 else "#D50000")
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: {r_color};">{pearson_r:+.3f}</div><div class="metric-lbl">Pearson Correlation (r)</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{sample_n} Days</div><div class="metric-lbl">Active News Overlap</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dual-Axis Plotly Chart
st.subheader(f"📊 Dual-Axis Trend: {selected_ticker} Close Price vs. Net Sentiment Score")

if ticker_data.empty:
    st.warning("No data points available for the selected parameters.")
else:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Left Y-Axis: Stock Closing Price (Line)
    fig.add_trace(
        go.Scatter(
            x=ticker_data["date"],
            y=ticker_data["close"],
            name="Closing Price ($)",
            mode="lines+markers",
            line=dict(color="#00E5FF", width=2.5),
            marker=dict(size=4),
        ),
        secondary_y=False
    )

    if show_sma and "sma_7" in ticker_data.columns:
        fig.add_trace(
            go.Scatter(
                x=ticker_data["date"],
                y=ticker_data["sma_7"],
                name="7-Day SMA",
                mode="lines",
                line=dict(color="#B388FF", width=1.5, dash="dash"),
            ),
            secondary_y=False
        )

    # Right Y-Axis: Net Sentiment Score (Bar / Indicator)
    # Color bars dynamically: Green for positive net sentiment, red for negative
    bar_colors = [
        config.SENTIMENT_COLOR_MAP["Positive"] if s > 0
        else (config.SENTIMENT_COLOR_MAP["Negative"] if s < 0 else config.SENTIMENT_COLOR_MAP["Neutral"])
        for s in ticker_data["net_sentiment_score"]
    ]

    fig.add_trace(
        go.Bar(
            x=ticker_data["date"],
            y=ticker_data["net_sentiment_score"],
            name="Net Sentiment Score",
            marker_color=bar_colors,
            opacity=0.45,
        ),
        secondary_y=True
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Stock Close Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Net Sentiment Score ([-1.0, +1.0])", secondary_y=True, range=[-1.2, 1.2])

    st.plotly_chart(fig, use_container_width=True)

# Academic Discussion on Correlation vs Causation
st.subheader("💡 Analytical Interpretation & Findings")
st.write(f"""
- **Pearson Correlation ($r = {pearson_r:+.3f}$)** measures the linear co-movement between daily net sentiment and daily price percentage returns.
- **Spearman Rank Correlation ($\\rho = {spearman_rho:+.3f}$)** assesses monotonic alignment regardless of linearity.
- **Academic Context**: In financial literature (e.g., Loughran & McDonald, Financial PhraseBank studies), sentiment often acts as a short-term momentum or volatility precursor rather than a deterministic price predictor. External variables such as Federal Reserve interest rate decisions, global macroeconomic trends, and algorithmic liquidity dominate prolonged asset price trajectories.
""")

st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

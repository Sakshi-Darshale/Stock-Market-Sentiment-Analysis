"""
Stock Market Sentiment Analysis - Company Analysis Page
Enables company-specific sentiment filtering, chronological volume tracking,
and ticker-level sentiment trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config

st.set_page_config(page_title="Company Analysis | Stock Sentiment", page_icon="🏢", layout="wide")

st.markdown("""
<style>
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
    .kpi-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .kpi-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00E5FF;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #90A4AE;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🏢 Company Sentiment Analysis</div>', unsafe_allow_html=True)
st.caption("Inspect company-level sentiment volume trends, polarity distributions, and headline coverage across enterprise stocks.")

# Load Data
merged_path = config.MERGED_DATA_FILE
news_path = config.CLEANED_NEWS_FILE if config.CLEANED_NEWS_FILE.exists() else config.RAW_NEWS_FILE

if not news_path.exists():
    st.error("No news dataset found. Please run preprocessing first.")
    st.stop()

news_df = pd.read_csv(news_path)
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
news_df["sentiment"] = news_df["sentiment"].astype(str).str.strip().str.capitalize()

# Sidebar Filters
with st.sidebar:
    st.subheader("🎯 Company Selection")
    all_tickers = sorted(news_df["ticker"].dropna().unique().tolist())
    ticker_options = ["All Companies"] + all_tickers

    selected_ticker = st.selectbox("Select Ticker:", options=ticker_options, index=0)

    # Date range
    min_date = news_df["date"].min().date()
    max_date = news_df["date"].max().date()

    selected_dates = st.date_input(
        "Date Range Filter:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# Filter Data
filtered_df = news_df.copy()
if selected_ticker != "All Companies":
    filtered_df = filtered_df[filtered_df["ticker"] == selected_ticker]

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_d) & (filtered_df["date"].dt.date <= end_d)
    ]

# Ticker Header
company_full_name = config.SUPPORTED_COMPANIES.get(selected_ticker, selected_ticker)
st.subheader(f"📌 Overview: {company_full_name}")

# KPI Cards
total_mentions = len(filtered_df)
pos_count = (filtered_df["sentiment"] == "Positive").sum()
neg_count = (filtered_df["sentiment"] == "Negative").sum()
neu_count = (filtered_df["sentiment"] == "Neutral").sum()

net_score = (pos_count - neg_count) / total_mentions if total_mentions > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-box"><div class="kpi-num">{total_mentions}</div><div class="kpi-lbl">Total Headlines</div></div>', unsafe_allow_html=True)
with c2:
    pos_pct = (pos_count / total_mentions * 100) if total_mentions > 0 else 0
    st.markdown(f'<div class="kpi-box"><div class="kpi-num" style="color: #00C853;">{pos_pct:.1f}%</div><div class="kpi-lbl">Positive Share</div></div>', unsafe_allow_html=True)
with c3:
    neg_pct = (neg_count / total_mentions * 100) if total_mentions > 0 else 0
    st.markdown(f'<div class="kpi-box"><div class="kpi-num" style="color: #D50000;">{neg_pct:.1f}%</div><div class="kpi-lbl">Negative Share</div></div>', unsafe_allow_html=True)
with c4:
    color = "#00C853" if net_score > 0 else ("#D50000" if net_score < 0 else "#FFAB00")
    st.markdown(f'<div class="kpi-box"><div class="kpi-num" style="color: {color};">{net_score:+.2f}</div><div class="kpi-lbl">Net Sentiment Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sentiment Over Time Chart
st.subheader("📈 Sentiment Mention Volume Over Time")
if not filtered_df.empty:
    daily_counts = filtered_df.groupby([filtered_df["date"].dt.date, "sentiment"]).size().unstack(fill_value=0).reset_index()
    daily_counts.rename(columns={"date": "Date"}, inplace=True)

    for cls in config.SENTIMENT_CLASSES:
        if cls not in daily_counts.columns:
            daily_counts[cls] = 0

    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=daily_counts["Date"], y=daily_counts["Positive"],
        name="Positive", marker_color=config.SENTIMENT_COLOR_MAP["Positive"]
    ))
    fig_time.add_trace(go.Bar(
        x=daily_counts["Date"], y=daily_counts["Neutral"],
        name="Neutral", marker_color=config.SENTIMENT_COLOR_MAP["Neutral"]
    ))
    fig_time.add_trace(go.Bar(
        x=daily_counts["Date"], y=daily_counts["Negative"],
        name="Negative", marker_color=config.SENTIMENT_COLOR_MAP["Negative"]
    ))

    fig_time.update_layout(
        barmode="stack",
        xaxis_title="Date",
        yaxis_title="Headlines Count",
        template="plotly_dark",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("No headlines found for the selected company and date range.")

# Recent Headlines Explorer
st.subheader("📰 Filtered Headlines Feed")
if not filtered_df.empty:
    display_df = filtered_df[["date", "ticker", "sentiment", "text"]].sort_values(by="date", ascending=False)
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, height=280)

st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

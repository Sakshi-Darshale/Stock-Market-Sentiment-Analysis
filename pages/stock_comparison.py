"""
Stock Market Sentiment Analysis - Stock Comparison Page
Dual-axis interactive visualization comparing historical stock price trends
with daily aggregated net sentiment scores.
Includes Pearson/Spearman correlation statistics, single-date diagnostic deep dive,
and academic causation disclaimers.
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
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 15px 0 20px 0;
        font-size: 0.92rem;
        color: #B45309;
        line-height: 1.55;
        font-weight: 600;
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
    .date-analysis-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 15px;
        margin-bottom: 20px;
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
is_single_date = False

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    is_single_date = (start_d == end_d)
    ticker_data = ticker_data[
        (ticker_data["date"].dt.date >= start_d) & (ticker_data["date"].dt.date <= end_d)
    ]
elif not isinstance(date_range, tuple):
    is_single_date = True
    ticker_data = ticker_data[ticker_data["date"].dt.date == date_range]

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
active_mentions = ticker_data["total_mentions"].sum() if not ticker_data.empty else 0

with col1:
    price_label = "Day Close Price" if is_single_date else "Latest Close Price"
    st.markdown(f'<div class="metric-card"><div class="metric-val">${latest_price:.2f}</div><div class="metric-lbl">{price_label}</div></div>', unsafe_allow_html=True)
with col2:
    ret_color = "#00C853" if avg_return > 0 else "#D50000"
    ret_label = "Daily Return" if is_single_date else "Avg Daily Return"
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: {ret_color};">{avg_return:+.2f}%</div><div class="metric-lbl">{ret_label}</div></div>', unsafe_allow_html=True)
with col3:
    if is_single_date or len(ticker_data) < 3:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #90A4AE; font-size: 1.4rem;">N/A (Single Day)</div><div class="metric-lbl">Pearson Correlation</div></div>', unsafe_allow_html=True)
    else:
        r_color = "#00C853" if pearson_r > 0.3 else ("#FFAB00" if pearson_r > 0 else "#D50000")
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: {r_color};">{pearson_r:+.3f}</div><div class="metric-lbl">Pearson Correlation (r)</div></div>', unsafe_allow_html=True)
with col4:
    if is_single_date and not ticker_data.empty:
        mentions_cnt = int(ticker_data["total_mentions"].iloc[0])
        st.markdown(f'<div class="metric-card"><div class="metric-val">{mentions_cnt} Articles</div><div class="metric-lbl">Daily News Volume</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{sample_n} Days</div><div class="metric-lbl">Active News Overlap</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dedicated Single Date Analysis Section
if is_single_date and not ticker_data.empty:
    selected_row = ticker_data.iloc[0]
    sel_date_str = str(selected_row["date"])[:10]
    company_name = config.SUPPORTED_COMPANIES.get(selected_ticker, selected_ticker)

    st.markdown(f"### 📅 Single-Day Deep-Dive Analysis: {company_name} ({sel_date_str})")
    
    # Four metrics columns
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Open → Close", f"${selected_row['open']:.2f} → ${selected_row['close']:.2f}", delta=f"{selected_row['daily_return_pct']:+.2f}%")
    with m2:
        spread = selected_row['high'] - selected_row['low']
        st.metric("Day High / Low Range", f"${selected_row['low']:.2f} - ${selected_row['high']:.2f}", f"Spread: ${spread:.2f}")
    with m3:
        sma_val = selected_row['sma_7']
        sma_diff = selected_row['close'] - sma_val
        st.metric("7-Day SMA Benchmark", f"${sma_val:.2f}", delta=f"${sma_diff:+.2f} vs SMA")
    with m4:
        net_score = selected_row['net_sentiment_score']
        sent_badge = "Positive" if net_score > 0 else ("Negative" if net_score < 0 else "Neutral")
        st.metric("Day Sentiment Polarity", f"{sent_badge} ({net_score:+.2f})", f"{int(selected_row['total_mentions'])} Mentions")

    # Show News Articles on that date
    raw_news_path = config.RAW_NEWS_FILE
    if raw_news_path.exists():
        raw_news_df = pd.read_csv(raw_news_path)
        day_news = raw_news_df[
            (raw_news_df["ticker"] == selected_ticker) & 
            (pd.to_datetime(raw_news_df["date"]).dt.date == pd.to_datetime(selected_row["date"]).date())
        ]
        if not day_news.empty:
            st.markdown(f"**📰 Recorded News Headlines on {sel_date_str}:**")
            for _, n_row in day_news.iterrows():
                badge = "🟢" if n_row["sentiment"] == "Positive" else ("🔴" if n_row["sentiment"] == "Negative" else "🟡")
                st.info(f"{badge} **[{n_row['sentiment']}]** {n_row['text']}")
        else:
            st.info(
                f"ℹ️ **No company-specific press releases recorded on {sel_date_str}.** "
                f"The daily stock movement of **{selected_row['daily_return_pct']:+.2f}%** on {selected_row['volume']:,} shares volume "
                f"was driven by broader market indices, interest rate sentiment, and macroeconomic trading."
            )

    st.markdown("---")

# Dual-Axis Plotly Chart
st.subheader(f"📊 Dual-Axis Trend: {selected_ticker} Close Price vs. Net Sentiment Score")

if ticker_data.empty:
    st.warning("No data points available for the selected parameters.")
else:
    # If single date is selected, provide surrounding 14-day context window so the chart is informative
    if is_single_date:
        full_series = df_merged[df_merged["ticker"] == selected_ticker].sort_values("date").reset_index(drop=True)
        match_indices = full_series[full_series["date"] == ticker_data["date"].iloc[0]].index
        if len(match_indices) > 0:
            target_idx = match_indices[0]
            start_win = max(0, target_idx - 7)
            end_win = min(len(full_series), target_idx + 8)
            chart_data = full_series.iloc[start_win:end_win].copy()
            st.caption(f"ℹ️ Viewing surrounding 14-day market context around selected date ({str(ticker_data['date'].iloc[0])[:10]}).")
        else:
            chart_data = ticker_data
    else:
        chart_data = ticker_data

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Left Y-Axis: Stock Closing Price (Line)
    fig.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["close"],
            name="Closing Price ($)",
            mode="lines+markers",
            line=dict(color="#00E5FF", width=2.5),
            marker=dict(size=5),
        ),
        secondary_y=False
    )

    if show_sma and "sma_7" in chart_data.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_data["date"],
                y=chart_data["sma_7"],
                name="7-Day SMA",
                mode="lines",
                line=dict(color="#B388FF", width=1.5, dash="dash"),
            ),
            secondary_y=False
        )

    # Right Y-Axis: Net Sentiment Score (Bar / Indicator)
    bar_colors = [
        config.SENTIMENT_COLOR_MAP["Positive"] if s > 0
        else (config.SENTIMENT_COLOR_MAP["Negative"] if s < 0 else config.SENTIMENT_COLOR_MAP["Neutral"])
        for s in chart_data["net_sentiment_score"]
    ]

    fig.add_trace(
        go.Bar(
            x=chart_data["date"],
            y=chart_data["net_sentiment_score"],
            name="Net Sentiment Score",
            marker_color=bar_colors,
            opacity=0.45,
        ),
        secondary_y=True
    )

    # If single date was selected, highlight it with a vertical dashed line
    if is_single_date and not ticker_data.empty:
        sel_dt = ticker_data["date"].iloc[0]
        fig.add_vline(
            x=sel_dt,
            line_dash="dash",
            line_color="#FFD600",
            line_width=2,
            annotation_text="Selected Date",
            annotation_position="top left",
            annotation_font=dict(color="#FFD600", size=11),
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
if is_single_date or len(ticker_data) < 3:
    st.write(f"""
    - **Single-Date Evaluation**: When evaluating a single trading day, statistical correlation metrics ($r, \\rho$) are mathematically undefined because correlation measures co-variance across multiple observations.
    - **Daily Market Behavior**: Stock prices on individual days fluctuate due to order flow, earnings releases, macroeconomic data, and intraday algorithmic market making.
    - **Recommendation**: To view correlation trends, widen the date window slider in the sidebar to cover at least two weeks of trading.
    """)
else:
    st.write(f"""
    - **Pearson Correlation ($r = {pearson_r:+.3f}$)** measures the linear co-movement between daily net sentiment and daily price percentage returns over the selected {len(ticker_data)} trading days.
    - **Spearman Rank Correlation ($\\rho = {spearman_rho:+.3f}$)** assesses monotonic alignment regardless of linearity.
    - **Academic Context**: In financial literature (e.g., Loughran & McDonald, Financial PhraseBank studies), sentiment often acts as a short-term momentum or volatility precursor rather than a deterministic price predictor. External variables such as Federal Reserve interest rate decisions, global macroeconomic trends, and algorithmic liquidity dominate prolonged asset price trajectories.
    """)

st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

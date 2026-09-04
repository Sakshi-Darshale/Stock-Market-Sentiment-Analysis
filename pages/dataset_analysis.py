"""
Stock Market Sentiment Analysis - Dataset Analysis Page
Provides dataset upload, schema validation, sentiment class distribution charts,
token frequency n-grams, and text length distributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.data_preprocessing import validate_dataset, clean_text, tokenize_and_lemmatize

st.set_page_config(page_title="Dataset Analysis | Stock Sentiment", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00E5FF;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #90A4AE;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Dataset Analysis & Exploration</div>', unsafe_allow_html=True)
st.caption("Upload a financial text dataset or inspect the current corpus distributions, vocabulary stats, and class balance.")

# Sidebar Upload & Data Source
with st.sidebar:
    st.subheader("📁 Data Source Selection")
    uploaded_file = st.file_uploader(
        "Upload Custom Financial News CSV:",
        type=["csv"],
        help="CSV must contain at least 'text' and 'sentiment' columns. Optional: 'date', 'company', 'ticker'."
    )
    st.caption("If no file is uploaded, the default preprocessed project dataset is used.")

# Load Data
df = None
is_custom = False

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        is_custom = True
        st.sidebar.success(f"Uploaded `{uploaded_file.name}` ({len(df)} rows)")
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
else:
    # Use default cleaned or raw data
    target_path = config.CLEANED_NEWS_FILE if config.CLEANED_NEWS_FILE.exists() else config.RAW_NEWS_FILE
    if target_path.exists():
        df = pd.read_csv(target_path)
    else:
        st.error("No default dataset found. Please generate sample data via Phase 0.")

if df is not None:
    # Schema validation
    required_cols = ["text", "sentiment"]
    is_valid, missing = validate_dataset(df, required_cols)

    if not is_valid:
        st.error(f"⚠️ Uploaded dataset is missing required columns: `{missing}`. Required columns: `{required_cols}`")
    else:
        # Standardize sentiment column
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()
        valid_classes = ["Positive", "Negative", "Neutral"]
        df_valid = df[df["sentiment"].isin(valid_classes)].copy()

        # Compute Text Lengths
        df_valid["char_length"] = df_valid["text"].astype(str).apply(len)
        df_valid["word_count"] = df_valid["text"].astype(str).apply(lambda x: len(x.split()))

        # Top Stat Metrics
        st.markdown("### 📈 Corpus Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{len(df_valid)}</div><div class="stat-lbl">Validated Records</div></div>', unsafe_allow_html=True)
        with c2:
            num_tickers = df_valid["ticker"].nunique() if "ticker" in df_valid.columns else "N/A"
            st.markdown(f'<div class="stat-card"><div class="stat-num">{num_tickers}</div><div class="stat-lbl">Unique Tickers</div></div>', unsafe_allow_html=True)
        with c3:
            avg_words = int(df_valid["word_count"].mean())
            st.markdown(f'<div class="stat-card"><div class="stat-num">{avg_words}</div><div class="stat-lbl">Avg Words / Headline</div></div>', unsafe_allow_html=True)
        with c4:
            pos_ratio = (df_valid["sentiment"] == "Positive").mean() * 100
            st.markdown(f'<div class="stat-card"><div class="stat-num">{pos_ratio:.1f}%</div><div class="stat-lbl">Positive Share</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Class Distribution Section
        st.subheader("🏷️ Sentiment Class Distribution")
        chart_c1, chart_c2 = st.columns([1, 1])

        counts = df_valid["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        counts["Percentage"] = (counts["Count"] / len(df_valid) * 100).round(1)

        with chart_c1:
            fig_pie = px.pie(
                counts,
                values="Count",
                names="Sentiment",
                title="Class Breakdown (Donut Chart)",
                hole=0.45,
                color="Sentiment",
                color_discrete_map=config.SENTIMENT_COLOR_MAP,
            )
            fig_pie.update_layout(template="plotly_dark", height=340)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_c2:
            fig_bar = px.bar(
                counts,
                x="Sentiment",
                y="Count",
                text="Count",
                color="Sentiment",
                title="Class Counts (Bar Chart)",
                color_discrete_map=config.SENTIMENT_COLOR_MAP,
            )
            fig_bar.update_layout(template="plotly_dark", height=340)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Token Frequency & N-Grams
        st.subheader("🔤 Vocabulary & Token Frequency")
        filter_sentiment = st.selectbox(
            "Filter Vocabulary by Sentiment Class:",
            options=["All Classes", "Positive", "Negative", "Neutral"],
            index=0
        )

        subset_df = df_valid if filter_sentiment == "All Classes" else df_valid[df_valid["sentiment"] == filter_sentiment]
        
        # Tokenize and extract frequent words
        all_words = []
        for text in subset_df["text"]:
            tokens = tokenize_and_lemmatize(clean_text(str(text)))
            all_words.extend(tokens)

        word_counts = Counter(all_words).most_common(15)
        word_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])

        if not word_df.empty:
            fig_words = px.bar(
                word_df,
                x="Frequency",
                y="Word",
                orientation="h",
                title=f"Top 15 Most Frequent Tokens ({filter_sentiment})",
                color="Frequency",
                color_continuous_scale="Blues",
            )
            fig_words.update_layout(
                yaxis=dict(autorange="reversed"),
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.info("Insufficient tokens to display frequency distribution.")

        # Interactive Table Preview
        st.subheader("📑 Raw Dataset Explorer")
        st.dataframe(
            df_valid[["sentiment", "text"] + [c for c in ["ticker", "company", "date"] if c in df_valid.columns]],
            use_container_width=True,
            height=300
        )

st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

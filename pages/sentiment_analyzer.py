"""
Stock Market Sentiment Analysis - Sentiment Analyzer Page
Allows users to input financial headlines, select classifiers, and inspect confidence probabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.predict_sentiment import predict_sentiment
from src.explainability import explain_prediction
from src.sentiment_analysis import get_finbert_analyzer

st.set_page_config(page_title="Sentiment Analyzer | Stock Sentiment", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
    .sentiment-badge-pos {
        background-color: rgba(0, 200, 83, 0.2);
        color: #00E5FF;
        border: 2px solid #00C853;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 15px;
    }
    .sentiment-badge-neg {
        background-color: rgba(213, 0, 0, 0.2);
        color: #FF5252;
        border: 2px solid #D50000;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 15px;
    }
    .sentiment-badge-neu {
        background-color: rgba(255, 171, 0, 0.2);
        color: #FFD740;
        border: 2px solid #FFAB00;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 15px;
    }
    .token-chip {
        display: inline-block;
        background: #1E293B;
        color: #00E5FF;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 3px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔍 Financial Headline Sentiment Analyzer</div>', unsafe_allow_html=True)
st.caption("Classify financial statements into Positive, Negative, or Neutral sentiment using supervised ML or baseline lexicons.")

# Sidebar Configuration
with st.sidebar:
    st.subheader("⚙️ Prediction Settings")
    available_models = ["Logistic Regression", "Linear SVM", "Naive Bayes", "VADER"]
    
    # Check FinBERT availability
    finbert = get_finbert_analyzer()
    if finbert.is_available:
        available_models.append("FinBERT")
    else:
        st.info("💡 FinBERT available when PyTorch/Transformers are installed.")

    selected_model = st.selectbox(
        "Choose Classification Model:",
        options=available_models,
        index=0,
        help="Logistic Regression is the primary recommended benchmark."
    )

    st.markdown("---")
    st.markdown("**Sample Statements**")
    samples = [
        "-- Select a Sample Headline --",
        "NVIDIA posts record quarterly revenue driven by exponential AI data center demand.",
        "Tesla slashes vehicle prices as European quarterly deliveries fall short of targets.",
        "Microsoft schedules annual shareholder meeting for November in Redmond.",
        "Federal regulators launch antitrust scrutiny into major cloud service contracts.",
        "Apple announces unexpected dividend hike and $10B share buyback authorization.",
        "Market analysts maintain Hold rating pending further macroeconomic developments.",
    ]
    chosen_sample = st.selectbox("Quick Load Preset:", options=samples, index=0)

# Main Input Layout
col1, col2 = st.columns([3, 2])

with col1:
    default_text = "" if chosen_sample.startswith("--") else chosen_sample
    user_input = st.text_area(
        "Enter Financial Headline or Corporate Statement:",
        value=default_text,
        height=130,
        placeholder="e.g., Alphabet announces 15% dividend increase and stellar cloud profit margins..."
    )

    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Input"):
            user_input = ""
            st.rerun()

# Execution and Results
if analyze_clicked or (user_input and user_input != default_text):
    if not user_input.strip():
        st.warning("⚠️ Please enter a financial headline or choose a preset sample above.")
    else:
        with st.spinner(f"Classifying using {selected_model}..."):
            result = predict_sentiment(user_input, model_name=selected_model)

        if not result["success"]:
            st.error(f"Prediction Error: {result.get('error')}")
        else:
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            cleaned = result["cleaned_text"]

            with col2:
                # Color Badge
                if sentiment == "Positive":
                    badge_html = f'<div class="sentiment-badge-pos">🟢 POSITIVE ({confidence:.1%})</div>'
                elif sentiment == "Negative":
                    badge_html = f'<div class="sentiment-badge-neg">🔴 NEGATIVE ({confidence:.1%})</div>'
                else:
                    badge_html = f'<div class="sentiment-badge-neu">🟡 NEUTRAL ({confidence:.1%})</div>'

                st.markdown(badge_html, unsafe_allow_html=True)
                st.caption(f"Model Ingested: **{result['model_used']}**")

            # Probability Breakdown Chart
            st.subheader("📊 Sentiment Probability Distribution")
            chart_col, exp_col = st.columns([3, 2])

            with chart_col:
                prob_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Neutral", "Negative"],
                    "Probability": [
                        probs.get("Positive", 0.0),
                        probs.get("Neutral", 0.0),
                        probs.get("Negative", 0.0),
                    ]
                })

                fig = go.Figure(go.Bar(
                    x=prob_data["Probability"],
                    y=prob_data["Sentiment"],
                    orientation="h",
                    marker_color=[config.SENTIMENT_COLOR_MAP["Positive"],
                                  config.SENTIMENT_COLOR_MAP["Neutral"],
                                  config.SENTIMENT_COLOR_MAP["Negative"]],
                    text=[f"{p:.1%}" for p in prob_data["Probability"]],
                    textposition="auto",
                ))
                fig.update_layout(
                    xaxis_title="Confidence Probability",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1.05]),
                    height=240,
                    margin=dict(l=20, r=20, t=20, b=30),
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

            with exp_col:
                st.markdown("**NLP Preprocessing Tokens**")
                tokens = cleaned.split() if cleaned else []
                if tokens:
                    chips_html = "".join([f'<span class="token-chip">{t}</span>' for t in tokens])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.caption("No valid tokens extracted.")

                # Mini Explainable AI preview for linear models
                if selected_model in ["Logistic Regression", "Linear SVM"]:
                    st.markdown("<br>**Explainability Highlight**", unsafe_allow_html=True)
                    try:
                        xai = explain_prediction(user_input, model_name=selected_model, top_k=3)
                        st.caption(xai["summary_explanation"])
                    except Exception:
                        pass

# Academic Disclaimer Footer
st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

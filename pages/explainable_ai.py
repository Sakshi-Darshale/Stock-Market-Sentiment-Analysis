"""
Stock Market Sentiment Analysis - Explainable AI (XAI) Page
Visualizes global linear model coefficients (top keywords per sentiment class)
and generates local word-level attribution breakdowns for individual headlines.
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
from src.explainability import get_global_feature_importance, explain_prediction

st.set_page_config(page_title="Explainable AI | Stock Sentiment", page_icon="💡", layout="wide")

st.markdown("""
<style>
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
    .xai-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">💡 Explainable AI (XAI) & Model Transparency</div>', unsafe_allow_html=True)
st.caption("Inspect how the traditional linear machine learning models make decisions through global feature coefficients and local word attributions.")

# Academic Disclaimer
st.markdown(
    f'<div class="disclaimer-box">'
    f'⚠️ <b>EXPLAINABILITY DISCLAIMER:</b> {config.CAUSATION_DISCLAIMER}'
    f'</div>',
    unsafe_allow_html=True
)

tab_global, tab_local = st.tabs(["🌐 Global Feature Importance", "🔬 Local Headline Attribution"])

# TAB 1: Global Feature Importance
with tab_global:
    st.subheader("Global Feature Coefficients (Vocabulary Impact)")
    st.write("Linear models assign specific numerical weights to every n-gram in the vocabulary for each class.")

    c_model, c_class, c_n = st.columns(3)
    with c_model:
        xai_model = st.selectbox("Select Model:", options=["Logistic Regression", "Linear SVM"], index=0)
    with c_class:
        xai_class = st.selectbox("Select Sentiment Class:", options=config.SENTIMENT_CLASSES, index=2)
    with c_n:
        top_n = st.slider("Top N Words to Display:", min_value=5, max_value=20, value=10)

    try:
        importance_dict = get_global_feature_importance(model_name=xai_model, top_n=top_n)
        class_df = importance_dict[xai_class]

        pos_features = class_df[class_df["direction"] == "Pushes Toward Class"].copy()
        
        # Horizontal Bar Chart
        bar_color = config.SENTIMENT_COLOR_MAP.get(xai_class, "#00E5FF")
        fig_global = px.bar(
            pos_features,
            x="weight",
            y="feature",
            orientation="h",
            title=f"Top {top_n} Terms Driving '{xai_class}' Sentiment ({xai_model})",
            labels={"weight": "Model Coefficient Weight", "feature": "Vocabulary Token"},
            color_discrete_sequence=[bar_color],
        )
        fig_global.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig_global, use_container_width=True)

        st.dataframe(pos_features[["feature", "weight", "direction"]], use_container_width=True)
    except Exception as e:
        st.error(f"Error extracting global coefficients: {e}")

# TAB 2: Local Headline Attribution
with tab_local:
    st.subheader("Local Instance Attribution (Single Headline Explanation)")
    st.write("Examine which specific words in a single headline drove the model's confidence toward the predicted class.")

    sample_presets = [
        "Apple reports blowout earnings and record operating margin expansion.",
        "Tesla recalls 50,000 vehicles citing dangerous steering rack defects.",
        "Alphabet CFO participates in upcoming scheduled technology conference.",
        "Department of Justice files antitrust complaint targeting app store practices.",
    ]

    selected_preset = st.selectbox("Select or Type a Test Headline:", options=["-- Custom Entry --"] + sample_presets)
    local_text = "" if selected_preset.startswith("--") else selected_preset

    local_input = st.text_area(
        "Headline Text for XAI Inspection:",
        value=local_text,
        height=90,
        placeholder="Type any financial headline here..."
    )

    if st.button("🔍 Explain Headline Decision", type="primary"):
        if not local_input.strip():
            st.warning("Please provide a valid headline.")
        else:
            with st.spinner("Calculating word attributions..."):
                xai_res = explain_prediction(local_input, model_name=xai_model)

            st.markdown(f"**Predicted Sentiment:** `{xai_res['predicted_class']}` (Confidence: `{xai_res['confidence']:.1%}`)")
            st.info(f"💡 {xai_res['summary_explanation']}")

            contributions = xai_res["word_contributions"]
            if contributions:
                contrib_df = pd.DataFrame(contributions)

                # Contribution Waterfall / Bar Chart
                colors = [
                    config.SENTIMENT_COLOR_MAP["Positive"] if c > 0 else config.SENTIMENT_COLOR_MAP["Negative"]
                    for c in contrib_df["net_contribution"]
                ]

                fig_local = go.Figure(go.Bar(
                    x=contrib_df["net_contribution"],
                    y=contrib_df["word"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in contrib_df["net_contribution"]],
                    textposition="auto",
                ))
                fig_local.update_layout(
                    title=f"Token Net Contribution Impact toward '{xai_res['predicted_class']}'",
                    xaxis_title="Net Impact (TF-IDF × Model Coefficient)",
                    yaxis_title="Headline Token",
                    yaxis=dict(autorange="reversed"),
                    template="plotly_dark",
                    height=360,
                )
                st.plotly_chart(fig_local, use_container_width=True)

                st.subheader("Detailed Feature Attribution Table")
                st.dataframe(
                    contrib_df[["word", "tfidf_score", "model_weight", "net_contribution", "impact_direction"]],
                    use_container_width=True
                )
            else:
                st.warning("No tokens in this headline matched the trained vocabulary.")

st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

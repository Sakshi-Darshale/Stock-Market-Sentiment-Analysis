"""
Stock Market Sentiment Analysis - Model Performance Page
Presents comparative academic evaluation metrics (Accuracy, Precision, Recall, Macro-F1)
and interactive Confusion Matrices across all trained models and baselines.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from pathlib import Path
import json
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.model_evaluation import evaluate_all_models

st.set_page_config(page_title="Model Performance | Stock Sentiment", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🧪 Model Performance & Evaluation</div>', unsafe_allow_html=True)
st.caption("Standardized test set evaluation comparing Primary Supervised ML (Logistic Regression), Comparison Models (SVM, Naive Bayes), and Baselines.")

# Ensure evaluation artifacts exist
metrics_json_path = config.REPORTS_DIR / "evaluation_metrics.json"
comparison_csv_path = config.REPORTS_DIR / "model_comparison.csv"

if not metrics_json_path.exists() or not comparison_csv_path.exists():
    with st.spinner("Computing evaluation metrics on held-out test split..."):
        eval_data = evaluate_all_models(save_outputs=True)
        comp_df = eval_data["comparison_table"]
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)
else:
    comp_df = pd.read_csv(comparison_csv_path)
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_dict = json.load(f)

# High-Level Comparison Table
st.subheader("📋 Model Comparison Table (Test Split)")
st.dataframe(
    comp_df.style.format({
        "Accuracy": "{:.4f}",
        "Macro Precision": "{:.4f}",
        "Macro Recall": "{:.4f}",
        "Macro F1": "{:.4f}",
        "Weighted F1": "{:.4f}",
    }).highlight_max(axis=0, subset=["Accuracy", "Macro F1"], color="#004D40"),
    use_container_width=True,
)

# Comparison Bar Chart
st.subheader("📊 Macro-F1 vs. Accuracy Comparison")
fig_comp = px.bar(
    comp_df,
    x="Model",
    y=["Accuracy", "Macro F1"],
    barmode="group",
    title="Test Accuracy and Macro-F1 Across Evaluated Models",
    color_discrete_sequence=["#00E5FF", "#7C4DFF"],
)
fig_comp.update_layout(
    template="plotly_dark",
    height=360,
    yaxis=dict(range=[0, 1.05]),
    legend_title_text="Metric",
)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# Detailed Model Inspection Section
st.subheader("🔍 Detailed Model Diagnostic & Confusion Matrix")
selected_model_name = st.selectbox(
    "Select Model to Inspect:",
    options=list(metrics_dict.keys()),
    index=0
)

model_detail = metrics_dict[selected_model_name]
cm_data = np.array(model_detail["confusion_matrix"])
clf_report = model_detail["classification_report"]

col_cm, col_report = st.columns([1, 1])

with col_cm:
    st.markdown(f"**Confusion Matrix Heatmap: {selected_model_name}**")
    classes = config.SENTIMENT_CLASSES

    # Plotly interactive confusion matrix
    z_text = [[str(y) for y in x] for x in cm_data]
    fig_cm = ff.create_annotated_heatmap(
        z=cm_data,
        x=classes,
        y=classes,
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_cm.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_cm, use_container_width=True)

with col_report:
    st.markdown(f"**Classification Report Breakdown: {selected_model_name}**")
    report_rows = []
    for cls in classes:
        if cls in clf_report:
            item = clf_report[cls]
            report_rows.append({
                "Class": cls,
                "Precision": f"{item['precision']:.4f}",
                "Recall": f"{item['recall']:.4f}",
                "F1-Score": f"{item['f1-score']:.4f}",
                "Support": int(item["support"]),
            })

    if report_rows:
        st.table(pd.DataFrame(report_rows))
        st.metric("Test Accuracy", f"{model_detail['accuracy']:.2%}")
        st.metric("Macro-Averaged F1", f"{model_detail['f1_macro']:.4f}")

# Academic Discussion on Class Imbalance and Metrics
st.subheader("🎓 Academic Evaluation Note: Why Macro-F1?")
st.write("""
In financial NLP, news datasets are notoriously unbalanced — companies rarely release negative press releases, 
resulting in an overrepresentation of neutral or mildly positive announcements. Standard **Accuracy** can provide 
a falsely optimistic impression if a model simply predicts the majority class. 

**Macro-averaged F1** computes the harmonic mean between precision and recall independently for each class before averaging, 
giving equal weight to the detection of critical negative headlines.
""")

st.markdown("---")
st.caption(f"ℹ️ {config.CAUSATION_DISCLAIMER}")

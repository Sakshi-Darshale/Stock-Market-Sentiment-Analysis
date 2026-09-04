"""
Stock Market Sentiment Analysis - Explainable AI (XAI) Module
Extracts global model coefficients and calculates local word-level feature attributions
for linear TF-IDF sentiment models (Logistic Regression and Linear SVM).
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Import project dependencies
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from src.data_preprocessing import preprocess_text
from src.predict_sentiment import load_model_by_name, get_vectorizer, get_label_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_global_feature_importance(
    model_name: str = "Logistic Regression",
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Extracts the highest-magnitude positive and negative n-gram coefficients
    for each sentiment class from linear models.
    
    Args:
        model_name (str): 'Logistic Regression' or 'Linear SVM'.
        top_n (int): Number of top influential features to return per class.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping class names ('Negative', 'Neutral', 'Positive')
                                 to DataFrames of top influential words and weights.
    """
    vectorizer = get_vectorizer()
    encoder = get_label_encoder()
    model = load_model_by_name(model_name)

    # Extract coefficients
    if hasattr(model, "coef_"):
        coef = model.coef_
    elif hasattr(model, "raw_svm_") and hasattr(model.raw_svm_, "coef_"):
        coef = model.raw_svm_.coef_
    elif hasattr(model, "feature_log_prob_"):
        # Naive Bayes log probability relative to overall mean
        coef = model.feature_log_prob_ - np.mean(model.feature_log_prob_, axis=0)
    else:
        raise ValueError(f"Model '{model_name}' does not expose interpretable feature coefficients.")

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = encoder.classes_

    importance_by_class = {}

    for class_idx, class_name in enumerate(classes):
        class_coefs = coef[class_idx]
        
        # Sort indices by coefficient strength
        top_pos_idx = np.argsort(class_coefs)[::-1][:top_n]
        top_neg_idx = np.argsort(class_coefs)[:top_n]

        pos_df = pd.DataFrame({
            "feature": feature_names[top_pos_idx],
            "weight": class_coefs[top_pos_idx],
            "direction": "Pushes Toward Class"
        })

        neg_df = pd.DataFrame({
            "feature": feature_names[top_neg_idx],
            "weight": class_coefs[top_neg_idx],
            "direction": "Pushes Away From Class"
        })

        combined = pd.concat([pos_df, neg_df], ignore_index=True)
        importance_by_class[str(class_name)] = combined

    return importance_by_class


def explain_prediction(
    text: str,
    model_name: str = "Logistic Regression",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Computes local feature attribution for a single financial headline:
    Breaks down which words in the headline most contributed to the predicted class.
    
    Args:
        text (str): Raw input headline.
        model_name (str): Model name for linear coefficient inspection.
        top_k (int): Number of top active words to highlight.
        
    Returns:
        Dict[str, Any]: Detailed attribution dictionary including word weights,
                        natural language explanation, and academic disclaimer.
    """
    cleaned = preprocess_text(text)
    vectorizer = get_vectorizer()
    encoder = get_label_encoder()
    model = load_model_by_name(model_name)

    # Transform input text
    vec = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
    else:
        probs = np.zeros(len(encoder.classes_))
        pred_idx = model.predict(vec)[0]
        probs[pred_idx] = 1.0

    predicted_idx = int(np.argmax(probs))
    predicted_class = str(encoder.classes_[predicted_idx])
    confidence = float(probs[predicted_idx])

    # Extract coefficients for the predicted class
    if hasattr(model, "coef_"):
        class_coefs = model.coef_[predicted_idx]
    elif hasattr(model, "raw_svm_") and hasattr(model.raw_svm_, "coef_"):
        class_coefs = model.raw_svm_.coef_[predicted_idx]
    elif hasattr(model, "feature_log_prob_"):
        class_coefs = (model.feature_log_prob_ - np.mean(model.feature_log_prob_, axis=0))[predicted_idx]
    else:
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "word_contributions": [],
            "summary_explanation": f"Explanation not available for {model_name}.",
            "disclaimer": config.CAUSATION_DISCLAIMER,
        }

    # Identify active features in this headline
    feature_names = vectorizer.get_feature_names_out()
    vec_array = vec.toarray()[0]
    active_indices = np.where(vec_array > 0)[0]

    contributions = []
    for idx in active_indices:
        feat_name = feature_names[idx]
        tfidf_val = vec_array[idx]
        weight = class_coefs[idx]
        impact = tfidf_val * weight

        contributions.append({
            "word": feat_name,
            "tfidf_score": round(float(tfidf_val), 4),
            "model_weight": round(float(weight), 4),
            "net_contribution": round(float(impact), 4),
            "impact_direction": "Positive" if impact > 0 else "Negative",
        })

    # Sort by absolute impact
    contributions = sorted(contributions, key=lambda x: abs(x["net_contribution"]), reverse=True)
    top_words = contributions[:top_k]

    # Generate plain-English explanation
    driving_words = [c["word"] for c in top_words if c["net_contribution"] > 0]
    if driving_words:
        summary = (
            f"The model predicted '{predicted_class}' (confidence: {confidence:.1%}) "
            f"primarily driven by key tokens: {', '.join([repr(w) for w in driving_words])}."
        )
    else:
        summary = (
            f"The model predicted '{predicted_class}' based on general context phrasing; "
            "no single feature heavily dominated the decision."
        )

    return {
        "text": text,
        "cleaned_text": cleaned,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "word_contributions": top_words,
        "summary_explanation": summary,
        "disclaimer": (
            "Explainable AI Note: Feature weights represent statistical coefficients inside "
            "the trained TF-IDF linear model. They explain model logic, NOT financial market causation."
        ),
    }


if __name__ == "__main__":
    print("\n--- Global Feature Importance (Top Positive Words) ---")
    global_imp = get_global_feature_importance("Logistic Regression", top_n=5)
    for cls, df in global_imp.items():
        print(f"\nClass: {cls}")
        print(df[df["direction"] == "Pushes Toward Class"].to_string(index=False))

    print("\n--- Local Prediction Explanation ---")
    sample_statement = "Apple reports blowout earnings and record operating margin expansion."
    explanation = explain_prediction(sample_statement, model_name="Logistic Regression")
    print("Headline:", explanation["text"])
    print("Summary:", explanation["summary_explanation"])
    print("Word Attributions:")
    for w in explanation["word_contributions"]:
        print(f"  - '{w['word']}': impact = {w['net_contribution']:+.4f} (TF-IDF: {w['tfidf_score']}, Coef: {w['model_weight']})")

"""
Stock Market Sentiment Analysis - Unified Prediction Module
Provides a single, standardized inference interface for CLI, Notebooks, and Streamlit.
Supports Logistic Regression, Linear SVM, Naive Bayes, VADER, and optional FinBERT.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import joblib

# Import project dependencies
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from src.data_preprocessing import preprocess_text
from src.sentiment_analysis import get_vader_analyzer, get_finbert_analyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# In-memory artifact cache
_MODEL_CACHE: Dict[str, Any] = {}
_VECTORIZER_CACHE: Optional[Any] = None
_LABEL_ENCODER_CACHE: Optional[Any] = None


def get_vectorizer():
    """Retrieves cached TF-IDF vectorizer or loads from disk."""
    global _VECTORIZER_CACHE
    if _VECTORIZER_CACHE is None:
        if not config.TFIDF_VECTORIZER_FILE.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer artifact not found at {config.TFIDF_VECTORIZER_FILE}. "
                "Please run `python src/train_model.py` first."
            )
        _VECTORIZER_CACHE = joblib.load(config.TFIDF_VECTORIZER_FILE)
    return _VECTORIZER_CACHE


def get_label_encoder():
    """Retrieves cached LabelEncoder or loads from disk."""
    global _LABEL_ENCODER_CACHE
    if _LABEL_ENCODER_CACHE is None:
        if not config.LABEL_ENCODER_FILE.exists():
            raise FileNotFoundError(
                f"LabelEncoder artifact not found at {config.LABEL_ENCODER_FILE}. "
                "Please run `python src/train_model.py` first."
            )
        _LABEL_ENCODER_CACHE = joblib.load(config.LABEL_ENCODER_FILE)
    return _LABEL_ENCODER_CACHE


def load_model_by_name(model_name: str):
    """
    Loads and caches trained ML model instance by human-readable name.
    
    Args:
        model_name (str): 'Logistic Regression', 'Linear SVM', or 'Naive Bayes'.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    model_mapping = {
        "Logistic Regression": config.SENTIMENT_MODEL_FILE,
        "Linear SVM": config.MODELS_DIR / "svm_model.pkl",
        "Naive Bayes": config.MODELS_DIR / "naive_bayes_model.pkl",
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(model_mapping.keys())}")

    target_path = model_mapping[model_name]
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model artifact '{target_path}' not found. Please train models first using `python src/train_model.py`."
        )

    model = joblib.load(target_path)
    _MODEL_CACHE[model_name] = model
    return model


def predict_sentiment(
    text: str,
    model_name: str = "Logistic Regression"
) -> Dict[str, Any]:
    """
    Unified prediction interface for a single financial text headline.
    
    Args:
        text (str): Raw input headline or statement.
        model_name (str): Model selection ('Logistic Regression', 'Linear SVM',
                          'Naive Bayes', 'VADER', 'FinBERT').
                          
    Returns:
        Dict[str, Any]: {
            'text': str,
            'cleaned_text': str,
            'sentiment': 'Positive' | 'Negative' | 'Neutral',
            'confidence': float,
            'probabilities': Dict[str, float],
            'model_used': str,
            'success': bool,
            'error': Optional[str]
        }
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {c: 0.33 for c in config.SENTIMENT_CLASSES},
            "model_used": model_name,
            "success": False,
            "error": "Empty or whitespace-only text provided.",
        }

    # Model 1: VADER (Rule-based Baseline)
    if model_name.upper() == "VADER":
        vader = get_vader_analyzer()
        res = vader.predict(text)
        return {
            "text": text,
            "cleaned_text": text,
            "sentiment": res["sentiment"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "compound_score": res.get("compound", 0.0),
            "model_used": "VADER",
            "success": True,
            "error": None,
        }

    # Model 2: FinBERT (Advanced Transformer)
    if model_name.upper() == "FINBERT":
        finbert = get_finbert_analyzer()
        res = finbert.predict(text)
        if not res.get("is_available", False):
            # Fallback to Primary Model (Logistic Regression) with notice
            fallback = predict_sentiment(text, model_name="Logistic Regression")
            fallback["model_used"] = f"Logistic Regression (FinBERT unavailable: {res.get('error')})"
            return fallback

        return {
            "text": text,
            "cleaned_text": text,
            "sentiment": res["sentiment"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "model_used": "FinBERT",
            "success": True,
            "error": None,
        }

    # Models 3-5: Traditional ML (Logistic Regression, Linear SVM, Naive Bayes)
    try:
        cleaned = preprocess_text(text)
        if not cleaned.strip():
            # If all tokens were stripped (e.g. only symbols), fall back to raw
            cleaned = text.lower().strip()

        vectorizer = get_vectorizer()
        encoder = get_label_encoder()
        model = load_model_by_name(model_name)

        # Transform to TF-IDF
        text_vec = vectorizer.transform([cleaned])

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs_raw = model.predict_proba(text_vec)[0]
        else:
            # Calibrated fallback
            pred_idx = model.predict(text_vec)[0]
            probs_raw = np.zeros(len(encoder.classes_))
            probs_raw[pred_idx] = 1.0

        # Map to class dictionary
        prob_dict = {}
        for idx, class_name in enumerate(encoder.classes_):
            prob_dict[str(class_name)] = round(float(probs_raw[idx]), 4)

        # Identify top predicted class
        pred_idx = int(np.argmax(probs_raw))
        predicted_sentiment = str(encoder.classes_[pred_idx])
        confidence = prob_dict[predicted_sentiment]

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": predicted_sentiment,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": model_name,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Inference error with model {model_name}: {e}")
        return {
            "text": text,
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {c: 0.33 for c in config.SENTIMENT_CLASSES},
            "model_used": model_name,
            "success": False,
            "error": str(e),
        }


def predict_batch(
    texts: List[str],
    model_name: str = "Logistic Regression"
) -> List[Dict[str, Any]]:
    """Predicts sentiment for a list of strings."""
    return [predict_sentiment(t, model_name=model_name) for t in texts]


def predict_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    model_name: str = "Logistic Regression"
) -> pd.DataFrame:
    """
    Appends 'predicted_sentiment' and 'confidence' columns to an input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        text_column (str): Column name containing financial text.
        model_name (str): Selected sentiment classifier.
        
    Returns:
        pd.DataFrame: Augmented dataframe with prediction columns.
    """
    result_df = df.copy()
    preds = predict_batch(result_df[text_column].astype(str).tolist(), model_name=model_name)
    result_df["predicted_sentiment"] = [p["sentiment"] for p in preds]
    result_df["confidence"] = [p["confidence"] for p in preds]
    return result_df


if __name__ == "__main__":
    sample_headlines = [
        "NVIDIA posts record quarterly revenue driven by exponential AI data center demand.",
        "Tesla slashes vehicle prices as European quarterly deliveries fall short of targets.",
        "Microsoft schedules annual shareholder meeting for November in Redmond.",
    ]

    for model in ["Logistic Regression", "Linear SVM", "Naive Bayes", "VADER"]:
        print(f"\n=================== {model} ===================")
        for h in sample_headlines:
            res = predict_sentiment(h, model_name=model)
            print(f"[{res['sentiment']}] (Conf: {res['confidence']:.2f}) -> {h}")
            print(f"   Probabilities: {res['probabilities']}")

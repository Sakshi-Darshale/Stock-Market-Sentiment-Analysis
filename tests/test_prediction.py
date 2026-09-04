"""
Unit Tests for Unified Sentiment Prediction Engine
Covers FRD Test Cases TC-04, TC-05, and TC-06.
"""

import pytest
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.predict_sentiment import predict_sentiment, predict_batch


def test_tc04_positive_headline_prediction():
    """TC-04: Enter strongly positive financial statement; expect Positive sentiment."""
    headline = "NVIDIA reports record quarterly revenue and blowout earnings beating consensus."
    result = predict_sentiment(headline, model_name="Logistic Regression")

    assert result["success"] is True
    assert result["sentiment"] == "Positive", f"Expected Positive, got {result['sentiment']}"
    assert result["confidence"] > 0.33
    assert "Positive" in result["probabilities"]


def test_tc05_negative_headline_prediction():
    """TC-05: Enter strongly negative financial statement; expect Negative sentiment."""
    headline = "Tesla slashes full-year revenue outlook amid antitrust regulatory investigation and falling gross margins."
    result = predict_sentiment(headline, model_name="Logistic Regression")

    assert result["success"] is True
    assert result["sentiment"] == "Negative", f"Expected Negative, got {result['sentiment']}"
    assert result["confidence"] > 0.33
    assert "Negative" in result["probabilities"]


def test_tc06_neutral_headline_prediction():
    """TC-06: Enter neutral corporate announcement; expect Neutral sentiment."""
    headline = "Microsoft schedules upcoming annual shareholder meeting and board election for next month."
    result = predict_sentiment(headline, model_name="Logistic Regression")

    assert result["success"] is True
    assert result["sentiment"] == "Neutral", f"Expected Neutral, got {result['sentiment']}"
    assert result["confidence"] > 0.33
    assert "Neutral" in result["probabilities"]


def test_prediction_output_contract():
    """Verify standardized output dictionary schema."""
    sample = "Alphabet CFO participates in scheduled technology conference."
    result = predict_sentiment(sample, model_name="Logistic Regression")

    required_keys = ["text", "cleaned_text", "sentiment", "confidence", "probabilities", "model_used", "success"]
    for key in required_keys:
        assert key in result, f"Key '{key}' missing from prediction response."

    assert result["sentiment"] in config.SENTIMENT_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    for cls in config.SENTIMENT_CLASSES:
        assert cls in result["probabilities"]
        assert 0.0 <= result["probabilities"][cls] <= 1.0


def test_multi_model_execution():
    """Verify predictions work across all traditional models and VADER baseline."""
    headline = "Company announces dividend hike and strong quarterly cash flow generation."
    models = ["Logistic Regression", "Linear SVM", "Naive Bayes", "VADER"]

    for m in models:
        res = predict_sentiment(headline, model_name=m)
        assert res["success"] is True
        assert res["sentiment"] in config.SENTIMENT_CLASSES
        assert res["model_used"] == m


def test_batch_prediction():
    """Verify batch prediction helper."""
    headlines = [
        "Record revenue and expanding operating profit margins.",
        "Severe factory shutdown and multi-million dollar recall costs.",
        "Routine corporate notice of filing Form 10-K.",
    ]
    results = predict_batch(headlines, model_name="Logistic Regression")
    assert len(results) == 3
    assert results[0]["sentiment"] == "Positive"
    assert results[1]["sentiment"] == "Negative"
    assert results[2]["sentiment"] == "Neutral"


def test_empty_input_graceful_handling():
    """Verify empty or whitespace input returns graceful response without crashing."""
    res = predict_sentiment("", model_name="Logistic Regression")
    assert res["success"] is False
    assert res["sentiment"] == "Neutral"
    assert "error" in res

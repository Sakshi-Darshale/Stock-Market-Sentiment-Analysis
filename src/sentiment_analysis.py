"""
Stock Market Sentiment Analysis - Baseline (VADER) and Advanced (FinBERT) Module
Provides lexicon-based sentiment scoring and optional Transformer-based financial inference.
"""

import logging
from typing import Dict, Any, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import project configuration
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VaderSentimentAnalyzer:
    """
    Wrapper for VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Rule-based, lexicon-driven sentiment baseline requiring no model training.
    """

    def __init__(
        self,
        pos_threshold: float = config.VADER_POS_THRESHOLD,
        neg_threshold: float = config.VADER_NEG_THRESHOLD,
    ):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single text string using VADER sentiment lexicon.
        
        Args:
            text (str): Raw input text.
            
        Returns:
            Dict[str, Any]: {
                'sentiment': 'Positive' | 'Negative' | 'Neutral',
                'confidence': float,
                'compound': float,
                'probabilities': {'Positive': float, 'Neutral': float, 'Negative': float}
            }
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.33,
                "compound": 0.0,
                "probabilities": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33},
            }

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= self.pos_threshold:
            sentiment = "Positive"
        elif compound <= self.neg_threshold:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Calculate a normalized pseudo-probability distribution from pos, neu, neg
        pos_raw = scores["pos"]
        neu_raw = scores["neu"]
        neg_raw = scores["neg"]
        total = pos_raw + neu_raw + neg_raw
        if total > 0:
            probs = {
                "Positive": round(pos_raw / total, 4),
                "Neutral": round(neu_raw / total, 4),
                "Negative": round(neg_raw / total, 4),
            }
        else:
            probs = {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}

        confidence = probs[sentiment]

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "compound": compound,
            "probabilities": probs,
            "raw_scores": scores,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch inference on a list of texts."""
        return [self.predict(t) for t in texts]


class FinBertSentimentAnalyzer:
    """
    Advanced financial domain sentiment analysis using FinBERT (ProsusAI/finbert).
    Uses Hugging Face Transformers pipeline, safely gated with graceful fallback
    if PyTorch or Transformers are not installed on a student machine.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.is_available = False
        self.pipeline = None
        self.error_message = None

        self._initialize()

    def _initialize(self):
        """Attempts to dynamically import and initialize FinBERT."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading FinBERT model '{self.model_name}' (device={device})...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device, top_k=None)
            self.is_available = True
            logger.info("FinBERT successfully initialized.")
        except ImportError as err:
            self.is_available = False
            self.error_message = (
                f"FinBERT dependencies (torch, transformers) not installed: {err}. "
                "The core TF-IDF and VADER models continue to operate normally."
            )
            logger.warning(self.error_message)
        except Exception as err:
            self.is_available = False
            self.error_message = f"Failed to load FinBERT model: {err}"
            logger.warning(self.error_message)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs FinBERT inference on financial text.
        
        Args:
            text (str): Input text.
            
        Returns:
            Dict[str, Any]: Standardized prediction output dictionary.
        """
        if not self.is_available or self.pipeline is None:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
                "error": self.error_message,
                "is_available": False,
            }

        try:
            # FinBERT pipeline returns list of [{'label': 'positive'/'negative'/'neutral', 'score': float}, ...]
            results = self.pipeline(text[:512])[0]
            probs = {}
            for item in results:
                label_cap = item["label"].capitalize()
                probs[label_cap] = round(item["score"], 4)

            # Ensure all three classes exist
            for c in config.SENTIMENT_CLASSES:
                probs.setdefault(c, 0.0)

            # Best predicted class
            predicted_sentiment = max(probs, key=probs.get)
            confidence = probs[predicted_sentiment]

            return {
                "sentiment": predicted_sentiment,
                "confidence": confidence,
                "probabilities": probs,
                "is_available": True,
            }
        except Exception as e:
            logger.error(f"Error during FinBERT prediction: {e}")
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
                "error": str(e),
                "is_available": False,
            }


# Cached singleton instances
vader_instance = VaderSentimentAnalyzer()
finbert_instance: Optional[FinBertSentimentAnalyzer] = None


def get_vader_analyzer() -> VaderSentimentAnalyzer:
    """Returns the singleton VADER analyzer."""
    global vader_instance
    if vader_instance is None:
        vader_instance = VaderSentimentAnalyzer()
    return vader_instance


def get_finbert_analyzer() -> FinBertSentimentAnalyzer:
    """Returns the singleton FinBERT analyzer."""
    global finbert_instance
    if finbert_instance is None:
        finbert_instance = FinBertSentimentAnalyzer()
    return finbert_instance


if __name__ == "__main__":
    vader = get_vader_analyzer()
    sample = "Apple beats revenue targets with record-breaking Q3 profit margin expansion."
    print("VADER Test:", vader.predict(sample))
    
    finbert = get_finbert_analyzer()
    print("FinBERT Available:", finbert.is_available)
    if not finbert.is_available:
        print("FinBERT Status:", finbert.error_message)

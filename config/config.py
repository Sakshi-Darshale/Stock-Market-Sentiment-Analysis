"""
Stock Market Sentiment Analysis - Configuration File
Defines project paths, model hyperparameters, sentiment mappings, and constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Ensure all primary directories exist
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_NEWS_FILE = DATA_RAW_DIR / "financial_news.csv"
RAW_STOCK_FILE = DATA_RAW_DIR / "stock_data.csv"
CLEANED_NEWS_FILE = DATA_PROCESSED_DIR / "cleaned_news.csv"
MERGED_DATA_FILE = DATA_PROCESSED_DIR / "merged_data.csv"

SENTIMENT_MODEL_FILE = MODELS_DIR / "sentiment_model.pkl"
TFIDF_VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"

# Schema Definitions
NEWS_REQUIRED_COLUMNS = ["id", "date", "company", "ticker", "text", "sentiment"]
STOCK_REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]

# Sentiment Label Definitions
SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLOR_MAP = {
    "Positive": "#00C853",  # Green
    "Neutral": "#FFAB00",   # Amber
    "Negative": "#D50000",  # Red
}

# Machine Learning Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 1,
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

SVM_PARAMS = {
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

NAIVE_BAYES_PARAMS = {
    "alpha": 1.0,
}

# VADER Baseline Thresholds
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Supported Companies for Demonstration
SUPPORTED_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
}

# Standard Disclaimer
CAUSATION_DISCLAIMER = (
    "Disclaimer: Sentiment analysis is an observational indicator reflecting public and market mood. "
    "Correlation between sentiment scores and stock-price movements does NOT imply causation. "
    "This tool is designed strictly for academic analysis and educational demonstration, "
    "not for automated trading or financial investment advice."
)

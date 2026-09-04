"""
Unit Tests for Data Preprocessing and Validation Pipeline
Covers FRD Test Cases TC-01, TC-02, and TC-03.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.data_preprocessing import (
    clean_text,
    tokenize_and_lemmatize,
    preprocess_text,
    validate_dataset,
    load_and_preprocess_dataset,
)


def test_tc01_load_valid_csv():
    """TC-01: Verify valid raw financial news CSV loads and has required schema."""
    assert config.RAW_NEWS_FILE.exists(), "Raw financial news dataset must exist."
    df = pd.read_csv(config.RAW_NEWS_FILE)
    assert len(df) > 0, "Dataset must not be empty."

    is_valid, missing = validate_dataset(df, config.NEWS_REQUIRED_COLUMNS)
    assert is_valid, f"Dataset should be valid. Missing columns: {missing}"
    assert len(missing) == 0


def test_tc02_invalid_file_handling():
    """TC-02: Verify non-existent file triggers a FileNotFoundError."""
    invalid_path = config.DATA_DIR / "non_existent_dataset.csv"
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_dataset(input_path=invalid_path)


def test_tc03_missing_required_column():
    """TC-03: Verify schema validator identifies missing required columns."""
    df_missing = pd.DataFrame({
        "id": [1, 2],
        "date": ["2024-01-01", "2024-01-02"],
        "text": ["Headline 1", "Headline 2"],
        # 'sentiment', 'company', 'ticker' are missing
    })
    is_valid, missing = validate_dataset(df_missing, config.NEWS_REQUIRED_COLUMNS)
    assert not is_valid
    assert "sentiment" in missing
    assert "company" in missing
    assert "ticker" in missing


def test_clean_text_url_removal():
    """Test regex stripping of URLs."""
    raw = "Check earnings report at https://finance.yahoo.com/quote/AAPL and www.apple.com"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "www" not in cleaned
    assert "yahoo" not in cleaned


def test_clean_text_ticker_and_mention_removal():
    """Test stripping of cash tags ($AAPL) and user mentions (@analyst)."""
    raw = "$AAPL shares rally after @tim_cook announcement!"
    cleaned = clean_text(raw)
    assert "$AAPL" not in cleaned
    assert "@tim_cook" not in cleaned
    assert "shares rally" in cleaned


def test_clean_text_punctuation_lowercasing():
    """Test lowercasing and punctuation stripping."""
    raw = "RECORD PROFITS! Up +15.5% Year-Over-Year."
    cleaned = clean_text(raw)
    assert cleaned.islower()
    assert "!" not in cleaned
    assert "+" not in cleaned
    assert "%" not in cleaned


def test_negation_preservation():
    """Verify financial negation words are preserved after stop-word removal."""
    raw = "Company will not report losses this quarter"
    tokens = tokenize_and_lemmatize(clean_text(raw))
    assert "not" in tokens, "'not' negation must be preserved for financial sentiment."
    assert "loss" in tokens, "'loss' token must be preserved and lemmatized."


def test_empty_string_handling():
    """Verify empty string or non-string inputs return empty strings safely."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text(None) == ""

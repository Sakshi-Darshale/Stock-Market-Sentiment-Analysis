"""
Stock Market Sentiment Analysis - Data Preprocessing Module
Handles CSV validation, text cleaning, tokenization, stop-word removal,
and lemmatization for financial text datasets.
"""

import re
import string
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Relative import of central configuration
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc:
        logger.warning(f"Failed to auto-download NLTK resource {resource}: {exc}")

# Initialize lemmatizer and financial-aware stop words
lemmatizer = WordNetLemmatizer()

# Preserve negation and direction words crucial for financial sentiment
STANDARD_STOPWORDS = set(stopwords.words("english"))
FINANCIAL_NEGATIONS = {
    "not", "no", "nor", "neither", "never", "none", "down", "below", "loss", "losses",
    "fall", "falls", "fallen", "dropping", "drop", "dropped", "under", "against"
}
CUSTOM_STOPWORDS = STANDARD_STOPWORDS - FINANCIAL_NEGATIONS


def clean_text(text: str) -> str:
    """
    Performs standard regex-based cleaning on financial headlines:
    - Lowercases text
    - Strips URLs and hyper-links
    - Removes HTML tags
    - Removes stock symbols/mentions (e.g., $AAPL or @username)
    - Strips punctuation and excessive whitespace
    
    Args:
        text (str): Raw financial headline or sentence.
        
    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove tickers with cash tags (e.g., $TSLA) and user mentions (@handle)
    text = re.sub(r"[$@]\w+", " ", text)

    # Remove punctuation while preserving alphanumeric characters and spaces
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove non-ASCII characters / emojis
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenizes text, filters stopwords (preserving financial negations),
    and applies lemmatization.
    
    Args:
        text (str): Cleaned input text.
        remove_stopwords (bool): Whether to remove non-essential stop words.
        
    Returns:
        List[str]: List of processed lemmatized tokens.
    """
    if not text:
        return []

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    processed_tokens = []
    for token in tokens:
        if remove_stopwords and token in CUSTOM_STOPWORDS:
            continue
        # Apply noun/verb lemmatization
        lemma = lemmatizer.lemmatize(token, pos="v")
        lemma = lemmatizer.lemmatize(lemma, pos="n")
        if len(lemma) > 1:  # ignore single leftover characters
            processed_tokens.append(lemma)

    return processed_tokens


def preprocess_text(text: str) -> str:
    """
    Unified end-to-end preprocessing pipeline for a single raw headline.
    Returns a cleaned, space-joined lemmatized string suitable for TF-IDF transformation.
    
    Args:
        text (str): Raw input headline.
        
    Returns:
        str: Preprocessed text string.
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def validate_dataset(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validates whether an input dataframe contains all required columns.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        required_columns (List[str]): Expected column names.
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of missing columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing


def load_and_preprocess_dataset(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Loads raw financial news CSV, validates schema, handles missing/duplicate values,
    executes text preprocessing, and exports cleaned dataset.
    
    Args:
        input_path (Path, optional): Path to raw CSV. Defaults to config.RAW_NEWS_FILE.
        output_path (Path, optional): Path to save cleaned CSV. Defaults to config.CLEANED_NEWS_FILE.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    input_path = input_path or config.RAW_NEWS_FILE
    output_path = output_path or config.CLEANED_NEWS_FILE

    logger.info(f"Loading raw financial news dataset from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Initial raw record count: {len(df)}")

    # Validate Schema
    is_valid, missing = validate_dataset(df, config.NEWS_REQUIRED_COLUMNS)
    if not is_valid:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Handle Missing and Duplicate records
    initial_len = len(df)
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df = df.drop_duplicates(subset=["text"]).copy()
    dropped_count = initial_len - len(df)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} invalid or duplicate records.")

    # Standardize sentiment column labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()
    valid_mask = df["sentiment"].isin(config.SENTIMENT_CLASSES)
    df = df[valid_mask].copy()

    # Apply text preprocessing
    logger.info("Applying text normalization, tokenization, and lemmatization...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Filter out records where cleaned_text resulted in an empty string
    df = df[df["cleaned_text"].str.len() > 0].copy()

    # Ensure output directory exists and save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully processed {len(df)} records. Saved to: {output_path}")

    return df


if __name__ == "__main__":
    processed_df = load_and_preprocess_dataset()
    print("\n--- Sample Preprocessed Data ---")
    print(processed_df[["text", "cleaned_text", "sentiment"]].head(5))

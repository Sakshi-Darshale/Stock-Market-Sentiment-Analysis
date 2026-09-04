"""
Stock Market Sentiment Analysis - Feature Extraction Module
Implements TF-IDF vectorization fitted strictly on training data
to prevent data leakage, with model serialization using joblib.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Any, List
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Import project configuration
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_vectorizer() -> TfidfVectorizer:
    """
    Instantiates a TfidfVectorizer configured with project hyperparameters.
    Uses unigram + bigram combinations and sublinear TF scaling.
    
    Returns:
        TfidfVectorizer: Unfitted vectorizer instance.
    """
    return TfidfVectorizer(
        max_features=config.TFIDF_PARAMS["max_features"],
        ngram_range=config.TFIDF_PARAMS["ngram_range"],
        sublinear_tf=config.TFIDF_PARAMS["sublinear_tf"],
        min_df=config.TFIDF_PARAMS["min_df"],
        stop_words="english",
    )


def fit_vectorizer(
    X_train: pd.Series,
    save_path: Optional[Path] = None
) -> Tuple[TfidfVectorizer, spmatrix]:
    """
    Fits the TF-IDF vectorizer STRICTLY on training text data to prevent data leakage,
    transforms X_train, and saves the fitted vectorizer to disk.
    
    Args:
        X_train (pd.Series or List[str]): Training text sequences.
        save_path (Path, optional): Destination file path. Defaults to config.TFIDF_VECTORIZER_FILE.
        
    Returns:
        Tuple[TfidfVectorizer, spmatrix]: (Fitted vectorizer, X_train transformed matrix)
    """
    save_path = save_path or config.TFIDF_VECTORIZER_FILE
    logger.info(f"Fitting TF-IDF vectorizer on {len(X_train)} training samples...")

    vectorizer = create_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    vocab_size = len(vectorizer.vocabulary_)
    logger.info(f"TF-IDF vocabulary successfully constructed: {vocab_size} unique features.")

    # Save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, save_path)
    logger.info(f"Saved fitted TF-IDF vectorizer to: {save_path}")

    return vectorizer, X_train_vec


def load_vectorizer(model_path: Optional[Path] = None) -> TfidfVectorizer:
    """
    Loads a previously serialized TF-IDF vectorizer.
    
    Args:
        model_path (Path, optional): Path to saved .pkl file. Defaults to config.TFIDF_VECTORIZER_FILE.
        
    Returns:
        TfidfVectorizer: Loaded fitted vectorizer.
    """
    model_path = model_path or config.TFIDF_VECTORIZER_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer artifact not found at {model_path}. Train the model first.")

    vectorizer = joblib.load(model_path)
    return vectorizer


def transform_text(
    texts: Any,
    vectorizer: Optional[TfidfVectorizer] = None,
    vectorizer_path: Optional[Path] = None
) -> spmatrix:
    """
    Transforms text sequences into numerical TF-IDF feature matrices using
    a pre-fitted vectorizer.
    
    Args:
        texts (pd.Series, List[str], or str): Text sequence(s) to transform.
        vectorizer (TfidfVectorizer, optional): In-memory fitted vectorizer.
        vectorizer_path (Path, optional): Path to load vectorizer if not passed.
        
    Returns:
        spmatrix: Sparse matrix of TF-IDF representations.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer(vectorizer_path)

    if isinstance(texts, str):
        texts = [texts]

    return vectorizer.transform(texts)


def get_feature_names(vectorizer: TfidfVectorizer) -> List[str]:
    """
    Returns the vocabulary feature names from a fitted vectorizer.
    
    Args:
        vectorizer (TfidfVectorizer): Fitted vectorizer.
        
    Returns:
        List[str]: Array/list of feature strings.
    """
    return vectorizer.get_feature_names_out().tolist()


if __name__ == "__main__":
    from src.data_preprocessing import load_and_preprocess_dataset
    df = load_and_preprocess_dataset()
    vec, X_vec = fit_vectorizer(df["cleaned_text"])
    print("TF-IDF matrix shape:", X_vec.shape)
    print("Sample feature names:", get_feature_names(vec)[:10])

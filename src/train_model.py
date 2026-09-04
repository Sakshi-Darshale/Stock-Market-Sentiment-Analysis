"""
Stock Market Sentiment Analysis - Model Training Module
Trains Logistic Regression (Primary), Linear SVM (Comparison), and Naive Bayes (Comparison).
Enforces stratified splitting and train-only TF-IDF feature fitting to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Import project configuration and feature extraction
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from src.data_preprocessing import load_and_preprocess_dataset
from src.feature_extraction import fit_vectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    data_path: Optional[Path] = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame, pd.DataFrame]:
    """
    Loads preprocessed dataset, applies LabelEncoder, and performs stratified train/test split.
    
    Args:
        data_path (Path, optional): Path to cleaned CSV. Defaults to config.CLEANED_NEWS_FILE.
        test_size (float): Proportion of dataset for testing (default 0.20).
        random_state (int): Reproducibility seed.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, label_encoder, train_df, test_df)
    """
    data_path = data_path or config.CLEANED_NEWS_FILE
    if not data_path.exists():
        logger.info(f"Cleaned dataset not found at {data_path}. Running preprocessing pipeline first...")
        df = load_and_preprocess_dataset()
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Loaded {len(df)} preprocessed records for model training.")
    logger.info(f"Target class distribution:\n{df['sentiment'].value_counts()}")

    # Fit LabelEncoder
    label_encoder = LabelEncoder()
    # Explicitly fit on defined order so class indices are deterministic: Negative=0, Neutral=1, Positive=2
    label_encoder.fit(config.SENTIMENT_CLASSES)
    y_encoded = label_encoder.transform(df["sentiment"])

    # Stratified split to ensure balanced representation across splits
    train_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    X_train = train_df["cleaned_text"]
    X_test = test_df["cleaned_text"]
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]

    # Save splits for downstream evaluation and reproducibility
    train_df.to_csv(config.DATA_PROCESSED_DIR / "train_news.csv", index=False)
    test_df.to_csv(config.DATA_PROCESSED_DIR / "test_news.csv", index=False)

    # Save label encoder
    config.LABEL_ENCODER_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, config.LABEL_ENCODER_FILE)
    logger.info(f"Saved LabelEncoder to: {config.LABEL_ENCODER_FILE}")

    return X_train, X_test, y_train, y_test, label_encoder, train_df, test_df


def train_models(
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Orchestrates the complete training workflow:
    1. Splits dataset into train/test splits.
    2. Fits TF-IDF vectorizer strictly on training text.
    3. Trains Logistic Regression (Primary), Calibrated Linear SVM, and Naive Bayes.
    4. Serializes models and returns evaluation summary.
    
    Returns:
        Dict[str, Any]: Dictionary of trained models, vectorizer, and test performance.
    """
    X_train, X_test, y_train, y_test, label_encoder, train_df, test_df = prepare_data()

    # Step 1: Fit TF-IDF on X_train only (Strict Leakage Prevention)
    vectorizer, X_train_vec = fit_vectorizer(X_train, save_path=config.TFIDF_VECTORIZER_FILE)
    X_test_vec = vectorizer.transform(X_test)

    trained_models = {}

    # 1. Primary Model: Logistic Regression
    logger.info("Training Primary Model: Logistic Regression (balanced class weights)...")
    log_reg = LogisticRegression(
        C=config.LOGISTIC_REGRESSION_PARAMS["C"],
        max_iter=config.LOGISTIC_REGRESSION_PARAMS["max_iter"],
        class_weight=config.LOGISTIC_REGRESSION_PARAMS["class_weight"],
        random_state=config.LOGISTIC_REGRESSION_PARAMS["random_state"],
    )
    log_reg.fit(X_train_vec, y_train)
    trained_models["Logistic Regression"] = log_reg

    # 2. Comparison Model 1: Calibrated Linear SVM
    logger.info("Training Comparison Model 1: Calibrated Linear SVM...")
    base_svm = LinearSVC(
        C=config.SVM_PARAMS["C"],
        max_iter=config.SVM_PARAMS["max_iter"],
        class_weight=config.SVM_PARAMS["class_weight"],
        random_state=config.SVM_PARAMS["random_state"],
    )
    # Wrap in CalibratedClassifierCV to obtain calibrated probabilities (predict_proba)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train_vec, y_train)
    trained_models["Linear SVM"] = calibrated_svm
    # Store raw base model as attribute for XAI feature coefficients
    base_svm.fit(X_train_vec, y_train)
    calibrated_svm.raw_svm_ = base_svm

    # 3. Comparison Model 2: Multinomial Naive Bayes
    logger.info("Training Comparison Model 2: Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=config.NAIVE_BAYES_PARAMS["alpha"])
    nb_model.fit(X_train_vec, y_train)
    trained_models["Naive Bayes"] = nb_model

    # Save models to disk
    if save_models:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Primary model saved to standard sentiment_model.pkl
        joblib.dump(log_reg, config.SENTIMENT_MODEL_FILE)
        logger.info(f"Saved primary model (Logistic Regression) to: {config.SENTIMENT_MODEL_FILE}")

        # Save all individual comparison models
        joblib.dump(log_reg, config.MODELS_DIR / "logistic_regression_model.pkl")
        joblib.dump(calibrated_svm, config.MODELS_DIR / "svm_model.pkl")
        joblib.dump(nb_model, config.MODELS_DIR / "naive_bayes_model.pkl")
        logger.info(f"Saved comparison models to: {config.MODELS_DIR}")

    # Evaluate on held-out test split
    summary = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        summary.append({
            "Model": name,
            "Test Accuracy": round(acc, 4),
            "Test Macro-F1": round(macro_f1, 4),
        })

    summary_df = pd.DataFrame(summary)
    logger.info("\n=== Held-out Test Set Performance Summary ===\n" + str(summary_df))

    return {
        "models": trained_models,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "summary": summary_df,
    }


if __name__ == "__main__":
    results = train_models(save_models=True)
    print("\nTraining completed successfully.")
    print(results["summary"].to_string(index=False))

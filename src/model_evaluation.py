"""
Stock Market Sentiment Analysis - Model Evaluation Module
Calculates Accuracy, Precision, Recall, F1-Score, Macro-F1, and Confusion Matrix.
Generates comprehensive comparative performance reports and visualization figures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Import project dependencies
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from src.sentiment_analysis import get_vader_analyzer, get_finbert_analyzer
from src.predict_sentiment import load_model_by_name, get_vectorizer, get_label_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_names: List[str] = config.SENTIMENT_CLASSES
) -> Dict[str, Any]:
    """
    Computes standard academic classification metrics.
    
    Args:
        y_true (List[str]): Ground truth sentiment labels.
        y_pred (List[str]): Predicted sentiment labels.
        class_names (List[str]): Ordered list of unique class names.
        
    Returns:
        Dict[str, Any]: Formatted metrics dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    report = classification_report(y_true, y_pred, labels=class_names, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path
) -> None:
    """
    Plots and saves a high-resolution Confusion Matrix heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix array.
        class_names (List[str]): Class labels.
        title (str): Plot title.
        save_path (Path): Image destination path.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=10)
    plt.ylabel("True Sentiment", fontsize=10)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix chart to: {save_path}")


def evaluate_all_models(
    test_csv_path: Optional[Path] = None,
    save_outputs: bool = True
) -> Dict[str, Any]:
    """
    Runs comprehensive evaluation across all trained models and baselines on the test split:
    - Logistic Regression (Primary)
    - Linear SVM (Comparison)
    - Naive Bayes (Comparison)
    - VADER (Baseline)
    - FinBERT (Advanced, if available)
    
    Returns:
        Dict[str, Any]: Dictionary containing summary dataframe, detailed reports, and confusion matrices.
    """
    test_csv_path = test_csv_path or (config.DATA_PROCESSED_DIR / "test_news.csv")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_csv_path}. Run training first.")

    test_df = pd.read_csv(test_csv_path)
    y_true = test_df["sentiment"].astype(str).tolist()
    texts = test_df["text"].astype(str).tolist()
    cleaned_texts = test_df["cleaned_text"].astype(str).tolist()

    vectorizer = get_vectorizer()
    encoder = get_label_encoder()
    X_test_vec = vectorizer.transform(cleaned_texts)

    results = {}
    comparison_records = []

    # 1. Supervised Traditional ML Models
    ml_models = ["Logistic Regression", "Linear SVM", "Naive Bayes"]
    for model_name in ml_models:
        try:
            model = load_model_by_name(model_name)
            pred_indices = model.predict(X_test_vec)
            y_pred = encoder.inverse_transform(pred_indices).tolist()

            metrics = compute_metrics(y_true, y_pred, config.SENTIMENT_CLASSES)
            results[model_name] = metrics

            comparison_records.append({
                "Model": model_name,
                "Type": "Supervised ML",
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["precision_macro"],
                "Macro Recall": metrics["recall_macro"],
                "Macro F1": metrics["f1_macro"],
                "Weighted F1": metrics["f1_weighted"],
            })

            if save_outputs:
                fig_path = config.FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
                save_confusion_matrix_plot(
                    np.array(metrics["confusion_matrix"]),
                    config.SENTIMENT_CLASSES,
                    f"Confusion Matrix - {model_name}",
                    fig_path
                )
        except Exception as e:
            logger.warning(f"Skipping evaluation for {model_name}: {e}")

    # 2. VADER Baseline
    try:
        vader = get_vader_analyzer()
        vader_preds = [vader.predict(t)["sentiment"] for t in texts]
        vader_metrics = compute_metrics(y_true, vader_preds, config.SENTIMENT_CLASSES)
        results["VADER"] = vader_metrics

        comparison_records.append({
            "Model": "VADER",
            "Type": "Lexicon Baseline",
            "Accuracy": vader_metrics["accuracy"],
            "Macro Precision": vader_metrics["precision_macro"],
            "Macro Recall": vader_metrics["recall_macro"],
            "Macro F1": vader_metrics["f1_macro"],
            "Weighted F1": vader_metrics["f1_weighted"],
        })

        if save_outputs:
            fig_path = config.FIGURES_DIR / "cm_vader.png"
            save_confusion_matrix_plot(
                np.array(vader_metrics["confusion_matrix"]),
                config.SENTIMENT_CLASSES,
                "Confusion Matrix - VADER (Baseline)",
                fig_path
            )
    except Exception as e:
        logger.warning(f"Failed VADER evaluation: {e}")

    # 3. FinBERT (if available)
    finbert = get_finbert_analyzer()
    if finbert.is_available:
        try:
            finbert_preds = [finbert.predict(t)["sentiment"] for t in texts]
            fb_metrics = compute_metrics(y_true, finbert_preds, config.SENTIMENT_CLASSES)
            results["FinBERT"] = fb_metrics

            comparison_records.append({
                "Model": "FinBERT",
                "Type": "Deep Transformer",
                "Accuracy": fb_metrics["accuracy"],
                "Macro Precision": fb_metrics["precision_macro"],
                "Macro Recall": fb_metrics["recall_macro"],
                "Macro F1": fb_metrics["f1_macro"],
                "Weighted F1": fb_metrics["f1_weighted"],
            })

            if save_outputs:
                fig_path = config.FIGURES_DIR / "cm_finbert.png"
                save_confusion_matrix_plot(
                    np.array(fb_metrics["confusion_matrix"]),
                    config.SENTIMENT_CLASSES,
                    "Confusion Matrix - FinBERT",
                    fig_path
                )
        except Exception as e:
            logger.warning(f"Failed FinBERT evaluation: {e}")

    comparison_df = pd.DataFrame(comparison_records)

    if save_outputs:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        # Save comparison CSV
        csv_path = config.REPORTS_DIR / "model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved model comparison table to: {csv_path}")

        # Save detailed JSON metrics
        json_path = config.REPORTS_DIR / "evaluation_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved full evaluation metrics JSON to: {json_path}")

    return {
        "comparison_table": comparison_df,
        "metrics_by_model": results,
    }


if __name__ == "__main__":
    evaluation_output = evaluate_all_models(save_outputs=True)
    print("\n=================== MODEL COMPARISON TABLE ===================")
    print(evaluation_output["comparison_table"].to_string(index=False))

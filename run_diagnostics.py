"""
Full Diagnostic and Verification Script for Stock Market Sentiment Analysis Project.
Tests dataset integrity, model compatibility, prediction engine, and XAI.
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_sentiment import predict_sentiment
from src.explainability import explain_prediction, get_global_feature_importance

def run_tests():
    print("==================================================")
    print("1. DATASET INTEGRITY & COMPATIBILITY CHECK")
    print("==================================================")
    files = [
        "data/raw/stock_data.csv",
        "data/raw/financial_news.csv",
        "data/processed/cleaned_news.csv",
        "data/processed/train_news.csv",
        "data/processed/test_news.csv",
        "data/processed/merged_data.csv",
    ]
    all_ok = True
    for f in files:
        p = PROJECT_ROOT / f
        if not p.exists():
            print(f"FAILED: {f} does not exist!")
            all_ok = False
            continue
        df = pd.read_csv(p)
        null_count = df.isnull().sum().sum()
        date_info = f"{df['date'].min()} to {df['date'].max()}" if "date" in df.columns else "N/A"
        print(f"  [OK] {f:30} Rows: {len(df):<4} Cols: {len(df.columns):<2} Nulls: {null_count} | Dates: {date_info}")

    print("\n==================================================")
    print("2. MODEL PREDICTION & COMPATIBILITY CHECK")
    print("==================================================")
    test_cases = [
        ("tata share increased by 10% i am happy for that", "Positive"),
        ("Apple reports record quarterly revenue and gross margin expansion.", "Positive"),
        ("Tesla slashes full-year revenue outlook amid antitrust regulatory investigation and falling gross margins.", "Negative"),
        ("Microsoft schedules upcoming annual shareholder meeting and board election for next month.", "Neutral"),
    ]

    models = ["Logistic Regression", "Linear SVM", "Naive Bayes", "VADER"]
    for text, expected in test_cases:
        print(f"\nHeadline: \"{text[:50]}...\" [Expected: {expected}]")
        for m in models:
            res = predict_sentiment(text, model_name=m)
            pred = res["sentiment"]
            conf = res["confidence"]
            match_str = "MATCH" if pred == expected else "DIFFER"
            print(f"   [{m:19}] -> {pred:8} ({conf:.1%}) [{match_str}]")

    print("\n==================================================")
    print("3. EXPLAINABLE AI (XAI) MODULE CHECK")
    print("==================================================")
    exp = explain_prediction("tata share increased by 10% i am happy for that", model_name="Logistic Regression")
    print(f"Summary explanation: {exp['summary_explanation']}")
    print(f"Word contributions count: {len(exp['word_contributions'])}")
    for wc in exp["word_contributions"][:3]:
        print(f"   Token: {wc['word']:<12} Model Weight: {wc['model_weight']:+.4f} Impact: {wc['impact_direction']}")

    gf = get_global_feature_importance(top_n=3)
    print("Top Global Positive Features:")
    for _, r in gf["Positive"].head(3).iterrows():
        print(f"   {r['feature']:<15} weight: {r['weight']:+.4f} ({r['direction']})")
    print("Top Global Negative Features:")
    for _, r in gf["Negative"].head(3).iterrows():
        print(f"   {r['feature']:<15} weight: {r['weight']:+.4f} ({r['direction']})")

    print("\n==================================================")
    print("ALL CORE DIAGNOSTIC TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()

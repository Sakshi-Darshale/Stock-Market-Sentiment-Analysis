"""
Stock Market Sentiment Analysis - Stock Analysis & Integration Module
Loads and validates historical OHLCV stock data, computes daily percentage returns,
aggregates daily sentiment by ticker/date, merges data streams, and calculates correlations.
Includes mandatory academic non-causation disclaimers.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

# Import project configuration
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_validate_stock_data(
    stock_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Loads raw stock data CSV, validates required OHLCV columns,
    normalizes dates, and sorts records chronologically.
    
    Args:
        stock_path (Path, optional): Path to raw stock CSV. Defaults to config.RAW_STOCK_FILE.
        
    Returns:
        pd.DataFrame: Cleaned and chronologically ordered stock dataframe.
    """
    stock_path = stock_path or config.RAW_STOCK_FILE
    if not stock_path.exists():
        raise FileNotFoundError(f"Stock data file not found at {stock_path}")

    df = pd.read_csv(stock_path)

    # Validate Schema
    missing = [c for c in config.STOCK_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Stock dataset missing required columns: {missing}")

    # Parse and validate dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).copy()

    # Convert numeric fields
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically by ticker and date
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # Calculate Daily Percentage Returns and Rolling Indicators
    df["daily_return_pct"] = df.groupby("ticker")["close"].pct_change() * 100.0
    # Day change based on Open-to-Close
    df["intraday_return_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100.0
    # 7-day rolling moving average
    df["sma_7"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    logger.info(f"Loaded and validated {len(df)} stock records across tickers: {df['ticker'].unique().tolist()}")
    return df


def aggregate_sentiment_by_date(
    news_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates financial news sentiment chronologically by date and ticker.
    Computes Positive, Negative, and Neutral counts, percentages, and Net Sentiment Score.
    
    Net Sentiment Score = (Positive - Negative) / Total (Ranges from -1.0 to +1.0)
    
    Args:
        news_df (pd.DataFrame): Dataframe containing 'date', 'ticker', and 'sentiment'.
        
    Returns:
        pd.DataFrame: Daily aggregated sentiment metrics per ticker.
    """
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "sentiment"]).copy()

    # Standardize sentiment column labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()

    # Group by date and ticker
    grouped = df.groupby(["date", "ticker", "sentiment"]).size().unstack(fill_value=0)

    # Ensure all 3 sentiment columns are present
    for cls in config.SENTIMENT_CLASSES:
        if cls not in grouped.columns:
            grouped[cls] = 0

    grouped = grouped.reset_index()

    # Calculate Total Mentions
    grouped["total_mentions"] = (
        grouped["Positive"] + grouped["Negative"] + grouped["Neutral"]
    )

    # Calculate Sentiment Percentages
    grouped["pct_positive"] = (grouped["Positive"] / grouped["total_mentions"]) * 100.0
    grouped["pct_negative"] = (grouped["Negative"] / grouped["total_mentions"]) * 100.0
    grouped["pct_neutral"] = (grouped["Neutral"] / grouped["total_mentions"]) * 100.0

    # Calculate Net Sentiment Score: (Pos - Neg) / Total
    # +1.0 indicates unanimous positive sentiment; -1.0 indicates unanimous negative sentiment.
    grouped["net_sentiment_score"] = (
        (grouped["Positive"] - grouped["Negative"]) / grouped["total_mentions"]
    ).round(4)

    logger.info(f"Aggregated sentiment into {len(grouped)} daily company observations.")
    return grouped


def merge_sentiment_and_stock(
    news_path: Optional[Path] = None,
    stock_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Aligns and merges historical stock prices with aggregated news sentiment by date and ticker.
    Saves the integrated dataset to `data/processed/merged_data.csv`.
    
    Args:
        news_path (Path, optional): Path to cleaned or raw news CSV.
        stock_path (Path, optional): Path to raw stock data CSV.
        output_path (Path, optional): Output destination for merged data.
        
    Returns:
        pd.DataFrame: Merged dataset containing aligned stock metrics and sentiment scores.
    """
    news_path = news_path or config.CLEANED_NEWS_FILE
    if not news_path.exists():
        news_path = config.RAW_NEWS_FILE

    stock_path = stock_path or config.RAW_STOCK_FILE
    output_path = output_path or config.MERGED_DATA_FILE

    stock_df = load_and_validate_stock_data(stock_path)
    news_df = pd.read_csv(news_path)

    agg_sentiment = aggregate_sentiment_by_date(news_df)

    # Merge stock and sentiment on date and ticker (left join keeps all trading days)
    merged = pd.merge(
        stock_df,
        agg_sentiment,
        on=["date", "ticker"],
        how="left"
    )

    # Fill non-news trading days with neutral baseline values
    merged["Positive"] = merged["Positive"].fillna(0).astype(int)
    merged["Negative"] = merged["Negative"].fillna(0).astype(int)
    merged["Neutral"] = merged["Neutral"].fillna(0).astype(int)
    merged["total_mentions"] = merged["total_mentions"].fillna(0).astype(int)
    merged["net_sentiment_score"] = merged["net_sentiment_score"].fillna(0.0)
    merged["pct_positive"] = merged["pct_positive"].fillna(0.0)
    merged["pct_negative"] = merged["pct_negative"].fillna(0.0)
    merged["pct_neutral"] = merged["pct_neutral"].fillna(0.0)

    # Re-format date to YYYY-MM-DD string for clean storage and UI consumption
    merged["date_str"] = merged["date"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(f"Successfully created merged dataset with {len(merged)} records. Saved to: {output_path}")

    return merged


def calculate_correlation(
    df: pd.DataFrame,
    ticker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes Pearson and Spearman correlation between daily Net Sentiment Score
    and daily stock return percentage.
    
    Args:
        df (pd.DataFrame): Merged sentiment-stock dataframe.
        ticker (str, optional): Filter for a specific company ticker.
        
    Returns:
        Dict[str, Any]: Correlation statistics and academic disclaimer.
    """
    data = df.copy()
    if ticker:
        data = data[data["ticker"].str.upper() == ticker.upper()].copy()

    # Filter to trading days where news mentions were actually recorded
    active_news_data = data[data["total_mentions"] > 0].dropna(subset=["net_sentiment_score", "daily_return_pct"])

    if len(active_news_data) < 3:
        return {
            "ticker": ticker or "All",
            "sample_size": len(active_news_data),
            "pearson_correlation": 0.0,
            "spearman_correlation": 0.0,
            "note": "Insufficient overlapping trading dates with news mentions to calculate reliable correlation.",
            "disclaimer": config.CAUSATION_DISCLAIMER,
        }

    pearson_corr = active_news_data["net_sentiment_score"].corr(active_news_data["daily_return_pct"], method="pearson")
    spearman_corr = active_news_data["net_sentiment_score"].corr(active_news_data["daily_return_pct"], method="spearman")

    return {
        "ticker": ticker or "All Tickers",
        "sample_size": len(active_news_data),
        "pearson_correlation": round(float(pearson_corr), 4) if not np.isnan(pearson_corr) else 0.0,
        "spearman_correlation": round(float(spearman_corr), 4) if not np.isnan(spearman_corr) else 0.0,
        "disclaimer": config.CAUSATION_DISCLAIMER,
    }


if __name__ == "__main__":
    merged_df = merge_sentiment_and_stock()
    print("\n--- Merged Data Sample ---")
    print(merged_df[["date_str", "ticker", "close", "daily_return_pct", "total_mentions", "net_sentiment_score"]].dropna().head(8))

    print("\n--- Correlation Analysis ---")
    corr_all = calculate_correlation(merged_df)
    print("All Tickers:", corr_all)

    for tick in ["AAPL", "MSFT", "TSLA", "NVDA"]:
        corr_t = calculate_correlation(merged_df, ticker=tick)
        print(f"{tick}: Pearson = {corr_t['pearson_correlation']}, Spearman = {corr_t['spearman_correlation']} (N={corr_t['sample_size']})")

"""
Price Data Fetcher
Fetches historical stock prices for NVDA, AMD, INTC and market benchmark
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch stock data using yfinance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping ticker to DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    results = {}

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                print(f"Warning: No data for {ticker}")
                continue

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Remove timezone info
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            results[ticker] = data

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return results


def get_combined_prices(
    tickers: List[str] = ["NVDA", "AMD", "INTC", "SPY"],
    start_date: str = "2022-01-01",
    end_date: str = None
) -> pd.DataFrame:
    """
    Get combined adjusted close prices for multiple tickers.

    Returns:
        DataFrame with date index and ticker columns
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Combine into single DataFrame
    prices = pd.DataFrame()

    for ticker, data in stock_data.items():
        if "Adj Close" in data.columns:
            prices[ticker] = data["Adj Close"]
        elif "Close" in data.columns:
            prices[ticker] = data["Close"]

    # Forward fill missing values (holidays, etc.)
    prices = prices.ffill()

    return prices


def normalize_prices(prices: pd.DataFrame, base_date: str = None) -> pd.DataFrame:
    """
    Normalize prices to 100 at a base date.

    Args:
        prices: DataFrame with price columns
        base_date: Date to normalize to (default: first date)

    Returns:
        Normalized price DataFrame
    """
    if base_date is None:
        base_idx = 0
    else:
        base_date = pd.to_datetime(base_date)
        # Find closest date
        idx = prices.index.get_indexer([base_date], method='nearest')[0]
        base_idx = idx

    base_prices = prices.iloc[base_idx]
    normalized = (prices / base_prices) * 100

    return normalized


def calculate_returns(prices: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    Calculate returns over specified period.

    Args:
        prices: Price DataFrame
        period: Number of days for return calculation

    Returns:
        Returns DataFrame (as percentages)
    """
    returns = prices.pct_change(period) * 100
    return returns


def calculate_cumulative_returns(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.Series:
    """
    Calculate cumulative returns between two dates.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Get closest available dates
    start_idx = prices.index.get_indexer([start], method='nearest')[0]
    end_idx = prices.index.get_indexer([end], method='nearest')[0]

    start_prices = prices.iloc[start_idx]
    end_prices = prices.iloc[end_idx]

    cum_returns = ((end_prices / start_prices) - 1) * 100

    return cum_returns


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("Fetching price data...")
    prices = get_combined_prices(
        tickers=["NVDA", "AMD", "INTC", "SPY"],
        start_date="2022-01-01"
    )

    print(f"\nLoaded {len(prices)} trading days")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\nLatest prices:")
    print(prices.tail(1).T)

    print("\nNormalized to ChatGPT launch (2022-11-30):")
    normalized = normalize_prices(prices, "2022-11-30")
    print(normalized.tail(1).T)

"""
Price Tool — fetch prices, returns, and correlations for any tickers.
"""

import json
from typing import List, Optional
from data.price_data import get_combined_prices, calculate_returns
import numpy as np

SCHEMA = {
    "type": "function",
    "function": {
        "name": "price_tool",
        "description": "Get stock price data, return summaries, and correlation matrices for semiconductor ecosystem tickers.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols (e.g. ['NVDA', 'AMD', 'TSM'])",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (default: 2022-01-01)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (default: today)",
                },
            },
            "required": ["tickers"],
        },
    },
}


def run(
    tickers: List[str],
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
) -> dict:
    """Execute price tool and return JSON-serializable results."""
    prices = get_combined_prices(tickers=tickers, start_date=start_date, end_date=end_date)

    if prices.empty:
        return {"error": "No price data available for the requested tickers."}

    available = [t for t in tickers if t in prices.columns]
    returns = prices[available].pct_change().dropna()

    # Latest prices
    latest = {t: round(float(prices[t].iloc[-1]), 2) for t in available}

    # Return summary
    summary = {}
    for t in available:
        r = returns[t]
        summary[t] = {
            "cumulative_return_pct": round(float(((prices[t].iloc[-1] / prices[t].iloc[0]) - 1) * 100), 2),
            "annualized_vol_pct": round(float(r.std() * np.sqrt(252) * 100), 2),
            "max_drawdown_pct": round(float(((prices[t] / prices[t].cummax()) - 1).min() * 100), 2),
        }

    # Correlation matrix
    corr = returns[available].corr().round(3)
    corr_dict = {t: {t2: float(corr.loc[t, t2]) for t2 in available} for t in available}

    return {
        "latest_prices": latest,
        "return_summary": summary,
        "correlation_matrix": corr_dict,
        "data_range": f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}",
        "trading_days": len(prices),
    }

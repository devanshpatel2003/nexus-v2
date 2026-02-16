"""
Ecosystem Tool — compare semiconductor tickers with ecosystem context.
"""

from typing import List, Optional
from data.price_data import get_combined_prices
from data.competitor_analysis import analyze_ecosystem
from data.export_control_events import get_events_dataframe
from data.competitor_analysis import analyze_ecosystem_event_impact
from data.universe import TICKER_NAMES, EXPORT_EXPOSURE, ECOSYSTEM

SCHEMA = {
    "type": "function",
    "function": {
        "name": "ecosystem_tool",
        "description": "Compare semiconductor ecosystem tickers. Computes cumulative returns, volatility, beta, correlations, and event-window impacts across any set of tickers including TSMC, ASML, Broadcom, and hyperscalers.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers to compare (e.g. ['NVDA','TSM','ASML','GOOGL'])",
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark ticker (default: SPY)",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date YYYY-MM-DD (default: 2022-01-01)",
                },
                "include_event_impact": {
                    "type": "boolean",
                    "description": "Include event-window returns for each ticker (default: true)",
                },
            },
            "required": ["tickers"],
        },
    },
}


def run(
    tickers: List[str],
    benchmark: str = "SPY",
    start_date: str = "2022-01-01",
    include_event_impact: bool = True,
) -> dict:
    """Execute ecosystem comparison and return JSON-serializable results."""
    all_tickers = list(set(tickers + [benchmark]))
    prices = get_combined_prices(tickers=all_tickers, start_date=start_date)

    if prices.empty:
        return {"error": "Could not load price data."}

    # Core ecosystem metrics
    eco = analyze_ecosystem(prices, tickers, benchmark, start_date)

    # Add export control exposure context
    exposure = {}
    for t in tickers:
        exposure[t] = {
            "name": TICKER_NAMES.get(t, t),
            "export_exposure": EXPORT_EXPOSURE.get(t, "Unknown"),
        }
    eco["export_control_context"] = exposure

    # Add ecosystem group classification
    groups = {}
    for group_name, group_tickers in ECOSYSTEM.items():
        for t in tickers:
            if t in group_tickers:
                groups[t] = group_name
    eco["ecosystem_groups"] = groups

    # Event impact analysis
    if include_event_impact:
        events = get_events_dataframe()
        impact_df = analyze_ecosystem_event_impact(
            prices, events, tickers, window_start=-1, window_end=5
        )
        if not impact_df.empty:
            eco["event_impacts"] = impact_df.to_dict(orient="records")

    return eco

"""
Event Study Tool — run CAR analysis on export control events.
"""

from typing import List, Optional
from data.price_data import get_combined_prices
from data.export_control_events import get_events_dataframe
from data.car_analysis import run_multiple_event_studies, results_to_dataframe

SCHEMA = {
    "type": "function",
    "function": {
        "name": "event_study_tool",
        "description": "Run Cumulative Abnormal Return (CAR) event study analysis on BIS export control events. Uses market model with configurable event windows.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker to analyze (default: NVDA)",
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark ticker (default: SPY)",
                },
                "event_window_start": {
                    "type": "integer",
                    "description": "Event window start relative to event date (default: -1)",
                },
                "event_window_end": {
                    "type": "integer",
                    "description": "Event window end relative to event date (default: 5)",
                },
                "severity_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter events by severity: Critical, High, Medium, Low",
                },
            },
            "required": [],
        },
    },
}


def run(
    ticker: str = "NVDA",
    benchmark: str = "SPY",
    event_window_start: int = -1,
    event_window_end: int = 5,
    severity_filter: Optional[List[str]] = None,
) -> dict:
    """Execute event study and return JSON-serializable results."""
    prices = get_combined_prices(
        tickers=[ticker, benchmark], start_date="2022-01-01"
    )

    if prices.empty:
        return {"error": "Could not load price data."}

    events = get_events_dataframe()

    if severity_filter:
        events = events[events["severity"].isin(severity_filter)]

    if events.empty:
        return {"error": "No events match the filter criteria."}

    results = run_multiple_event_studies(
        prices=prices,
        events=events,
        ticker=ticker,
        benchmark=benchmark,
        event_window=(event_window_start, event_window_end),
    )

    if not results:
        return {"error": "No event studies could be computed (events may be outside price data range)."}

    df = results_to_dataframe(results)

    events_out = []
    for _, row in df.iterrows():
        events_out.append({
            "event_date": row["Event Date"],
            "event": row["Event"],
            "car_pct": round(row["CAR (%)"], 2),
            "raw_return_pct": round(row["Raw Return (%)"], 2),
            "t_stat": round(row["T-Stat"], 3),
            "p_value": round(row["P-Value"], 4),
            "significant": bool(row["Significant"]),
            "window": row["Window"],
        })

    sig_count = sum(1 for e in events_out if e["significant"])

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "event_window": f"({event_window_start}, {event_window_end})",
        "events_analyzed": len(events_out),
        "results": events_out,
        "summary": {
            "average_car_pct": round(df["CAR (%)"].mean(), 2),
            "significant_events": f"{sig_count}/{len(events_out)}",
            "worst_car": round(df["CAR (%)"].min(), 2),
            "best_car": round(df["CAR (%)"].max(), 2),
        },
        "methodology": "Market model (OLS, 120-day estimation window). Significance at p<0.05 (two-tailed t-test).",
    }

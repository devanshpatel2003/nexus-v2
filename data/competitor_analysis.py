"""
Competitor Analysis Module
Analyzes competitive dynamics between NVDA, AMD, and INTC around export control events
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompetitiveShiftResult:
    """Results from competitive shift analysis around an event."""
    event_date: str
    event_title: str
    event_severity: str
    nvda_return: float
    amd_return: float
    intc_return: float
    spy_return: float
    nvda_vs_amd: float  # NVDA return minus AMD return
    nvda_vs_intc: float
    amd_vs_intc: float
    nvda_alpha: float  # NVDA excess return vs SPY
    amd_alpha: float
    intc_alpha: float
    winner: str  # Which stock performed best


def calculate_window_returns(
    prices: pd.DataFrame,
    event_date: str,
    window_start: int = -1,
    window_end: int = 5
) -> Dict[str, float]:
    """
    Calculate returns for each ticker over an event window.
    """
    event_date_dt = pd.to_datetime(event_date)

    # Find event date index
    if event_date_dt not in prices.index:
        idx = prices.index.get_indexer([event_date_dt], method='nearest')[0]
    else:
        idx = prices.index.get_loc(event_date_dt)

    # Define window
    start_idx = max(0, idx + window_start)
    end_idx = min(len(prices) - 1, idx + window_end)

    # Calculate returns
    returns = {}
    for ticker in prices.columns:
        start_price = prices[ticker].iloc[start_idx]
        end_price = prices[ticker].iloc[end_idx]
        returns[ticker] = ((end_price / start_price) - 1) * 100

    return returns


def analyze_competitive_shift(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    window_start: int = -1,
    window_end: int = 5
) -> List[CompetitiveShiftResult]:
    """
    Analyze competitive dynamics around each event.
    """
    results = []

    price_start = prices.index.min()
    price_end = prices.index.max()

    for _, event in events.iterrows():
        try:
            event_date = pd.to_datetime(str(event["date"])[:10])

            # Skip events outside price data range
            if event_date < price_start or event_date > price_end:
                continue

            returns = calculate_window_returns(
                prices, str(event["date"])[:10],
                window_start, window_end
            )

            nvda = returns.get("NVDA", 0)
            amd = returns.get("AMD", 0)
            intc = returns.get("INTC", 0)
            spy = returns.get("SPY", 0)

            # Determine winner
            perf = {"NVDA": nvda, "AMD": amd, "INTC": intc}
            winner = max(perf, key=perf.get)

            results.append(CompetitiveShiftResult(
                event_date=event_date.strftime("%Y-%m-%d"),
                event_title=event["title"],
                event_severity=event["severity"],
                nvda_return=nvda,
                amd_return=amd,
                intc_return=intc,
                spy_return=spy,
                nvda_vs_amd=nvda - amd,
                nvda_vs_intc=nvda - intc,
                amd_vs_intc=amd - intc,
                nvda_alpha=nvda - spy,
                amd_alpha=amd - spy,
                intc_alpha=intc - spy,
                winner=winner
            ))

        except Exception as e:
            print(f"Error processing {event['title']}: {e}")

    return results


def results_to_dataframe(results: List[CompetitiveShiftResult]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    if not results:
        return pd.DataFrame()

    records = []
    for r in results:
        records.append({
            "Event Date": r.event_date,
            "Event": r.event_title,
            "Severity": r.event_severity,
            "NVDA %": r.nvda_return,
            "AMD %": r.amd_return,
            "INTC %": r.intc_return,
            "SPY %": r.spy_return,
            "NVDA-AMD": r.nvda_vs_amd,
            "NVDA-INTC": r.nvda_vs_intc,
            "AMD-INTC": r.amd_vs_intc,
            "NVDA Alpha": r.nvda_alpha,
            "AMD Alpha": r.amd_alpha,
            "INTC Alpha": r.intc_alpha,
            "Winner": r.winner,
        })

    return pd.DataFrame(records)


def calculate_rolling_correlation(
    prices: pd.DataFrame,
    ticker1: str = "NVDA",
    ticker2: str = "AMD",
    window: int = 30
) -> pd.Series:
    """
    Calculate rolling correlation between two tickers.
    """
    returns = prices.pct_change()
    correlation = returns[ticker1].rolling(window=window).corr(returns[ticker2])
    return correlation


def calculate_relative_strength(
    prices: pd.DataFrame,
    ticker: str = "NVDA",
    benchmark: str = "AMD"
) -> pd.Series:
    """
    Calculate relative strength (ticker / benchmark ratio).
    """
    return prices[ticker] / prices[benchmark]


def get_market_share_proxy(
    prices: pd.DataFrame,
    tickers: List[str] = ["NVDA", "AMD", "INTC"]
) -> pd.DataFrame:
    """
    Calculate market cap weight proxy based on price movements.
    Normalized to sum to 100%.
    """
    # Use cumulative returns as proxy
    returns = (prices[tickers] / prices[tickers].iloc[0]) * 100
    total = returns.sum(axis=1)
    weights = returns.div(total, axis=0) * 100
    return weights


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def get_competitive_summary(results: List[CompetitiveShiftResult]) -> Dict:
    """
    Get summary statistics from competitive analysis.
    """
    if not results:
        return {}

    df = results_to_dataframe(results)

    # Filter to restriction-type events
    restriction_events = df[df["Severity"].isin(["Critical", "High"])]

    return {
        "total_events": len(df),
        "nvda_avg_return": df["NVDA %"].mean(),
        "amd_avg_return": df["AMD %"].mean(),
        "intc_avg_return": df["INTC %"].mean(),
        "nvda_wins": (df["Winner"] == "NVDA").sum(),
        "amd_wins": (df["Winner"] == "AMD").sum(),
        "intc_wins": (df["Winner"] == "INTC").sum(),
        "nvda_vs_amd_avg": df["NVDA-AMD"].mean(),
        "nvda_vs_intc_avg": df["NVDA-INTC"].mean(),
        "amd_vs_intc_avg": df["AMD-INTC"].mean(),
        # Restriction events only
        "restriction_nvda_avg": restriction_events["NVDA %"].mean() if len(restriction_events) > 0 else 0,
        "restriction_amd_avg": restriction_events["AMD %"].mean() if len(restriction_events) > 0 else 0,
        "restriction_amd_wins": (restriction_events["Winner"] == "AMD").sum() if len(restriction_events) > 0 else 0,
        "restriction_count": len(restriction_events),
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    from price_data import get_combined_prices
    from export_control_events import get_events_dataframe

    print("Loading data...")
    prices = get_combined_prices(start_date="2022-01-01")
    events = get_events_dataframe()

    print("\nAnalyzing competitive dynamics...")
    results = analyze_competitive_shift(prices, events, window_start=-1, window_end=5)

    df = results_to_dataframe(results)
    print("\n" + "=" * 100)
    print("COMPETITIVE SHIFT ANALYSIS")
    print("=" * 100)
    print(df[["Event Date", "Event", "NVDA %", "AMD %", "INTC %", "Winner"]].to_string(index=False))

    summary = get_competitive_summary(results)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"NVDA wins: {summary['nvda_wins']} | AMD wins: {summary['amd_wins']} | INTC wins: {summary['intc_wins']}")
    print(f"Avg NVDA-AMD spread: {summary['nvda_vs_amd_avg']:+.2f}%")
    print(f"\nDuring restriction events (Critical/High severity):")
    print(f"  NVDA avg: {summary['restriction_nvda_avg']:+.2f}%")
    print(f"  AMD avg: {summary['restriction_amd_avg']:+.2f}%")
    print(f"  AMD wins: {summary['restriction_amd_wins']} / {summary['restriction_count']}")

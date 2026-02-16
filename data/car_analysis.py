"""
CAR (Cumulative Abnormal Returns) Analysis
Event study methodology for export control announcements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class EventStudyResult:
    """Results from an event study analysis."""
    event_date: str
    event_title: str
    ticker: str
    car: float  # Cumulative Abnormal Return (%)
    raw_return: float  # Raw return over window (%)
    benchmark_return: float  # Benchmark return (%)
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    window_start: str
    window_end: str
    estimation_window_days: int


def calculate_expected_returns(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    estimation_start: int,
    estimation_end: int
) -> Tuple[float, float]:
    """
    Calculate expected returns using market model.
    R_i = alpha + beta * R_m + epsilon
    """
    # Get estimation window data
    stock_est = stock_returns.iloc[estimation_start:estimation_end].dropna()
    market_est = market_returns.iloc[estimation_start:estimation_end].dropna()

    # Align the series
    common_idx = stock_est.index.intersection(market_est.index)
    stock_est = stock_est.loc[common_idx]
    market_est = market_est.loc[common_idx]

    if len(stock_est) < 20:
        return 0.0, 1.0

    # OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        market_est.values, stock_est.values
    )

    return intercept, slope


def run_event_study(
    prices: pd.DataFrame,
    event_date: str,
    event_title: str,
    ticker: str = "NVDA",
    benchmark: str = "SPY",
    event_window_start: int = -1,
    event_window_end: int = 5,
    estimation_window: int = 120
) -> EventStudyResult:
    """
    Run event study for a single event.
    """
    event_date_dt = pd.to_datetime(event_date)

    # Calculate daily returns
    returns = prices.pct_change() * 100

    # Find event date index
    if event_date_dt not in returns.index:
        idx = returns.index.get_indexer([event_date_dt], method='nearest')[0]
        event_idx = idx
    else:
        event_idx = returns.index.get_loc(event_date_dt)

    # Define windows
    event_start_idx = max(0, event_idx + event_window_start)
    event_end_idx = min(len(returns) - 1, event_idx + event_window_end)
    estimation_end_idx = max(0, event_start_idx - 1)
    estimation_start_idx = max(0, estimation_end_idx - estimation_window)

    # Get returns
    stock_returns = returns[ticker]
    market_returns = returns[benchmark]

    # Calculate alpha and beta from estimation window
    alpha, beta = calculate_expected_returns(
        stock_returns, market_returns,
        estimation_start_idx, estimation_end_idx
    )

    # Calculate abnormal returns
    expected = alpha + beta * market_returns
    abnormal_returns = stock_returns - expected

    # Calculate CAR over event window
    event_ar = abnormal_returns.iloc[event_start_idx:event_end_idx + 1]
    car = event_ar.sum()

    # Raw returns
    event_raw = stock_returns.iloc[event_start_idx:event_end_idx + 1]
    raw_return = event_raw.sum()

    # Benchmark returns
    event_benchmark = market_returns.iloc[event_start_idx:event_end_idx + 1]
    benchmark_return = event_benchmark.sum()

    # Statistical significance
    estimation_ar = abnormal_returns.iloc[estimation_start_idx:estimation_end_idx]
    std_ar = estimation_ar.std()

    if std_ar > 0 and len(estimation_ar) > 2:
        window_length = event_end_idx - event_start_idx + 1
        t_stat = car / (std_ar * np.sqrt(window_length))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(estimation_ar) - 1))
    else:
        t_stat = 0.0
        p_value = 1.0

    # Window dates
    window_start = returns.index[event_start_idx].strftime("%Y-%m-%d")
    window_end = returns.index[event_end_idx].strftime("%Y-%m-%d")

    return EventStudyResult(
        event_date=event_date_dt.strftime("%Y-%m-%d"),
        event_title=event_title,
        ticker=ticker,
        car=float(car),
        raw_return=float(raw_return),
        benchmark_return=float(benchmark_return),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=p_value < 0.05,
        window_start=window_start,
        window_end=window_end,
        estimation_window_days=estimation_end_idx - estimation_start_idx
    )


def run_multiple_event_studies(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    ticker: str = "NVDA",
    benchmark: str = "SPY",
    event_window: Tuple[int, int] = (-1, 5)
) -> List[EventStudyResult]:
    """
    Run event studies for multiple events.
    """
    results = []

    # Get the date range of our price data
    price_start = prices.index.min()
    price_end = prices.index.max()

    for _, event in events.iterrows():
        try:
            event_date = pd.to_datetime(str(event["date"])[:10])

            # Skip events outside our price data range
            if event_date < price_start or event_date > price_end:
                continue

            result = run_event_study(
                prices=prices,
                event_date=str(event["date"])[:10],
                event_title=event["title"],
                ticker=ticker,
                benchmark=benchmark,
                event_window_start=event_window[0],
                event_window_end=event_window[1]
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing event {event['title']}: {e}")

    return results


def results_to_dataframe(results: List[EventStudyResult]) -> pd.DataFrame:
    """Convert event study results to DataFrame."""
    if not results:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "Event Date", "Event", "Ticker", "CAR (%)", "Raw Return (%)",
            "Benchmark (%)", "T-Stat", "P-Value", "Significant", "Window"
        ])

    records = []
    for r in results:
        records.append({
            "Event Date": r.event_date,
            "Event": r.event_title,
            "Ticker": r.ticker,
            "CAR (%)": r.car,
            "Raw Return (%)": r.raw_return,
            "Benchmark (%)": r.benchmark_return,
            "T-Stat": r.t_statistic,
            "P-Value": r.p_value,
            "Significant": r.is_significant,
            "Window": f"{r.window_start} to {r.window_end}",
        })
    return pd.DataFrame(records)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    from price_data import get_combined_prices
    from export_control_events import get_events_dataframe

    print("Loading price data...")
    prices = get_combined_prices(start_date="2022-01-01")
    print(f"Price data: {prices.index.min()} to {prices.index.max()}")

    print("\nLoading events...")
    events = get_events_dataframe()
    print(f"Events: {len(events)} total")

    print("\nRunning event studies for NVDA...")
    results = run_multiple_event_studies(
        prices=prices,
        events=events,
        ticker="NVDA",
        event_window=(-1, 5)
    )

    print(f"\nProcessed {len(results)} events")

    if results:
        df = results_to_dataframe(results)
        print("\n" + "=" * 80)
        print("EVENT STUDY RESULTS: NVDA")
        print("=" * 80)
        print(df[["Event Date", "Event", "CAR (%)", "Significant"]].to_string(index=False))
        print(f"\nAverage CAR: {df['CAR (%)'].mean():.2f}%")

# Event Study Methodology
doc_id: method_event_study

## Overview
NEXUS uses the market model for event study analysis, a standard approach in empirical finance.

## Market Model
Expected return: R_i = alpha + beta * R_market + epsilon

- Estimation window: 120 trading days prior to event window
- Event window: configurable (default: -1 to +5 days relative to event)
- Benchmark: S&P 500 (SPY) by default

## Cumulative Abnormal Return (CAR)
CAR = sum of (actual return - expected return) over the event window

## Statistical Testing
- T-statistic: CAR / (sigma_AR * sqrt(window_length))
- Two-tailed t-test with degrees of freedom = estimation window length - 1
- Significance threshold: p < 0.05

## Limitations
1. Only 1 of 11 events was statistically significant (May 2023 earnings). Individual event studies with short windows often lack statistical power.
2. Events near weekends or holidays use nearest trading day approximation.
3. Overlapping event windows can confound results.
4. The market model assumes linear relationship between stock and benchmark returns.
5. 120-day estimation window may capture structural breaks (e.g., AI boom changing NVDA's beta).

## Interpretation Guidelines
- CAR measures the stock's return in excess of what the market model predicts.
- Positive CAR = stock outperformed expectations; negative CAR = underperformed.
- Statistical significance matters: non-significant CARs should be interpreted cautiously.
- For aggregated analysis, consider grouping events by type (restriction vs. earnings) for more power.

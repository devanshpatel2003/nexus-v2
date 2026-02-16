# Volatility Surface Methodology
doc_id: method_volatility

## Implied Volatility
IV is extracted from the options chain via yfinance (Black-Scholes implied). NEXUS displays IV across:
- Strike prices (moneyness: % from spot)
- Expirations (days to expiry)

## Volatility Skew
Skew = IV(OTM puts) - IV(ATM options)
- Positive skew: crash protection demand exceeds upside speculation.
- Higher skew before export control events indicates hedging activity.
- Measured at 30-day and 60-day tenors.

## Term Structure
ATM volatility plotted across expirations:
- Contango (normal): long-term IV > short-term IV
- Backwardation (inverted): short-term IV > long-term IV — pricing near-term event risk
- Inverted structure before BIS announcements = market pricing imminent regulatory risk

## Historical Realized Volatility
Proxy for historical IV using rolling standard deviation of returns:
- RV(20): 20-day annualized realized vol
- RV(60): 60-day annualized realized vol
- True historical IV requires paid data sources (e.g., CBOE, OptionMetrics)

## Limitations
1. Live options chain requires market hours for fresh data
2. yfinance IV can be missing for illiquid strikes
3. RV ≠ IV: realized vol is backward-looking, implied vol is forward-looking
4. Black-Scholes assumptions (constant vol, no dividends, no early exercise) are approximations

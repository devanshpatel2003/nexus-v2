"""
Options Data Module
Fetches options chain data and calculates implied volatility surfaces
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def fetch_options_chain(ticker: str = "NVDA") -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fetch current options chain using yfinance.

    Returns:
        Tuple of (calls_df, puts_df, current_price)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    stock = yf.Ticker(ticker)
    current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice', 100)

    # Get all expiration dates
    expirations = stock.options

    if not expirations:
        raise ValueError(f"No options data available for {ticker}")

    all_calls = []
    all_puts = []

    for exp_date in expirations[:8]:  # Limit to 8 expirations for performance
        try:
            opt = stock.option_chain(exp_date)

            calls = opt.calls.copy()
            calls['expiration'] = exp_date
            calls['type'] = 'call'
            all_calls.append(calls)

            puts = opt.puts.copy()
            puts['expiration'] = exp_date
            puts['type'] = 'put'
            all_puts.append(puts)

        except Exception as e:
            print(f"Error fetching {exp_date}: {e}")
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    return calls_df, puts_df, current_price


def calculate_days_to_expiry(expiration: str) -> int:
    """Calculate days until expiration."""
    exp_date = pd.to_datetime(expiration)
    today = pd.Timestamp.now().normalize()
    return max(1, (exp_date - today).days)


def calculate_moneyness(strike: float, spot: float, option_type: str = 'call') -> float:
    """
    Calculate moneyness as percentage from spot.
    Positive = OTM for calls, ITM for puts
    """
    return ((strike / spot) - 1) * 100


def build_vol_surface(
    options_df: pd.DataFrame,
    spot_price: float,
    option_type: str = 'put'
) -> pd.DataFrame:
    """
    Build implied volatility surface from options data.

    Returns DataFrame with columns: strike, expiration, days_to_expiry, moneyness, implied_vol
    """
    if options_df.empty:
        return pd.DataFrame()

    records = []

    for _, row in options_df.iterrows():
        try:
            strike = row['strike']
            expiration = row['expiration']

            # Get implied volatility (yfinance provides this)
            iv = row.get('impliedVolatility', 0)

            if iv is None or iv <= 0 or iv > 5:  # Filter unrealistic IVs
                continue

            days = calculate_days_to_expiry(expiration)
            moneyness = calculate_moneyness(strike, spot_price, option_type)

            # Filter to reasonable moneyness range (-30% to +30%)
            if abs(moneyness) > 30:
                continue

            records.append({
                'strike': strike,
                'expiration': expiration,
                'days_to_expiry': days,
                'moneyness': moneyness,
                'implied_vol': iv * 100,  # Convert to percentage
                'bid': row.get('bid', 0),
                'ask': row.get('ask', 0),
                'volume': row.get('volume', 0),
                'open_interest': row.get('openInterest', 0),
            })

        except Exception as e:
            continue

    return pd.DataFrame(records)


def get_vol_surface_matrix(
    vol_surface: pd.DataFrame,
    moneyness_bins: List[float] = None,
    expiry_bins: List[int] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Convert vol surface to matrix for heatmap visualization.

    Returns:
        Tuple of (vol_matrix, moneyness_labels, expiry_labels)
    """
    if vol_surface.empty:
        return np.array([[]]), [], []

    if moneyness_bins is None:
        moneyness_bins = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    if expiry_bins is None:
        expiry_bins = [7, 14, 30, 45, 60, 90, 120, 180]

    # Bin the data
    vol_surface['moneyness_bin'] = pd.cut(
        vol_surface['moneyness'],
        bins=moneyness_bins + [100],  # Add upper bound
        labels=moneyness_bins
    )

    vol_surface['expiry_bin'] = pd.cut(
        vol_surface['days_to_expiry'],
        bins=[0] + expiry_bins + [1000],
        labels=['<7'] + [f'{expiry_bins[i]}' for i in range(len(expiry_bins))]
    )

    # Pivot to matrix
    pivot = vol_surface.pivot_table(
        values='implied_vol',
        index='moneyness_bin',
        columns='expiry_bin',
        aggfunc='mean'
    )

    # Clean up
    pivot = pivot.dropna(how='all', axis=0).dropna(how='all', axis=1)

    return pivot.values, list(pivot.index), list(pivot.columns)


def calculate_skew(vol_surface: pd.DataFrame, days_to_expiry: int = 30) -> Dict:
    """
    Calculate volatility skew metrics for a given expiration.

    Skew = IV(OTM puts) - IV(ATM)
    """
    # Filter to near the target expiry
    mask = (vol_surface['days_to_expiry'] >= days_to_expiry - 10) & \
           (vol_surface['days_to_expiry'] <= days_to_expiry + 10)

    subset = vol_surface[mask]

    if subset.empty:
        return {'skew': 0, 'atm_vol': 0, 'otm_put_vol': 0}

    # ATM: moneyness near 0
    atm = subset[abs(subset['moneyness']) < 3]['implied_vol'].mean()

    # OTM puts: moneyness -10% to -5%
    otm_puts = subset[(subset['moneyness'] >= -15) & (subset['moneyness'] <= -5)]['implied_vol'].mean()

    skew = otm_puts - atm if not np.isnan(otm_puts) and not np.isnan(atm) else 0

    return {
        'skew': skew,
        'atm_vol': atm if not np.isnan(atm) else 0,
        'otm_put_vol': otm_puts if not np.isnan(otm_puts) else 0,
        'days_to_expiry': days_to_expiry
    }


def calculate_term_structure(vol_surface: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ATM volatility term structure.
    """
    # Get ATM options (moneyness close to 0)
    atm = vol_surface[abs(vol_surface['moneyness']) < 5].copy()

    if atm.empty:
        return pd.DataFrame()

    # Group by expiration and get mean IV
    term = atm.groupby('days_to_expiry').agg({
        'implied_vol': 'mean',
        'volume': 'sum',
        'open_interest': 'sum'
    }).reset_index()

    term = term.sort_values('days_to_expiry')

    return term


def get_historical_iv(ticker: str = "NVDA", period: str = "1y") -> pd.DataFrame:
    """
    Get historical implied volatility proxy using historical volatility.
    (True historical IV requires paid data sources)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required")

    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return pd.DataFrame()

    # Calculate realized volatility as IV proxy
    returns = hist['Close'].pct_change()

    # 20-day realized vol (annualized)
    rv_20 = returns.rolling(20).std() * np.sqrt(252) * 100

    # 60-day realized vol
    rv_60 = returns.rolling(60).std() * np.sqrt(252) * 100

    result = pd.DataFrame({
        'date': hist.index,
        'close': hist['Close'],
        'rv_20': rv_20,
        'rv_60': rv_60
    })

    return result


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("Fetching NVDA options chain...")

    try:
        calls, puts, spot = fetch_options_chain("NVDA")
        print(f"Spot price: ${spot:.2f}")
        print(f"Calls: {len(calls)} contracts")
        print(f"Puts: {len(puts)} contracts")

        print("\nBuilding put vol surface...")
        vol_surface = build_vol_surface(puts, spot, 'put')
        print(f"Vol surface: {len(vol_surface)} points")

        if not vol_surface.empty:
            print("\nSample vol surface data:")
            print(vol_surface.head(10).to_string())

            print("\nSkew analysis (30-day):")
            skew = calculate_skew(vol_surface, 30)
            print(f"  ATM Vol: {skew['atm_vol']:.1f}%")
            print(f"  OTM Put Vol: {skew['otm_put_vol']:.1f}%")
            print(f"  Skew: {skew['skew']:+.1f}%")

            print("\nTerm structure:")
            term = calculate_term_structure(vol_surface)
            print(term.to_string())

    except Exception as e:
        print(f"Error: {e}")
        print("This may be due to market hours or data availability.")

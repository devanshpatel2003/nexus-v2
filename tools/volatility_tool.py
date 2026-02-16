"""
Volatility Tool — IV surface, skew, term structure, historical vol.
"""

from typing import List, Optional
from data.options_data import (
    fetch_options_chain, build_vol_surface,
    calculate_skew, calculate_term_structure, get_historical_iv,
)

SCHEMA = {
    "type": "function",
    "function": {
        "name": "volatility_tool",
        "description": "Get implied volatility surface data, skew metrics, term structure, and historical realized volatility for a semiconductor ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker (default: NVDA)",
                },
                "option_type": {
                    "type": "string",
                    "enum": ["put", "call"],
                    "description": "Option type for surface (default: put)",
                },
                "skew_days": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Days-to-expiry for skew calc (default: [30, 60])",
                },
            },
            "required": [],
        },
    },
}


def run(
    ticker: str = "NVDA",
    option_type: str = "put",
    skew_days: Optional[List[int]] = None,
) -> dict:
    """Execute volatility analysis and return JSON-serializable results."""
    if skew_days is None:
        skew_days = [30, 60]

    result = {"ticker": ticker, "option_type": option_type}

    # Try live options chain
    try:
        calls, puts, spot_price = fetch_options_chain(ticker)
        result["spot_price"] = round(float(spot_price), 2)

        options_df = puts if option_type == "put" else calls
        vol_surface = build_vol_surface(options_df, spot_price, option_type)

        if not vol_surface.empty:
            # Skew metrics
            skew_data = {}
            for d in skew_days:
                s = calculate_skew(vol_surface, d)
                skew_data[f"{d}d"] = {
                    "atm_vol": round(s["atm_vol"], 1),
                    "otm_put_vol": round(s["otm_put_vol"], 1),
                    "skew": round(s["skew"], 1),
                }
            result["skew"] = skew_data

            # Term structure
            term = calculate_term_structure(vol_surface)
            if not term.empty:
                result["term_structure"] = [
                    {
                        "days_to_expiry": int(row["days_to_expiry"]),
                        "implied_vol": round(float(row["implied_vol"]), 1),
                    }
                    for _, row in term.head(8).iterrows()
                ]

            result["surface_points"] = len(vol_surface)
        else:
            result["note"] = "Vol surface could not be built from available options."

    except Exception as e:
        result["options_error"] = f"Live options unavailable: {str(e)[:100]}"

    # Historical realized volatility (always available)
    try:
        hist = get_historical_iv(ticker, period="2y")
        if not hist.empty:
            result["historical_vol"] = {
                "current_rv20": round(float(hist["rv_20"].iloc[-1]), 1),
                "current_rv60": round(float(hist["rv_60"].iloc[-1]), 1),
                "average_rv20": round(float(hist["rv_20"].mean()), 1),
            }
    except Exception:
        pass

    return result

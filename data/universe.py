"""
Semiconductor Ecosystem Universe
Expanded ticker groups for NEXUS v2 analysis.
"""

from typing import Dict, List


# Ecosystem groups
ECOSYSTEM: Dict[str, List[str]] = {
    "gpu_leaders": ["NVDA", "AMD"],
    "legacy_semi": ["INTC"],
    "foundry_equipment": ["TSM", "ASML"],
    "networking_broadband": ["AVGO"],
    "hyperscalers": ["GOOGL", "AMZN", "MSFT"],
    "benchmark": ["SPY", "SMH"],
}

# Flat list of all tickers
ALL_TICKERS: List[str] = sorted(set(
    t for group in ECOSYSTEM.values() for t in group
))

# Core tickers for event study (directly affected by export controls)
CORE_TICKERS: List[str] = ["NVDA", "AMD", "INTC", "SPY"]

# Extended tickers for ecosystem analysis
EXTENDED_TICKERS: List[str] = [
    "NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO",
    "GOOGL", "AMZN", "MSFT", "SMH", "SPY",
]

# Display names
TICKER_NAMES: Dict[str, str] = {
    "NVDA": "Nvidia",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "TSM": "Taiwan Semiconductor (TSMC)",
    "ASML": "ASML Holdings",
    "AVGO": "Broadcom",
    "GOOGL": "Alphabet (Google TPU)",
    "AMZN": "Amazon (Trainium/Inferentia)",
    "MSFT": "Microsoft (Maia/Cobalt)",
    "SPY": "S&P 500 ETF",
    "SMH": "VanEck Semiconductor ETF",
}

# Colors for charts
TICKER_COLORS: Dict[str, str] = {
    "NVDA": "#76b900",
    "AMD": "#ed1c24",
    "INTC": "#0071c5",
    "TSM": "#e8530e",
    "ASML": "#00a3e0",
    "AVGO": "#cc0000",
    "GOOGL": "#4285f4",
    "AMZN": "#ff9900",
    "MSFT": "#00a4ef",
    "SPY": "#888888",
    "SMH": "#aaaaaa",
}

# Export control exposure (qualitative)
EXPORT_EXPOSURE: Dict[str, str] = {
    "NVDA": "Direct — primary target of BIS controls on AI accelerators",
    "AMD": "Direct — MI series GPUs restricted alongside Nvidia",
    "INTC": "Moderate — Gaudi accelerators + foundry services exposure",
    "TSM": "Indirect — manufactures restricted chips, caught in entity lists",
    "ASML": "Indirect — EUV lithography equipment export restrictions",
    "AVGO": "Low — networking/custom silicon, limited China AI exposure",
    "GOOGL": "Indirect — TPU not exported, but cloud AI services affected",
    "AMZN": "Indirect — Trainium internal, but AWS China operations affected",
    "MSFT": "Indirect — Azure AI services in restricted regions",
}


def get_group_tickers(group: str) -> List[str]:
    """Get tickers for a named ecosystem group."""
    return ECOSYSTEM.get(group, [])


def get_ticker_info(ticker: str) -> Dict:
    """Get display name, color, and export exposure for a ticker."""
    return {
        "ticker": ticker,
        "name": TICKER_NAMES.get(ticker, ticker),
        "color": TICKER_COLORS.get(ticker, "#888888"),
        "export_exposure": EXPORT_EXPOSURE.get(ticker, "Unknown"),
    }

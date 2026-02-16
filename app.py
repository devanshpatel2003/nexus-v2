"""
NEXUS v2: AI-Powered Finance Research Assistant
Intelligent chat interface grounded in semiconductor export control case study.
Built for MGMT 69000: Mastering AI for Finance — Purdue University.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from data.export_control_events import (
    get_events_dataframe, get_event_summary, Severity, EventType,
)
from data.price_data import get_combined_prices, normalize_prices
from data.car_analysis import (
    run_multiple_event_studies, results_to_dataframe as car_to_dataframe,
)
from data.competitor_analysis import (
    analyze_competitive_shift, results_to_dataframe as competitor_to_dataframe,
    get_competitive_summary, calculate_rolling_correlation, calculate_relative_strength,
)
from data.options_data import (
    fetch_options_chain, build_vol_surface, get_vol_surface_matrix,
    calculate_skew, calculate_term_structure, get_historical_iv,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEXUS v2 — Semiconductor Export Control Research",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS
# ============================================================

NAVY = "#0f1d33"
NAVY_LIGHT = "#1a2744"
NAVY_MID = "#2a3a5c"
BLUE_ACCENT = "#2563eb"
SLATE = "#475569"
BORDER = "#e2e8f0"
BG_SUBTLE = "#f8fafc"

# ============================================================
# INSTITUTIONAL CSS
# ============================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ── */
html, body, [class*="st-"],
p, span, li, td, th, label, input, textarea, select, button {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* ── Main-area text colors (scoped — NOT sidebar, NOT navy boxes) ── */
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] li,
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] td {{
    font-size: 0.84rem !important;
    line-height: 1.65 !important;
    color: {NAVY_LIGHT} !important;
}}
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] strong,
[data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] b {{
    color: {NAVY} !important;
}}

/* ── Override: white text inside navy-bg containers ── */
.takeaway-box, .takeaway-box p, .takeaway-box li, .takeaway-box strong, .takeaway-box b {{
    color: rgba(255,255,255,0.88) !important;
}}
.takeaway-box strong, .takeaway-box b {{
    color: #ffffff !important;
}}
.brand-header strong, .brand-header b {{
    color: rgba(255,255,255,0.90) !important;
}}
.spot-banner, .spot-banner * {{
    color: rgba(255,255,255,0.88) !important;
}}

/* ── Hide chrome ── */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* ── Layout ── */
.stApp > header {{ display: none; }}
.stApp {{ background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }}
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}}
[data-testid="stAppViewContainer"] {{
    background: transparent;
}}

/* ── Brand Header (full-bleed blue) ── */
.brand-header {{
    background: {BLUE_ACCENT};
    /* Full-bleed: break out of max-width container edge-to-edge */
    margin-left: calc(-50vw + 50%) !important;
    margin-right: calc(-50vw + 50%) !important;
    padding-left: calc(50vw - 50%) !important;
    padding-right: calc(50vw - 50%) !important;
    margin-top: -1rem;
    margin-bottom: 1.2rem;
    padding-top: 1.2rem;
    padding-bottom: 0.9rem;
    border-bottom: 3px solid {NAVY};
}}
/* Inner content stays at original width */
.brand-header .header-inner {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2rem;
}}
.brand-header .top-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}}
.brand-header .logo {{
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
}}
.brand-header .logo span {{ color: #ffffff; font-weight: 300; }}
.brand-header .tagline {{
    font-size: 0.66rem;
    color: rgba(255,255,255,0.80);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 3px;
}}
.brand-header .header-meta {{
    text-align: right;
    font-size: 0.64rem;
    color: rgba(255,255,255,0.70);
    letter-spacing: 0.04em;
    line-height: 1.7;
}}
.brand-header .header-meta strong {{
    color: rgba(255,255,255,0.92) !important;
    font-weight: 600;
}}
.brand-header .desc-bar {{
    margin-top: 0.65rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(255,255,255,0.22);
    font-size: 0.74rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.55;
}}

/* ── Metric Cards ── */
[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(15,29,51,0.05);
    border-top: 2px solid {NAVY_LIGHT};
    transition: box-shadow 0.15s ease;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 2px 8px rgba(15,29,51,0.08);
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: {SLATE} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: {NAVY} !important;
    letter-spacing: -0.02em !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.74rem !important;
}}

/* ── Tab Navigation ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: #f1f5f9;
    border-radius: 6px;
    padding: 3px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
    height: 40px;
    border-radius: 5px;
    font-weight: 500;
    font-size: 0.8rem !important;
    color: {SLATE} !important;
    padding: 0 16px;
    white-space: nowrap;
    border-bottom: none !important;
}}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    background: {NAVY_LIGHT} !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(15,29,51,0.18);
    border-bottom: none !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {NAVY};
    border-right: 1px solid rgba(255,255,255,0.06);
}}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b {{
    color: rgba(255,255,255,0.88) !important;
}}
section[data-testid="stSidebar"] p {{
    font-size: 0.78rem !important;
}}
section[data-testid="stSidebar"] label {{
    color: rgba(255,255,255,0.50) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: rgba(255,255,255,0.35) !important;
    margin-top: 1.4rem !important;
}}
section[data-testid="stSidebar"] [data-testid="stAlert"] {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 0.76rem;
}}
section[data-testid="stSidebar"] [data-testid="stAlert"] p,
section[data-testid="stSidebar"] [data-testid="stAlert"] span {{
    color: rgba(255,255,255,0.78) !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.85) !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
}}

/* ── Section Headers (unified) ── */
.section-header {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {NAVY};
    margin-bottom: 0.9rem;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid {NAVY_LIGHT};
    display: block;
}}

/* ── Callout Boxes ── */
.callout {{
    background: {BG_SUBTLE};
    border: 1px solid {BORDER};
    border-left: 3px solid {NAVY_LIGHT};
    border-radius: 5px;
    padding: 14px 18px;
    margin: 0.6rem 0 1.1rem 0;
    font-size: 0.82rem;
    line-height: 1.65;
    color: #334155;
}}
.callout strong {{ color: {NAVY} !important; }}
.callout-blue {{
    background: #eff6ff;
    border-color: #bfdbfe;
    border-left-color: {BLUE_ACCENT};
}}
.callout-method {{
    background: #f5f3ff;
    border-color: #ddd6fe;
    border-left-color: #7c3aed;
}}

/* ── Observation / Takeaway ── */
.obs-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 0.6rem 0 1rem 0;
}}
.obs-card {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #334155;
    box-shadow: 0 1px 3px rgba(15,29,51,0.04);
}}
.obs-card .obs-label {{
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: {NAVY_MID};
    margin-bottom: 5px;
}}
.obs-card strong {{ color: {NAVY} !important; }}
.takeaway-box {{
    background: {NAVY};
    border-radius: 6px;
    padding: 18px 22px;
    margin: 1rem 0 1.5rem 0;
    color: rgba(255,255,255,0.88);
    font-size: 0.82rem;
    line-height: 1.65;
}}
.takeaway-box .tk-title {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.40);
    margin-bottom: 8px;
}}
.takeaway-box ul {{ margin: 6px 0 0 0; padding-left: 18px; }}
.takeaway-box li {{ margin-bottom: 5px; color: rgba(255,255,255,0.85) !important; }}
.takeaway-box strong {{ color: #ffffff !important; }}

/* ── Spot Price Banner ── */
.spot-banner {{
    background: {NAVY};
    border-radius: 6px;
    padding: 13px 22px;
    margin-bottom: 1.1rem;
    display: inline-block;
    border: 1px solid {NAVY_MID};
}}
.spot-banner .label {{
    color: rgba(255,255,255,0.50) !important;
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-weight: 600;
}}
.spot-banner .price {{
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-left: 8px;
}}
.spot-banner .meta {{
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.74rem;
    margin-left: 12px;
}}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15,29,51,0.04);
}}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {{
    border-radius: 8px;
    border: 1px solid {BORDER};
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15,29,51,0.03);
    gap: 0.8rem !important;
}}
/* User messages — subtle tint */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: {BG_SUBTLE};
    border-color: {BORDER};
}}
/* Hide default emoji avatars, replace with clean initials */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {{
    display: none !important;
}}
/* Custom avatar badge */
[data-testid="stChatMessage"] > div:first-child {{
    min-width: 32px !important;
    max-width: 32px !important;
    height: 32px !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    flex-shrink: 0 !important;
}}
/* Chat text styling */
[data-testid="stChatMessage"] p {{
    font-size: 0.84rem !important;
    line-height: 1.7 !important;
    color: {NAVY_LIGHT} !important;
    margin: 0 !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}}
[data-testid="stChatMessage"] strong {{
    color: {NAVY} !important;
}}
/* Chat input bar */
[data-testid="stChatInput"] {{
    border-top: 1px solid {BORDER} !important;
    padding-top: 0.8rem !important;
}}
[data-testid="stChatInput"] textarea {{
    font-size: 0.84rem !important;
    color: {NAVY_LIGHT} !important;
    border-radius: 8px !important;
    border-color: {BORDER} !important;
    padding: 0.65rem 1rem !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {SLATE} !important;
    opacity: 0.7 !important;
}}

/* ── Buttons ── */
.stButton > button {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.8rem;
    transition: all 0.15s ease;
    background: #ffffff;
    color: {NAVY_LIGHT};
    box-shadow: 0 1px 2px rgba(15,29,51,0.04);
}}
.stButton > button:hover {{
    border-color: {NAVY_MID};
    background: #f8fafc;
    box-shadow: 0 2px 6px rgba(15,29,51,0.08);
    color: {NAVY};
}}
hr {{
    border-color: {BORDER} !important;
    margin: 1.2rem 0 !important;
}}
.stAlert {{
    border-radius: 6px;
    font-size: 0.82rem;
}}
/* ── Caption uniformity ── */
[data-testid="stCaptionContainer"] p {{
    font-size: 0.78rem !important;
    color: {SLATE} !important;
    line-height: 1.55 !important;
}}

/* ── Widget overrides ── */
.stSelectbox [data-baseweb="select"] > div {{
    font-size: 0.82rem !important;
}}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{
    font-size: 0.74rem !important;
    color: {SLATE} !important;
}}
/* Expander headers */
[data-testid="stExpander"] summary {{
    background: {BG_SUBTLE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
}}
[data-testid="stExpander"] summary span {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: {NAVY_LIGHT} !important;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    border: 1px solid {BORDER} !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
    padding: 0.8rem 1rem !important;
}}
/* Code blocks in evidence */
.stCodeBlock {{
    border-radius: 5px !important;
    font-size: 0.78rem !important;
}}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15,29,51,0.04);
    margin-bottom: 0.3rem;
}}

/* ── Spinner text ── */
.stSpinner > div {{
    font-size: 0.82rem !important;
    color: {SLATE} !important;
}}

/* ── Footer ── */
.pro-footer {{
    text-align: center;
    font-size: 0.68rem;
    color: {SLATE};
    letter-spacing: 0.03em;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid {BORDER};
    margin-top: 3rem;
    line-height: 1.8;
}}
.pro-footer strong {{ color: {NAVY_LIGHT} !important; }}

/* ── Suggestion buttons (chat tab) ── */
.stTabs [data-testid="stVerticalBlockBorderWrapper"] .stButton > button {{
    text-align: left !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: {NAVY_LIGHT} !important;
    padding: 0.55rem 1rem !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.45 !important;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=12, color=NAVY_LIGHT),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=11, color=SLATE),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left",
    ),
    margin=dict(l=55, r=20, t=60, b=45),
    xaxis=dict(
        gridcolor="#f1f5f9", zerolinecolor="#e2e8f0",
        linecolor="#e2e8f0", linewidth=1,
        tickfont=dict(size=11, color=SLATE),
    ),
    yaxis=dict(
        gridcolor="#f1f5f9", zerolinecolor="#e2e8f0",
        linecolor="#e2e8f0", linewidth=1,
        tickfont=dict(size=11, color=SLATE),
    ),
    hoverlabel=dict(bgcolor=NAVY, font_size=12, font_family="Inter", font_color="#ffffff"),
)

TITLE_STYLE = dict(font=dict(size=14, color=NAVY, family="Inter"), x=0, xanchor="left", y=0.97)

COLORS = {
    "NVDA": "#16a34a", "AMD": "#dc2626", "INTC": "#2563eb", "SPY": "#94a3b8",
    "TSM": "#ea580c", "ASML": "#0891b2", "AVGO": "#be123c",
    "GOOGL": "#4285f4", "AMZN": "#f59e0b", "MSFT": "#0ea5e9",
    "positive": "#16a34a", "negative": "#dc2626", "accent": NAVY_LIGHT,
    "muted": "#94a3b8", "grid": "#f1f5f9",
}

SEVERITY_COLORS = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626", "Critical": "#7c3aed"}

# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="brand-header">
  <div class="header-inner">
    <div class="top-row">
        <div>
            <div class="logo"><span>N</span>EXUS<span style="font-weight:400; font-size:0.65em; color:rgba(255,255,255,0.55); margin-left:6px;">v2.0</span></div>
            <div class="tagline">Semiconductor Export Control Research Platform</div>
        </div>
        <div class="header-meta">
            <strong>Purdue University</strong> &middot; Daniels School of Business<br>
            MGMT 69000: Mastering AI for Finance &middot; Prof. Cinder Zhang<br>
            Spring 2026
        </div>
    </div>
    <div class="desc-bar">
        Quantitative analysis of U.S. Bureau of Industry &amp; Security (BIS) semiconductor export
        restrictions and their impact on equity valuations, competitive dynamics, and options-implied
        risk premia across the global semiconductor supply chain. Powered by RAG-grounded AI with
        live market data tools.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab1, tab2, tab3, tab4 = st.tabs([
    "  Research Chat  ",
    "  Event Database  ",
    "  Price Reaction  ",
    "  Ecosystem Analysis  ",
    "  Volatility Surface  ",
])

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
        <div style="font-size:1.2rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">
            <span style="color:{BLUE_ACCENT};">N</span>EXUS <span style="color:{BLUE_ACCENT};">v2</span>
        </div>
        <div style="font-size:0.58rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:0.12em; margin-top:4px; font-weight:500;">
            Purdue Daniels School of Business
        </div>
        <div style="font-size:0.52rem; color:rgba(255,255,255,0.22); margin-top:2px; letter-spacing:0.06em;">
            MGMT 69000 &middot; Mastering AI for Finance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### System Status")
    try:
        from core.rag.vector_store import get_collection
        count = get_collection().count()
        st.success(f"RAG Index: {count} chunks loaded")
    except Exception:
        st.warning("RAG Index: Not built")

    try:
        from core.config import get_openai_key
        get_openai_key()
        st.success("OpenAI API: Connected")
    except Exception:
        st.error("OpenAI API: Key missing")

# ============================================================
# LOAD DATA (cached)
# ============================================================

@st.cache_data(ttl=3600)
def load_events():
    return get_events_dataframe()

@st.cache_data(ttl=3600)
def load_prices():
    return get_combined_prices(tickers=["NVDA", "AMD", "INTC", "SPY"], start_date="2022-01-01")

@st.cache_data(ttl=3600)
def load_car_results(_prices, _events, ticker, window):
    return run_multiple_event_studies(prices=_prices, events=_events, ticker=ticker, event_window=window)

@st.cache_data(ttl=3600)
def load_competitor_results(_prices, _events, window):
    return analyze_competitive_shift(prices=_prices, events=_events, window_start=window[0], window_end=window[1])

@st.cache_data(ttl=300)
def load_options_data(ticker):
    return fetch_options_chain(ticker)

@st.cache_data(ttl=3600)
def load_historical_vol(ticker):
    return get_historical_iv(ticker, period="2y")

events_df = load_events()
summary = get_event_summary()

# ============================================================
# TAB: RESEARCH CHAT
# ============================================================

with tab_chat:
    st.markdown('<div class="section-header">AI Research Assistant &middot; RAG-Grounded Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-blue">
        <strong>How this works:</strong> The research assistant retrieves relevant passages from 15 curated
        case study documents using vector similarity search (RAG), then augments responses with live
        market data via four callable tools: <strong>Price Analysis</strong>, <strong>Event Study (CAR)</strong>,
        <strong>Volatility Surface</strong>, and <strong>Ecosystem Comparison</strong>. Every response must
        cite its sources — either retrieved documents or tool outputs. The system will not fabricate data.
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []

    for msg in st.session_state.messages:
        avatar = "N" if msg["role"] == "assistant" else "\U0001F464"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if not st.session_state.messages:
        st.markdown('<div class="section-header">Suggested Queries</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="callout">
            Select a pre-built query below to explore the platform's capabilities, or type your own question.
            The assistant can answer factual questions from case materials, run quantitative analyses on
            demand, and synthesize findings across multiple data sources.
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        suggestions = [
            ("What was the market impact of the October 7 export controls?", "Document Retrieval"),
            ("Run a CAR analysis for NVDA around Critical events", "Event Study Tool"),
            ("Compare NVDA, TSM, and ASML performance since 2022", "Ecosystem Tool"),
            ("What is the current volatility skew for NVDA puts?", "Volatility Tool"),
            ("How did hyperscalers react to export control announcements?", "Multi-Source"),
            ("Explain the event study methodology and its limitations", "Methodology"),
        ]
        for i, (q, tag) in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"{tag}  |  {q}", key=f"suggest_{i}", width="stretch"):
                    st.session_state.pending_question = q
                    st.rerun()

    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        user_input = st.chat_input("Ask about export controls, CAR analysis, volatility, ecosystem...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="\U0001F464"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="N"):
            with st.spinner("Retrieving context & executing tools..."):
                try:
                    from core.chat.agent import run_agent
                    history = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.messages[:-1][-10:]]
                    result = run_agent(user_input, history)
                    response_text = result["response"]
                    st.markdown(response_text)

                    if result["tools_called"] or result["citations"]:
                        with st.expander("Evidence & Sources", expanded=False):
                            if result["citations"]:
                                st.markdown("**Retrieved Documents:**")
                                for cid in result["citations"][:5]:
                                    st.code(cid, language=None)
                            if result["tools_called"]:
                                st.markdown("**Tools Executed:**")
                                for tc in result["tools_called"]:
                                    st.code(
                                        f"{tc['tool']}({json.dumps(tc['arguments'], default=str)[:120]})",
                                        language="python",
                                    )

                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.agent_logs.append(result)
                except ImportError:
                    st.warning("Chat requires `openai` and `chromadb`. Run: `pip install openai chromadb`")
                    st.session_state.messages.append({"role": "assistant", "content": "Dependencies missing."})
                except Exception as e:
                    st.error(f"Error: {str(e)[:300]}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)[:200]}"})

    with st.sidebar:
        st.markdown("### Chat Controls")
        if st.button("Clear Conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.agent_logs = []
            st.rerun()
        if st.session_state.agent_logs:
            with st.expander("Agent Telemetry"):
                for i, log in enumerate(st.session_state.agent_logs):
                    cites = len(log.get("citations", []))
                    tools = len(log.get("tools_called", []))
                    tool_names = ", ".join(t["tool"] for t in log.get("tools_called", [])) or "none"
                    st.markdown(
                        f"**Turn {i+1}** — {cites} citations, {tools} tools (`{tool_names}`)"
                    )

# ============================================================
# TAB 1: EVENT DATABASE
# ============================================================

with tab1:
    st.markdown('<div class="section-header">BIS Export Control Event Database &middot; 2022 – 2026</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Context:</strong> Since August 2022, the U.S. Bureau of Industry and Security (BIS) has
        enacted a series of increasingly restrictive semiconductor export controls targeting China. These
        regulations limit the sale of advanced AI-capable chips (e.g., NVIDIA A100/H100), chip-making
        equipment (ASML lithography systems), and related technologies. This database tracks each major
        regulatory action and its observed 5-day market reaction on NVDA and AMD equities. Events are
        classified by <strong>severity</strong> (Low / Medium / High / Critical) and
        <strong>type</strong> (Initial Restriction, Expansion, Enforcement, Investigation, Allied Coordination).
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Event Filters")
        date_range = st.date_input(
            "Date Range",
            value=(events_df["date"].min(), events_df["date"].max()),
            min_value=events_df["date"].min(),
            max_value=events_df["date"].max(),
            key="events_date_range",
        )
        severity_options = ["All"] + [s.value for s in Severity]
        selected_severity = st.selectbox("Severity", severity_options, key="severity_filter")
        type_options = ["All"] + [t.value for t in EventType]
        selected_type = st.selectbox("Event Type", type_options, key="type_filter")

    filtered_df = events_df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(date_range[0])) &
            (filtered_df["date"] <= pd.to_datetime(date_range[1]))
        ]
    if selected_severity != "All":
        filtered_df = filtered_df[filtered_df["severity"] == selected_severity]
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["event_type"] == selected_type]

    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Events", len(filtered_df))
    with m2:
        avg_nvda = filtered_df['nvda_reaction_pct'].mean()
        st.metric("Avg NVDA Impact", f"{avg_nvda:+.1f}%")
    with m3:
        avg_amd = filtered_df['amd_reaction_pct'].mean()
        st.metric("Avg AMD Impact", f"{avg_amd:+.1f}%")
    with m4:
        n_critical = len(filtered_df[filtered_df["severity"] == "Critical"])
        st.metric("Critical Events", n_critical)

    # Computed observations
    neg_events = len(filtered_df[filtered_df['nvda_reaction_pct'] < 0])
    worst_event = filtered_df.loc[filtered_df['nvda_reaction_pct'].idxmin()] if len(filtered_df) > 0 else None
    best_event = filtered_df.loc[filtered_df['nvda_reaction_pct'].idxmax()] if len(filtered_df) > 0 else None

    if len(filtered_df) > 0:
        st.markdown(f"""
        <div class="obs-grid">
            <div class="obs-card">
                <div class="obs-label">Directional Bias</div>
                <strong>{neg_events}</strong> of <strong>{len(filtered_df)}</strong> events
                ({neg_events/len(filtered_df)*100:.0f}%) produced negative NVDA returns within 5 trading days,
                suggesting export controls carry a {'predominantly negative' if neg_events > len(filtered_df)/2 else 'mixed'} price signal for affected firms.
            </div>
            <div class="obs-card">
                <div class="obs-label">Severity Asymmetry</div>
                Critical events average <strong>{filtered_df[filtered_df['severity']=='Critical']['nvda_reaction_pct'].mean():+.1f}%</strong>
                NVDA impact vs. <strong>{filtered_df[filtered_df['severity']!='Critical']['nvda_reaction_pct'].mean():+.1f}%</strong> for
                non-critical events — {'confirming' if abs(filtered_df[filtered_df['severity']=='Critical']['nvda_reaction_pct'].mean()) > abs(filtered_df[filtered_df['severity']!='Critical']['nvda_reaction_pct'].mean()) else 'contradicting the expectation'} that severity classification predicts magnitude.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Timeline chart
    fig_timeline = go.Figure()
    for severity in ["Low", "Medium", "High", "Critical"]:
        sdf = filtered_df[filtered_df["severity"] == severity]
        if len(sdf) > 0:
            fig_timeline.add_trace(go.Scatter(
                x=sdf["date"], y=sdf["nvda_reaction_pct"], mode="markers",
                name=severity,
                marker=dict(
                    size=14 if severity in ["Critical", "High"] else 10,
                    color=SEVERITY_COLORS[severity],
                    line=dict(width=1.5, color="#ffffff"),
                    symbol="diamond" if severity == "Critical" else "circle",
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>NVDA: %{y:+.1f}%<extra></extra>",
                customdata=sdf[["title"]].values,
            ))
    fig_timeline.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Export Control Events vs NVDA 5-Day Reaction", **TITLE_STYLE),
        xaxis_title="", yaxis_title="NVDA Reaction (%)",
        height=420,
    )
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="#cbd5e1", line_width=1)
    st.plotly_chart(fig_timeline, width="stretch")

    st.markdown("""
    <div class="callout callout-method">
        <strong>Chart Interpretation:</strong> Each marker represents a BIS regulatory action. The y-axis
        shows NVDA's cumulative return over the 5 trading days following the announcement. Diamond markers
        denote Critical-severity events. Points below the zero line indicate negative price reactions.
        Marker size scales with severity — larger markers correspond to more impactful regulatory actions.
    </div>
    """, unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-header">Event Detail Table</div>', unsafe_allow_html=True)
    display_df = filtered_df[["date", "title", "severity", "event_type", "nvda_reaction_pct", "amd_reaction_pct"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Event", "Severity", "Type", "NVDA %", "AMD %"]
    st.dataframe(
        display_df, width="stretch", hide_index=True,
        column_config={
            "NVDA %": st.column_config.NumberColumn(format="%+.1f%%"),
            "AMD %": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

    # Takeaway
    if len(filtered_df) > 0 and worst_event is not None:
        st.markdown(f"""
        <div class="takeaway-box">
            <div class="tk-title">Key Takeaways &middot; Event Database</div>
            <ul>
                <li>The most adverse event was <strong>{worst_event['title']}</strong> ({worst_event['date'].strftime('%Y-%m-%d')}),
                    which produced a <strong>{worst_event['nvda_reaction_pct']:+.1f}%</strong> NVDA reaction.</li>
                <li>The strongest positive reaction was <strong>{best_event['title']}</strong> ({best_event['date'].strftime('%Y-%m-%d')})
                    at <strong>{best_event['nvda_reaction_pct']:+.1f}%</strong>, suggesting markets sometimes view restrictions
                    as competitively beneficial or price in relief when outcomes are less severe than feared.</li>
                <li>NVDA and AMD show a <strong>{filtered_df[['nvda_reaction_pct','amd_reaction_pct']].corr().iloc[0,1]:.2f}</strong>
                    cross-sectional correlation in event reactions, indicating {'tightly coupled' if filtered_df[['nvda_reaction_pct','amd_reaction_pct']].corr().iloc[0,1] > 0.7 else 'moderately linked'} risk exposure.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2: PRICE REACTION
# ============================================================

with tab2:
    st.markdown('<div class="section-header">Price Performance &middot; Cumulative Abnormal Returns</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Overview:</strong> This section examines how semiconductor equities have performed relative
        to a chosen normalization date and quantifies abnormal returns around each export control event
        using the <strong>Market Model</strong> framework. The normalized chart shows indexed performance
        (base = 100), while the CAR analysis isolates returns that cannot be explained by broad market
        movements (SPY benchmark), providing a cleaner signal of event-specific impact.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading market data..."):
        try:
            prices = load_prices()
            data_loaded = True
        except Exception as e:
            st.error(f"Error: {e}")
            data_loaded = False

    if data_loaded and len(prices) > 0:
        with st.sidebar:
            st.markdown("### Price Settings")
            normalize_to = st.selectbox(
                "Normalize To",
                ["First Date", "ChatGPT Launch (2022-11-30)", "Oct 7 Controls (2022-10-07)"],
                index=1, key="normalize_to",
            )
            car_ticker = st.selectbox("CAR Ticker", ["NVDA", "AMD", "INTC"], key="car_ticker")
            win_start = st.slider("Window Start", -5, 0, -1, key="win_start")
            win_end = st.slider("Window End", 1, 10, 5, key="win_end")

        base_date = "2022-11-30" if "ChatGPT" in normalize_to else ("2022-10-07" if "Oct 7" in normalize_to else None)
        normalized = normalize_prices(prices, base_date)

        # Compute stats for observations
        latest_prices = normalized.iloc[-1]
        total_return_nvda = latest_prices.get("NVDA", 100) - 100
        total_return_amd = latest_prices.get("AMD", 100) - 100
        total_return_spy = latest_prices.get("SPY", 100) - 100

        fig = go.Figure()
        for t in ["NVDA", "AMD", "INTC", "SPY"]:
            if t in normalized.columns:
                fig.add_trace(go.Scatter(
                    x=normalized.index, y=normalized[t], name=t,
                    line=dict(
                        color=COLORS[t],
                        width=2.5 if t != "SPY" else 1.5,
                        dash="dot" if t == "SPY" else "solid",
                    ),
                ))
        for _, ev in events_df.iterrows():
            if ev["severity"] == "Critical":
                fig.add_vline(x=ev["date"], line_dash="dash", line_color="#c4b5fd", line_width=1)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"Normalized Price Performance (100 = {base_date or 'Start'})", **TITLE_STYLE),
            yaxis_title="Indexed Price", height=460,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="#cbd5e1", line_width=1)
        st.plotly_chart(fig, width="stretch")

        st.markdown(f"""
        <div class="obs-grid">
            <div class="obs-card">
                <div class="obs-label">Cumulative Returns Since Base</div>
                NVDA: <strong>{total_return_nvda:+.0f}%</strong> &nbsp;|&nbsp;
                AMD: <strong>{total_return_amd:+.0f}%</strong> &nbsp;|&nbsp;
                SPY: <strong>{total_return_spy:+.0f}%</strong><br>
                {'NVDA has dramatically outperformed the broad market despite repeated export restrictions, suggesting AI demand dominates regulatory headwinds.' if total_return_nvda > total_return_spy else 'NVDA has underperformed SPY, suggesting export restrictions have weighed on growth expectations.'}
            </div>
            <div class="obs-card">
                <div class="obs-label">Relative Outperformance</div>
                NVDA excess return over SPY: <strong>{total_return_nvda - total_return_spy:+.0f}pp</strong><br>
                NVDA excess return over AMD: <strong>{total_return_nvda - total_return_amd:+.0f}pp</strong><br>
                Purple dashed lines mark Critical-severity BIS events.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # CAR section
        st.markdown('<div class="section-header">Cumulative Abnormal Returns (CAR) Analysis</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="callout callout-method">
            <strong>Methodology:</strong> We estimate expected returns using the <strong>Market Model</strong>
            (R<sub>i,t</sub> = &alpha; + &beta; &middot; R<sub>m,t</sub> + &epsilon;) with a 120-day estimation
            window ending 10 days before each event. Abnormal returns are computed as AR<sub>t</sub> =
            R<sub>actual</sub> - R<sub>expected</sub>, then cumulated over the [{win_start}, +{win_end}] event window.
            Statistical significance is tested at the 95% level (|t| > 1.96). This isolates the
            event-specific price impact after removing the component attributable to market-wide movements.
        </div>
        """, unsafe_allow_html=True)

        car_results = load_car_results(prices, events_df, car_ticker, (win_start, win_end))
        car_df = car_to_dataframe(car_results)

        if len(car_df) > 0:
            avg_car = car_df['CAR (%)'].mean()
            n_sig = car_df['Significant'].sum()
            n_neg_car = len(car_df[car_df['CAR (%)'] < 0])
            worst_car = car_df['CAR (%)'].min()
            best_car = car_df['CAR (%)'].max()

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Average CAR", f"{avg_car:+.2f}%")
            with m2:
                st.metric("Significant Events", f"{n_sig} / {len(car_df)}")
            with m3:
                st.metric("Worst CAR", f"{worst_car:+.2f}%")
            with m4:
                st.metric("Best CAR", f"{best_car:+.2f}%")

            st.markdown("")

            fig_car = go.Figure()
            fig_car.add_trace(go.Bar(
                x=car_df["Event Date"], y=car_df["CAR (%)"],
                marker_color=[COLORS["negative"] if x < 0 else COLORS["positive"] for x in car_df["CAR (%)"]],
                marker_line=dict(width=0),
                hovertemplate="<b>%{customdata}</b><br>CAR: %{y:+.2f}%<extra></extra>",
                customdata=car_df["Event"],
            ))
            fig_car.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"{car_ticker} Cumulative Abnormal Return by Event [{win_start}, +{win_end}]", **TITLE_STYLE),
                yaxis_title="CAR (%)", height=380,
            )
            fig_car.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
            st.plotly_chart(fig_car, width="stretch")

            st.markdown(f"""
            <div class="takeaway-box">
                <div class="tk-title">Key Takeaways &middot; Price Reaction & CAR</div>
                <ul>
                    <li>The average CAR for {car_ticker} across all events is <strong>{avg_car:+.2f}%</strong>,
                        {'indicating a net negative abnormal reaction to export control announcements' if avg_car < 0 else 'suggesting markets have, on average, absorbed these events without sustained abnormal losses'}.</li>
                    <li><strong>{n_sig} of {len(car_df)}</strong> events produced statistically significant abnormal returns
                        (|t| > 1.96), {'a majority — confirming that export controls are material information events' if n_sig > len(car_df)/2 else 'a minority — suggesting many events were anticipated or offset by other factors'}.</li>
                    <li><strong>{n_neg_car} of {len(car_df)}</strong> events show negative CAR, while <strong>{len(car_df) - n_neg_car}</strong>
                        show positive CAR. This mixed picture may reflect varying market expectations — events perceived as
                        less restrictive than feared can trigger relief rallies.</li>
                    <li>The [{win_start}, +{win_end}] window captures {'pre-event leakage and post-event drift' if win_start < 0 else 'only post-announcement returns'}.
                        Adjusting the window in the sidebar can reveal whether abnormal returns concentrate on the announcement
                        day or develop over subsequent sessions.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 3: ECOSYSTEM ANALYSIS
# ============================================================

with tab3:
    st.markdown('<div class="section-header">Semiconductor Ecosystem &middot; Competitive Dynamics</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Overview:</strong> Export controls do not affect all semiconductor firms equally. This analysis
        examines the competitive dynamics between NVDA (GPU leader, primary target), AMD (direct competitor,
        partially affected), and INTC (legacy manufacturer, potential beneficiary via domestic fab incentives).
        By comparing event-window returns, rolling correlations, and relative price strength, we can assess
        whether restrictions create competitive redistribution or systemic sector-wide repricing.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading ecosystem data..."):
        try:
            prices = load_prices()
            data_loaded = True
        except Exception:
            data_loaded = False

    if data_loaded and len(prices) > 0:
        with st.sidebar:
            st.markdown("### Ecosystem Settings")
            cwin_start = st.slider("Window Start", -5, 0, -1, key="cwin_start")
            cwin_end = st.slider("Window End", 1, 10, 5, key="cwin_end")
            corr_win = st.slider("Correlation Window (days)", 10, 60, 30, key="corr_win")

        comp_results = load_competitor_results(prices, events_df, (cwin_start, cwin_end))
        comp_df = competitor_to_dataframe(comp_results)
        comp_summary = get_competitive_summary(comp_results)

        nvda_wins = comp_summary.get('nvda_wins', 0)
        amd_wins = comp_summary.get('amd_wins', 0)
        intc_wins = comp_summary.get('intc_wins', 0)
        spread = comp_summary.get('nvda_vs_amd_avg', 0)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("NVDA Wins", f"{nvda_wins} / {len(comp_df)}")
        with m2:
            st.metric("AMD Wins", f"{amd_wins} / {len(comp_df)}")
        with m3:
            st.metric("INTC Wins", f"{intc_wins} / {len(comp_df)}")
        with m4:
            st.metric("NVDA-AMD Spread", f"{spread:+.1f}%")

        if len(comp_df) > 0:
            st.markdown(f"""
            <div class="obs-grid">
                <div class="obs-card">
                    <div class="obs-label">Competitive Shift Signal</div>
                    {'AMD outperforms NVDA in more event windows, suggesting the market views AMD as a relative beneficiary of NVDA-targeted restrictions.' if amd_wins > nvda_wins else 'NVDA outperforms AMD in more event windows despite being the primary restriction target — indicating strong demand resilience or market confidence in workaround strategies.'}
                    The average NVDA-AMD spread of <strong>{spread:+.1f}%</strong> {'favors AMD' if spread < 0 else 'favors NVDA'} per event.
                </div>
                <div class="obs-card">
                    <div class="obs-label">INTC Positioning</div>
                    Intel "wins" (smallest loss or best gain) in <strong>{intc_wins}</strong> event windows.
                    {'As a domestically-focused manufacturer potentially benefiting from reshoring incentives (CHIPS Act), INTC sometimes decouples from the restriction narrative.' if intc_wins >= 2 else 'INTC rarely outperforms peers around export control events, suggesting minimal competitive benefit from restrictions.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        fig_comp = go.Figure()
        for ticker, color in [("NVDA", COLORS["NVDA"]), ("AMD", COLORS["AMD"]), ("INTC", COLORS["INTC"])]:
            fig_comp.add_trace(go.Bar(
                name=ticker, x=comp_df["Event Date"], y=comp_df[f"{ticker} %"],
                marker_color=color, marker_line=dict(width=0),
            ))
        fig_comp.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            title=dict(text=f"Event-Window Returns: NVDA vs AMD vs INTC [{cwin_start}, +{cwin_end}]", **TITLE_STYLE),
            yaxis_title="Return (%)", height=420,
        )
        fig_comp.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
        st.plotly_chart(fig_comp, width="stretch")

        st.markdown("""
        <div class="callout callout-method">
            <strong>Reading the chart:</strong> Each cluster of three bars represents one export control event.
            Green (NVDA), Red (AMD), and Blue (INTC) show the raw return for each ticker over the specified
            event window. When NVDA's bar is more negative than AMD's, the market is differentially penalizing
            the primary target. When all three move together, the event is repricing the sector broadly.
        </div>
        """, unsafe_allow_html=True)

        # Correlation + Relative Strength
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Rolling Return Correlation</div>', unsafe_allow_html=True)
            correlation = calculate_rolling_correlation(prices, "NVDA", "AMD", corr_win)
            avg_corr = correlation.mean()
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=correlation.index, y=correlation.values, name="NVDA-AMD",
                line=dict(color=BLUE_ACCENT, width=2),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
            ))
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"NVDA-AMD {corr_win}-Day Return Correlation", **TITLE_STYLE),
                yaxis_title="Correlation", height=360,
            )
            fig_corr.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
            st.plotly_chart(fig_corr, width="stretch")
            st.caption(
                f"Average correlation: **{avg_corr:.2f}**. "
                f"{'High correlation suggests both stocks respond similarly to macro/sector factors, making differentiation around events more meaningful.' if avg_corr > 0.6 else 'Moderate correlation allows for meaningful competitive divergence analysis around events.'}"
            )

        with col2:
            st.markdown('<div class="section-header">Relative Price Strength</div>', unsafe_allow_html=True)
            relative = calculate_relative_strength(prices, "NVDA", "AMD")
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=relative.index, y=relative.values, name="NVDA/AMD",
                line=dict(color=NAVY_LIGHT, width=2),
                fill="tozeroy", fillcolor="rgba(26,39,68,0.06)",
            ))
            fig_rel.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="NVDA / AMD Price Ratio (Relative Strength)", **TITLE_STYLE),
                yaxis_title="Ratio", height=360,
            )
            st.plotly_chart(fig_rel, width="stretch")
            ratio_start = relative.iloc[0] if len(relative) > 0 else 1
            ratio_end = relative.iloc[-1] if len(relative) > 0 else 1
            ratio_change = (ratio_end / ratio_start - 1) * 100
            st.caption(
                f"Ratio moved from **{ratio_start:.2f}** to **{ratio_end:.2f}** ({ratio_change:+.0f}%). "
                f"{'A rising ratio means NVDA is outpacing AMD — export controls have not eroded its competitive premium.' if ratio_change > 0 else 'A falling ratio means AMD is gaining ground on NVDA, potentially benefiting from restriction-driven market share shifts.'}"
            )

        # Takeaway
        st.markdown(f"""
        <div class="takeaway-box">
            <div class="tk-title">Key Takeaways &middot; Ecosystem Analysis</div>
            <ul>
                <li>Export controls produce {'sector-wide' if abs(spread) < 1 else 'differentiated'} repricing across
                    GPU competitors. The average NVDA-AMD event-window spread is <strong>{spread:+.1f}%</strong>.</li>
                <li>The NVDA-AMD return correlation averages <strong>{avg_corr:.2f}</strong>, meaning
                    {'the two stocks largely move together, limiting pure competitive transfer' if avg_corr > 0.7 else 'there is room for competitive divergence around policy events'}.</li>
                <li>The NVDA/AMD price ratio has {'expanded' if ratio_change > 0 else 'compressed'} by
                    <strong>{abs(ratio_change):.0f}%</strong> over the period, indicating
                    {'NVDA maintains its competitive premium despite being the primary restriction target' if ratio_change > 0 else 'AMD is narrowing the gap, possibly capturing diverted demand'}.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 4: VOLATILITY SURFACE
# ============================================================

with tab4:
    st.markdown('<div class="section-header">Options Implied Volatility &middot; Event Risk Premium Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Overview:</strong> Implied volatility (IV) extracted from options prices is a forward-looking
        measure of expected risk. Unlike historical volatility (backward-looking), IV embeds the market's
        collective expectation of future price moves, including the probability of tail events like new
        export restrictions. This section analyzes the <strong>volatility surface</strong> (IV across strikes
        and expirations), <strong>skew</strong> (the premium for downside protection), <strong>term structure</strong>
        (how risk expectations vary by horizon), and <strong>historical realized volatility</strong> for context.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Options Settings")
        vol_ticker = st.selectbox("Ticker", ["NVDA", "AMD", "INTC"], key="vol_ticker")
        option_type = st.selectbox("Option Type", ["put", "call"], key="option_type")

    with st.spinner(f"Fetching {vol_ticker} options chain..."):
        try:
            calls, puts, spot_price = load_options_data(vol_ticker)
            options_loaded = True
        except Exception as e:
            st.info(f"Live options chain unavailable — showing historical volatility. ({str(e)[:80]})")
            options_loaded = False

    if options_loaded:
        st.markdown(
            f'<div class="spot-banner">'
            f'<span class="label">Spot Price</span>'
            f'<span class="price">${spot_price:.2f}</span>'
            f'<span class="meta">Calls: {len(calls)} &middot; Puts: {len(puts)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        options_df = puts if option_type == "put" else calls
        vol_surface = build_vol_surface(options_df, spot_price, option_type)

        if not vol_surface.empty:
            skew_30 = calculate_skew(vol_surface, 30)
            skew_60 = calculate_skew(vol_surface, 60)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("ATM IV (30d)", f"{skew_30['atm_vol']:.1f}%")
            with m2:
                st.metric("OTM Put IV (30d)", f"{skew_30['otm_put_vol']:.1f}%")
            with m3:
                st.metric("Skew (30d)", f"{skew_30['skew']:+.1f}%",
                           delta="Put premium" if skew_30['skew'] > 0 else "Normal")
            with m4:
                st.metric("Skew (60d)", f"{skew_60['skew']:+.1f}%")

            skew_val = skew_30['skew']
            atm_val = skew_30['atm_vol']
            st.markdown(f"""
            <div class="obs-grid">
                <div class="obs-card">
                    <div class="obs-label">Skew Interpretation</div>
                    The 30-day skew of <strong>{skew_val:+.1f}%</strong> means OTM puts trade at
                    {'a significant premium to ATM options — indicating elevated demand for downside hedging, consistent with institutional positioning ahead of potential regulatory actions.' if skew_val > 3 else 'a modest premium to ATM — within normal ranges, suggesting no exceptional event hedging demand at present.' if skew_val > 0 else 'a discount to ATM — unusual and may indicate call-skew from speculative upside positioning.'}
                </div>
                <div class="obs-card">
                    <div class="obs-label">Absolute Volatility Level</div>
                    ATM 30-day IV of <strong>{atm_val:.1f}%</strong> annualized
                    {'is elevated (>50%), pricing in high uncertainty — this often precedes or follows major regulatory announcements.' if atm_val > 50 else 'is within typical range for a high-growth semiconductor name.' if atm_val > 30 else 'is relatively low, suggesting the market sees limited near-term event risk.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3D Surface
            st.markdown('<div class="section-header">3D Implied Volatility Surface</div>', unsafe_allow_html=True)

            fig_3d = go.Figure(data=[go.Scatter3d(
                x=vol_surface['moneyness'],
                y=vol_surface['days_to_expiry'],
                z=vol_surface['implied_vol'],
                mode='markers',
                marker=dict(
                    size=4, color=vol_surface['implied_vol'],
                    colorscale=[[0, "#2563eb"], [0.5, "#f59e0b"], [1, "#dc2626"]],
                    colorbar=dict(title="IV %", thickness=15, len=0.6),
                    opacity=0.85, line=dict(width=0),
                ),
                hovertemplate="Moneyness: %{x:.1f}%<br>DTE: %{y}<br>IV: %{z:.1f}%<extra></extra>",
            )])
            fig_3d.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                title=dict(text=f"{vol_ticker} {option_type.upper()} Implied Volatility Surface",
                           font=dict(size=14, color=NAVY, family="Inter")),
                scene=dict(
                    xaxis=dict(title="Moneyness (%)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
                    yaxis=dict(title="Days to Expiry", backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
                    zaxis=dict(title="Implied Vol (%)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                ),
                font=dict(family="Inter", color=NAVY_LIGHT),
                height=580, margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_3d, width="stretch")

            st.markdown("""
            <div class="callout callout-method">
                <strong>Surface Interpretation:</strong> The x-axis (moneyness) shows how far a strike is
                from spot — negative values are out-of-the-money puts (downside protection). The y-axis shows
                days to expiry. The z-axis (color) shows implied volatility. A well-functioning market shows
                a "smile" shape across strikes and typically upward-sloping term structure. Elevated IV in the
                near-term, low-moneyness region signals institutional demand for short-dated crash protection.
            </div>
            """, unsafe_allow_html=True)

            # Heatmap
            st.markdown('<div class="section-header">Volatility Heatmap</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([2.5, 1])
            with col1:
                pivot_df = vol_surface.pivot_table(
                    values='implied_vol',
                    index=pd.cut(vol_surface['moneyness'], bins=[-25, -15, -10, -5, 0, 5, 10, 15, 25]),
                    columns=pd.cut(vol_surface['days_to_expiry'], bins=[0, 14, 30, 60, 90, 180, 365]),
                    aggfunc='mean',
                )
                if not pivot_df.empty:
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=[str(c) for c in pivot_df.columns],
                        y=[str(i) for i in pivot_df.index],
                        colorscale=[[0, "#dbeafe"], [0.5, "#fde68a"], [1, "#fca5a5"]],
                        colorbar=dict(title="IV %", thickness=12),
                        hovertemplate="Moneyness: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>",
                    ))
                    fig_heat.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(text="IV Heatmap — Strike Moneyness x Days to Expiry", **TITLE_STYLE),
                        xaxis_title="Days to Expiry", yaxis_title="Moneyness (%)", height=420,
                    )
                    st.plotly_chart(fig_heat, width="stretch")

            with col2:
                st.markdown("""
                **Reading the Heatmap**

                - **Red/warm zones** — High IV. Expensive options reflecting elevated risk expectations.
                - **Blue/cool zones** — Low IV. Cheaper options, lower perceived risk.
                - **Bottom-left concentration** — Near-term OTM puts showing high IV is the classic "event premium" pattern. Institutional hedgers buy short-dated downside protection ahead of anticipated regulatory actions.

                **Export Control Signal:** When the bottom-left corner is significantly warmer than the rest of the surface, the market is pricing in near-term tail risk from potential BIS announcements.
                """)

            # Term structure + Smile
            term_structure = calculate_term_structure(vol_surface)
            smile_data = vol_surface[
                (vol_surface['days_to_expiry'] >= 20) & (vol_surface['days_to_expiry'] <= 40)
            ].sort_values('moneyness')

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">ATM Term Structure</div>', unsafe_allow_html=True)
                if not term_structure.empty:
                    fig_term = go.Figure()
                    fig_term.add_trace(go.Scatter(
                        x=term_structure['days_to_expiry'], y=term_structure['implied_vol'],
                        mode='lines+markers', name='ATM IV',
                        line=dict(color=BLUE_ACCENT, width=2.5),
                        marker=dict(size=8, color=BLUE_ACCENT, line=dict(width=1.5, color="#ffffff")),
                    ))
                    fig_term.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(text="ATM Implied Volatility by Expiry", **TITLE_STYLE),
                        xaxis_title="Days to Expiry", yaxis_title="IV (%)", height=380,
                    )
                    st.plotly_chart(fig_term, width="stretch")

                    if len(term_structure) >= 2:
                        short_term = term_structure[term_structure['days_to_expiry'] <= 30]['implied_vol'].mean()
                        long_term = term_structure[term_structure['days_to_expiry'] > 60]['implied_vol'].mean()
                        if short_term > long_term * 1.1:
                            st.metric("Structure", "Inverted (Backwardation)")
                            st.caption("Short-term IV exceeds long-term IV. This inversion typically signals that the market is pricing a specific near-term catalyst — often a pending regulatory announcement or earnings event.")
                        elif long_term > short_term * 1.1:
                            st.metric("Structure", "Contango (Normal)")
                            st.caption("Long-term IV exceeds short-term. This is the normal state — uncertainty grows with time horizon. No immediate event premium detected in the term structure.")
                        else:
                            st.metric("Structure", "Flat")
                            st.caption("Relatively flat term structure. The market sees similar risk across horizons, with no strong near-term event signal.")

            with col2:
                st.markdown('<div class="section-header">30-Day Volatility Smile</div>', unsafe_allow_html=True)
                if not smile_data.empty:
                    fig_smile = go.Figure()
                    fig_smile.add_trace(go.Scatter(
                        x=smile_data['moneyness'], y=smile_data['implied_vol'],
                        mode='markers+lines', name='30d IV',
                        line=dict(color="#7c3aed", width=2),
                        marker=dict(size=6, color="#7c3aed", line=dict(width=1.5, color="#ffffff")),
                    ))
                    fig_smile.add_vline(x=0, line_dash="dash", line_color="#cbd5e1")
                    fig_smile.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(text="IV Smile — 30-Day Expiry Cross-Section", **TITLE_STYLE),
                        xaxis_title="Moneyness (%) — Negative = OTM Puts",
                        yaxis_title="IV (%)", height=380,
                    )
                    st.plotly_chart(fig_smile, width="stretch")
                    st.caption("The \"smile\" shape reflects higher IV for strikes far from spot (both puts and calls). A steeper left side (negative moneyness) indicates strong demand for downside hedging — the defining feature of export control event premium.")

        else:
            st.warning("Could not build volatility surface from available options data.")

    # Historical vol (always shown)
    st.markdown('<div class="section-header">Historical Realized Volatility</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-method">
        <strong>Methodology:</strong> Realized volatility is computed as the annualized standard deviation of
        daily log returns over rolling 20-day and 60-day windows. Unlike implied volatility (forward-looking),
        realized vol measures what <em>actually happened</em>. Comparing the two reveals whether the market
        over- or under-estimated risk — the "volatility risk premium."
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading historical volatility..."):
        hist_vol = load_historical_vol(vol_ticker)

    if not hist_vol.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_vol['date'], y=hist_vol['rv_20'], name='20-Day RV',
            line=dict(color="#dc2626", width=1.5),
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_vol['date'], y=hist_vol['rv_60'], name='60-Day RV',
            line=dict(color=BLUE_ACCENT, width=1.5),
        ))
        for _, ev in events_df.iterrows():
            if ev["severity"] in ["Critical", "High"]:
                fig_hist.add_vline(x=ev["date"], line_dash="dash", line_color="#c4b5fd", line_width=1)
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"{vol_ticker} Historical Realized Volatility (purple lines = Critical/High events)", **TITLE_STYLE),
            xaxis_title="", yaxis_title="Annualized Volatility (%)", height=400,
        )
        st.plotly_chart(fig_hist, width="stretch")

        current_rv20 = hist_vol['rv_20'].iloc[-1]
        current_rv60 = hist_vol['rv_60'].iloc[-1]
        avg_rv = hist_vol['rv_20'].mean()
        max_rv = hist_vol['rv_20'].max()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Current 20d RV", f"{current_rv20:.1f}%")
        with m2:
            st.metric("Current 60d RV", f"{current_rv60:.1f}%")
        with m3:
            st.metric("20d RV vs Average", f"{avg_rv:.1f}%", delta=f"{current_rv20 - avg_rv:+.1f}%")

        # Volatility takeaway
        st.markdown(f"""
        <div class="takeaway-box">
            <div class="tk-title">Key Takeaways &middot; Volatility Analysis</div>
            <ul>
                <li>Current 20-day realized vol of <strong>{current_rv20:.1f}%</strong> is
                    {'above' if current_rv20 > avg_rv else 'below'} the historical average of
                    <strong>{avg_rv:.1f}%</strong> — {'indicating heightened near-term price swings' if current_rv20 > avg_rv else 'suggesting a relatively calm period'}.</li>
                <li>The historical peak was <strong>{max_rv:.1f}%</strong>, typically coinciding with
                    major restriction announcements (visible as purple vertical lines on the chart).</li>
                <li>{'The 20d RV exceeds 60d RV, confirming a recent volatility spike that has not yet been smoothed by the longer window.' if current_rv20 > current_rv60 else 'The 60d RV exceeds 20d RV, suggesting the most recent period is calmer than the trailing two months.'}</li>
                <li>Comparing implied vol (ATM 30d IV{': ' + str(round(skew_30["atm_vol"], 1)) + '%' if options_loaded and not vol_surface.empty else ''})
                    against realized vol ({current_rv20:.1f}%) reveals
                    {'a positive volatility risk premium — the market expects more volatility than has recently materialized, consistent with embedded event-risk pricing.' if options_loaded and not vol_surface.empty and skew_30['atm_vol'] > current_rv20 else 'the relative positioning of forward vs. backward-looking risk measures.'}
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div class="pro-footer">
    <strong>NEXUS v2</strong> &middot; AI-Powered Semiconductor Export Control Research Platform<br>
    Purdue University &middot; Daniels School of Business &middot; MGMT 69000: Mastering AI for Finance &middot; Prof. Cinder Zhang<br>
    Data: Bureau of Industry & Security, SEC EDGAR, Yahoo Finance &middot; Methodology: Market Model CAR (120-day estimation window), Black-Scholes Implied Volatility, ChromaDB Vector Retrieval + OpenAI GPT-4o-mini<br>
    Built with Streamlit, Plotly, ChromaDB, and OpenAI API &middot; All analyses are for educational purposes only and do not constitute investment advice.
</div>
""", unsafe_allow_html=True)

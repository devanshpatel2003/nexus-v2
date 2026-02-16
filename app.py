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

# Import data modules
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
# INSTITUTIONAL CSS — White / Navy Blue
# ============================================================

NAVY = "#0f1d33"
NAVY_LIGHT = "#1a2744"
NAVY_MID = "#2a3a5c"
BLUE_ACCENT = "#2563eb"
SLATE = "#475569"
BORDER = "#e2e8f0"
BG_SUBTLE = "#f8fafc"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* Hide default branding */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Main container */
.block-container {{
    padding-top: 0.8rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}}

/* ── Brand Header ── */
.brand-header {{
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
    margin: -1rem -1rem 1.5rem -1rem;
    padding: 1.2rem 2rem;
    border-bottom: 3px solid {BLUE_ACCENT};
}}
.brand-header .logo {{
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
}}
.brand-header .logo span {{
    color: {BLUE_ACCENT};
    font-weight: 800;
}}
.brand-header .tagline {{
    font-size: 0.7rem;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 2px;
}}

/* ── Metric Cards ── */
[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    border-left: 3px solid {NAVY_LIGHT};
}}

[data-testid="stMetricLabel"] {{
    font-size: 0.68rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {SLATE} !important;
}}

[data-testid="stMetricValue"] {{
    font-size: 1.4rem !important;
    font-weight: 700;
    color: {NAVY} !important;
    letter-spacing: -0.02em;
}}

[data-testid="stMetricDelta"] {{
    font-size: 0.72rem !important;
}}

/* ── Tab Navigation ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {BG_SUBTLE};
    border-radius: 8px;
    padding: 3px;
    border: 1px solid {BORDER};
}}

.stTabs [data-baseweb="tab"] {{
    height: 42px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.82rem;
    letter-spacing: 0.01em;
    color: {SLATE};
    padding: 0 18px;
    white-space: nowrap;
}}

.stTabs [aria-selected="true"] {{
    background: {NAVY_LIGHT} !important;
    color: #ffffff !important;
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(15,29,51,0.2);
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}}

section[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.85) !important;
}}

section[data-testid="stSidebar"] label {{
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.75rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.12);
    color: #ffffff;
}}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.4) !important;
    margin-top: 1.4rem;
}}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {{
    border-radius: 10px;
    border: 1px solid {BORDER};
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    background: #ffffff;
}}

/* ── Buttons ── */
.stButton > button {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.8rem;
    transition: all 0.15s ease;
    background: #ffffff;
    color: {NAVY_LIGHT};
}}

.stButton > button:hover {{
    border-color: {NAVY_LIGHT};
    background: {BG_SUBTLE};
    box-shadow: 0 1px 4px rgba(15,29,51,0.08);
    color: {NAVY};
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    font-size: 0.8rem;
    font-weight: 500;
    color: {SLATE};
}}

/* ── Dividers ── */
hr {{
    border-color: {BORDER} !important;
    margin: 1.5rem 0 !important;
}}

/* ── Section Headers ── */
.section-header {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {NAVY_LIGHT};
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {NAVY_LIGHT};
    display: block;
}}

/* ── Severity Badges ── */
.badge-critical {{ background: #faf5ff; color: #7c3aed; padding: 3px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; border: 1px solid #e9d5ff; }}
.badge-high {{ background: #fef2f2; color: #dc2626; padding: 3px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; border: 1px solid #fecaca; }}
.badge-medium {{ background: #fffbeb; color: #d97706; padding: 3px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; border: 1px solid #fde68a; }}
.badge-low {{ background: #f0fdf4; color: #16a34a; padding: 3px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; border: 1px solid #bbf7d0; }}

/* ── Spot Price Banner ── */
.spot-banner {{
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
    border-radius: 8px;
    padding: 14px 22px;
    margin-bottom: 1.2rem;
    display: inline-block;
    border: 1px solid {NAVY_MID};
}}
.spot-banner .label {{
    color: rgba(255,255,255,0.5);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}}
.spot-banner .price {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin-left: 8px;
}}
.spot-banner .meta {{
    color: rgba(255,255,255,0.35);
    font-size: 0.72rem;
    margin-left: 12px;
}}

/* ── Footer ── */
.pro-footer {{
    text-align: center;
    font-size: 0.65rem;
    color: {SLATE};
    letter-spacing: 0.04em;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid {BORDER};
    margin-top: 3rem;
    line-height: 1.8;
}}

/* ── Sidebar status overrides ── */
section[data-testid="stSidebar"] [data-testid="stAlert"] {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 0.75rem;
}}

/* ── Info/warning/error boxes on main area ── */
.stAlert {{
    border-radius: 8px;
    font-size: 0.82rem;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME — Institutional Light
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
    <div class="logo"><span>N</span>EXUS <span>v2</span></div>
    <div class="tagline">AI-Powered Semiconductor Export Control Research</div>
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
# TAB: RESEARCH CHAT
# ============================================================

with tab_chat:
    st.markdown('<div class="section-header">Grounded Research Assistant</div>', unsafe_allow_html=True)
    st.caption(
        "Ask questions about export controls, event studies, volatility, "
        "or the semiconductor ecosystem. Answers are grounded in case materials and live market tools."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.messages:
        st.markdown("")
        st.markdown('<div class="section-header">Suggested Queries</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        suggestions = [
            ("What was the market impact of the October 7 export controls?", "RAG"),
            ("Run a CAR analysis for NVDA around Critical events", "Event Study"),
            ("Compare NVDA, TSM, and ASML performance since 2022", "Ecosystem"),
            ("What is the current volatility skew for NVDA puts?", "Volatility"),
            ("How did hyperscalers react to export control announcements?", "Ecosystem"),
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
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
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
# TAB 1: EVENT DATABASE
# ============================================================

with tab1:
    st.markdown('<div class="section-header">BIS Export Control Timeline &middot; 2022 – 2026</div>', unsafe_allow_html=True)

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

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Events", len(filtered_df))
    with m2:
        st.metric("Avg NVDA Impact", f"{filtered_df['nvda_reaction_pct'].mean():+.1f}%")
    with m3:
        st.metric("Avg AMD Impact", f"{filtered_df['amd_reaction_pct'].mean():+.1f}%")
    with m4:
        st.metric("Critical Events", len(filtered_df[filtered_df["severity"] == "Critical"]))

    st.markdown("")

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

# ============================================================
# TAB 2: PRICE REACTION
# ============================================================

with tab2:
    st.markdown('<div class="section-header">Normalized Price Performance &middot; CAR Analysis</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="section-header">Cumulative Abnormal Returns</div>', unsafe_allow_html=True)

        car_results = load_car_results(prices, events_df, car_ticker, (win_start, win_end))
        car_df = car_to_dataframe(car_results)

        if len(car_df) > 0:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Average CAR", f"{car_df['CAR (%)'].mean():+.2f}%")
            with m2:
                st.metric("Significant Events", f"{car_df['Significant'].sum()} / {len(car_df)}")
            with m3:
                st.metric("Worst CAR", f"{car_df['CAR (%)'].min():+.2f}%")
            with m4:
                st.metric("Best CAR", f"{car_df['CAR (%)'].max():+.2f}%")

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

# ============================================================
# TAB 3: ECOSYSTEM ANALYSIS
# ============================================================

with tab3:
    st.markdown('<div class="section-header">Competitive Dynamics &middot; Semiconductor Ecosystem</div>', unsafe_allow_html=True)

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

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("NVDA Wins", f"{comp_summary.get('nvda_wins', 0)} / {len(comp_df)}")
        with m2:
            st.metric("AMD Wins", f"{comp_summary.get('amd_wins', 0)} / {len(comp_df)}")
        with m3:
            st.metric("INTC Wins", f"{comp_summary.get('intc_wins', 0)} / {len(comp_df)}")
        with m4:
            st.metric("NVDA-AMD Spread", f"{comp_summary.get('nvda_vs_amd_avg', 0):+.1f}%")

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
            title=dict(text=f"Event-Window Returns [{cwin_start}, +{cwin_end}]", **TITLE_STYLE),
            yaxis_title="Return (%)", height=420,
        )
        fig_comp.add_hline(y=0, line_dash="dash", line_color="#cbd5e1")
        st.plotly_chart(fig_comp, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Rolling Correlation</div>', unsafe_allow_html=True)
            correlation = calculate_rolling_correlation(prices, "NVDA", "AMD", corr_win)
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

        with col2:
            st.markdown('<div class="section-header">Relative Strength</div>', unsafe_allow_html=True)
            relative = calculate_relative_strength(prices, "NVDA", "AMD")
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=relative.index, y=relative.values, name="NVDA/AMD",
                line=dict(color=NAVY_LIGHT, width=2),
                fill="tozeroy", fillcolor=f"rgba(26,39,68,0.06)",
            ))
            fig_rel.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="NVDA / AMD Price Ratio", **TITLE_STYLE),
                yaxis_title="Ratio", height=360,
            )
            st.plotly_chart(fig_rel, width="stretch")

# ============================================================
# TAB 4: VOLATILITY SURFACE
# ============================================================

with tab4:
    st.markdown('<div class="section-header">Options Implied Volatility &middot; Event Premium Analysis</div>', unsafe_allow_html=True)

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

            st.markdown("")

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
                title=dict(text=f"{vol_ticker} {option_type.upper()} Volatility Surface",
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
                        title=dict(text="IV Heatmap (Strike x Expiry)", **TITLE_STYLE),
                        xaxis_title="Days to Expiry", yaxis_title="Moneyness (%)", height=420,
                    )
                    st.plotly_chart(fig_heat, width="stretch")

            with col2:
                st.markdown("""
                **Interpreting the Heatmap**

                - **Red zones** — High IV, expensive options, elevated risk pricing
                - **Blue zones** — Low IV, cheaper options
                - **OTM put skew** (negative moneyness) typically shows higher IV due to crash hedging demand

                **Event Premium Signal:**
                Elevated OTM put IV before BIS announcements indicates institutional hedging activity.
                """)

            term_structure = calculate_term_structure(vol_surface)
            smile_data = vol_surface[
                (vol_surface['days_to_expiry'] >= 20) & (vol_surface['days_to_expiry'] <= 40)
            ].sort_values('moneyness')

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">Term Structure</div>', unsafe_allow_html=True)
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
                        title=dict(text="ATM Implied Volatility Term Structure", **TITLE_STYLE),
                        xaxis_title="Days to Expiry", yaxis_title="IV (%)", height=380,
                    )
                    st.plotly_chart(fig_term, width="stretch")

                    if len(term_structure) >= 2:
                        short_term = term_structure[term_structure['days_to_expiry'] <= 30]['implied_vol'].mean()
                        long_term = term_structure[term_structure['days_to_expiry'] > 60]['implied_vol'].mean()
                        if short_term > long_term * 1.1:
                            st.metric("Structure", "Inverted (Backwardation)")
                            st.info("Short-term IV > long-term. Market pricing near-term event risk.")
                        elif long_term > short_term * 1.1:
                            st.metric("Structure", "Contango")
                            st.info("Normal structure — no immediate event premium detected.")
                        else:
                            st.metric("Structure", "Flat")

            with col2:
                st.markdown('<div class="section-header">Volatility Smile</div>', unsafe_allow_html=True)
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
                        title=dict(text="30-Day Volatility Smile", **TITLE_STYLE),
                        xaxis_title="Moneyness (%) — Negative = OTM Puts",
                        yaxis_title="IV (%)", height=380,
                    )
                    st.plotly_chart(fig_smile, width="stretch")

        else:
            st.warning("Could not build volatility surface from available options data.")

    st.markdown('<div class="section-header">Historical Realized Volatility</div>', unsafe_allow_html=True)

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
            title=dict(text=f"{vol_ticker} Realized Volatility (purple lines = restriction events)", **TITLE_STYLE),
            xaxis_title="", yaxis_title="Annualized Vol (%)", height=400,
        )
        st.plotly_chart(fig_hist, width="stretch")

        m1, m2, m3 = st.columns(3)
        current_rv20 = hist_vol['rv_20'].iloc[-1]
        current_rv60 = hist_vol['rv_60'].iloc[-1]
        avg_rv = hist_vol['rv_20'].mean()
        with m1:
            st.metric("Current 20d RV", f"{current_rv20:.1f}%")
        with m2:
            st.metric("Current 60d RV", f"{current_rv60:.1f}%")
        with m3:
            st.metric("20d RV vs Average", f"{avg_rv:.1f}%", delta=f"{current_rv20 - avg_rv:+.1f}%")

# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div class="pro-footer">
    NEXUS v2 &middot; AI-Powered Semiconductor Export Control Research &middot; MGMT 69000 &middot; Purdue University<br>
    Data Sources: Bureau of Industry & Security, SEC Filings, Yahoo Finance<br>
    Methodology: Market Model CAR (120-day estimation) &middot; Black-Scholes IV &middot; ChromaDB RAG + OpenAI
</div>
""", unsafe_allow_html=True)

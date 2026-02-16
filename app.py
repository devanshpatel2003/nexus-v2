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
    page_title="NEXUS v2 — AI Finance Research Assistant",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# INSTITUTIONAL CSS
# ============================================================

st.markdown("""
<style>
/* ---- Global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---- Hide default Streamlit branding ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---- Main container ---- */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(118,185,0,0.06) 0%, rgba(30,30,60,0.4) 100%);
    border: 1px solid rgba(118,185,0,0.15);
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(250,250,250,0.55) !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}

[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
}

/* ---- Tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: rgba(15,15,30,0.5);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
    color: rgba(250,250,250,0.5);
    padding: 0 20px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(118,185,0,0.2) 0%, rgba(118,185,0,0.08) 100%) !important;
    color: #76b900 !important;
    border: 1px solid rgba(118,185,0,0.3);
    font-weight: 600;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #0e1225 100%);
    border-right: 1px solid rgba(118,185,0,0.1);
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(118,185,0,0.7);
    margin-top: 1.2rem;
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    overflow: hidden;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.04);
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}

/* ---- Buttons ---- */
.stButton > button {
    border: 1px solid rgba(118,185,0,0.25);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.8rem;
    transition: all 0.2s ease;
    background: rgba(118,185,0,0.06);
}

.stButton > button:hover {
    border-color: rgba(118,185,0,0.5);
    background: rgba(118,185,0,0.12);
    box-shadow: 0 0 20px rgba(118,185,0,0.08);
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    font-size: 0.82rem;
    font-weight: 500;
    color: rgba(250,250,250,0.65);
}

/* ---- Dividers ---- */
hr {
    border-color: rgba(255,255,255,0.04) !important;
    margin: 1.5rem 0 !important;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(118,185,0,0.6);
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(118,185,0,0.12);
}

/* ---- Badge pills ---- */
.badge-critical { background: rgba(155,89,182,0.2); color: #bb6bd9; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
.badge-high { background: rgba(231,76,60,0.2); color: #e74c3c; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
.badge-medium { background: rgba(243,156,18,0.2); color: #f39c12; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
.badge-low { background: rgba(46,204,113,0.2); color: #2ecc71; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }

/* ---- Logo / Brand Bar ---- */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.6rem 0 1rem 0;
    border-bottom: 1px solid rgba(118,185,0,0.12);
    margin-bottom: 1.2rem;
}
.brand-bar .logo {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #fafafa;
}
.brand-bar .logo span { color: #76b900; }
.brand-bar .tagline {
    font-size: 0.68rem;
    color: rgba(250,250,250,0.35);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ---- Chat suggestion cards ---- */
.suggestion-card {
    background: rgba(118,185,0,0.04);
    border: 1px solid rgba(118,185,0,0.12);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.suggestion-card:hover {
    background: rgba(118,185,0,0.1);
    border-color: rgba(118,185,0,0.3);
}

/* ---- Footer ---- */
.pro-footer {
    text-align: center;
    font-size: 0.65rem;
    color: rgba(250,250,250,0.2);
    letter-spacing: 0.05em;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.03);
    margin-top: 3rem;
}
.pro-footer a { color: rgba(118,185,0,0.4); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME (institutional dark)
# ============================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,25,0.4)",
    font=dict(family="Inter, sans-serif", size=12, color="rgba(250,250,250,0.7)"),
    title=dict(font=dict(size=14, color="rgba(250,250,250,0.85)"), x=0, xanchor="left"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=11, color="rgba(250,250,250,0.55)"),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left",
    ),
    margin=dict(l=50, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    hoverlabel=dict(bgcolor="rgba(20,20,40,0.95)", font_size=12, font_family="Inter"),
)

COLORS = {
    "NVDA": "#76b900", "AMD": "#ed1c24", "INTC": "#0071c5", "SPY": "#555555",
    "TSM": "#e8530e", "ASML": "#00a3e0", "AVGO": "#cc0000",
    "GOOGL": "#4285f4", "AMZN": "#ff9900", "MSFT": "#00a4ef",
    "positive": "#00d97e", "negative": "#e63757", "accent": "#76b900",
    "muted": "rgba(250,250,250,0.25)", "grid": "rgba(255,255,255,0.04)",
}

SEVERITY_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Critical": "#bb6bd9"}

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="brand-bar">
    <div>
        <div class="logo"><span>N</span>EXUS <span>v2</span></div>
        <div class="tagline">AI-Powered Semiconductor Export Control Research</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tab navigation
tab_chat, tab1, tab2, tab3, tab4 = st.tabs([
    "  Research Chat  ",
    "  Event Database  ",
    "  Price Reaction  ",
    "  Ecosystem Analysis  ",
    "  Volatility Surface  ",
])

# ============================================================
# SIDEBAR — Brand + Info
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid rgba(118,185,0,0.1); margin-bottom: 1rem;">
        <div style="font-size:1.3rem; font-weight:700; letter-spacing:-0.02em;">
            <span style="color:#76b900;">N</span>EXUS <span style="color:#76b900;">v2</span>
        </div>
        <div style="font-size:0.6rem; color:rgba(250,250,250,0.3); text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;">
            Purdue Daniels School of Business
        </div>
        <div style="font-size:0.55rem; color:rgba(250,250,250,0.2); margin-top:2px;">
            MGMT 69000 &middot; Mastering AI for Finance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
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
    st.markdown(
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
                if st.button(f"**{tag}** — {q}", key=f"suggest_{i}", use_container_width=True):
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
        st.markdown('<div class="section-header">Chat Controls</div>', unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True):
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
        st.markdown('<div class="section-header">Event Filters</div>', unsafe_allow_html=True)
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

    # Metrics row
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
                    line=dict(width=1, color="rgba(255,255,255,0.2)"),
                    symbol="diamond" if severity == "Critical" else "circle",
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>NVDA: %{y:+.1f}%<extra></extra>",
                customdata=sdf[["title"]].values,
            ))
    fig_timeline.update_layout(
        **PLOTLY_LAYOUT,
        title="Export Control Events vs NVDA 5-Day Reaction",
        xaxis_title="", yaxis_title="NVDA Reaction (%)",
        height=420,
    )
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Table
    display_df = filtered_df[["date", "title", "severity", "event_type", "nvda_reaction_pct", "amd_reaction_pct"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Event", "Severity", "Type", "NVDA %", "AMD %"]
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
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
            st.markdown('<div class="section-header">Price Settings</div>', unsafe_allow_html=True)
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

        # Price chart
        fig = go.Figure()
        for t in ["NVDA", "AMD", "INTC", "SPY"]:
            if t in normalized.columns:
                fig.add_trace(go.Scatter(
                    x=normalized.index, y=normalized[t], name=t,
                    line=dict(
                        color=COLORS[t],
                        width=2.5 if t != "SPY" else 1,
                        dash="dot" if t == "SPY" else "solid",
                    ),
                ))
        for _, ev in events_df.iterrows():
            if ev["severity"] == "Critical":
                fig.add_vline(x=ev["date"], line_dash="dash", line_color="rgba(187,107,217,0.3)", line_width=1)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Normalized Price Performance (100 = {base_date or 'Start'})",
            yaxis_title="Indexed Price", height=460,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.1)", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        # CAR section
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
                title=f"{car_ticker} Cumulative Abnormal Return by Event [{win_start}, +{win_end}]",
                yaxis_title="CAR (%)", height=380,
            )
            fig_car.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
            st.plotly_chart(fig_car, use_container_width=True)

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
            st.markdown('<div class="section-header">Ecosystem Settings</div>', unsafe_allow_html=True)
            cwin_start = st.slider("Window Start", -5, 0, -1, key="cwin_start")
            cwin_end = st.slider("Window End", 1, 10, 5, key="cwin_end")
            corr_win = st.slider("Correlation Window (days)", 10, 60, 30, key="corr_win")

        comp_results = load_competitor_results(prices, events_df, (cwin_start, cwin_end))
        comp_df = competitor_to_dataframe(comp_results)
        comp_summary = get_competitive_summary(comp_results)

        # Scorecard
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

        # Grouped bar
        fig_comp = go.Figure()
        for ticker, color in [("NVDA", COLORS["NVDA"]), ("AMD", COLORS["AMD"]), ("INTC", COLORS["INTC"])]:
            fig_comp.add_trace(go.Bar(
                name=ticker, x=comp_df["Event Date"], y=comp_df[f"{ticker} %"],
                marker_color=color, marker_line=dict(width=0),
            ))
        fig_comp.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            title=f"Event-Window Returns [{cwin_start}, +{cwin_end}]",
            yaxis_title="Return (%)", height=420,
        )
        fig_comp.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Two-column: correlation + relative strength
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Rolling Correlation</div>', unsafe_allow_html=True)
            correlation = calculate_rolling_correlation(prices, "NVDA", "AMD", corr_win)
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=correlation.index, y=correlation.values, name="NVDA-AMD",
                line=dict(color="#00a3e0", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,163,224,0.06)",
            ))
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                title=f"NVDA-AMD {corr_win}-Day Return Correlation",
                yaxis_title="Correlation", height=360,
            )
            fig_corr.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.1)")
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Relative Strength</div>', unsafe_allow_html=True)
            relative = calculate_relative_strength(prices, "NVDA", "AMD")
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=relative.index, y=relative.values, name="NVDA/AMD",
                line=dict(color=COLORS["NVDA"], width=1.5),
                fill="tozeroy", fillcolor="rgba(118,185,0,0.06)",
            ))
            fig_rel.update_layout(
                **PLOTLY_LAYOUT,
                title="NVDA / AMD Price Ratio",
                yaxis_title="Ratio", height=360,
            )
            st.plotly_chart(fig_rel, use_container_width=True)

# ============================================================
# TAB 4: VOLATILITY SURFACE
# ============================================================

with tab4:
    st.markdown('<div class="section-header">Options Implied Volatility &middot; Event Premium Analysis</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="section-header">Options Settings</div>', unsafe_allow_html=True)
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
        # Spot banner
        st.markdown(
            f'<div style="background:rgba(118,185,0,0.08); border:1px solid rgba(118,185,0,0.2); '
            f'border-radius:8px; padding:10px 18px; margin-bottom:1rem; display:inline-block;">'
            f'<span style="color:rgba(250,250,250,0.5); font-size:0.7rem; text-transform:uppercase; '
            f'letter-spacing:0.08em;">Spot Price</span> &nbsp; '
            f'<span style="font-size:1.3rem; font-weight:700; color:#76b900;">${spot_price:.2f}</span>'
            f'&nbsp;&nbsp; <span style="color:rgba(250,250,250,0.3); font-size:0.75rem;">'
            f'Calls: {len(calls)} &middot; Puts: {len(puts)}</span></div>',
            unsafe_allow_html=True,
        )

        options_df = puts if option_type == "put" else calls
        vol_surface = build_vol_surface(options_df, spot_price, option_type)

        if not vol_surface.empty:
            # Skew metrics
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

            # 3D Surface
            st.markdown('<div class="section-header">3D Implied Volatility Surface</div>', unsafe_allow_html=True)

            fig_3d = go.Figure(data=[go.Scatter3d(
                x=vol_surface['moneyness'],
                y=vol_surface['days_to_expiry'],
                z=vol_surface['implied_vol'],
                mode='markers',
                marker=dict(
                    size=4, color=vol_surface['implied_vol'],
                    colorscale=[[0, "#00d97e"], [0.5, "#f6c343"], [1, "#e63757"]],
                    colorbar=dict(title="IV %", thickness=15, len=0.6),
                    opacity=0.85, line=dict(width=0),
                ),
                hovertemplate="Moneyness: %{x:.1f}%<br>DTE: %{y}<br>IV: %{z:.1f}%<extra></extra>",
            )])
            fig_3d.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                title=dict(text=f"{vol_ticker} {option_type.upper()} Volatility Surface", font=dict(size=14)),
                scene=dict(
                    xaxis=dict(title="Moneyness (%)", backgroundcolor="rgba(10,10,25,0.3)", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Days to Expiry", backgroundcolor="rgba(10,10,25,0.3)", gridcolor="rgba(255,255,255,0.05)"),
                    zaxis=dict(title="Implied Vol (%)", backgroundcolor="rgba(10,10,25,0.3)", gridcolor="rgba(255,255,255,0.05)"),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                ),
                font=dict(family="Inter", color="rgba(250,250,250,0.7)"),
                height=580, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            # Heatmap + legend
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
                        colorscale=[[0, "#00d97e"], [0.5, "#f6c343"], [1, "#e63757"]],
                        colorbar=dict(title="IV %", thickness=12),
                        hovertemplate="Moneyness: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>",
                    ))
                    fig_heat.update_layout(
                        **PLOTLY_LAYOUT,
                        title="IV Heatmap (Strike x Expiry)",
                        xaxis_title="Days to Expiry", yaxis_title="Moneyness (%)", height=420,
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

            with col2:
                st.markdown("""
                **Interpreting the Heatmap**

                - **Red zones** — High IV, expensive options, elevated risk pricing
                - **Green zones** — Low IV, cheaper options
                - **OTM put skew** (negative moneyness) typically shows higher IV due to crash hedging demand

                **Event Premium Signal:**
                Elevated OTM put IV before BIS announcements indicates institutional hedging activity.
                """)

            # Term structure + Smile side by side
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
                        line=dict(color="#00a3e0", width=2.5),
                        marker=dict(size=8, line=dict(width=1, color="rgba(255,255,255,0.3)")),
                    ))
                    fig_term.update_layout(
                        **PLOTLY_LAYOUT,
                        title="ATM Implied Volatility Term Structure",
                        xaxis_title="Days to Expiry", yaxis_title="IV (%)", height=380,
                    )
                    st.plotly_chart(fig_term, use_container_width=True)

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
                        line=dict(color="#bb6bd9", width=2),
                        marker=dict(size=6, line=dict(width=1, color="rgba(255,255,255,0.2)")),
                    ))
                    fig_smile.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
                    fig_smile.update_layout(
                        **PLOTLY_LAYOUT,
                        title="30-Day Volatility Smile",
                        xaxis_title="Moneyness (%) — Negative = OTM Puts",
                        yaxis_title="IV (%)", height=380,
                    )
                    st.plotly_chart(fig_smile, use_container_width=True)

        else:
            st.warning("Could not build volatility surface from available options data.")

    # Historical vol (always shown)
    st.markdown('<div class="section-header">Historical Realized Volatility</div>', unsafe_allow_html=True)

    with st.spinner("Loading historical volatility..."):
        hist_vol = load_historical_vol(vol_ticker)

    if not hist_vol.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_vol['date'], y=hist_vol['rv_20'], name='20-Day RV',
            line=dict(color="#e63757", width=1.5),
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_vol['date'], y=hist_vol['rv_60'], name='60-Day RV',
            line=dict(color="#00a3e0", width=1.5),
        ))
        for _, ev in events_df.iterrows():
            if ev["severity"] in ["Critical", "High"]:
                fig_hist.add_vline(x=ev["date"], line_dash="dash", line_color="rgba(187,107,217,0.25)", line_width=1)
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{vol_ticker} Realized Volatility (purple lines = restriction events)",
            xaxis_title="", yaxis_title="Annualized Vol (%)", height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

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

st.markdown("""
<div class="pro-footer">
    NEXUS v2 &middot; AI-Powered Finance Research Assistant &middot; MGMT 69000 &middot; Purdue University<br>
    Data: BIS, SEC, Yahoo Finance &middot; CAR: Market Model (120d) &middot; Vol: Black-Scholes Implied &middot; RAG: ChromaDB + OpenAI
</div>
""", unsafe_allow_html=True)

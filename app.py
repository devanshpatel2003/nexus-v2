"""
NEXUS: Nvidia Export Controls Research & Trading System
Section 1: Export Control Event Database
Section 2: Event-Price Reaction Chart with CAR Analysis
Section 3: Competitor Comparison Panel
Section 4: Volatility Surface View
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import our modules
from data.export_control_events import (
    get_events_dataframe,
    get_event_summary,
    Severity,
    EventType,
)
from data.price_data import (
    get_combined_prices,
    normalize_prices,
)
from data.car_analysis import (
    run_multiple_event_studies,
    results_to_dataframe as car_to_dataframe,
)
from data.competitor_analysis import (
    analyze_competitive_shift,
    results_to_dataframe as competitor_to_dataframe,
    get_competitive_summary,
    calculate_rolling_correlation,
    calculate_relative_strength,
)
from data.options_data import (
    fetch_options_chain,
    build_vol_surface,
    get_vol_surface_matrix,
    calculate_skew,
    calculate_term_structure,
    get_historical_iv,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEXUS - Export Control Analysis",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HEADER & NAVIGATION
# ============================================================

st.title("🔒 NEXUS: Nvidia Export Controls Research System")

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Event Database",
    "📈 Price Reaction",
    "⚔️ Competitor Analysis",
    "🌊 Volatility Surface"
])

# ============================================================
# LOAD DATA (cached)
# ============================================================

@st.cache_data(ttl=3600)
def load_events():
    return get_events_dataframe()

@st.cache_data(ttl=3600)
def load_prices():
    return get_combined_prices(
        tickers=["NVDA", "AMD", "INTC", "SPY"],
        start_date="2022-01-01"
    )

@st.cache_data(ttl=3600)
def load_car_results(_prices, _events, ticker, window):
    return run_multiple_event_studies(
        prices=_prices, events=_events, ticker=ticker, event_window=window
    )

@st.cache_data(ttl=3600)
def load_competitor_results(_prices, _events, window):
    return analyze_competitive_shift(
        prices=_prices, events=_events,
        window_start=window[0], window_end=window[1]
    )

@st.cache_data(ttl=300)  # 5 min cache for options
def load_options_data(ticker):
    return fetch_options_chain(ticker)

@st.cache_data(ttl=3600)
def load_historical_vol(ticker):
    return get_historical_iv(ticker, period="2y")

# Load core data
events_df = load_events()
summary = get_event_summary()

# ============================================================
# TAB 1: EVENT DATABASE
# ============================================================

with tab1:
    st.markdown("**Curated BIS export control announcements (2022-2026)**")

    st.sidebar.header("Event Filters")

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(events_df["date"].min(), events_df["date"].max()),
        min_value=events_df["date"].min(),
        max_value=events_df["date"].max(),
        key="events_date_range"
    )

    severity_options = ["All"] + [s.value for s in Severity]
    selected_severity = st.sidebar.selectbox("Severity", severity_options, key="severity_filter")

    type_options = ["All"] + [t.value for t in EventType]
    selected_type = st.sidebar.selectbox("Event Type", type_options, key="type_filter")

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(filtered_df))
    with col2:
        st.metric("Avg NVDA", f"{filtered_df['nvda_reaction_pct'].mean():+.1f}%")
    with col3:
        st.metric("Avg AMD", f"{filtered_df['amd_reaction_pct'].mean():+.1f}%")
    with col4:
        st.metric("Critical", len(filtered_df[filtered_df["severity"] == "Critical"]))

    st.markdown("---")

    severity_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Critical": "#9b59b6"}
    fig_timeline = go.Figure()
    for severity in ["Low", "Medium", "High", "Critical"]:
        sdf = filtered_df[filtered_df["severity"] == severity]
        if len(sdf) > 0:
            fig_timeline.add_trace(go.Scatter(
                x=sdf["date"], y=sdf["nvda_reaction_pct"], mode="markers",
                name=severity, marker=dict(size=15, color=severity_colors[severity]),
                hovertemplate="<b>%{customdata[0]}</b><br>%{y:+.1f}%<extra></extra>",
                customdata=sdf[["title"]].values,
            ))
    fig_timeline.update_layout(title="Events vs NVDA Reaction", height=400, hovermode="closest")
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_timeline, use_container_width=True)

    display_df = filtered_df[["date", "title", "severity", "event_type", "nvda_reaction_pct", "amd_reaction_pct"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Event", "Severity", "Type", "NVDA %", "AMD %"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2: PRICE REACTION
# ============================================================

with tab2:
    st.markdown("**Stock prices with CAR analysis**")

    with st.spinner("Loading prices..."):
        try:
            prices = load_prices()
            data_loaded = True
        except Exception as e:
            st.error(f"Error: {e}")
            data_loaded = False

    if data_loaded and len(prices) > 0:
        st.sidebar.markdown("---")
        st.sidebar.header("Chart Settings")

        normalize_to = st.sidebar.selectbox(
            "Normalize To",
            ["First Date", "ChatGPT Launch (2022-11-30)", "Oct 7 Controls (2022-10-07)"],
            index=1, key="normalize_to"
        )

        car_ticker = st.sidebar.selectbox("CAR Ticker", ["NVDA", "AMD", "INTC"], key="car_ticker")
        win_start = st.sidebar.slider("Window Start", -5, 0, -1, key="win_start")
        win_end = st.sidebar.slider("Window End", 1, 10, 5, key="win_end")

        base_date = "2022-11-30" if "ChatGPT" in normalize_to else ("2022-10-07" if "Oct 7" in normalize_to else None)
        normalized = normalize_prices(prices, base_date)

        fig = go.Figure()
        colors = {"NVDA": "#76b900", "AMD": "#ed1c24", "INTC": "#0071c5", "SPY": "#888"}
        for t in ["NVDA", "AMD", "INTC", "SPY"]:
            if t in normalized.columns:
                fig.add_trace(go.Scatter(
                    x=normalized.index, y=normalized[t], name=t,
                    line=dict(color=colors[t], width=2 if t != "SPY" else 1, dash="dot" if t == "SPY" else "solid")
                ))

        for _, ev in events_df.iterrows():
            if ev["severity"] == "Critical":
                fig.add_vline(x=ev["date"], line_dash="dash", line_color="red", opacity=0.3)

        fig.update_layout(title=f"Normalized Prices (100 = {base_date or 'Start'})", height=450)
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        car_results = load_car_results(prices, events_df, car_ticker, (win_start, win_end))
        car_df = car_to_dataframe(car_results)

        if len(car_df) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg CAR", f"{car_df['CAR (%)'].mean():+.2f}%")
            with col2:
                st.metric("Significant", f"{car_df['Significant'].sum()}/{len(car_df)}")
            with col3:
                st.metric("Worst", f"{car_df['CAR (%)'].min():+.2f}%")

            fig_car = go.Figure()
            fig_car.add_trace(go.Bar(
                x=car_df["Event Date"], y=car_df["CAR (%)"],
                marker_color=["#e74c3c" if x < 0 else "#2ecc71" for x in car_df["CAR (%)"]],
                hovertemplate="<b>%{customdata}</b><br>CAR: %{y:+.2f}%<extra></extra>",
                customdata=car_df["Event"]
            ))
            fig_car.update_layout(title=f"{car_ticker} CAR by Event", height=350)
            fig_car.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_car, use_container_width=True)

# ============================================================
# TAB 3: COMPETITOR ANALYSIS
# ============================================================

with tab3:
    st.markdown("**NVDA vs AMD vs INTC competitive dynamics**")

    with st.spinner("Loading..."):
        try:
            prices = load_prices()
            data_loaded = True
        except:
            data_loaded = False

    if data_loaded and len(prices) > 0:
        st.sidebar.markdown("---")
        st.sidebar.header("Competitor Settings")
        cwin_start = st.sidebar.slider("Window Start", -5, 0, -1, key="cwin_start")
        cwin_end = st.sidebar.slider("Window End", 1, 10, 5, key="cwin_end")
        corr_win = st.sidebar.slider("Corr Window", 10, 60, 30, key="corr_win")

        comp_results = load_competitor_results(prices, events_df, (cwin_start, cwin_end))
        comp_df = competitor_to_dataframe(comp_results)
        comp_summary = get_competitive_summary(comp_results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NVDA Wins", f"{comp_summary.get('nvda_wins', 0)}/{len(comp_df)}")
        with col2:
            st.metric("AMD Wins", f"{comp_summary.get('amd_wins', 0)}/{len(comp_df)}")
        with col3:
            st.metric("INTC Wins", f"{comp_summary.get('intc_wins', 0)}/{len(comp_df)}")
        with col4:
            st.metric("NVDA-AMD Spread", f"{comp_summary.get('nvda_vs_amd_avg', 0):+.1f}%")

        st.markdown("---")

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="NVDA", x=comp_df["Event Date"], y=comp_df["NVDA %"], marker_color="#76b900"))
        fig_comp.add_trace(go.Bar(name="AMD", x=comp_df["Event Date"], y=comp_df["AMD %"], marker_color="#ed1c24"))
        fig_comp.add_trace(go.Bar(name="INTC", x=comp_df["Event Date"], y=comp_df["INTC %"], marker_color="#0071c5"))
        fig_comp.update_layout(barmode="group", title="Returns by Event", height=400)
        fig_comp.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_comp, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            correlation = calculate_rolling_correlation(prices, "NVDA", "AMD", corr_win)
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=correlation.index, y=correlation.values, name="Correlation", line=dict(color="#3498db")))
            fig_corr.update_layout(title=f"NVDA-AMD {corr_win}d Correlation", height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            relative = calculate_relative_strength(prices, "NVDA", "AMD")
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=relative.index, y=relative.values, name="NVDA/AMD", line=dict(color="#76b900")))
            fig_rel.update_layout(title="NVDA/AMD Ratio", height=350)
            st.plotly_chart(fig_rel, use_container_width=True)

# ============================================================
# TAB 4: VOLATILITY SURFACE
# ============================================================

with tab4:
    st.markdown("""
    **Options implied volatility surface with event premium analysis.**
    Visualize where the market prices risk around export control announcements.
    """)

    st.sidebar.markdown("---")
    st.sidebar.header("Options Settings")

    vol_ticker = st.sidebar.selectbox("Ticker", ["NVDA", "AMD", "INTC"], key="vol_ticker")
    option_type = st.sidebar.selectbox("Option Type", ["put", "call"], key="option_type")

    # Load options data
    st.subheader(f"🌊 {vol_ticker} Implied Volatility Surface")

    with st.spinner(f"Fetching {vol_ticker} options chain..."):
        try:
            calls, puts, spot_price = load_options_data(vol_ticker)
            options_loaded = True
        except Exception as e:
            st.warning(f"Could not load live options data: {e}")
            st.info("Options data requires market hours and active trading. Showing historical volatility instead.")
            options_loaded = False

    if options_loaded:
        st.success(f"Spot Price: ${spot_price:.2f} | Calls: {len(calls)} | Puts: {len(puts)}")

        # Build vol surface
        options_df = puts if option_type == "put" else calls
        vol_surface = build_vol_surface(options_df, spot_price, option_type)

        if not vol_surface.empty:
            # ============================================================
            # KEY METRICS
            # ============================================================

            col1, col2, col3, col4 = st.columns(4)

            skew_30 = calculate_skew(vol_surface, 30)
            skew_60 = calculate_skew(vol_surface, 60)

            with col1:
                st.metric("ATM Vol (30d)", f"{skew_30['atm_vol']:.1f}%")

            with col2:
                st.metric("OTM Put Vol (30d)", f"{skew_30['otm_put_vol']:.1f}%")

            with col3:
                st.metric("Skew (30d)", f"{skew_30['skew']:+.1f}%",
                         delta="Put premium" if skew_30['skew'] > 0 else "Normal")

            with col4:
                st.metric("Skew (60d)", f"{skew_60['skew']:+.1f}%")

            st.markdown("---")

            # ============================================================
            # 3D VOLATILITY SURFACE
            # ============================================================

            st.subheader("📊 3D Volatility Surface")

            fig_3d = go.Figure(data=[go.Scatter3d(
                x=vol_surface['moneyness'],
                y=vol_surface['days_to_expiry'],
                z=vol_surface['implied_vol'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=vol_surface['implied_vol'],
                    colorscale='RdYlGn_r',
                    colorbar=dict(title="IV %"),
                    opacity=0.8
                ),
                hovertemplate=(
                    "Moneyness: %{x:.1f}%<br>"
                    "Days to Expiry: %{y}<br>"
                    "IV: %{z:.1f}%<extra></extra>"
                )
            )])

            fig_3d.update_layout(
                title=f"{vol_ticker} {option_type.upper()} Implied Volatility Surface",
                scene=dict(
                    xaxis_title="Moneyness (%)",
                    yaxis_title="Days to Expiry",
                    zaxis_title="Implied Vol (%)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                ),
                height=600,
            )

            st.plotly_chart(fig_3d, use_container_width=True)

            # ============================================================
            # HEATMAP
            # ============================================================

            st.subheader("🔥 Volatility Heatmap")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Create pivot for heatmap
                pivot_df = vol_surface.pivot_table(
                    values='implied_vol',
                    index=pd.cut(vol_surface['moneyness'], bins=[-25, -15, -10, -5, 0, 5, 10, 15, 25]),
                    columns=pd.cut(vol_surface['days_to_expiry'], bins=[0, 14, 30, 60, 90, 180, 365]),
                    aggfunc='mean'
                )

                if not pivot_df.empty:
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=[str(c) for c in pivot_df.columns],
                        y=[str(i) for i in pivot_df.index],
                        colorscale='RdYlGn_r',
                        colorbar=dict(title="IV %"),
                        hovertemplate="Moneyness: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>"
                    ))

                    fig_heat.update_layout(
                        title="IV Heatmap (Strike × Expiry)",
                        xaxis_title="Days to Expiry",
                        yaxis_title="Moneyness (%)",
                        height=450
                    )

                    st.plotly_chart(fig_heat, use_container_width=True)

            with col2:
                st.markdown("**Reading the Heatmap:**")
                st.markdown("""
                - **Red** = High IV (expensive options)
                - **Green** = Low IV (cheap options)
                - **OTM Puts** (negative moneyness) typically show higher IV
                - **Skew** = difference between OTM put vol and ATM vol

                **Event Premium:**
                Watch for elevated OTM put IV before export control announcements — this indicates hedging demand.
                """)

            # ============================================================
            # TERM STRUCTURE
            # ============================================================

            st.markdown("---")
            st.subheader("📈 Volatility Term Structure")

            term_structure = calculate_term_structure(vol_surface)

            if not term_structure.empty:
                col1, col2 = st.columns(2)

                with col1:
                    fig_term = go.Figure()

                    fig_term.add_trace(go.Scatter(
                        x=term_structure['days_to_expiry'],
                        y=term_structure['implied_vol'],
                        mode='lines+markers',
                        name='ATM IV',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=10)
                    ))

                    fig_term.update_layout(
                        title="ATM Implied Volatility Term Structure",
                        xaxis_title="Days to Expiry",
                        yaxis_title="Implied Volatility (%)",
                        height=400
                    )

                    st.plotly_chart(fig_term, use_container_width=True)

                with col2:
                    # Term structure interpretation
                    if len(term_structure) >= 2:
                        short_term = term_structure[term_structure['days_to_expiry'] <= 30]['implied_vol'].mean()
                        long_term = term_structure[term_structure['days_to_expiry'] > 60]['implied_vol'].mean()

                        if short_term > long_term * 1.1:
                            structure_type = "Inverted (Backwardation)"
                            interpretation = "Short-term IV higher than long-term. Market pricing near-term event risk."
                        elif long_term > short_term * 1.1:
                            structure_type = "Contango"
                            interpretation = "Long-term IV higher. Normal structure, no immediate event premium."
                        else:
                            structure_type = "Flat"
                            interpretation = "Similar IV across expirations."

                        st.metric("Term Structure", structure_type)
                        st.info(interpretation)

                        st.markdown("**Open Interest:**")
                        st.dataframe(
                            term_structure[['days_to_expiry', 'implied_vol', 'open_interest']].head(8),
                            use_container_width=True, hide_index=True
                        )

            # ============================================================
            # SKEW SMILE
            # ============================================================

            st.markdown("---")
            st.subheader("😊 Volatility Smile / Skew")

            # Get 30-day options for smile
            smile_data = vol_surface[
                (vol_surface['days_to_expiry'] >= 20) &
                (vol_surface['days_to_expiry'] <= 40)
            ].sort_values('moneyness')

            if not smile_data.empty:
                fig_smile = go.Figure()

                fig_smile.add_trace(go.Scatter(
                    x=smile_data['moneyness'],
                    y=smile_data['implied_vol'],
                    mode='markers+lines',
                    name='30d IV',
                    line=dict(color='#9b59b6', width=2),
                    marker=dict(size=8)
                ))

                fig_smile.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig_smile.update_layout(
                    title="30-Day Volatility Smile",
                    xaxis_title="Moneyness (%) — Negative = OTM Puts",
                    yaxis_title="Implied Volatility (%)",
                    height=400
                )

                # Add annotations
                fig_smile.add_annotation(x=-15, y=smile_data['implied_vol'].max(),
                                        text="OTM Puts<br>(Crash protection)", showarrow=False)
                fig_smile.add_annotation(x=15, y=smile_data['implied_vol'].max(),
                                        text="OTM Calls<br>(Upside bets)", showarrow=False)

                st.plotly_chart(fig_smile, use_container_width=True)

        else:
            st.warning("Could not build vol surface from available data.")

    # ============================================================
    # HISTORICAL VOLATILITY (Always show)
    # ============================================================

    st.markdown("---")
    st.subheader("📉 Historical Realized Volatility")

    with st.spinner("Loading historical volatility..."):
        hist_vol = load_historical_vol(vol_ticker)

    if not hist_vol.empty:
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Scatter(
            x=hist_vol['date'], y=hist_vol['rv_20'],
            name='20-Day RV', line=dict(color='#e74c3c', width=2)
        ))

        fig_hist.add_trace(go.Scatter(
            x=hist_vol['date'], y=hist_vol['rv_60'],
            name='60-Day RV', line=dict(color='#3498db', width=2)
        ))

        # Add event markers
        for _, ev in events_df.iterrows():
            if ev["severity"] in ["Critical", "High"]:
                fig_hist.add_vline(x=ev["date"], line_dash="dash", line_color="red", opacity=0.3)

        fig_hist.update_layout(
            title=f"{vol_ticker} Realized Volatility (Red lines = restriction events)",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Current vol metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_rv20 = hist_vol['rv_20'].iloc[-1]
            st.metric("Current 20d RV", f"{current_rv20:.1f}%")
        with col2:
            current_rv60 = hist_vol['rv_60'].iloc[-1]
            st.metric("Current 60d RV", f"{current_rv60:.1f}%")
        with col3:
            avg_rv = hist_vol['rv_20'].mean()
            st.metric("Average 20d RV", f"{avg_rv:.1f}%",
                     delta=f"{current_rv20 - avg_rv:+.1f}% vs avg")

    # ============================================================
    # INTERPRETATION
    # ============================================================

    st.markdown("---")
    st.subheader("📝 Options Intelligence")

    st.markdown(f"""
    **Key Observations for {vol_ticker}:**

    1. **Volatility Skew:** OTM puts typically trade at a premium to ATM options.
       This "skew" reflects crash protection demand. Higher skew = more hedging activity.

    2. **Event Premium:** Before export control announcements, watch for:
       - Elevated short-term IV vs long-term (inverted term structure)
       - Increased OTM put skew
       - Rising open interest in protective puts

    3. **Trading Implications:**
       - **High skew + inverted term structure** = Market pricing near-term risk
       - **Flat skew + contango** = No imminent event expected
       - Consider selling rich OTM puts after event passes (vol crush)

    4. **Historical Context:**
       Realized volatility spikes during restriction announcements provide
       reference points for expected moves on future events.
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("""
**NEXUS** | Nvidia Export Controls Research & Trading System |
Data: BIS, SEC, Yahoo Finance | Options: Live chain data |
CAR: Market model (120d estimation) | Vol: Black-Scholes implied
""")

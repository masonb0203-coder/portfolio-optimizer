"""
Portfolio Optimizer — Streamlit App
====================================
Entry point. Run with:  streamlit run app.py
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #2a2d35;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextArea label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stToggle label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7a8499 !important;
    }

    .sidebar-section {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a9eff !important;
        border-bottom: 1px solid #2a2d35;
        padding-bottom: 6px;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .app-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        color: #ffffff;
        line-height: 1.3;
    }
    .app-subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 300;
        color: #7a8499;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    div[data-testid="stSidebar"] .stButton > button {
        background-color: #4a9eff;
        color: #000000 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border: none;
        border-radius: 3px;
        width: 100%;
        padding: 10px;
        margin-top: 8px;
        transition: background-color 0.15s ease;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #73b4ff;
        color: #000000 !important;
    }
    div[data-testid="stSidebar"] .stButton > button:disabled {
        background-color: #2a2d35 !important;
        color: #555 !important;
    }

    .ticker-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .ticker-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        background: #1e2130;
        border: 1px solid #2a2d35;
        border-radius: 3px;
        padding: 2px 7px;
        color: #4a9eff;
    }

    .status-ready {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #3ddc84;
        background: #0d1f16;
        border: 1px solid #1e4d2e;
        border-radius: 3px;
        padding: 8px 10px;
        margin-top: 10px;
        line-height: 1.6;
    }
    .status-warn {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #ffb347;
        background: #1f1800;
        border: 1px solid #4d3800;
        border-radius: 3px;
        padding: 8px 10px;
        margin-top: 10px;
        line-height: 1.6;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 1px solid #2a2d35;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 10px 20px;
        border-radius: 0;
        color: #7a8499;
    }
    .stTabs [aria-selected="true"] {
        color: #4a9eff !important;
        border-bottom: 2px solid #4a9eff !important;
        background: transparent !important;
    }

    .phase-intro {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 300;
        color: #9aa3b5;
        max-width: 620px;
        line-height: 1.7;
        margin-top: 8px;
    }
    .phase-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        color: #ffffff;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Default tickers ───────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "VTI", "QQQ", "VTV", "IWM", "VEA",
    "VWO", "BND", "TLT", "TIP", "VNQ", "GLD", "PDBC",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">Portfolio<br>Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Markowitz · PCA · Black-Litterman</div>',
        unsafe_allow_html=True,
    )

    # ── Asset Universe ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Asset Universe</div>', unsafe_allow_html=True)

    tickers_input = st.text_area(
        "Tickers (space or comma separated)",
        value=" ".join(DEFAULT_TICKERS),
        height=90,
        help="Any valid Yahoo Finance ticker symbol. 8–15 assets recommended.",
    )

    # Parse tickers
    raw = tickers_input.replace(",", " ").upper().split()
    tickers = [t.strip() for t in raw if t.strip()]

    # Render as tag pills
    if tickers:
        tags_html = (
            '<div class="ticker-tags">'
            + "".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
            + "</div>"
        )
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── Historical Window ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Historical Window</div>', unsafe_allow_html=True)

    lookback_years = st.slider(
        "Lookback (years)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Years of daily price history used to estimate μ and Σ.",
    )
    st.caption(f"≈ {lookback_years * 252:,} trading days of data")

    # ── Parameters ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Parameters</div>', unsafe_allow_html=True)

    risk_free_rate = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=15.0,
        value=4.3,
        step=0.1,
        format="%.1f",
        help="Approximate annualized yield on 3-month US Treasuries. Update this to reflect current rates.",
    ) / 100

    # ── Phase 3 Options ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Phase 3 — Black-Litterman</div>', unsafe_allow_html=True)

    use_position_cap = st.toggle(
        "Enable position cap",
        value=True,
        help="Hard upper bound on any single asset's weight.",
    )

    max_weight = 1.0
    if use_position_cap:
        max_weight_pct = st.slider(
            "Max weight per asset (%)",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
            help="No single asset can exceed this allocation.",
        )
        max_weight = max_weight_pct / 100
        st.caption(
            f"Cap: {max_weight_pct}% → at least {int(1 / max_weight)} assets must be held"
        )

    # ── Backtest Options ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Backtest</div>', unsafe_allow_html=True)

    train_years = st.slider(
        "Training window (years)",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Rolling lookback window used at each rebalancing date. Must be shorter than total lookback.",
    )
    train_days = train_years * 252

    rebal_freq = st.selectbox(
        "Rebalancing frequency",
        options=[
            "Monthly (21 days)",
            "Quarterly (63 days)",
            "Semi-annual (126 days)",
            "Annual (252 days)",
        ],
        index=1,
        help="How often the optimizer re-runs and portfolio weights are updated.",
    )
    rebal_days_map = {
        "Monthly (21 days)": 21,
        "Quarterly (63 days)": 63,
        "Semi-annual (126 days)": 126,
        "Annual (252 days)": 252,
    }
    rebal_days = rebal_days_map[rebal_freq]

    transaction_cost = st.number_input(
        "Transaction cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="One-way cost applied to each dollar of turnover at every rebalance.",
    ) / 100

    # ── Validation & Run ──────────────────────────────────────────────────────
    oos_days = lookback_years * 252 - train_days
    config_valid = len(tickers) >= 2 and oos_days > rebal_days

    if not config_valid:
        if len(tickers) < 2:
            msg = "⚠ Need at least 2 tickers."
        else:
            msg = "⚠ Lookback must exceed training window by at least one rebalancing period."
        st.markdown(f'<div class="status-warn">{msg}</div>', unsafe_allow_html=True)
    else:
        oos_years = round(oos_days / 252, 1)
        st.markdown(
            f'<div class="status-ready">'
            f"✓ Config valid<br>"
            f"{len(tickers)} assets · {lookback_years}yr lookback<br>"
            f"~{oos_years}yr out-of-sample window"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("▶  Run Optimizer", disabled=not config_valid)

# ── Persist config in session state on run ────────────────────────────────────
if run_button:
    st.session_state["config"] = {
        "tickers": tickers,
        "lookback_years": lookback_years,
        "trading_days": 252,
        "risk_free_rate": risk_free_rate,
        "max_weight": max_weight,
        "use_position_cap": use_position_cap,
        "train_days": train_days,
        "rebal_days": rebal_days,
        "transaction_cost": transaction_cost,
    }
    st.session_state["run"] = True

has_run = st.session_state.get("run", False)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Phase 1 — Mean-Variance",
    "Phase 2 — PCA Cleaning",
    "Phase 3 — Black-Litterman",
    "Backtest",
])

with tab1:
    st.markdown('<div class="phase-title">Phase 1 — Mean-Variance Optimization</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown(
            '<div class="phase-intro">'
            "Estimates expected returns (μ) and a covariance matrix (Σ) from historical prices, "
            "then finds the efficient frontier via constrained quadratic optimization (SLSQP). "
            "Produces the minimum variance portfolio and maximum Sharpe ratio portfolio."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Optimizer logic arrives in Phase C.")

with tab2:
    st.markdown('<div class="phase-title">Phase 2 — PCA Covariance Cleaning</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown(
            '<div class="phase-intro">'
            "Applies Marchenko-Pastur spectral theory to separate signal eigenvalues from noise in Σ. "
            "Noise eigenvalues — statistically indistinguishable from a random matrix — are replaced "
            "with their mean, producing a cleaner Σ that stabilizes portfolio weights."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Optimizer logic arrives in Phase C.")

with tab3:
    st.markdown('<div class="phase-title">Phase 3 — Black-Litterman Equilibrium</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown(
            '<div class="phase-intro">'
            "Replaces historical μ with market-implied equilibrium returns: Π = δ × Σ_clean × w_mkt. "
            "Assets whose historical returns were inflated by a recent run (e.g. GLD) are corrected "
            "to what the market consensus actually implies, producing more realistic allocations."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Optimizer logic arrives in Phase C.")

with tab4:
    st.markdown('<div class="phase-title">Backtest</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown(
            '<div class="phase-intro">'
            "Rolling out-of-sample backtest with quarterly rebalancing and a fixed training window. "
            "Compares all three phases against Equal Weight and VTI Buy-and-Hold. "
            "No look-ahead bias — each rebalancing decision uses only data available at that date."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Optimizer logic arrives in Phase C.")

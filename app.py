"""
Portfolio Optimizer — Streamlit App
====================================
Entry point. Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from optimizer import (
    fetch_prices,
    run_optimization,
    run_backtest,
    compute_metrics,
)

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

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    section[data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #2a2d35;
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
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
    }
    div[data-testid="stSidebar"] .stButton > button:hover { background-color: #73b4ff; }
    div[data-testid="stSidebar"] .stButton > button:disabled {
        background-color: #2a2d35 !important;
        color: #555 !important;
    }
    .ticker-tags { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
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
    .stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 1px solid #2a2d35; }
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
    .phase-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .phase-intro {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 300;
        color: #9aa3b5;
        max-width: 620px;
        line-height: 1.7;
        margin-top: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["VTI","QQQ","VTV","IWM","VEA","VWO","BND","TLT","TIP","VNQ","GLD","PDBC"]
PHASE_COLORS    = {"Phase 1": "#c0392b", "Phase 2": "#2d6a2d", "Phase 3": "#2980b9"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_colors(n):
    return cm.tab10(np.linspace(0, 1, n))

def _metrics_table(phase_dict, rf, label):
    w   = phase_dict["w_msr"]
    m   = phase_dict["msr_metrics"]
    return {
        "Portfolio"  : label,
        "Return"     : f"{m['return']*100:.2f}%",
        "Volatility" : f"{m['volatility']*100:.2f}%",
        "Sharpe"     : f"{m['sharpe']:.3f}",
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">Portfolio<br>Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Markowitz · PCA · Black-Litterman</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Asset Universe</div>', unsafe_allow_html=True)
    tickers_input = st.text_area(
        "Tickers (space or comma separated)",
        value=" ".join(DEFAULT_TICKERS), height=90,
        help="Any valid Yahoo Finance ticker. 8–15 assets recommended.",
    )
    raw     = tickers_input.replace(",", " ").upper().split()
    tickers = [t.strip() for t in raw if t.strip()]
    if tickers:
        st.markdown(
            '<div class="ticker-tags">'
            + "".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
            + "</div>", unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section">Historical Window</div>', unsafe_allow_html=True)
    lookback_years = st.slider("Lookback (years)", 1, 10, 5)
    st.caption(f"≈ {lookback_years * 252:,} trading days")

    st.markdown('<div class="sidebar-section">Parameters</div>', unsafe_allow_html=True)
    risk_free_rate = st.number_input(
        "Risk-free rate (%)", 0.0, 15.0, 4.3, 0.1, format="%.1f",
        help="Approx. annualised 3-month US Treasury yield.",
    ) / 100

    st.markdown('<div class="sidebar-section">Phase 3 — Black-Litterman</div>', unsafe_allow_html=True)
    use_cap = st.toggle("Enable position cap", value=True)
    max_weight = 1.0
    if use_cap:
        max_weight = st.slider("Max weight per asset (%)", 5, 100, 30, 5) / 100
        st.caption(f"Cap: {max_weight*100:.0f}% → min {int(1/max_weight)} assets held")

    st.markdown('<div class="sidebar-section">Backtest</div>', unsafe_allow_html=True)
    train_years = st.slider("Training window (years)", 1, 5, 2)
    train_days  = train_years * 252
    rebal_freq  = st.selectbox(
        "Rebalancing frequency",
        ["Monthly (21 days)", "Quarterly (63 days)", "Semi-annual (126 days)", "Annual (252 days)"],
        index=1,
    )
    rebal_days = {"Monthly (21 days)":21,"Quarterly (63 days)":63,"Semi-annual (126 days)":126,"Annual (252 days)":252}[rebal_freq]
    transaction_cost = st.number_input("Transaction cost (%)", 0.0, 1.0, 0.05, 0.01, format="%.2f") / 100

    oos_days    = lookback_years * 252 - train_days
    config_ok   = len(tickers) >= 2 and oos_days > rebal_days
    if not config_ok:
        msg = "⚠ Need at least 2 tickers." if len(tickers) < 2 else "⚠ Lookback must exceed training window."
        st.markdown(f'<div class="status-warn">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="status-ready">✓ Config valid<br>'
            f'{len(tickers)} assets · {lookback_years}yr lookback<br>'
            f'~{oos_days/252:.1f}yr out-of-sample</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("▶  Run Optimizer", disabled=not config_ok)

# ── Run on button press ───────────────────────────────────────────────────────
if run_button:
    st.session_state["run"]    = False
    st.session_state["error"]  = None
    st.session_state["results"] = None

    with st.spinner("Downloading price data…"):
        try:
            prices = fetch_prices(tickers, lookback_years)
        except Exception as e:
            st.session_state["error"] = f"Data error: {e}"
            st.rerun()

    with st.spinner("Running optimisation (phases 1–3)…"):
        try:
            opt = run_optimization(prices, risk_free_rate, max_weight)
        except Exception as e:
            st.session_state["error"] = f"Optimisation error: {e}"
            st.rerun()

    with st.spinner("Running backtest…"):
        try:
            bt_results, bt_logs = run_backtest(
                prices, train_days, rebal_days, transaction_cost, max_weight, risk_free_rate
            )
        except Exception as e:
            st.session_state["error"] = f"Backtest error: {e}"
            st.rerun()

    st.session_state["run"]        = True
    st.session_state["opt"]        = opt
    st.session_state["bt_results"] = bt_results
    st.session_state["bt_logs"]    = bt_logs
    st.session_state["tickers"]    = tickers
    st.session_state["rf"]         = risk_free_rate
    st.session_state["max_weight"] = max_weight

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(st.session_state["error"])

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Phase 1 — Mean-Variance",
    "Phase 2 — PCA Cleaning",
    "Phase 3 — Black-Litterman",
    "Backtest",
])

has_run = st.session_state.get("run", False)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="phase-title">Phase 1 — Mean-Variance Optimization</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Estimates expected returns (μ) and a covariance matrix (Σ) from historical prices, then finds the efficient frontier via constrained quadratic optimization (SLSQP). Produces the minimum variance and maximum Sharpe ratio portfolios.</div>', unsafe_allow_html=True)
    else:
        opt     = st.session_state["opt"]
        tkrs    = opt["tickers"]
        n       = len(tkrs)
        p1      = opt["phase1"]
        mu_vec  = opt["mu_vec"]
        cov_mat = opt["cov_mat"]
        rf      = st.session_state["rf"]
        colors  = _bar_colors(n)

        # ── Metrics row ───────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        mvp = p1["mvp_metrics"]; msr = p1["msr_metrics"]
        c1.metric("MVP Return",     f"{mvp['return']*100:.2f}%")
        c1.metric("MVP Volatility", f"{mvp['volatility']*100:.2f}%")
        c1.metric("MVP Sharpe",     f"{mvp['sharpe']:.3f}")
        c2.metric("Max SR Return",     f"{msr['return']*100:.2f}%")
        c2.metric("Max SR Volatility", f"{msr['volatility']*100:.2f}%")
        c2.metric("Max SR Sharpe",     f"{msr['sharpe']:.3f}")

        # ── Efficient frontier ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        sc = ax.scatter(p1["frontier_vols"]*100, p1["frontier_rets"]*100,
                        c=p1["frontier_sharpes"], cmap="RdYlGn", s=14, zorder=3)
        plt.colorbar(sc, ax=ax).set_label("Sharpe Ratio", color="#9aa3b5")
        # Individual assets
        asset_vols = np.sqrt(np.diag(cov_mat))*100
        asset_rets = mu_vec*100
        for i, t in enumerate(tkrs):
            ax.scatter(asset_vols[i], asset_rets[i], color=colors[i], s=80, zorder=6, edgecolors="white", linewidths=0.6)
            ax.annotate(t, (asset_vols[i], asset_rets[i]), xytext=(4,3), textcoords="offset points", fontsize=8, color=colors[i], fontweight="bold")
        # Key portfolios
        ax.scatter(mvp["volatility"]*100, mvp["return"]*100, marker="*", s=360, color="gold", edgecolors="black", zorder=7, label="Min Variance")
        ax.scatter(msr["volatility"]*100, msr["return"]*100, marker="*", s=360, color="limegreen", edgecolors="black", zorder=7, label="Max Sharpe")
        ax.scatter([0], [rf*100], color="steelblue", s=50, zorder=5)
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d35")
        ax.tick_params(colors="#9aa3b5"); ax.xaxis.label.set_color("#9aa3b5"); ax.yaxis.label.set_color("#9aa3b5")
        ax.set_title("Efficient Frontier — Phase 1", color="#ffffff", fontsize=12)
        ax.set_xlabel("Annualised Volatility (%)"); ax.set_ylabel("Annualised Return (%)")
        ax.legend(fontsize=9, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax.grid(True, alpha=0.15, color="#2a2d35")
        st.pyplot(fig); plt.close()

        # ── Weight comparison ─────────────────────────────────────────────────
        st.subheader("Portfolio Weights")
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig2.patch.set_facecolor("#0f1117")
        for ax2, (w, title) in zip(axes, [(p1["w_mvp"],"Min Variance"),(p1["w_msr"],"Max Sharpe")]):
            ax2.set_facecolor("#0f1117")
            ax2.barh(tkrs, w*100, color=colors, edgecolor="white", linewidth=0.5)
            ax2.set_title(title, color="#ffffff")
            ax2.tick_params(colors="#9aa3b5")
            for spine in ax2.spines.values(): spine.set_edgecolor("#2a2d35")
            ax2.invert_yaxis()
            ax2.set_xlabel("Weight (%)", color="#9aa3b5")
            ax2.grid(True, axis="x", alpha=0.15, color="#2a2d35")
            for i, val in enumerate(w):
                if val > 0.01:
                    ax2.text(val*100+0.3, i, f"{val*100:.1f}%", va="center", fontsize=8, color="#9aa3b5")
        fig2.tight_layout()
        st.pyplot(fig2); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="phase-title">Phase 2 — PCA Covariance Cleaning</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Applies Marchenko-Pastur spectral theory to separate signal eigenvalues from noise in Σ. Noise eigenvalues are replaced with their mean, preserving the total variance while removing estimation error.</div>', unsafe_allow_html=True)
    else:
        opt      = st.session_state["opt"]
        tkrs     = opt["tickers"]
        n        = len(tkrs)
        p1       = opt["phase1"]
        p2       = opt["phase2"]
        pca      = opt["pca_info"]
        colors   = _bar_colors(n)
        rf       = st.session_state["rf"]

        # ── PCA stats ─────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Signal eigenvalues", pca["n_signal"])
        c2.metric("Noise eigenvalues",  pca["n_noise"])
        c3.metric("MP threshold λ⁺",    f"{pca['lambda_plus']:.4f}")

        # ── Eigenvalue spectrum ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        evs    = pca["eigenvalues"]
        lc     = pca["lambda_clean"]
        x      = np.arange(1, n+1)
        w_bar  = 0.38
        bar_colors_ev = ["#2d6a2d" if evs[i] > pca["lambda_plus"] else "#555" for i in range(n)]
        ax.bar(x - w_bar/2, evs,  w_bar, color="#c0392b", alpha=0.8, label="Raw Σ",     edgecolor="white", linewidth=0.4)
        ax.bar(x + w_bar/2, lc,   w_bar, color="#2d6a2d", alpha=0.9, label="Σ_clean",   edgecolor="white", linewidth=0.4)
        ax.axhline(pca["lambda_plus"], color="crimson", linestyle="--", linewidth=1.5, label=f"MP λ⁺ = {pca['lambda_plus']:.4f}")
        ax.set_xticks(x); ax.set_xticklabels([f"λ{i}" for i in x], color="#9aa3b5")
        ax.tick_params(colors="#9aa3b5")
        ax.set_title("Eigenvalue Spectrum: Raw vs Cleaned", color="#ffffff")
        ax.legend(facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white", fontsize=9)
        ax.grid(True, axis="y", alpha=0.15, color="#2a2d35")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d35")
        st.pyplot(fig); plt.close()

        # ── Overlaid frontiers ────────────────────────────────────────────────
        st.subheader("Phase 1 vs Phase 2 Frontiers")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#0f1117")
        ax2.plot(p1["frontier_vols"]*100, p1["frontier_rets"]*100, color="#c0392b", lw=2, label="Phase 1 — Raw Σ")
        ax2.plot(p2["frontier_vols"]*100, p2["frontier_rets"]*100, color="#2d6a2d", lw=2, linestyle="--", label="Phase 2 — Σ_clean")
        ax2.scatter(p1["msr_metrics"]["volatility"]*100, p1["msr_metrics"]["return"]*100, marker="*", s=280, color="#e74c3c", edgecolors="black", zorder=5)
        ax2.scatter(p2["msr_metrics"]["volatility"]*100, p2["msr_metrics"]["return"]*100, marker="D", s=100, color="#27ae60", edgecolors="black", zorder=5)
        ax2.tick_params(colors="#9aa3b5"); ax2.set_xlabel("Volatility (%)", color="#9aa3b5"); ax2.set_ylabel("Return (%)", color="#9aa3b5")
        ax2.set_title("Efficient Frontier Comparison", color="#ffffff")
        ax2.legend(facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white", fontsize=9)
        ax2.grid(True, alpha=0.15, color="#2a2d35")
        for spine in ax2.spines.values(): spine.set_edgecolor("#2a2d35")
        st.pyplot(fig2); plt.close()

        # ── Weight comparison ─────────────────────────────────────────────────
        st.subheader("Max Sharpe Weights: Phase 1 vs Phase 2")
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig3.patch.set_facecolor("#0f1117")
        for ax3, (w, title) in zip(axes3, [(p1["w_msr"],"Phase 1"),(p2["w_msr"],"Phase 2")]):
            ax3.set_facecolor("#0f1117")
            ax3.barh(tkrs, w*100, color=colors, edgecolor="white", linewidth=0.5)
            ax3.set_title(title, color="#ffffff")
            ax3.tick_params(colors="#9aa3b5")
            for spine in ax3.spines.values(): spine.set_edgecolor("#2a2d35")
            ax3.invert_yaxis(); ax3.set_xlabel("Weight (%)", color="#9aa3b5")
            ax3.grid(True, axis="x", alpha=0.15, color="#2a2d35")
        fig3.tight_layout()
        st.pyplot(fig3); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="phase-title">Phase 3 — Black-Litterman Equilibrium</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Replaces historical μ with market-implied equilibrium returns Π = δ × Σ_clean × w_mkt. Assets whose historical returns were inflated by a recent bull run are corrected to market consensus, producing more realistic allocations.</div>', unsafe_allow_html=True)
    else:
        opt    = st.session_state["opt"]
        tkrs   = opt["tickers"]
        n      = len(tkrs)
        p1     = opt["phase1"]
        p2     = opt["phase2"]
        p3     = opt["phase3"]
        mu_vec = opt["mu_vec"]
        pi     = opt["pi"]
        rf     = st.session_state["rf"]
        colors = _bar_colors(n)

        # ── BL stats ──────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk aversion δ",      f"{opt['delta']:.3f}")
        c2.metric("Max SR Return (BL)",   f"{p3['msr_metrics']['return']*100:.2f}%")
        c3.metric("Max SR Sharpe (BL)",   f"{p3['msr_metrics']['sharpe']:.3f}")

        # ── μ vs Π bar chart ──────────────────────────────────────────────────
        st.subheader("Historical μ vs Equilibrium Π")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        x = np.arange(n); w_bar = 0.38
        ax.bar(x - w_bar/2, mu_vec*100, w_bar, color="#c0392b", alpha=0.8, label="Historical μ", edgecolor="white", linewidth=0.4)
        ax.bar(x + w_bar/2, pi*100,     w_bar, color="#2d6a2d", alpha=0.9, label="Equilibrium Π", edgecolor="white", linewidth=0.4)
        ax.axhline(rf*100, color="steelblue", linestyle=":", linewidth=1.2, label=f"Risk-free {rf*100:.1f}%")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tkrs, rotation=30, ha="right", color="#9aa3b5")
        ax.tick_params(colors="#9aa3b5"); ax.set_ylabel("Annualised Return (%)", color="#9aa3b5")
        ax.set_title("Historical μ vs Black-Litterman Π", color="#ffffff")
        ax.legend(facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white", fontsize=9)
        ax.grid(True, axis="y", alpha=0.15, color="#2a2d35")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d35")
        st.pyplot(fig); plt.close()

        # ── 3-way Max Sharpe weight comparison ───────────────────────────────
        st.subheader("Max Sharpe Weights — All Three Phases")
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        fig2.patch.set_facecolor("#0f1117")
        mw = st.session_state["max_weight"]
        for ax2, (w, title) in zip(axes2, [
            (p1["w_msr"], "Phase 1\nRaw Σ · hist μ"),
            (p2["w_msr"], "Phase 2\nΣ_clean · hist μ"),
            (p3["w_msr"], f"Phase 3\nΣ_clean · BL Π · {mw*100:.0f}% cap"),
        ]):
            ax2.set_facecolor("#0f1117")
            ax2.barh(tkrs, w*100, color=colors, edgecolor="white", linewidth=0.5)
            if "Phase 3" in title:
                ax2.axvline(mw*100, color="white", linestyle=":", linewidth=1, alpha=0.5)
            ax2.set_title(title, color="#ffffff", fontsize=9)
            ax2.tick_params(colors="#9aa3b5")
            for spine in ax2.spines.values(): spine.set_edgecolor("#2a2d35")
            ax2.invert_yaxis(); ax2.set_xlabel("Weight (%)", color="#9aa3b5")
            ax2.grid(True, axis="x", alpha=0.15, color="#2a2d35")
            for i, val in enumerate(w):
                if val > 0.01:
                    ax2.text(val*100+0.3, i, f"{val*100:.1f}%", va="center", fontsize=7.5, color="#9aa3b5")
        fig2.tight_layout()
        st.pyplot(fig2); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="phase-title">Backtest</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Rolling out-of-sample backtest comparing all three phases against Equal Weight and VTI Buy-and-Hold. No look-ahead bias — each rebalancing decision uses only data available at that date.</div>', unsafe_allow_html=True)
    else:
        bt   = st.session_state["bt_results"]
        rf   = st.session_state["rf"]

        BT_COLORS = {
            "Phase 1 — Raw Σ, hist μ"   : ("#c0392b", "-",  2.0),
            "Phase 2 — Σ_clean, hist μ" : ("#2d6a2d", "--", 2.0),
            "Phase 3 — BL Π, cap"       : ("#2980b9", "-.", 2.4),
            "Equal Weight"               : ("#e67e22", ":",  1.8),
            "VTI Buy-and-Hold"           : ("#7f8c8d", "-",  1.4),
        }

        # ── Metrics table ─────────────────────────────────────────────────────
        metrics_rows = []
        for label, pv in bt.items():
            m = compute_metrics(pv, rf)
            metrics_rows.append({"Strategy": label, **m})
        df = pd.DataFrame(metrics_rows).set_index("Strategy")
        df.columns = ["Return %","CAGR %","Vol %","Sharpe","Max DD %","Calmar"]
        st.dataframe(df.style.format({
            "Return %":"{:.2f}","CAGR %":"{:.2f}","Vol %":"{:.2f}",
            "Sharpe":"{:.3f}","Max DD %":"{:.2f}","Calmar":"{:.3f}",
        }), use_container_width=True)

        # ── Cumulative return chart ────────────────────────────────────────────
        st.subheader("Cumulative Returns")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        for label, pv in bt.items():
            color, ls, lw = BT_COLORS.get(label, ("#888","-",1.5))
            ax.plot(pv.index, pv.values, color=color, linestyle=ls, linewidth=lw,
                    label=f"{label}  ({(pv.iloc[-1]-1)*100:+.1f}%)")
        ax.axhline(1.0, color="#555", linewidth=0.7, linestyle=":")
        ax.tick_params(colors="#9aa3b5"); ax.set_ylabel("Portfolio Value ($1.00 start)", color="#9aa3b5")
        ax.set_title("Out-of-Sample Cumulative Returns", color="#ffffff")
        ax.legend(fontsize=8, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax.grid(True, alpha=0.15, color="#2a2d35")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d35")
        st.pyplot(fig); plt.close()

        # ── Drawdown chart ────────────────────────────────────────────────────
        st.subheader("Drawdown from Peak")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#0f1117")
        for label, pv in bt.items():
            color, ls, lw = BT_COLORS.get(label, ("#888","-",1.5))
            dd = (pv - pv.cummax()) / pv.cummax() * 100
            ax2.plot(dd.index, dd.values, color=color, linestyle=ls, linewidth=lw, label=label)
        ax2.axhline(0, color="#555", linewidth=0.8)
        ax2.tick_params(colors="#9aa3b5"); ax2.set_ylabel("Drawdown (%)", color="#9aa3b5")
        ax2.set_title("Drawdown from Peak", color="#ffffff")
        ax2.legend(fontsize=8, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax2.grid(True, alpha=0.15, color="#2a2d35")
        for spine in ax2.spines.values(): spine.set_edgecolor("#2a2d35")
        st.pyplot(fig2); plt.close()

"""
Portfolio Optimizer — Streamlit App (Final)
============================================
Polished version with:
  - Step-by-step progress bar during computation
  - Correlation heatmap (Phase 1)
  - Rolling 1-year Sharpe chart (Backtest)
  - Live ticker validation before run
  - CSV download for metrics + weights
  - Phase comparison summary table (Phase 3)
"""

import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf

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

# ── CSS ───────────────────────────────────────────────────────────────────────
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
    .ticker-tag-invalid {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        background: #2a1010;
        border: 1px solid #5a2020;
        border-radius: 3px;
        padding: 2px 7px;
        color: #ff6b6b;
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
        max-width: 680px;
        line-height: 1.7;
        margin-top: 8px;
        margin-bottom: 20px;
    }
    .summary-table th {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #7a8499;
    }
    /* Download button */
    .stDownloadButton > button {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em;
        background: #1e2130 !important;
        border: 1px solid #2a2d35 !important;
        color: #4a9eff !important;
        border-radius: 3px !important;
    }
    .stDownloadButton > button:hover {
        background: #2a2d35 !important;
        border-color: #4a9eff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["VTI","QQQ","VTV","IWM","VEA","VWO","BND","TLT","TIP","VNQ","GLD","PDBC"]
BT_COLORS = {
    "Phase 1 — Raw Σ, hist μ"   : ("#c0392b", "-",  2.0),
    "Phase 2 — Σ_clean, hist μ" : ("#2d6a2d", "--", 2.0),
    "Phase 3 — BL Π, cap"       : ("#2980b9", "-.", 2.4),
    "Equal Weight"               : ("#e67e22", ":",  1.8),
    "VTI Buy-and-Hold"           : ("#7f8c8d", "-",  1.4),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _bar_colors(n):
    return cm.tab10(np.linspace(0, 1, n))

def _dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d35")
    ax.tick_params(colors="#9aa3b5")
    ax.xaxis.label.set_color("#9aa3b5")
    ax.yaxis.label.set_color("#9aa3b5")
    return fig, ax

def _dark_fig_multi(rows, cols, figsize, sharey=False, sharex=False):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=sharey, sharex=sharex)
    fig.patch.set_facecolor("#0f1117")
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor("#0f1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d35")
        ax.tick_params(colors="#9aa3b5")
        ax.xaxis.label.set_color("#9aa3b5")
        ax.yaxis.label.set_color("#9aa3b5")
    return fig, axes

def _validate_tickers(tickers):
    """Quick single-row download to verify each ticker exists on Yahoo Finance."""
    valid, invalid = [], []
    for t in tickers:
        try:
            test = yf.download(t, period="5d", progress=False, auto_adjust=True)
            (valid if not test.empty else invalid).append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid

def _weights_to_csv(tickers, phases):
    """Build a CSV of weights across all three phases for download."""
    rows = []
    for t, w1, w2, w3 in zip(
        tickers,
        phases[0]["w_msr"],
        phases[1]["w_msr"],
        phases[2]["w_msr"],
    ):
        rows.append({
            "Ticker"         : t,
            "Phase1_MaxSR %": round(w1*100, 2),
            "Phase2_MaxSR %": round(w2*100, 2),
            "Phase3_MaxSR %": round(w3*100, 2),
        })
    return pd.DataFrame(rows).to_csv(index=False)

def _metrics_to_csv(bt_results, rf):
    rows = []
    for label, pv in bt_results.items():
        m = compute_metrics(pv, rf)
        rows.append({"Strategy": label, **m})
    return pd.DataFrame(rows).to_csv(index=False)

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

    # Show tag pills (all tentatively valid until checked)
    if tickers:
        st.markdown(
            '<div class="ticker-tags">'
            + "".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
            + "</div>", unsafe_allow_html=True,
        )

    # Live ticker validation button
    if st.button("✓ Validate tickers", help="Check all tickers against Yahoo Finance before running."):
        with st.spinner("Checking tickers…"):
            v, inv = _validate_tickers(tickers)
        st.session_state["validated_tickers"] = v
        st.session_state["invalid_tickers"]   = inv
        if inv:
            st.markdown(
                '<div class="ticker-tags">'
                + "".join(f'<span class="ticker-tag">{t}</span>' for t in v)
                + "".join(f'<span class="ticker-tag-invalid">{t} ✗</span>' for t in inv)
                + "</div>", unsafe_allow_html=True,
            )
            st.warning(f"Remove invalid tickers: {', '.join(inv)}")
        else:
            st.success(f"All {len(v)} tickers valid ✓")

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
    rebal_days = {
        "Monthly (21 days)": 21, "Quarterly (63 days)": 63,
        "Semi-annual (126 days)": 126, "Annual (252 days)": 252,
    }[rebal_freq]
    transaction_cost = st.number_input(
        "Transaction cost (%)", 0.0, 1.0, 0.05, 0.01, format="%.2f"
    ) / 100

    oos_days  = lookback_years * 252 - train_days
    config_ok = len(tickers) >= 2 and oos_days > rebal_days
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

# ── Run computation ───────────────────────────────────────────────────────────
if run_button:
    st.session_state["run"]     = False
    st.session_state["error"]   = None
    st.session_state["results"] = None

    progress = st.progress(0, text="Step 1 / 4 — Downloading price data…")

    try:
        prices = fetch_prices(tickers, lookback_years)
    except Exception as e:
        st.session_state["error"] = f"Data error: {e}"
        progress.empty()
        st.rerun()

    progress.progress(25, text="Step 2 / 4 — Running Phase 1 & 2 optimisation…")

    try:
        opt = run_optimization(prices, risk_free_rate, max_weight)
    except Exception as e:
        st.session_state["error"] = f"Optimisation error: {e}"
        progress.empty()
        st.rerun()

    progress.progress(60, text="Step 3 / 4 — Running backtest (all strategies)…")

    try:
        bt_results, bt_logs = run_backtest(
            prices, train_days, rebal_days, transaction_cost, max_weight, risk_free_rate
        )
    except Exception as e:
        st.session_state["error"] = f"Backtest error: {e}"
        progress.empty()
        st.rerun()

    progress.progress(95, text="Step 4 / 4 — Building charts…")

    st.session_state.update({
        "run"        : True,
        "opt"        : opt,
        "bt_results" : bt_results,
        "bt_logs"    : bt_logs,
        "tickers"    : tickers,
        "rf"         : risk_free_rate,
        "max_weight" : max_weight,
        "prices"     : prices,
    })
    progress.progress(100, text="Done ✓")
    progress.empty()

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
        log_ret = opt["log_returns"]

        # ── Metrics ───────────────────────────────────────────────────────────
        mvp = p1["mvp_metrics"]; msr = p1["msr_metrics"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("MVP Return",     f"{mvp['return']*100:.2f}%")
            st.metric("MVP Volatility", f"{mvp['volatility']*100:.2f}%")
            st.metric("MVP Sharpe",     f"{mvp['sharpe']:.3f}")
        with c2:
            st.metric("Max SR Return",     f"{msr['return']*100:.2f}%")
            st.metric("Max SR Volatility", f"{msr['volatility']*100:.2f}%")
            st.metric("Max SR Sharpe",     f"{msr['sharpe']:.3f}")

        # ── Efficient frontier ────────────────────────────────────────────────
        st.subheader("Efficient Frontier")
        fig, ax = _dark_fig((11, 6))
        sc = ax.scatter(
            p1["frontier_vols"]*100, p1["frontier_rets"]*100,
            c=p1["frontier_sharpes"], cmap="RdYlGn", s=14, zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sharpe Ratio", color="#9aa3b5")
        cbar.ax.yaxis.set_tick_params(color="#9aa3b5")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9aa3b5")

        asset_vols = np.sqrt(np.diag(cov_mat))*100
        asset_rets = mu_vec*100
        for i, t in enumerate(tkrs):
            ax.scatter(asset_vols[i], asset_rets[i], color=colors[i], s=80, zorder=6, edgecolors="white", linewidths=0.6)
            ax.annotate(t, (asset_vols[i], asset_rets[i]), xytext=(4,3), textcoords="offset points", fontsize=8, color=colors[i], fontweight="bold")

        ax.scatter(mvp["volatility"]*100, mvp["return"]*100, marker="*", s=360, color="gold",      edgecolors="black", zorder=7, label="Min Variance")
        ax.scatter(msr["volatility"]*100, msr["return"]*100, marker="*", s=360, color="limegreen", edgecolors="black", zorder=7, label="Max Sharpe")
        ax.scatter([0], [rf*100], color="steelblue", s=50, zorder=5)
        ax.annotate(f"Risk-free {rf*100:.1f}%", (0, rf*100), xytext=(4, -12), textcoords="offset points", fontsize=8, color="steelblue")
        ax.set_title("Efficient Frontier — Phase 1", color="#ffffff", fontsize=12)
        ax.set_xlabel("Annualised Volatility (%)")
        ax.set_ylabel("Annualised Return (%)")
        ax.legend(fontsize=9, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax.grid(True, alpha=0.15, color="#2a2d35")
        ax.set_xlim(left=0)
        st.pyplot(fig); plt.close()

        # ── Correlation heatmap ───────────────────────────────────────────────
        st.subheader("Correlation Matrix")
        corr = log_ret.corr().values
        fig2, ax2 = _dark_fig((9, 7))
        im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        cbar2 = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Correlation", color="#9aa3b5")
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#9aa3b5")
        ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
        ax2.set_xticklabels(tkrs, rotation=45, ha="right", fontsize=9, color="#9aa3b5")
        ax2.set_yticklabels(tkrs, fontsize=9, color="#9aa3b5")
        ax2.set_title("Asset Correlation Matrix", color="#ffffff", fontsize=12)
        for i in range(n):
            for j in range(n):
                ax2.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(corr[i,j]) > 0.6 else "#9aa3b5")
        fig2.tight_layout()
        st.pyplot(fig2); plt.close()

        # ── Weight comparison ─────────────────────────────────────────────────
        st.subheader("Portfolio Weights")
        fig3, axes3 = _dark_fig_multi(1, 2, (12, 4), sharey=True)
        for ax3, (w, title) in zip(axes3, [(p1["w_mvp"],"Min Variance"),(p1["w_msr"],"Max Sharpe")]):
            ax3.barh(tkrs, w*100, color=colors, edgecolor="white", linewidth=0.5)
            ax3.set_title(title, color="#ffffff")
            ax3.invert_yaxis()
            ax3.set_xlabel("Weight (%)")
            ax3.grid(True, axis="x", alpha=0.15, color="#2a2d35")
            for i, val in enumerate(w):
                if val > 0.01:
                    ax3.text(val*100+0.3, i, f"{val*100:.1f}%", va="center", fontsize=8, color="#9aa3b5")
        fig3.tight_layout()
        st.pyplot(fig3); plt.close()

        # ── Download weights ──────────────────────────────────────────────────
        csv_w = _weights_to_csv(tkrs, [p1, opt["phase2"], opt["phase3"]])
        st.download_button(
            "⬇ Download weights (all phases) CSV",
            data=csv_w, file_name="portfolio_weights.csv", mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="phase-title">Phase 2 — PCA Covariance Cleaning</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Applies Marchenko-Pastur spectral theory to separate signal eigenvalues from noise in Σ. Noise eigenvalues — statistically indistinguishable from a random matrix — are replaced with their mean, preserving total variance while removing estimation error.</div>', unsafe_allow_html=True)
    else:
        opt    = st.session_state["opt"]
        tkrs   = opt["tickers"]
        n      = len(tkrs)
        p1     = opt["phase1"]
        p2     = opt["phase2"]
        pca    = opt["pca_info"]
        colors = _bar_colors(n)
        rf     = st.session_state["rf"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Signal eigenvalues", pca["n_signal"])
        c2.metric("Noise eigenvalues",  pca["n_noise"])
        c3.metric("MP threshold λ⁺",    f"{pca['lambda_plus']:.4f}")

        # ── Eigenvalue spectrum ───────────────────────────────────────────────
        st.subheader("Eigenvalue Spectrum")
        fig, ax = _dark_fig((10, 4))
        evs   = pca["eigenvalues"]
        lc    = pca["lambda_clean"]
        x     = np.arange(1, n+1)
        w_bar = 0.38
        ax.bar(x - w_bar/2, evs, w_bar, color="#c0392b", alpha=0.8, label="Raw Σ",    edgecolor="white", linewidth=0.4)
        ax.bar(x + w_bar/2, lc,  w_bar, color="#2d6a2d", alpha=0.9, label="Σ_clean",  edgecolor="white", linewidth=0.4)
        ax.axhline(pca["lambda_plus"], color="crimson", linestyle="--", linewidth=1.5, label=f"MP λ⁺ = {pca['lambda_plus']:.4f}")
        ax.set_xticks(x); ax.set_xticklabels([f"λ{i}" for i in x])
        ax.set_title("Eigenvalue Spectrum: Raw vs Cleaned", color="#ffffff")
        ax.legend(facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white", fontsize=9)
        ax.grid(True, axis="y", alpha=0.15, color="#2a2d35")
        st.pyplot(fig); plt.close()

        # ── Overlaid frontiers ────────────────────────────────────────────────
        st.subheader("Phase 1 vs Phase 2 Frontiers")
        fig2, ax2 = _dark_fig((10, 5))
        ax2.plot(p1["frontier_vols"]*100, p1["frontier_rets"]*100, color="#c0392b", lw=2, label="Phase 1 — Raw Σ")
        ax2.plot(p2["frontier_vols"]*100, p2["frontier_rets"]*100, color="#2d6a2d", lw=2, linestyle="--", label="Phase 2 — Σ_clean")
        ax2.scatter(p1["msr_metrics"]["volatility"]*100, p1["msr_metrics"]["return"]*100, marker="*", s=280, color="#e74c3c", edgecolors="black", zorder=5, label="P1 Max Sharpe")
        ax2.scatter(p2["msr_metrics"]["volatility"]*100, p2["msr_metrics"]["return"]*100, marker="D", s=100, color="#27ae60", edgecolors="black", zorder=5, label="P2 Max Sharpe")
        ax2.set_xlabel("Volatility (%)"); ax2.set_ylabel("Return (%)")
        ax2.set_title("Efficient Frontier Comparison", color="#ffffff")
        ax2.legend(facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white", fontsize=9)
        ax2.grid(True, alpha=0.15, color="#2a2d35")
        ax2.set_xlim(left=0)
        st.pyplot(fig2); plt.close()

        # ── Weight comparison ─────────────────────────────────────────────────
        st.subheader("Max Sharpe Weights: Phase 1 vs Phase 2")
        fig3, axes3 = _dark_fig_multi(1, 2, (12, 4), sharey=True)
        for ax3, (w, title) in zip(axes3, [(p1["w_msr"],"Phase 1"),(p2["w_msr"],"Phase 2")]):
            ax3.barh(tkrs, w*100, color=colors, edgecolor="white", linewidth=0.5)
            ax3.set_title(title, color="#ffffff")
            ax3.invert_yaxis(); ax3.set_xlabel("Weight (%)")
            ax3.grid(True, axis="x", alpha=0.15, color="#2a2d35")
            for i, val in enumerate(w):
                if val > 0.01:
                    ax3.text(val*100+0.3, i, f"{val*100:.1f}%", va="center", fontsize=8, color="#9aa3b5")
        fig3.tight_layout()
        st.pyplot(fig3); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="phase-title">Phase 3 — Black-Litterman Equilibrium</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Replaces historical μ with market-implied equilibrium returns Π = δ × Σ_clean × w_mkt. Assets whose historical returns were inflated by a recent bull run are corrected to market consensus, producing more realistic and diversified allocations.</div>', unsafe_allow_html=True)
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
        mw     = st.session_state["max_weight"]
        colors = _bar_colors(n)

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk aversion δ",    f"{opt['delta']:.3f}")
        c2.metric("Max SR Return (BL)", f"{p3['msr_metrics']['return']*100:.2f}%")
        c3.metric("Max SR Sharpe (BL)", f"{p3['msr_metrics']['sharpe']:.3f}")

        # ── μ vs Π ────────────────────────────────────────────────────────────
        st.subheader("Historical μ vs Equilibrium Π")
        fig, ax = _dark_fig((11, 4))
        x = np.arange(n); w_bar = 0.38
        ax.bar(x - w_bar/2, mu_vec*100, w_bar, color="#c0392b", alpha=0.8, label="Historical μ", edgecolor="white", linewidth=0.4)
        ax.bar(x + w_bar/2, pi*100,     w_bar, color="#2d6a2d", alpha=0.9, label="Equilibrium Π", edgecolor="white", linewidth=0.4)
        ax.axhline(rf*100, color="steelblue", linestyle=":", linewidth=1.2, label=f"Risk-free {rf*100:.1f}%")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tkrs, rotation=30, ha="right")
        ax.set_ylabel("Annualised Return (%)")
        ax.set_title("Historical μ vs Black-Litterman Π", color="#ffffff")
        ax.legend(facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white", fontsize=9)
        ax.grid(True, axis="y", alpha=0.15, color="#2a2d35")
        st.pyplot(fig); plt.close()

        # ── 3-way weight comparison ───────────────────────────────────────────
        st.subheader("Max Sharpe Weights — All Three Phases")
        fig2, axes2 = _dark_fig_multi(1, 3, (16, 4), sharey=True)
        for ax2, (w, title) in zip(axes2, [
            (p1["w_msr"], "Phase 1\nRaw Σ · hist μ"),
            (p2["w_msr"], "Phase 2\nΣ_clean · hist μ"),
            (p3["w_msr"], f"Phase 3\nΣ_clean · BL Π · {mw*100:.0f}% cap"),
        ]):
            ax2.barh(tkrs, w*100, color=colors, edgecolor="white", linewidth=0.5)
            if "Phase 3" in title:
                ax2.axvline(mw*100, color="white", linestyle=":", linewidth=1, alpha=0.4)
            ax2.set_title(title, color="#ffffff", fontsize=9)
            ax2.invert_yaxis(); ax2.set_xlabel("Weight (%)")
            ax2.grid(True, axis="x", alpha=0.15, color="#2a2d35")
            for i, val in enumerate(w):
                if val > 0.01:
                    ax2.text(val*100+0.3, i, f"{val*100:.1f}%", va="center", fontsize=7.5, color="#9aa3b5")
        fig2.tight_layout()
        st.pyplot(fig2); plt.close()

        # ── Phase comparison summary table ────────────────────────────────────
        st.subheader("Phase Comparison — Max Sharpe Portfolio")
        rows = []
        for label, phase in [("Phase 1", p1), ("Phase 2", p2), ("Phase 3", p3)]:
            m = phase["msr_metrics"]
            rows.append({
                "Phase"      : label,
                "Return"     : f"{m['return']*100:.2f}%",
                "Volatility" : f"{m['volatility']*100:.2f}%",
                "Sharpe"     : f"{m['sharpe']:.3f}",
                **{t: f"{w*100:.1f}%" for t, w in zip(tkrs, phase["w_msr"])},
            })
        summary_df = pd.DataFrame(rows).set_index("Phase")
        st.dataframe(summary_df, use_container_width=True)

        # ── Download weights ──────────────────────────────────────────────────
        csv_w = _weights_to_csv(tkrs, [p1, p2, p3])
        st.download_button(
            "⬇ Download weights CSV",
            data=csv_w, file_name="portfolio_weights.csv", mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="phase-title">Backtest</div>', unsafe_allow_html=True)
    if not has_run:
        st.markdown('<div class="phase-intro">Rolling out-of-sample backtest comparing all three phases against Equal Weight and VTI Buy-and-Hold. No look-ahead bias — each rebalancing decision uses only data available at that date.</div>', unsafe_allow_html=True)
    else:
        bt = st.session_state["bt_results"]
        rf = st.session_state["rf"]

        # ── Metrics table ─────────────────────────────────────────────────────
        st.subheader("Performance Summary")
        rows = []
        for label, pv in bt.items():
            m = compute_metrics(pv, rf)
            rows.append({"Strategy": label, **m})
        df = pd.DataFrame(rows).set_index("Strategy")
        df.columns = ["Return %", "CAGR %", "Vol %", "Sharpe", "Max DD %", "Calmar"]

        # Highlight best in each column
        def highlight_best(s):
            if s.name in ["Max DD %"]:
                best = s.max()   # least negative = best
            else:
                best = s.max()
            return ["background-color: #0d2a1a; color: #3ddc84; font-weight: 600"
                    if v == best else "" for v in s]

        st.dataframe(
            df.style
              .apply(highlight_best)
              .format({c: "{:.2f}" for c in ["Return %","CAGR %","Vol %","Max DD %"]}
                      | {"Sharpe": "{:.3f}", "Calmar": "{:.3f}"}),
            use_container_width=True,
        )

        csv_m = _metrics_to_csv(bt, rf)
        st.download_button(
            "⬇ Download metrics CSV",
            data=csv_m, file_name="backtest_metrics.csv", mime="text/csv",
        )

        # ── Cumulative return ─────────────────────────────────────────────────
        st.subheader("Cumulative Returns")
        fig, ax = _dark_fig((12, 5))
        for label, pv in bt.items():
            color, ls, lw = BT_COLORS.get(label, ("#888","-",1.5))
            ax.plot(pv.index, pv.values, color=color, linestyle=ls, linewidth=lw,
                    label=f"{label}  ({(pv.iloc[-1]-1)*100:+.1f}%)")
        ax.axhline(1.0, color="#555", linewidth=0.7, linestyle=":")
        ax.set_ylabel("Portfolio Value ($1.00 start)")
        ax.set_title("Out-of-Sample Cumulative Returns", color="#ffffff")
        ax.legend(fontsize=8, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax.grid(True, alpha=0.15, color="#2a2d35")
        st.pyplot(fig); plt.close()

        # ── Drawdown ──────────────────────────────────────────────────────────
        st.subheader("Drawdown from Peak")
        fig2, ax2 = _dark_fig((12, 4))
        for label, pv in bt.items():
            color, ls, lw = BT_COLORS.get(label, ("#888","-",1.5))
            dd = (pv - pv.cummax()) / pv.cummax() * 100
            ax2.plot(dd.index, dd.values, color=color, linestyle=ls, linewidth=lw, label=label)
        ax2.axhline(0, color="#555", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown from Peak", color="#ffffff")
        ax2.legend(fontsize=8, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax2.grid(True, alpha=0.15, color="#2a2d35")
        st.pyplot(fig2); plt.close()

        # ── Rolling 1-year Sharpe ─────────────────────────────────────────────
        st.subheader("Rolling 1-Year Sharpe Ratio")
        fig3, ax3 = _dark_fig((12, 4))
        for label, pv in bt.items():
            color, ls, lw = BT_COLORS.get(label, ("#888","-",1.5))
            daily  = pv.pct_change().dropna()
            r_roll = daily.rolling(252).mean() * 252
            v_roll = daily.rolling(252).std()  * np.sqrt(252)
            s_roll = (r_roll - rf) / v_roll
            ax3.plot(s_roll.index, s_roll.values, color=color, linestyle=ls, linewidth=lw, label=label)
        ax3.axhline(0, color="#555", linewidth=0.8)
        ax3.axhline(1, color="#888", linewidth=0.7, linestyle=":", alpha=0.6)
        ax3.set_ylabel("Sharpe Ratio (trailing 1yr)")
        ax3.set_title("Rolling 1-Year Sharpe Ratio", color="#ffffff")
        ax3.legend(fontsize=8, facecolor="#1e2130", edgecolor="#2a2d35", labelcolor="white")
        ax3.grid(True, alpha=0.15, color="#2a2d35")
        st.pyplot(fig3); plt.close()

        # ── Download metrics ──────────────────────────────────────────────────
        st.download_button(
            "⬇ Download full metrics CSV",
            data=csv_m, file_name="backtest_metrics.csv", mime="text/csv",
            key="dl_metrics_bottom",
        )

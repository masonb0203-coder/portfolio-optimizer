"""
optimizer.py — Core computation module for the Portfolio Optimizer
===================================================================
All math is extracted directly from portfolio_optimizer.ipynb.
No Streamlit imports here — pure Python/NumPy/pandas/SciPy only.

Public API
----------
fetch_prices(tickers, lookback_years)           → pd.DataFrame
compute_stats(prices, trading_days)             → (log_returns, mu, cov)
pca_clean_cov(cov_mat, n, T)                   → np.ndarray
bl_equilibrium(cov_clean, tickers, mu_vec, rf)  → (w_mkt, delta, pi)
run_optimization(mu_vec, cov_mat, n, rf,
                 max_weight, n_frontier)         → dict
run_backtest(prices, strategies, config)        → (results, rebal_logs)
compute_metrics(pv_series, rf)                  → dict
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# ── 1. Data ───────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], lookback_years: int) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Returns a clean DataFrame: rows = trading days, cols = tickers.
    Forward-fills single-day gaps, drops rows with any remaining NaN
    (assets not yet listed at start of window).

    Raises
    ------
    ValueError  if no valid price data is returned.
    """
    from datetime import datetime, timedelta

    end   = datetime.today()
    start = end - timedelta(days=365 * lookback_years + 10)  # small buffer

    raw = yf.download(
        tickers    = tickers,
        start      = start.strftime("%Y-%m-%d"),
        end        = end.strftime("%Y-%m-%d"),
        auto_adjust= True,
        progress   = False,
    )

    # yfinance returns MultiIndex when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices_raw = raw["Close"].copy()
    else:
        prices_raw = raw[["Close"]].copy()
        prices_raw.columns = tickers

    # Keep only requested tickers (in order)
    available = [t for t in tickers if t in prices_raw.columns]
    if not available:
        raise ValueError("No price data returned. Check ticker symbols.")
    prices_raw = prices_raw[available]

    prices = prices_raw.ffill().dropna()

    if prices.empty:
        raise ValueError(
            "Price data is empty after cleaning. "
            "Try fewer tickers or a shorter lookback."
        )

    return prices


# ── 2. Statistics ─────────────────────────────────────────────────────────────

def compute_stats(
    prices: pd.DataFrame,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute log returns, annualised μ vector, and annualised covariance matrix Σ.

    Returns
    -------
    log_returns : pd.DataFrame  shape (T-1, n)
    mu_vec      : np.ndarray    shape (n,)   annualised expected returns
    cov_mat     : np.ndarray    shape (n, n) annualised sample covariance
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu_vec      = log_returns.mean().values * trading_days
    cov_mat     = log_returns.cov().values  * trading_days
    return log_returns, mu_vec, cov_mat


# ── 3. PCA Covariance Cleaning (Marchenko-Pastur) ────────────────────────────

def pca_clean_cov(
    cov_mat: np.ndarray,
    n: int,
    T: int,
) -> np.ndarray:
    """
    Apply Marchenko-Pastur spectral cleaning to a covariance matrix.

    Algorithm
    ---------
    1. Eigendecompose Σ = Q Λ Q^T  (eigh — symmetric, stable)
    2. Compute MP upper threshold  λ⁺ = σ²(1 + √(n/T))²
    3. Replace noise eigenvalues (λ ≤ λ⁺) with their mean
       — preserves tr(Σ) exactly
    4. Reconstruct Σ_clean = Q Λ_clean Q^T

    Returns
    -------
    cov_clean : np.ndarray  shape (n, n)  PSD, symmetric
    """
    eigenvalues_raw, Q_raw = np.linalg.eigh(cov_mat)

    # Sort descending
    idx         = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[idx]
    Q           = Q_raw[:, idx]

    # Marchenko-Pastur threshold
    q           = n / T
    sigma2      = np.trace(cov_mat) / n
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2

    noise_mask  = eigenvalues <= lambda_plus
    noise_mean  = eigenvalues[noise_mask].mean() if noise_mask.any() else eigenvalues[-1]

    lambda_clean              = eigenvalues.copy()
    lambda_clean[noise_mask]  = noise_mean

    cov_clean = Q @ np.diag(lambda_clean) @ Q.T
    cov_clean = (cov_clean + cov_clean.T) / 2  # symmetrise floating-point drift

    # Diagnostics dict (callers can ignore)
    info = {
        "eigenvalues"   : eigenvalues,
        "lambda_clean"  : lambda_clean,
        "lambda_plus"   : lambda_plus,
        "n_signal"      : int((~noise_mask).sum()),
        "n_noise"       : int(noise_mask.sum()),
        "Q"             : Q,
    }
    return cov_clean, info


# ── 4. Black-Litterman Equilibrium Returns ───────────────────────────────────

def bl_equilibrium(
    cov_clean: np.ndarray,
    tickers: list[str],
    mu_vec: np.ndarray,
    rf: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute Black-Litterman market-implied equilibrium returns.

    Steps
    -----
    1. Fetch market caps (marketCap for stocks, totalAssets for ETFs)
    2. Normalise to within-universe weights w_mkt
    3. Estimate risk aversion δ = (w_mkt^T μ - rf) / (w_mkt^T Σ w_mkt)
       clamped to [1.0, 5.0]
    4. Π = δ × Σ_clean × w_mkt

    Returns
    -------
    w_mkt : np.ndarray  market cap weights (sum to 1)
    delta : float       risk aversion coefficient
    pi    : np.ndarray  equilibrium return vector
    """
    caps = []
    for t in tickers:
        info = yf.Ticker(t).info
        cap  = (
            info.get("marketCap")
            or info.get("totalAssets")
            or info.get("netAssets")
            or 0
        )
        caps.append(max(cap, 0))

    caps  = np.array(caps, dtype=float)
    total = caps.sum()
    if total == 0:
        w_mkt = np.ones(len(tickers)) / len(tickers)
    else:
        w_mkt = caps / total

    mkt_ret = float(w_mkt @ mu_vec)
    mkt_var = float(w_mkt @ cov_clean @ w_mkt)
    delta   = float(np.clip((mkt_ret - rf) / mkt_var if mkt_var > 1e-8 else 2.5, 1.0, 5.0))

    pi = delta * cov_clean @ w_mkt
    return w_mkt, delta, pi


# ── 5. Portfolio metric helpers ───────────────────────────────────────────────

def _port_return(w, mu):     return float(w @ mu)
def _port_vol(w, cov):       return float(np.sqrt(w @ cov @ w))
def _port_sharpe(w, mu, cov, rf): return (_port_return(w, mu) - rf) / _port_vol(w, cov)


def _max_sharpe(mu_vec, cov_mat, n, rf, max_weight=1.0):
    """Find the maximum Sharpe ratio portfolio via SLSQP."""
    bounds     = tuple((0.0, max_weight) for _ in range(n))
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    w0         = np.ones(n) / n

    def neg_sharpe(w):
        r = float(w @ mu_vec)
        v = float(np.sqrt(w @ cov_mat @ w))
        return -(r - rf) / v if v > 1e-10 else 0.0

    res = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=[constraint],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return np.clip(res.x, 0, max_weight) if res.success else w0


def _min_variance(cov_mat, n, max_weight=1.0):
    """Find the minimum variance portfolio via SLSQP."""
    bounds     = tuple((0.0, max_weight) for _ in range(n))
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    w0         = np.ones(n) / n

    res = minimize(
        lambda w: float(np.sqrt(w @ cov_mat @ w)),
        w0, method="SLSQP",
        bounds=bounds, constraints=[constraint],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return np.clip(res.x, 0, max_weight) if res.success else w0


def _efficient_frontier(mu_vec, cov_mat, n, rf, max_weight=1.0, n_points=200):
    """
    Trace the efficient frontier by solving 200 target-return problems.
    Returns arrays of (vols, rets, sharpes, weights).
    """
    w_mvp    = _min_variance(cov_mat, n, max_weight)
    ret_min  = _port_return(w_mvp, mu_vec)
    ret_max  = mu_vec.max() if max_weight == 1.0 else (mu_vec * max_weight).sum()

    bounds     = tuple((0.0, max_weight) for _ in range(n))
    constraint_sum = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    w0         = np.ones(n) / n

    vols, rets, sharpes, weights = [], [], [], []

    for r_target in np.linspace(ret_min, ret_max, n_points):
        constraint_ret = {
            "type": "eq",
            "fun" : lambda w, r=r_target: float(w @ mu_vec) - r,
        }
        res = minimize(
            lambda w: float(np.sqrt(w @ cov_mat @ w)),
            w0, method="SLSQP",
            bounds=bounds,
            constraints=[constraint_sum, constraint_ret],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            w = res.x
            vols.append(_port_vol(w, cov_mat))
            rets.append(_port_return(w, mu_vec))
            sharpes.append(_port_sharpe(w, mu_vec, cov_mat, rf))
            weights.append(w)

    return (
        np.array(vols),
        np.array(rets),
        np.array(sharpes),
        np.array(weights),
    )


# ── 6. Full optimisation run (all 3 phases) ───────────────────────────────────

def run_optimization(
    prices: pd.DataFrame,
    rf: float,
    max_weight: float = 1.0,
    n_frontier: int = 200,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """
    Run all three optimisation phases on the supplied price data.

    Returns a results dict with keys:
        tickers, log_returns, mu_vec, cov_mat,
        cov_clean, pca_info,
        w_mkt, delta, pi,
        phase1, phase2, phase3
    Each phase sub-dict has keys:
        w_mvp, w_msr,
        mvp_metrics, msr_metrics,
        frontier_vols, frontier_rets, frontier_sharpes, frontier_weights
    """
    tickers = list(prices.columns)
    n       = len(tickers)

    # ── Stats ────────────────────────────────────────────────────────────────
    log_returns, mu_vec, cov_mat = compute_stats(prices, trading_days)
    T = len(log_returns)

    # ── PCA cleaning ─────────────────────────────────────────────────────────
    cov_clean, pca_info = pca_clean_cov(cov_mat, n, T)

    # ── BL equilibrium ────────────────────────────────────────────────────────
    w_mkt, delta, pi = bl_equilibrium(cov_clean, tickers, mu_vec, rf)

    def _phase_results(mu, cov, cap):
        w_mvp = _min_variance(cov, n, cap)
        w_msr = _max_sharpe(mu, cov, n, rf, cap)
        fv, fr, fs, fw = _efficient_frontier(mu, cov, n, rf, cap, n_frontier)

        def metrics(w):
            r = _port_return(w, mu)
            v = _port_vol(w, cov)
            return {"return": r, "volatility": v, "sharpe": (r - rf) / v}

        return {
            "w_mvp"             : w_mvp,
            "w_msr"             : w_msr,
            "mvp_metrics"       : metrics(w_mvp),
            "msr_metrics"       : metrics(w_msr),
            "frontier_vols"     : fv,
            "frontier_rets"     : fr,
            "frontier_sharpes"  : fs,
            "frontier_weights"  : fw,
        }

    # Phase 1 — raw Σ, historical μ, no cap
    phase1 = _phase_results(mu_vec, cov_mat,   1.0)
    # Phase 2 — Σ_clean, historical μ, no cap
    phase2 = _phase_results(mu_vec, cov_clean, 1.0)
    # Phase 3 — Σ_clean, BL Π, position cap
    phase3 = _phase_results(pi,     cov_clean, max_weight)

    return {
        "tickers"     : tickers,
        "log_returns" : log_returns,
        "mu_vec"      : mu_vec,
        "cov_mat"     : cov_mat,
        "cov_clean"   : cov_clean,
        "pca_info"    : pca_info,
        "w_mkt"       : w_mkt,
        "delta"       : delta,
        "pi"          : pi,
        "phase1"      : phase1,
        "phase2"      : phase2,
        "phase3"      : phase3,
    }


# ── 7. Backtest ───────────────────────────────────────────────────────────────

def _backtest_single(
    prices: pd.DataFrame,
    strategy: str,
    train_days: int,
    rebal_days: int,
    tc: float,
    max_weight: float,
    rf: float,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.Series, list[dict]]:
    """
    Rolling out-of-sample backtest for one strategy.

    strategy options: 'p1' | 'p2' | 'p3' | 'equal' | 'vti_bh'
    """
    n         = prices.shape[1]
    tickers   = list(prices.columns)
    all_dates = prices.index
    N         = len(all_dates)

    vti_col = tickers.index("VTI") if "VTI" in tickers else 0

    rebal_indices   = list(range(train_days, N - 1, rebal_days))
    pv              = 1.0
    current_weights = np.ones(n) / n
    pv_dict         = {}
    rebal_log       = []

    for k, rebal_idx in enumerate(rebal_indices):
        # Training window
        train_prices  = prices.iloc[rebal_idx - train_days : rebal_idx]
        train_log_ret = np.log(train_prices / train_prices.shift(1)).dropna()
        T_train       = len(train_log_ret)
        mu_train      = train_log_ret.mean().values * trading_days
        cov_train     = train_log_ret.cov().values  * trading_days

        # Compute new weights
        if strategy == "equal":
            new_w = np.ones(n) / n

        elif strategy == "vti_bh":
            new_w          = np.zeros(n)
            new_w[vti_col] = 1.0

        elif strategy == "p1":
            new_w = _max_sharpe(mu_train, cov_train, n, rf, 1.0)

        elif strategy == "p2":
            cov_c, _ = pca_clean_cov(cov_train, n, T_train)
            new_w    = _max_sharpe(mu_train, cov_c, n, rf, 1.0)

        elif strategy == "p3":
            cov_c, _        = pca_clean_cov(cov_train, n, T_train)
            _, _, pi_train  = bl_equilibrium(cov_c, tickers, mu_train, rf)
            new_w           = _max_sharpe(pi_train, cov_c, n, rf, max_weight)

        else:
            new_w = np.ones(n) / n

        # Transaction cost on turnover
        turnover  = 0.5 * np.sum(np.abs(new_w - current_weights))
        pv       *= 1 - tc * turnover

        # Test window
        next_idx   = rebal_indices[k + 1] if k + 1 < len(rebal_indices) else N - 1
        test_slice = prices.iloc[rebal_idx : next_idx + 1]

        rebal_log.append({
            "date"    : all_dates[rebal_idx],
            "weights" : new_w.copy(),
            "turnover": turnover,
        })

        pv_dict[test_slice.index[0]] = pv
        w = new_w.copy()

        for j in range(1, len(test_slice)):
            date        = test_slice.index[j]
            simple_rets = (test_slice.iloc[j].values / test_slice.iloc[j - 1].values) - 1.0
            pv         *= 1.0 + float(w @ simple_rets)
            pv_dict[date] = pv
            w = w * (1.0 + simple_rets)
            s = w.sum()
            if s > 0:
                w /= s

        current_weights = w

    pv_series = pd.Series(pv_dict).sort_index()
    pv_series = pv_series / pv_series.iloc[0]
    return pv_series, rebal_log


def run_backtest(
    prices: pd.DataFrame,
    train_days: int,
    rebal_days: int,
    transaction_cost: float,
    max_weight: float,
    rf: float,
    trading_days: int = TRADING_DAYS,
) -> tuple[dict, dict]:
    """
    Run all five strategies through the backtest engine.

    Returns
    -------
    results    : dict  label → pd.Series (portfolio value, normalised to 1.0)
    rebal_logs : dict  label → list of rebalancing dicts
    """
    strategies = {
        "Phase 1 — Raw Σ, hist μ"   : ("p1",     1.0),
        "Phase 2 — Σ_clean, hist μ" : ("p2",     1.0),
        "Phase 3 — BL Π, cap"       : ("p3",     max_weight),
        "Equal Weight"               : ("equal",  1.0),
        "VTI Buy-and-Hold"           : ("vti_bh", 1.0),
    }

    results    = {}
    rebal_logs = {}

    for label, (code, cap) in strategies.items():
        pv, log = _backtest_single(
            prices           = prices,
            strategy         = code,
            train_days       = train_days,
            rebal_days       = rebal_days,
            tc               = transaction_cost,
            max_weight       = cap,
            rf               = rf,
            trading_days     = trading_days,
        )
        results[label]    = pv
        rebal_logs[label] = log

    # Align to common date range and re-normalise
    common_start = max(pv.index[0] for pv in results.values())
    common_end   = min(pv.index[-1] for pv in results.values())
    results = {k: v[common_start:common_end] for k, v in results.items()}
    results = {k: v / v.iloc[0] for k, v in results.items()}

    return results, rebal_logs


# ── 8. Performance metrics ────────────────────────────────────────────────────

def compute_metrics(pv_series: pd.Series, rf: float) -> dict:
    """
    Compute key out-of-sample performance statistics.

    Returns dict with:
        total_return, cagr, volatility, sharpe, max_drawdown, calmar
    """
    rets     = pv_series.pct_change().dropna()
    n_years  = len(pv_series) / TRADING_DAYS
    total    = float(pv_series.iloc[-1]) - 1.0
    cagr     = float(pv_series.iloc[-1]) ** (1.0 / n_years) - 1.0
    vol      = float(rets.std()) * np.sqrt(TRADING_DAYS)
    sharpe   = (cagr - rf) / vol if vol > 0 else 0.0

    rolling_max  = pv_series.cummax()
    drawdown     = (pv_series - rolling_max) / rolling_max
    max_dd       = float(drawdown.min())
    calmar       = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    return {
        "Total Return" : round(total  * 100, 2),
        "CAGR"         : round(cagr   * 100, 2),
        "Volatility"   : round(vol    * 100, 2),
        "Sharpe"       : round(sharpe,        3),
        "Max Drawdown" : round(max_dd  * 100, 2),
        "Calmar"       : round(calmar,         3),
    }

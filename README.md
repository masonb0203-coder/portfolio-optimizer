# Portfolio Optimizer

A quantitative portfolio optimizer built with Python and Streamlit, progressing through three phases of increasing sophistication.

## What it does

**Phase 1 — Mean-Variance Optimization (Markowitz)**
Estimates expected returns (μ) and a covariance matrix (Σ) from historical prices, then finds the efficient frontier via constrained quadratic optimization.

**Phase 2 — PCA Covariance Cleaning**
Applies Marchenko-Pastur spectral theory to separate signal eigenvalues from noise in Σ, producing a cleaner covariance estimate that stabilizes portfolio weights.

**Phase 3 — Black-Litterman Equilibrium Returns**
Replaces historical μ with market-implied equilibrium returns (Π = δ × Σ_clean × w_mkt), grounding the optimizer in market consensus rather than noisy historical means.

**Backtest**
Rolling out-of-sample backtest with quarterly rebalancing, a 2-year training window, and 0.05% transaction costs — comparing all three phases against Equal Weight and VTI Buy-and-Hold benchmarks.

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, set the main file to `app.py`
5. Click Deploy

## Project structure

```
portfolio-optimizer/
├── app.py              # Streamlit entry point
├── optimizer.py        # Core computation module (Phase B)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Tech stack

- **Streamlit** — UI framework
- **yfinance** — free historical price data from Yahoo Finance
- **NumPy / pandas** — matrix operations and time-series handling
- **SciPy** — constrained optimization (SLSQP)
- **Matplotlib** — charts embedded in the Streamlit UI

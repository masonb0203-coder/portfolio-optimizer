"""
Portfolio Optimizer — Streamlit App
====================================
Entry point. Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Portfolio Optimizer")
st.caption("Mean-Variance · PCA Covariance Cleaning · Black-Litterman")

st.info("App coming soon — Phase B (controls) and Phase C (optimizer) are next.")

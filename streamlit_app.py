import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")

st.title("📊 Monte Carlo Portfolio Simulator")
st.markdown("""
This tool simulates future portfolio performance using **Geometric Brownian Motion (GBM)**.
It automatically cleans data, handles missing tickers, and re-normalizes weights.
""")

# --- 2. Sidebar: User Inputs ---
st.sidebar.header("Configuration")

# Default values (Updated to Indian Tickers for INR context)
default_tickers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, GOLDBEES.NS"
default_weights = "0.4, 0.3, 0.2, 0.1"

ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)
weights_input = st.sidebar.text_input("Enter Weights (comma separated)", default_weights)

st.sidebar.markdown("---")
# Updated label to Rupees
initial_investment = st.sidebar.number_input("Initial Investment (₹)", value=100000, step=5000)
years = st.sidebar.slider("Years to Forecast", min_value=1, max_value=10, value=1)
simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)

# --- 3. Helper Function to Process Inputs ---
def process_inputs(ticker_str, weight_str):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    try:
        weights = np.array([float(w) for w in weight_str.split(",")])
    except ValueError:
        return None, None

    if len(tickers) != len(weights):
        st.error(f"Error: You provided {len(tickers)} tickers but {len(weights)} weights. Please ensure they match.")
        return None, None

    return tickers, weights

# --- 4. Main Execution ---
if st.sidebar.button("Run Simulation", type="primary"):
    tickers, weights = process_inputs(ticker_input, weights_input)

    if tickers and weights is not None:
        with st.spinner('Fetching and aligning market data...'):
            try:
                # A. Fetch Data
                raw_data = yf.download(tickers, period="5y", auto_adjust=True)['Close']

                # Handle single-ticker edge case
                if isinstance(raw_data, pd.Series):
                    raw_data = raw_data.to_frame()
                    raw_data.columns = tickers

                # B. Robust Data Cleaning
                # 1. Remove tickers that returned NO data (all NaNs)
                found_tickers = raw_data.columns[~raw_data.isnull().all()].tolist()
                missing = list(set(tickers) - set(found_tickers))

                if missing:
                    st.warning(f"⚠️ Skipped (No Data found): {', '.join(missing)}")

                # 2. Filter data to found tickers only
                data = raw_data[found_tickers]

                # 3. Fill forward to handle holiday gaps, then drop remaining NaNs
                data = data.ffill().dropna()

                if data.empty or len(data) < 30:
                    st.error("❌ Error: Not enough overlapping historical data. Try older tickers or a shorter timeframe.")
                    st.stop()

                # C. Re-align weights based on surviving tickers
                indices_to_keep = [tickers.index(t) for t in found_tickers]
                final_weights = weights[indices_to_keep]
                
                # Normalize weights to sum to 1.0
                weight_sum = np.sum(final_weights)
                if weight_sum == 0:
                    st.error("Error: All valid assets have 0 weight.")
                    st.stop()
                final_weights = final_weights / weight_sum 

                # D. Calculate Statistics
                log_returns = np.log(1 + data.pct_change()).dropna()
                mean_daily = log_returns.mean().to_numpy()
                std_daily = log_returns.std().to_numpy()
                corr_matrix = log_returns.corr().to_numpy()

                # E. Cholesky Decomposition
                epsilon = 1e-8
                try:
                    L = np.linalg.cholesky(corr_matrix + epsilon * np.eye(len(found_tickers)))
                except np.linalg.LinAlgError:
                    st.warning("⚠️ Correlation matrix is not positive definite. Falling back to uncorrelated shocks.")
                    L = np.eye(len(found_tickers))

                # F. Vectorized Simulation Engine
                days = int(252 * years)
                num_assets = len(found_tickers)
                
                # 1. Generate random shocks
                uncorr_shocks = np.random.normal(0, 1, (simulations, days, num_assets))
                
                # 2. Correlate the shocks
                corr_shocks = np.einsum('ij, mdj -> mdi', L, uncorr_shocks)
                
                # 3. Calculate Daily Returns
                drift = mean_daily -

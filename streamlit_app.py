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
""")

# --- 2. Sidebar: User Inputs ---
st.sidebar.header("Configuration")

default_tickers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, GOLDBEES.NS"
default_weights = "0.4, 0.3, 0.2, 0.1"

ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)
weights_input = st.sidebar.text_input("Enter Weights (comma separated)", default_weights)

st.sidebar.markdown("---")
initial_investment = st.sidebar.number_input("Initial Investment (₹)", value=100000, step=5000)
years = st.sidebar.slider("Years to Forecast", min_value=1, max_value=10, value=1)
simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)

# --- 3. Helper Function ---
def process_inputs(ticker_str, weight_str):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    try:
        weights = np.array([float(w) for w in weight_str.split(",")])
    except ValueError:
        return None, None
    if len(tickers) != len(weights):
        st.error(f"Error: {len(tickers)} tickers vs {len(weights)} weights.")
        return None, None
    return tickers, weights

# --- 4. Main Execution ---
if st.sidebar.button("Run Simulation", type="primary"):
    tickers, weights = process_inputs(ticker_input, weights_input)

    if tickers and weights is not None:
        with st.spinner('Fetching market data...'):
            try:
                # A. Fetch Data
                raw_data = yf.download(tickers, period="5y", auto_adjust=True)['Close']

                if isinstance(raw_data, pd.Series):
                    raw_data = raw_data.to_frame()
                    raw_data.columns = tickers

                # B. Data Cleaning
                found_tickers = raw_data.columns[~raw_data.isnull().all()].tolist()
                data = raw_data[found_tickers].ffill().dropna()

                if data.empty or len(data) < 30:
                    st.error("Not enough historical data overlap.")
                    st.stop()

                # C. Weights Alignment
                indices_to_keep = [tickers.index(t) for t in found_tickers]
                final_weights = weights[indices_to_keep]
                final_weights = final_weights / np.sum(final_weights) 

                # D. Statistics
                log_returns = np.log(1 + data.pct_change()).dropna()
                mean_daily = log_returns.mean().to_numpy()
                std_daily = log_returns.std().to_numpy()
                corr_matrix = log_returns.corr().to_numpy()

                # E. Cholesky
                epsilon = 1e-8
                L = np.linalg.cholesky(corr_matrix + epsilon * np.eye(len(found_tickers)))

                # F. Vectorized Simulation
                days = int(252 * years)
                num_assets = len(found_tickers)
                uncorr_shocks = np.random.normal(0, 1, (simulations, days, num_assets))
                corr_shocks = np.einsum('ij, mdj -> mdi', L, uncorr_shocks)
                
                drift = mean_daily - 0.5 * std_daily**2
                daily_returns = np.exp(drift + std_daily * corr_shocks)
                path_multipliers = np.cumprod(daily_returns, axis=1)
                
                current_prices = data.iloc[-1].to_numpy()
                price_paths = current_prices * path_multipliers
                
                shares = (initial_investment * final_weights) / current_prices
                portfolio_values = np.dot(price_paths, shares)

                # --- 5. UI Results ---
                st.success("✅ Simulation Complete")
                final_vals = portfolio_values[:, -1]
                expected_val = np.mean(final_vals)
                var_95 = np.percentile(final_vals, 5)

                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Value", f"₹{expected_val:,.2f}")
                col2.metric("VaR (95%)", f"₹{var_95:,.2f}")
                col3.metric("Min Simulation", f"₹{np.min(final_vals):,.2f}")

                tab1, tab2 = st.tabs(["📉 Paths", "📊 Distribution"])
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(portfolio_values[:100, :].T, color='royalblue', alpha=0.1)
                    ax.set_ylabel("Value (₹)")
                    st.pyplot(fig)
                with tab2:
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    # Histogram of final outcomes
                    ax2.hist(final_vals, bins=50, color='teal', alpha=0.7, edgecolor='white')
                    
                    # --- ADDING VaR LINE ---
                    ax2.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f"VaR 95%: ₹{var_95:,.0f}")
                    
                    # Adding an arrow or annotation for clarity
                    ax2.annotate('95% Confidence Loss Threshold', 
                                 xy=(var_95, 0), xytext=(var_95*0.8, 10),
                                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                 horizontalalignment='right')
                    
                    ax2.axvline(initial_investment, color='black', linewidth=1, label="Initial Investment")
                    ax2.set_title("Distribution of Final Outcomes & Risk Threshold")
                    ax2.set_xlabel("Final Portfolio Value (₹)")
                    ax2.set_ylabel("Frequency")
                    ax2.legend()
                    st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error: {e}")

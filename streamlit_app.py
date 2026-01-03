import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================================
# --- ⚙️ CONFIGURATION BLOCK ---
# ==========================================================
st.title("📈 Monte Carlo Portfolio Simulation")

TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'GOLDBEES.NS']
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
YEARS = 1
SIMULATIONS = 2000  # Lowered slightly for faster cloud performance
INITIAL_INVESTMENT = 100000 
# ==========================================================

def run_simulation():
    # 1. Setup & Data Fetching
    st.write(f"Fetching data for: `{TICKERS}`...")
    
    # DOWNLOAD FIX: Handle errors and empty data
    try:
        data = yf.download(TICKERS, period="3y", auto_adjust=True)['Close']
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return

    # Handle single ticker edge cases
    if len(TICKERS) == 1:
        data = data.to_frame()
        data.columns = TICKERS
        
    # DATA CLEANING FIX: Fill missing values before dropping
    # This prevents one bad day/ticker from wiping out the whole dataset
    data = data.ffill()
    data = data.dropna()

    # CHECK: Stop if data is empty after cleaning
    if data.empty:
        st.error("❌ No data found. Try different tickers or check yfinance.")
        return

    # 2. Statistics & Correlation
    log_returns = np.log(1 + data.pct_change()).dropna()
    mean_daily = log_returns.mean().to_numpy()
    std_daily = log_returns.std().to_numpy()
    corr_matrix = log_returns.corr().to_numpy()
    
    # Cholesky Decomposition for correlated assets
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        st.warning("⚠️ Matrix not positive definite. Using uncorrelated shocks.")
        L = np.eye(len(TICKERS)) # Fallback

    # 3. Vectorized Simulation Engine
    st.write(f"🚀 Simulating {SIMULATIONS} paths...")
    days = int(252 * YEARS)
    num_assets = len(TICKERS)
    
    # Generate Correlated Shocks
    uncorr_shocks = np.random.normal(0, 1, (SIMULATIONS, days, num_assets))
    corr_shocks = np.einsum('ij, mdj -> mdi', L, uncorr_shocks)
    
    # Calculate Price Paths using Geometric Brownian Motion
    drift = mean_daily - 0.5 * std_daily**2
    daily_returns = np.exp(drift + std_daily * corr_shocks)
    
    # Accumulate returns over time
    path_multipliers = np.cumprod(daily_returns, axis=1)
    
    # Apply to starting prices
    current_prices = data.iloc[-1].to_numpy()
    price_paths = current_prices * path_multipliers
    
    # 4. Portfolio Value Calculation
    shares = (INITIAL_INVESTMENT * WEIGHTS) / current_prices
    portfolio_values = np.dot(price_paths, shares)
    
    # 5. Metrics
    final_vals = portfolio_values[:, -1]
    expected_val = np.mean(final_vals)
    var_95 = np.percentile(final_vals, 5)
    cvar_95 = final_vals[final_vals <= var_95].mean()

    # 6. Output & Visualization (Converted for Streamlit)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Value", f"₹{expected_val:,.2f}")
    col2.metric("VaR (95%)", f"₹{var_95:,.2f}")
    col3.metric("Conditional VaR", f"₹{cvar_95:,.2f}")
    st.write("---")

    # Visualizing the distribution of outcomes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: The Simulation "Fan"
    ax1.plot(portfolio_values[:150, :].T, alpha=0.15, color='royalblue')
    ax1.set_title(f"Monte Carlo Paths ({YEARS} Year Forecast)")
    ax1.set_ylabel("Portfolio Value (₹)")
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Final Distribution
    ax2.hist(final_vals, bins=60, color='teal', alpha=0.7, edgecolor='white')
    ax2.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: ₹{var_95:,.0f}')
    ax2.axvline(expected_val, color='gold', linewidth=2, label=f'Expected: ₹{expected_val:,.0f}')
    ax2.set_title("Distribution of Final Outcomes")
    ax2.set_xlabel("Final Value (₹)")
    ax2.legend()
    
    plt.tight_layout()
    
    # STREAMLIT FIX: Use st.pyplot instead of plt.show
    st.pyplot(fig)

if __name__ == "__main__":
    run_simulation()

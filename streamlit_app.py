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

# Default values
default_tickers = "SPY, TLT, GLD, BTC-USD"
default_weights = "0.4, 0.3, 0.2, 0.1"

ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)
weights_input = st.sidebar.text_input("Enter Weights (comma separated)", default_weights)

st.sidebar.markdown("---")
initial_investment = st.sidebar.number_input("Initial Investment ($)", value=10000, step=1000)
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
                # Note: We download a bit more history to ensure we have enough overlap after cleaning
                raw_data = yf.download(tickers, period="5y", auto_adjust=True)['Close']

                # Handle single-ticker edge case (Series -> DataFrame)
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
                # We need to filter the original weights to match the tickers that actually had data
                indices_to_keep = [tickers.index(t) for t in found_tickers]
                final_weights = weights[indices_to_keep]
                
                # Normalize weights to sum to 1.0 (in case we dropped a ticker)
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

                # E. Cholesky Decomposition (with stability epsilon)
                # This ensures the matrix is positive semi-definite
                epsilon = 1e-8
                try:
                    L = np.linalg.cholesky(corr_matrix + epsilon * np.eye(len(found_tickers)))
                except np.linalg.LinAlgError:
                    st.warning("⚠️ Correlation matrix is not positive definite. Falling back to uncorrelated shocks.")
                    L = np.eye(len(found_tickers))

                # F. Vectorized Simulation Engine (Fast!)
                days = int(252 * years)
                num_assets = len(found_tickers)
                
                # 1. Generate random shocks
                # Shape: (Simulations, Days, Assets)
                uncorr_shocks = np.random.normal(0, 1, (simulations, days, num_assets))
                
                # 2. Correlate the shocks using Cholesky
                corr_shocks = np.einsum('ij, mdj -> mdi', L, uncorr_shocks)
                
                # 3. Calculate Daily Returns (Geometric Brownian Motion)
                drift = mean_daily - 0.5 * std_daily**2
                daily_returns = np.exp(drift + std_daily * corr_shocks)
                
                # 4. Cumulative Returns Path
                # Prepend a row of 1s (starting point)
                path_multipliers = np.cumprod(daily_returns, axis=1)
                
                # 5. Apply to current prices
                current_prices = data.iloc[-1].to_numpy()
                price_paths = current_prices * path_multipliers
                
                # 6. Calculate Portfolio Value Path
                # shares = (Investment * Weight) / Price
                shares = (initial_investment * final_weights) / current_prices
                portfolio_values = np.dot(price_paths, shares)

                # --- 5. Results & Visualization ---
                st.success(f"✅ Simulation complete: {len(found_tickers)} assets over {years} year(s).")

                final_vals = portfolio_values[:, -1]
                expected_val = np.mean(final_vals)
                var_95 = np.percentile(final_vals, 5)
                max_drawdown = (np.min(final_vals) - initial_investment)

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Value", f"${expected_val:,.2f}", delta=f"{((expected_val/initial_investment)-1)*100:.1f}%")
                col2.metric("VaR (95%)", f"${var_95:,.2f}", delta_color="inverse")
                col3.metric("Min Simulation Value", f"${np.min(final_vals):,.2f}", delta_color="inverse")

                # Charts
                tab1, tab2 = st.tabs(["📉 Simulation Paths", "📊 Distribution"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    # Plot only first 100 paths to keep chart clean
                    ax.plot(portfolio_values[:100, :].T, color='royalblue', alpha=0.1)
                    ax.set_title(f"Monte Carlo Paths (First 100 of {simulations})")
                    ax.set_ylabel("Portfolio Value ($)")
                    ax.set_xlabel("Trading Days")
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)

                with tab2:
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.hist(final_vals, bins=50, color='teal', alpha=0.7, edgecolor='white')
                    ax2.axvline(var_95, color='red', linestyle='--', label=f"VaR 95%: ${var_95:,.0f}")
                    ax2.axvline(initial_investment, color='black', linewidth=1, label="Initial Inv.")
                    ax2.set_title("Distribution of Final Outcomes")
                    ax2.legend()
                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"Critical Simulation Error: {e}")

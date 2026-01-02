import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================================
# --- ⚙️ CONFIGURATION BLOCK (Edit your settings here) ---
# ==========================================================
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'GOLDBEES.NS']
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
YEARS = 1
SIMULATIONS = 10000        # High number for better accuracy
INITIAL_INVESTMENT = 100000 
# ==========================================================

def run_simulation():
    # 1. Setup & Data Fetching
    print(f"Fetching data for: {TICKERS}...")
    data = yf.download(TICKERS, period="3y", auto_adjust=True)['Close']
    
    # Handle single ticker edge cases
    if len(TICKERS) == 1:
        data = data.to_frame()
        data.columns = TICKERS
        
    data = data.ffill().dropna()
    
    # 2. Statistics & Correlation
    log_returns = np.log(1 + data.pct_change()).dropna()
    mean_daily = log_returns.mean().to_numpy()
    std_daily = log_returns.std().to_numpy()
    corr_matrix = log_returns.corr().to_numpy()
    
    # Cholesky Decomposition for correlated assets
    L = np.linalg.cholesky(corr_matrix)

    # 3. Vectorized Simulation Engine
    print(f"🚀 Simulating {SIMULATIONS} paths...")
    days = int(252 * YEARS)
    num_assets = len(TICKERS)
    
    # Generate Correlated Shocks
    # Shape: (Simulations, Days, Assets)
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

    # 6. Output & Visualization
    print("\n" + "="*30)
    print(f"EXPECTED VALUE: ₹{expected_val:,.2f}")
    print(f"VALUE AT RISK (95%): ₹{var_95:,.2f}")
    print(f"CONDITIONAL VaR: ₹{cvar_95:,.2f}")
    print("="*30)

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
    plt.show()

if __name__ == "__main__":
    run_simulation()
    

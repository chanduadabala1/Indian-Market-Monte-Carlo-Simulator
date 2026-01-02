import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_user_inputs():
    print("\n--- ⚙️ PORTFOLIO CONFIGURATION ⚙️ ---")
    
    # 1. Tickers
    default_tickers = "RELIANCE.NS, INFY.NS, TCS.NS, GOLDBEES.NS"
    ticker_str = input(f"Enter Tickers (comma separated) [Default: {default_tickers}]: ").strip()
    if not ticker_str:
        tickers = [t.strip() for t in default_tickers.split(',')]
    else:
        tickers = [t.strip() for t in ticker_str.split(',')]
        
    # 2. Weights
    print(f"Detected {len(tickers)} assets.")
    default_weights = "0.4, 0.2, 0.2, 0.2"
    weight_str = input(f"Enter Weights (comma separated) [Default: {default_weights}]: ").strip()
    if not weight_str:
        weights = np.array([float(w) for w in default_weights.split(',')])
    else:
        weights = np.array([float(w) for w in weight_str.split(',')])
    
    # Validation
    if len(tickers) != len(weights):
        raise ValueError(f"Mismatch: You entered {len(tickers)} tickers but {len(weights)} weights.")
    weights /= np.sum(weights) # Normalize to 1.0

    # 3. Simulation Params
    try:
        years = float(input("Enter Simulation Years [Default: 1]: ") or 1)
        sims = int(input("Enter # of Simulations [Default: 5000]: ") or 5000)
        investment = float(input("Enter Initial Investment (₹) [Default: 100000]: ") or 100000)
    except ValueError:
        print("Invalid number entered. Using defaults.")
        years, sims, investment = 1, 5000, 100000

    return tickers, weights, years, sims, investment

# --- MAIN EXECUTION ---
try:
    tickers, weights, years, simulations, investment = get_user_inputs()
    
    print(f"\nFetching data for: {tickers}...")
    # Fetch data (Auto-adjust handles splits/dividends)
    data = yf.download(tickers, period="3y", auto_adjust=True)['Close']
    
    # Handle single ticker edge case or missing data
    if len(tickers) == 1:
        data = data.to_frame()
        data.columns = tickers
    
    data = data.ffill().dropna()
    
    if data.empty:
        raise ValueError("No data fetched. Check ticker spelling.")

    # --- STATISTICS ---
    log_returns = np.log(1 + data.pct_change()).dropna()
    mean_daily = log_returns.mean().to_numpy()
    std_daily = log_returns.std().to_numpy()
    corr_matrix = log_returns.corr().to_numpy()
    
    # Cholesky Decomposition (for correlation)
    L = np.linalg.cholesky(corr_matrix)

    # --- VECTORIZED SIMULATION (The Fast Part) ---
    print(f"🚀 Running {simulations} simulations for {years} years...")
    
    days = int(252 * years)
    num_assets = len(tickers)
    
    # 1. Generate uncorrelated random shocks for ALL simulations at once
    # Shape: (Simulations, Days, Assets)
    uncorr_shocks = np.random.normal(0, 1, (simulations, days, num_assets))
    
    # 2. Apply Correlation (Einstein Summation for speed)
    # This multiplies the Cholesky matrix 'L' with the random shocks
    corr_shocks = np.einsum('ij, mdj -> mdi', L, uncorr_shocks)
    
    # 3. Calculate Cumulative Returns (Geometric Brownian Motion)
    # Formula: P_t = P_0 * exp( cumsum( drift + sigma * Z ) )
    drift = mean_daily - 0.5 * std_daily**2
    
    # Broadcast drift and std to match shock shape
    daily_log_returns = drift + std_daily * corr_shocks
    cumulative_returns = np.exp(np.cumsum(daily_log_returns, axis=1))
    
    # 4. Apply to Starting Prices
    current_prices = data.iloc[-1].to_numpy()
    # Add a starting row of 1.0 (or actual prices) for t=0
    # Shape: (Simulations, Days, Assets)
    price_paths = current_prices * cumulative_returns
    
    # 5. Calculate Portfolio Value
    # Shares purchased at t=0
    shares = (investment * weights) / current_prices
    
    # Portfolio Value = Sum (Price_asset * Shares_asset) across assets
    # Shape: (Simulations, Days)
    portfolio_values = np.dot(price_paths, shares)
    
    # --- RESULTS & METRICS ---
    final_values = portfolio_values[:, -1]
    expected_val = np.mean(final_values)
    median_val = np.median(final_values)
    var_95 = np.percentile(final_values, 5)  # 5th percentile (95% confidence)
    cvar_95 = final_values[final_values <= var_95].mean() # Conditional VaR

    print("\n--- 📊 SIMULATION RESULTS ---")
    print(f"Initial Investment:  ₹{investment:,.2f}")
    print(f"Expected Value:      ₹{expected_val:,.2f} (Mean)")
    print(f"Median Outcome:      ₹{median_val:,.2f} (More robust than mean)")
    print(f"VaR (95% Risk):      ₹{var_95:,.2f} (Worst case in 95% of times)")
    print(f"CVaR (Tail Risk):    ₹{cvar_95:,.2f} (Avg loss in worst 5% scenarios)")

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 6))
    
    # Plot 1: 100 Sample Paths
    plt.subplot(1, 2, 1)
    plt.plot(portfolio_values[:100, :].T, alpha=0.3, linewidth=1)
    plt.title(f"Simulation Paths (First 100 of {simulations})")
    plt.ylabel("Portfolio Value (₹)")
    plt.xlabel("Trading Days")
    plt.grid(alpha=0.3)
    
    # Plot 2: Final Distribution
    plt.subplot(1, 2, 2)
    plt.hist(final_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR: ₹{var_95/1000:.0f}k')
    plt.axvline(expected_val, color='green', linestyle='-', linewidth=2, label=f'Mean: ₹{expected_val/1000:.0f}k')
    plt.axvline(investment, color='black', linestyle=':', linewidth=2, label='Initial')
    plt.title("Final Portfolio Value Distribution")
    plt.xlabel("Value (₹)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n❌ Error: {e}")

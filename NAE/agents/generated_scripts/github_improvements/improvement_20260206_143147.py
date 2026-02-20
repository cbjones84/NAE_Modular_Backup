"""
Auto-implemented improvement from GitHub
Source: bhanukaranwal/Options-Trading-Bot/data_fetcher.py
Implemented: 2026-02-06T14:31:49.617833
Usefulness Score: 80
Keywords: def , strategy, backtest, calculate, simulate, volatility, size, greek
"""

# Original source: bhanukaranwal/Options-Trading-Bot
# Path: data_fetcher.py


# Function: generate_sample_options_data
def generate_sample_options_data(days=365, symbol="NIFTY", initial_price=25000):
    """
    Generates a realistic but simulated 1-minute options data CSV for backtesting.
    """
    logging.info(f"Generating {days} days of 1-minute sample data for {symbol}...")
    
    start_date = datetime.now() - timedelta(days=days)
    minutes = days * 24 * 60
    timestamps = pd.to_datetime(pd.date_range(start=start_date, periods=minutes, freq='T'))
    
    # Simulate underlying price movement (Geometric Brownian Motion)
    drift = 0.05 / (365 * 24 * 60) # 5% annual drift
    volatility = 0.20 / np.sqrt(365 * 24 * 60) # 20% annual volatility
    returns = np.exp(drift + volatility * np.random.randn(minutes))
    underlying_prices = initial_price * returns.cumprod()

    # Create a base DataFrame
    df = pd.DataFrame({'timestamp': timestamps, 'underlying_price': underlying_prices})
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'underlying_price': 'close'}, inplace=True)
    
    # Simulate OHLC
    df['open'] = df['close'].shift(1)
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 5, size=len(df))
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 5, size=len(df))
    df.dropna(inplace=True)
    
    # For simplicity in this example, we won't generate a full options chain.
    # The backtester will use this underlying price data and a strategy
    # that dynamically calculates option prices. For a more advanced simulation,
    # you would generate data for multiple strikes and expiries.
    
    filepath = "nifty_options_data.csv"
    df.to_csv(filepath)
    logging.info(f"Sample data saved to {filepath}")
    return df



# Function: load_historical_data
def load_historical_data(filepath):
    """Loads historical data from a CSV file into a pandas DataFrame."""
    try:
        logging.info(f"Loading historical data from {filepath}...")
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        # Ensure standard column names for backtrader
        df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low'}, inplace=True)
        df['volume'] = 0  # Add dummy volume if not present
        df['openinterest'] = 0 # Add dummy open interest
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found at {filepath}. Please generate it first.")
        raise



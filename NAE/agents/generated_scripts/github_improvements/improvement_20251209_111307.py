"""
Auto-implemented improvement from GitHub
Source: Nikkitaseth/ProjectAlpha/Program.py
Implemented: 2025-12-09T11:13:07.007289
Usefulness Score: 100
Keywords: def , optimize, calculate, compute, simulate, loss, risk, sharpe, volatility, var, size, stop, loss
"""

# Original source: Nikkitaseth/ProjectAlpha
# Path: Program.py


# Function: get_Portfolio
def get_Portfolio(tickers, start_date, end_date):
    stock_data = data.DataReader(tickers, data_source='yahoo', start = start_date, end = end_date)['Adj Close']
    
    return stock_data

# Change shape of DataFrame with optimized weights so they can be used as a source for VaR


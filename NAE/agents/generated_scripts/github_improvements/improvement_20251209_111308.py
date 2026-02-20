"""
Auto-implemented improvement from GitHub
Source: gayathriravi318/VaR-calculation/IIQF full project.py
Implemented: 2025-12-09T11:13:08.008899
Usefulness Score: 80
Keywords: def , calculate, simulate, risk, volatility, var
"""

# Original source: gayathriravi318/VaR-calculation
# Path: IIQF full project.py


# Function: calculate_parametric_var
def calculate_parametric_var(start_date, end_date, weights, confidence_level=0.99):
    # Tickers and Weights
    tickers = ['DLF.NS', 'NTPC.NS', 'HDFCBANK.NS']
    
    # Download historical data for the given tickers
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate portfolio variance and standard deviation
    portfolio_variance = np.dot(weights.T, np.dot(returns.cov(), weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    # Calculate Z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)
    
    # Calculate the Parametric VaR
    var = z_score * portfolio_std * np.sqrt(len(portfolio_returns))
    
    return var

# Define the weights for DLF, NTPC, and HDFC Bank


# Function: calculate_historical_var
def calculate_historical_var(start_date, end_date, weights, confidence_level=0.99):
    # Tickers and Weights
    tickers = ['DLF.NS', 'NTPC.NS', 'HDFCBANK.NS']
    
    # Download historical data for the given tickers
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Sort portfolio returns
    sorted_returns = np.sort(portfolio_returns)
    
    # Calculate the VaR using historical simulation
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[var_index]  # VaR is a positive number
    
    return var

# Define the weights for DLF, NTPC, and HDFC Bank


# Function: monte_carlo_option_pricing
def monte_carlo_option_pricing(S, K, T, r, sigma, option_type='call', simulations=10000):
    """
    Calculate the price of a European option using Monte Carlo simulation.

    Parameters:
    S : float : Current price of the underlying asset (Nifty)
    K : float : Strike price of the option
    T : float : Time to maturity in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying asset (annualized)
    option_type : str : 'call' for call option, 'put' for put option
    simulations : int : Number of simulations to run

    Returns:
    option_price : float : The simulated option price
    """
    # Generate random price paths using Geometric Brownian Motion (GBM)
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs for call or put option
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    
    # Calculate the present value of the expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

# Parameters for Nifty Option


# Function: bond_price
def bond_price(face_value, coupon_rate, yield_curve, maturity, payment_frequency=1):
    """
    Calculate the price of a bond given a yield curve.

    face_value : float : The face value of the bond.
    coupon_rate : float : The annual coupon rate.
    yield_curve : float : The yield for a given maturity.
    maturity : int : The bond's maturity in years.
    payment_frequency : int : Number of coupon payments per year.

    Returns:
    bond_price : float : The price of the bond.
    """
    coupon_payment = (coupon_rate / payment_frequency) * face_value
    price = 0
    
    for t in range(1, maturity * payment_frequency + 1):
        discount_factor = 1 / (1 + yield_curve / payment_frequency) ** t
        price += coupon_payment * discount_factor

    # Add the face value's present value
    price += face_value / (1 + yield_curve / payment_frequency) ** (maturity * payment_frequency)
    
    return price



# Function: dv01
def dv01(face_value, coupon_rate, yield_curve, maturity, shift=0.0001, payment_frequency=1):
    """
    Calculate the DV01 for a bond.
    
    face_value : float : The face value of the bond.
    coupon_rate : float : The annual coupon rate.
    yield_curve : float : The yield for a given maturity.
    maturity : int : The bond's maturity in years.
    shift : float : The interest rate shift in decimal (1 basis point = 0.0001).
    payment_frequency : int : Number of coupon payments per year.

    Returns:
    dv01_value : float : The DV01 of the bond.
    """
    price_original = bond_price(face_value, coupon_rate, yield_curve, maturity, payment_frequency)
    price_shifted = bond_price(face_value, coupon_rate, yield_curve + shift, maturity, payment_frequency)
    
    dv01_value = price_original - price_shifted
    return dv01_value



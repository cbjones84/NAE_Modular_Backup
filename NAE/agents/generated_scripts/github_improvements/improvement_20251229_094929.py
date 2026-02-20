"""
Auto-implemented improvement from GitHub
Source: quantgalore/volatility-surface/vol-surface-backtest.py
Implemented: 2025-12-29T09:49:29.890194
Usefulness Score: 80
Keywords: def , optimize, calculate, risk, volatility, size
"""

# Original source: quantgalore/volatility-surface
# Path: vol-surface-backtest.py


# Function: black_scholes
def black_scholes(option_type, S, K, t, r, q, sigma):
    """
    Calculate the Black-Scholes option price.
    
    :param option_type: 'call' for call option, 'put' for put option.
    :param S: Current stock price.
    :param K: Strike price.
    :param t: Time to expiration (in years).
    :param r: Risk-free interest rate (annualized).
    :param q: Dividend yield (annualized).
    :param sigma: Stock price volatility (annualized).
    
    :return: Option price.
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    
    if option_type == 'call':
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")



# Function: call_implied_vol
def call_implied_vol(S, K, t, r, option_price):
    q = 0.015
    option_type = "call"
    
    def f_call(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    call_newton_vol = optimize.newton(f_call, x0=0.15, tol=0.05, maxiter=50)
    
    return call_newton_vol
                


# Function: f_call
def f_call(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    call_newton_vol = optimize.newton(f_call, x0=0.15, tol=0.05, maxiter=50)
    
    return call_newton_vol
                


# Function: put_implied_vol
def put_implied_vol(S, K, t, r, option_price):
    q = 0.015
    option_type = "put"
    
    def f_put(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price
    
    put_newton_vol = optimize.newton(f_put, x0=0.15, tol=0.05, maxiter=50)
    
    return put_newton_vol        



# Function: f_put
def f_put(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price
    
    put_newton_vol = optimize.newton(f_put, x0=0.15, tol=0.05, maxiter=50)
    
    return put_newton_vol        



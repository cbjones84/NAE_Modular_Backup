"""
Auto-implemented improvement from GitHub
Source: simeen2000/Project-on-Options-and-Portfolio-Optimization/Exercise_1_oblig(4).py
Implemented: 2025-12-09T11:47:05.797804
Usefulness Score: 100
Keywords: def , optimize, calculate, compute, train, fit, volatility, var, size
"""

# Original source: simeen2000/Project-on-Options-and-Portfolio-Optimization
# Path: Exercise_1_oblig(4).py


# Function: portfolio_return_volatility
def portfolio_return_volatility(weights, expected_weekly_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_weekly_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Function to calculate portfolio variance


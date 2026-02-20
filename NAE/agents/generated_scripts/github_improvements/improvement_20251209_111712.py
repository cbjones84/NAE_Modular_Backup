"""
Auto-implemented improvement from GitHub
Source: ApurvShah007/portfolio-optimizer/basic_portfolio_functions.py
Implemented: 2025-12-09T11:17:12.582306
Usefulness Score: 100
Keywords: def , optimize, train, loss, risk, sharpe, volatility, var, loss
"""

# Original source: ApurvShah007/portfolio-optimizer
# Path: basic_portfolio_functions.py


# Function: basicStats
def basicStats(df, weights, start):
    #Calculating the essential Values for the uder entered portfolio
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_volatility = np.sqrt(port_variance)
    annual_return = np.sum(returns.mean()*weights) * 252

    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_vols = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(annual_return, 2)*100)+'%'

    df = df.pct_change()[1:]
    df_spy = web.DataReader('SPY', data_source='yahoo', start = start, end = datetime.today().strftime('%Y-%m-%d'))['Adj Close']
    df_spy = df_spy.pct_change()[1:]
    port_ret = (df * weights).sum(axis = 1)
    (beta, alpha) = stats.linregress(df_spy.values, port_ret.values)[0:2]
                
    #This prints the stats for the portfolio passed in by the user
    print("The basic stats of the portfolio: ")
    print("Expected annual return : ", percent_ret)
    print('Annual volatility/standard deviation/risk : ',percent_vols)
    print('Annual variance : ',percent_var)
    print("Portfolio Beta :", round(beta, 4))



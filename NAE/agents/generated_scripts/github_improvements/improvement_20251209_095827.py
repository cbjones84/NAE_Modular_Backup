"""
Auto-implemented improvement from GitHub
Source: giuseppecangemi/quantfinance/Put's_greek.py
Implemented: 2025-12-09T09:58:27.148593
Usefulness Score: 80
Keywords: def , risk, volatility, greek
"""

# Original source: giuseppecangemi/quantfinance
# Path: Put's_greek.py


# Function: Delta_Put
def Delta_Put(Price, T, volatility, strike, risk_free):
    d1 = ((np.log(Price / strike)) + (risk_free + ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    Delta = scipy.stats.norm.cdf(d1)-1
    return Delta



# Function: Gamma_Put
def Gamma_Put(Price, T, volatility, strike, risk_free):
    d1 = ((np.log(Price / strike)) + (risk_free + ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    phi = (np.exp((-d1**2)/2)/(np.sqrt(2*math.pi)))
    Gamma = (phi)/(Price*volatility*np.sqrt(T))
    return Gamma


# Function: Theta_Put
def Theta_Put (Price, T, volatility, strike, risk_free):
    d1 = ((np.log(Price / strike)) + (risk_free + ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    d2 = ((np.log(Price / strike)) + (risk_free - ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    phi = (np.exp((-d1**2)/2)/(np.sqrt(2*math.pi)))
    Nd2 =scipy.stats.norm.cdf(-d2)
    Theta = -((Price*phi*volatility)/(2*np.sqrt(T)))+(risk_free*strike*np.exp(-risk_free*T)*Nd2)
    return Theta


# Function: Vega_put
def Vega_put(Price, T, volatility, strike, risk_free):
    d1 = ((np.log(Price / strike)) + (risk_free + ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    Nd1 = (np.exp((-d1**2)/2))/(np.sqrt(2*np.pi))
    Vega = Price*np.sqrt(T)*Nd1
    return Vega


# Function: Rho_put
def Rho_put(Price, T, volatility, strike, risk_free):
    d1 = ((np.log(Price / strike)) + (risk_free + ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    d2 = ((np.log(Price / strike)) + (risk_free - ((volatility ** 2) / 2)) * T) / (volatility * np.sqrt(T))
    Rho = Rho = (-strike*T*np.exp(-risk_free*T))*scipy.stats.norm.cdf(-d2)
    return Rho


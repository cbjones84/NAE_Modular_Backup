"""
Auto-implemented improvement from GitHub
Source: dipanb/quant_models/fin_functions.py
Implemented: 2025-12-09T11:14:04.470648
Usefulness Score: 100
Keywords: def , class , optimize, calculate, simulate, model, train, fit, risk, sharpe, var, position, size
"""

# Original source: dipanb/quant_models
# Path: fin_functions.py


# Function: generate_portfolios
def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_wts = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random(len(stocks))
        w/=np.sum(w)
        portfolio_wts.append(w)
        portfolio_means.append(np.sum(returns.mean()*w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T,np.dot(returns.cov(),w))*NUM_TRADING_DAYS))

        return np.array(portfolio_wts),np.array(portfolio_means),np.array(portfolio_risks)

    def show_portfolios(returns,vols):
        plt.figure(figsize=(10,6))
        plt.scatter(vols,returns,c=returns/vols,marker='o')
        plt.grid(True)
        plt.colorbar(label = 'Sharpe Ratio')




# Function: optimize_portfolio
def optimize_portfolio(wts,returns):
    constraints = {'type':'eq','fun':lambda x: np.sum(x)-1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_fn_sharpe, x0=wts[0],args=returns,method='SLSOP',
                          bounds = bounds,constraints=constraints)



# Function: __init__
def __init__(self,stocks,start_date,end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.beta = None
        self.exp_return = None

    def donwload_data(self):
        stock_data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            stock_data[stock] = ticker['Adj Close']
        return pd.DataFrame(stock_data)

    def initialize(self):
        stocks_data = self.donwload_data()
        #Using monthly returns
        stocks_data = stocks_data.resample('M').last()
        self.data = pd.DataFrame({'s_adjclose':stocks_data[self.stocks[0]],
                                  'm_adjclose':stocks_data[self.stocks[1]]})
        self.data[['s_returns','m_returns']] = np.log(self.data[['s_adjclose','m_adjclose']]/self.data[['s_adjclose','m_adjclose']].shift(1))
        self.data = self.data[1:]

    def calculate_beta(self):
        cov_mat = np.cov(self.data['s_returns'],self.data['m_returns'])
        self.beta = cov_mat[0,1]/cov_mat[1,1]


    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        self.exp_return = RISK_FREE_RATE + beta*(self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)



# Function: donwload_data
def donwload_data(self):
        stock_data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            stock_data[stock] = ticker['Adj Close']
        return pd.DataFrame(stock_data)

    def initialize(self):
        stocks_data = self.donwload_data()
        #Using monthly returns
        stocks_data = stocks_data.resample('M').last()
        self.data = pd.DataFrame({'s_adjclose':stocks_data[self.stocks[0]],
                                  'm_adjclose':stocks_data[self.stocks[1]]})
        self.data[['s_returns','m_returns']] = np.log(self.data[['s_adjclose','m_adjclose']]/self.data[['s_adjclose','m_adjclose']].shift(1))
        self.data = self.data[1:]

    def calculate_beta(self):
        cov_mat = np.cov(self.data['s_returns'],self.data['m_returns'])
        self.beta = cov_mat[0,1]/cov_mat[1,1]


    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        self.exp_return = RISK_FREE_RATE + beta*(self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)



# Function: initialize
def initialize(self):
        stocks_data = self.donwload_data()
        #Using monthly returns
        stocks_data = stocks_data.resample('M').last()
        self.data = pd.DataFrame({'s_adjclose':stocks_data[self.stocks[0]],
                                  'm_adjclose':stocks_data[self.stocks[1]]})
        self.data[['s_returns','m_returns']] = np.log(self.data[['s_adjclose','m_adjclose']]/self.data[['s_adjclose','m_adjclose']].shift(1))
        self.data = self.data[1:]

    def calculate_beta(self):
        cov_mat = np.cov(self.data['s_returns'],self.data['m_returns'])
        self.beta = cov_mat[0,1]/cov_mat[1,1]


    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        self.exp_return = RISK_FREE_RATE + beta*(self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)



# Function: calculate_beta
def calculate_beta(self):
        cov_mat = np.cov(self.data['s_returns'],self.data['m_returns'])
        self.beta = cov_mat[0,1]/cov_mat[1,1]


    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        self.exp_return = RISK_FREE_RATE + beta*(self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)



# Function: regression
def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'],self.data['s_returns'],deg=1)
        self.exp_return = RISK_FREE_RATE + beta*(self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE)



# Function: __init__
def __init__(self,S0,E,T,rf,sigma,iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = stock_price-self.E
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)

    def put_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = self.E-stock_price
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)



# Function: call_option_sim
def call_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = stock_price-self.E
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)

    def put_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = self.E-stock_price
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)



# Function: put_option_sim
def put_option_sim(self):
        options_data = np.zeros([self.iterations,2])
        rand = np.random.normal(0,1,[1,self.iterations])
        stock_price = self.S0*np.exp(self.T*(self.rf-0.5*self.sigma**2)+self.sigma*np.sqrt(self.T)*rand)
        options_data[:.1] = self.E-stock_price
        average = np.sum(np.amax(options_data,axis=1))/float(self.iterations)
        return average*np.exp(-self.rf*self.T)



# Function: calculate_var
def calculate_var(position,c,mu,sigma,n=1):
    z = stats.norm.ppf(1-c)
    var = position *(mu*n-sigma*z*np.sqrt(n))
    return var



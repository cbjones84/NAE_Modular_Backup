"""
Auto-implemented improvement from GitHub
Source: VithuQFin/Market-Risk-Models/VaR&KDE.py
Implemented: 2025-12-09T11:14:03.908044
Usefulness Score: 80
Keywords: def , calculate, compute, risk, var, size
"""

# Original source: VithuQFin/Market-Risk-Models
# Path: VaR&KDE.py


# Function: read_and_prepare_data
def read_and_prepare_data(csv_path):
    """
    Reads the dataset, formats it, and computes daily returns.

    Returns:
        df_clean (DataFrame): Cleaned dataset with computed returns.
    """
    df = pd.read_csv(csv_path, sep=';', header=0, decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Close'] = df['Close'].astype(str).str.replace(',', '.').astype(float)

    # Compute daily returns
    df['rendements'] = df['Close'].pct_change()
    df.dropna(inplace=True)  # Remove first NaN due to pct_change()

    return df


###############################################################################
# 2) CALCULATE NON-PARAMETRIC VAR
###############################################################################



# Function: compute_var
def compute_var(returns, alpha=0.95):
    """
    Computes the non-parametric Value-at-Risk (VaR) using the empirical CDF.

    Returns:
        non_parametric_var (float): Estimated VaR.
    """
    x_range = np.linspace(min(returns), max(returns), 100000)
    cdf_values = [empirical_cdf(x, returns) for x in x_range]

    for i, cdf in enumerate(cdf_values):
        if cdf >= (1 - alpha):
            return x_range[i]
    return None


###############################################################################
# 3) KERNEL DENSITY ESTIMATION
###############################################################################



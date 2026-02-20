"""
Auto-implemented improvement from GitHub
Source: 4Rbelaez/prediction-markets/main.py
Implemented: 2026-02-02T13:43:07.575916
Usefulness Score: 100
Keywords: def , class , optimize, calculate, compute, simulate, model, predict, fit, volatility, var, size
"""

# Original source: 4Rbelaez/prediction-markets
# Path: main.py


# Function: __init__
def __init__(self, transaction_cost: float = 0.02):
        """Initialize detector with transaction costs"""
        self.transaction_cost = transaction_cost

    def detect_simple_arbitrage(self, market: Market):
        """Detect if buying both YES and NO guarantees profit"""
        total_cost = market.yes_price + market.no_price
        net_cost = total_cost * (1 + self.transaction_cost)

        if net_cost < 1.0:
            profit_margin = (1.0 - net_cost) / net_cost

            return {
                'market': market.name,
                'type': 'simple_arbitrage',
                'total_cost': total_cost,
                'net_cost': net_cost,
                'profit_margin': profit_margin,
                'guaranteed_return': profit_margin * 100
            }

        return None



# Function: detect_simple_arbitrage
def detect_simple_arbitrage(self, market: Market):
        """Detect if buying both YES and NO guarantees profit"""
        total_cost = market.yes_price + market.no_price
        net_cost = total_cost * (1 + self.transaction_cost)

        if net_cost < 1.0:
            profit_margin = (1.0 - net_cost) / net_cost

            return {
                'market': market.name,
                'type': 'simple_arbitrage',
                'total_cost': total_cost,
                'net_cost': net_cost,
                'profit_margin': profit_margin,
                'guaranteed_return': profit_margin * 100
            }

        return None



# Function: __init__
def __init__(self, initial_price: float, volatility: float, periods: int):
        """
        Initialize the binomial price model

         Args:
            initial_price: Starting probability/price (0-1)
            volatility: Price volatility (standard deviation)
            periods: Number of time periods to simulate
            """
        self.p0 = initial_price
        self.sigma = volatility
        self.periods = periods

    def simulate_paths(self, n_simulations: int = 1000):
        """
        Simulate multiple price paths using Bernoulli random variables

        Returns:
            Array of shape (n_simulations, periods+1) with price paths
        """
        # Create array to store all paths
        paths = np.zeros((n_simulations, self.periods + 1))

        # Set starting price for all paths
        paths[:, 0] = self.p0

        # Calculate up and down factors
        u = np.exp(self.sigma)
        d = np.exp(-self.sigma)

        # Simulate each path
        for sim in range(n_simulations):
            for t in range(1, self.periods + 1):
                # Bernoulli trial: flip a coin (0 or 1)
                # Use np.random.binomial(1, 0.5) to get 0 or 1
                coin_flip = np.random.binomial(1, 0.5)

                if coin_flip == 1:
                    # Price goes up
                    paths[sim, t] = paths[sim, t-1] * u
                else:
                    # Price goes down
                    paths[sim, t] = paths[sim, t-1] * d

                # Clip price to valid range [0.01, 0.99]
                paths[sim, t] = np.clip(paths[sim, t], 0.01, 0.99)

        return paths



# Function: simulate_paths
def simulate_paths(self, n_simulations: int = 1000):
        """
        Simulate multiple price paths using Bernoulli random variables

        Returns:
            Array of shape (n_simulations, periods+1) with price paths
        """
        # Create array to store all paths
        paths = np.zeros((n_simulations, self.periods + 1))

        # Set starting price for all paths
        paths[:, 0] = self.p0

        # Calculate up and down factors
        u = np.exp(self.sigma)
        d = np.exp(-self.sigma)

        # Simulate each path
        for sim in range(n_simulations):
            for t in range(1, self.periods + 1):
                # Bernoulli trial: flip a coin (0 or 1)
                # Use np.random.binomial(1, 0.5) to get 0 or 1
                coin_flip = np.random.binomial(1, 0.5)

                if coin_flip == 1:
                    # Price goes up
                    paths[sim, t] = paths[sim, t-1] * u
                else:
                    # Price goes down
                    paths[sim, t] = paths[sim, t-1] * d

                # Clip price to valid range [0.01, 0.99]
                paths[sim, t] = np.clip(paths[sim, t], 0.01, 0.99)

        return paths



# Function: __init__
def __init__(self, bankroll: float = 10000.0):
        """
        Initialize optimizer with starting bankroll

        Args:
            bankroll: Total money available to bet
        """
        self.bankroll = bankroll

    def kelly_fraction(self, true_prob: float, market_price: float) -> float:
        """
        Calculate Kelly fraction for binary bet

        Args:
            true_prob: Your estimate true probability of event
            market_price: Market's implied probability (price)

        Returns:
            Optimal fraction of bankroll to bet
        """
        # If market price >= true probability, no edge, dont bet
        if market_price >= true_prob:
            return 0.0

        # Calculate odds
        # b = (1 / market_price) - 1
        b = (1 / market_price) - 1

        # Calculate probabilities
        p = true_prob
        q = 1 - p

        # Calculate Kelly fraction
        # kelly = (p * b - q) / b
        kelly = (p * b - q) / b

        # Use fractional Kelly for safety (half Kelly)
        # Cap at 25% of bankroll maximum
        return max(0, min(kelly * 0.5, 0.25))

    def optimize_portfolio(self, markets: list, true_probs: np.ndarray) -> dict:
        """
        Optimize allocation across multiple markets

        Args:
            markets: List of Market objects
            true_probs: Array of your estimated true probabilities

        Returns:
            Dictionary Mapping markets names to dollar amounts to bet
        """
        allocations = {}
        remaining_bankroll = self.bankroll

        # Loop through each market
        for i, market in enumerate(markets):
            # Calculate Kelly fraction for this market
            kelly_frac = self.kelly_fraction(true_probs[i], market.yes_price)

            # Calculate bet size (kelly_frac * remaining_bankroll)
            bet_size = kelly_frac * remaining_bankroll

            # Store allocation
            allocations[market.name] = bet_size

            # Update remaining bankroll
            remaining_bankroll -= bet_size

        return allocations



# Function: __init__
def __init__(self):
        """Initialize analyzer with empty history"""
        self.history = []

    def add_market_observation(self, market_price: float, outcome: int):
        """
        Record a market price and its eventual outcome

        Args:
            market_price: Historical market price (0-1)
            outcome: Actual outcome (0 or 1)
        """

        # Append tuple (market_price, outcome) to self.history
        self.history.append((market_price, outcome))

    def compute_brier_score(self) -> float:
        """
        Compute Brier score to measure forecast accuracy
        Lower is better (0 = perfect, 1 = worst)

        Formula: (1/n) * sum((price - outcome)^2)
        """
        if not self.history:
            return None

        # Calculate Brier score
        # For each (price, outcome), compute (price - outcome)^2
        # Then tale the mean
        scores = [(price - outcome) ** 2 for price, outcome in self.history]

        return np.mean(scores)

    def binomial_calibration_test(self, price_bins: int = 10):
        """
        Test if market prices are well-calibrated using binomial test

        Markets trading at 0.7 should resolve to 1 about 70% of the time

        Returns:
            DataFrame with calibration statistics
        """

        if len(self.history) < 10:
            return None

        # Convert history to DataFrame
        import pandas as pd
        df = pd.DataFrame(self.history, columns=["price", "outcome"])

        # Create price bins
        df['price_bin'] = pd.cut(df['price'], bins=price_bins)

        results = []

        # Group by bins and test each bin
        for bin_name, group in df.groupby('price_bin', observed=True):
            if len(group) < 3:  # Need at least 3 observations
                continue

            # Calculate average price in this bin
            avg_price = group['price'].mean()

            # Calculate actual outcome rate
            actual_rate = group['outcome'].mean()

            # Get sample size
            n = len(group)

            # Perform binomial test
            # Test: Are the outcomes consistent with avg_price?
            n_successes = group['outcome'].sum()
            p_value = stats.binomtest(n_successes, n, avg_price).pvalue

            # Append results
            results.append({
                'price_range': str(bin_name),
                'avg_market_price': actual_rate,
                'actual_outcome_rate': actual_rate,
                'n_observations': n,
                'p_value': p_value,
                'calibrated': p_value < 0.05,  # If p > 0.05, we accept calibration
            })

        return pd.DataFrame(results)



# Function: add_market_observation
def add_market_observation(self, market_price: float, outcome: int):
        """
        Record a market price and its eventual outcome

        Args:
            market_price: Historical market price (0-1)
            outcome: Actual outcome (0 or 1)
        """

        # Append tuple (market_price, outcome) to self.history
        self.history.append((market_price, outcome))

    def compute_brier_score(self) -> float:
        """
        Compute Brier score to measure forecast accuracy
        Lower is better (0 = perfect, 1 = worst)

        Formula: (1/n) * sum((price - outcome)^2)
        """
        if not self.history:
            return None

        # Calculate Brier score
        # For each (price, outcome), compute (price - outcome)^2
        # Then tale the mean
        scores = [(price - outcome) ** 2 for price, outcome in self.history]

        return np.mean(scores)

    def binomial_calibration_test(self, price_bins: int = 10):
        """
        Test if market prices are well-calibrated using binomial test

        Markets trading at 0.7 should resolve to 1 about 70% of the time

        Returns:
            DataFrame with calibration statistics
        """

        if len(self.history) < 10:
            return None

        # Convert history to DataFrame
        import pandas as pd
        df = pd.DataFrame(self.history, columns=["price", "outcome"])

        # Create price bins
        df['price_bin'] = pd.cut(df['price'], bins=price_bins)

        results = []

        # Group by bins and test each bin
        for bin_name, group in df.groupby('price_bin', observed=True):
            if len(group) < 3:  # Need at least 3 observations
                continue

            # Calculate average price in this bin
            avg_price = group['price'].mean()

            # Calculate actual outcome rate
            actual_rate = group['outcome'].mean()

            # Get sample size
            n = len(group)

            # Perform binomial test
            # Test: Are the outcomes consistent with avg_price?
            n_successes = group['outcome'].sum()
            p_value = stats.binomtest(n_successes, n, avg_price).pvalue

            # Append results
            results.append({
                'price_range': str(bin_name),
                'avg_market_price': actual_rate,
                'actual_outcome_rate': actual_rate,
                'n_observations': n,
                'p_value': p_value,
                'calibrated': p_value < 0.05,  # If p > 0.05, we accept calibration
            })

        return pd.DataFrame(results)



# Function: binomial_calibration_test
def binomial_calibration_test(self, price_bins: int = 10):
        """
        Test if market prices are well-calibrated using binomial test

        Markets trading at 0.7 should resolve to 1 about 70% of the time

        Returns:
            DataFrame with calibration statistics
        """

        if len(self.history) < 10:
            return None

        # Convert history to DataFrame
        import pandas as pd
        df = pd.DataFrame(self.history, columns=["price", "outcome"])

        # Create price bins
        df['price_bin'] = pd.cut(df['price'], bins=price_bins)

        results = []

        # Group by bins and test each bin
        for bin_name, group in df.groupby('price_bin', observed=True):
            if len(group) < 3:  # Need at least 3 observations
                continue

            # Calculate average price in this bin
            avg_price = group['price'].mean()

            # Calculate actual outcome rate
            actual_rate = group['outcome'].mean()

            # Get sample size
            n = len(group)

            # Perform binomial test
            # Test: Are the outcomes consistent with avg_price?
            n_successes = group['outcome'].sum()
            p_value = stats.binomtest(n_successes, n, avg_price).pvalue

            # Append results
            results.append({
                'price_range': str(bin_name),
                'avg_market_price': actual_rate,
                'actual_outcome_rate': actual_rate,
                'n_observations': n,
                'p_value': p_value,
                'calibrated': p_value < 0.05,  # If p > 0.05, we accept calibration
            })

        return pd.DataFrame(results)



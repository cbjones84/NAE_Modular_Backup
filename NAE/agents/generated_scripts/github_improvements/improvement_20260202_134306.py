"""
Auto-implemented improvement from GitHub
Source: 4Rbelaez/prediction-markets/backtesting.py
Implemented: 2026-02-02T13:43:06.719809
Usefulness Score: 100
Keywords: def , class , strategy, backtest, calculate, simulate, model, predict, fit, loss, position, size, loss
"""

# Original source: 4Rbelaez/prediction-markets
# Path: backtesting.py


# Function: close_trade
def close_trade(self, outcome: int, exit_time: int):
        """
        Close the trade and calculate P&L

        Args:
            outcome: 1 if event happened, 0 if not
            exit_time: Time period when closed
        """
        self.outcome = outcome
        self.exit_time = exit_time

        # Calculate P&L
        # If outcome = 1: we win, get back position_size / entry_price
        # If outcome = 0: we lose, get back 0
        if outcome == 1:
            self.pnl = (self.position_size / self.entry_price) - self.position_size
        else:
            self.pnl = -self.position_size




# Function: __init__
def __init__(self, initial_bankroll: float = 10000):
        """
        Initialize backtester

        Args:
            initial_bankroll: Starting capital
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.trades: List[Trade] = []
        self.equity_curve = [initial_bankroll]
        self.time = 0

    def run_strategy(self, markets: List[HistoricalMarket],
                     strategy_func, entry_time: int = 5):
        """
        Run a trading strategy on historical data

        Args:
            markets: List of HistoricalMarket objects
            strategy_func: Function that takes (market, time) and returns bet_size
            entry_time: What time period to enter trades
        """
        print(f"Running backtest with ${self.initial_bankroll:,.0f} starting capital...")
        print(f"Testing on {len(markets)} markets")
        print("=" * 70)

        # Enter trades at entry_time
        for market in markets:
            price_at_entry = market.prices[entry_time]

            # Call strategy function to get bet size
            bet_size = strategy_func(market, entry_time, price_at_entry)

            if bet_size > 0 and bet_size <= self.bankroll:
                # Create trade
                trade = Trade(
                    market_name=market.name,
                    entry_time=entry_time,
                    entry_price=price_at_entry,
                    position_size=bet_size
                )

                # Execute trade
                self.bankroll -= bet_size
                self.trades.append(trade)

        # Close all trades at the end
        for trade in self.trades:
            for market in markets:
                if market.name == trade.market_name:
                    trade.close_trade(market.outcome, len(market.prices) - 1)
                    self.bankroll += trade.position_size + trade.pnl
                    break

        # Update equity curve
        self.equity_curve.append(self.bankroll)

        # Calculate and display metrics
        self._display_results()

    def _display_results(self):
        """Display backtest results"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        # Calculate metrics
        total_return = self.bankroll - self.initial_bankroll
        return_pct = (total_return / self.initial_bankroll) * 100

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Display
        print(f"\nInitial Bankroll:    ${self.initial_bankroll:>10,.2f}")
        print(f"Final Bankroll:      ${self.bankroll:>10,.2f}")
        print(f"Total Return:        ${total_return:>10,.2f} ({return_pct:+.2f}%)")
        print(f"\nTotal Trades:        {len(self.trades):>10}")
        print(f"Winning Trades:      {len(winning_trades):>10}")
        print(f"Losing Trades:       {len(losing_trades):>10}")
        print(f"Win Rate:            {win_rate:>10.1f}%")
        print(f"\nAverage Win:         ${avg_win:>10,.2f}")
        print(f"Average Loss:        ${avg_loss:>10,.2f}")

        # Best and worst trades
        if self.trades:
            best_trade = max(self.trades, key=lambda t: t.pnl)
            worst_trade = min(self.trades, key=lambda t: t.pnl)
            print(f"\nBest Trade:          ${best_trade.pnl:>10,.2f} ({best_trade.market_name})")
            print(f"Worst Trade:         ${worst_trade.pnl:>10,.2f} ({worst_trade.market_name})")

        print("=" * 70)

    def plot_results(self):
        """Plot backtest results"""
        if len(self.trades) == 0:
            print("No trades to plot!")
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        # TODO: Plot equity over time
        ax1.plot([self.initial_bankroll, self.bankroll], 'b-o', linewidth=3, markersize=10)
        ax1.axhline(self.initial_bankroll, color='gray', linestyle='--',
                    label='Initial Bankroll')
        ax1.fill_between([0, 1], self.initial_bankroll,
                         [self.initial_bankroll, self.bankroll],
                         alpha=0.3, color='green' if self.bankroll > self.initial_bankroll else 'red')

        # Formatting
        ax1.set_ylabel('Bankroll ($)', fontsize=12)
        ax1.set_title('Equity Curve', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Start', 'End'])

        # BOTTOM PLOT: P&L distribution
        # Create histogram of trade P&Ls
        pnls = [t.pnl for t in self.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, edgecolor='black', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)

        # Formatting
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax2.set_title('Trade P&L Distribution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


# Example strategies


# Function: run_strategy
def run_strategy(self, markets: List[HistoricalMarket],
                     strategy_func, entry_time: int = 5):
        """
        Run a trading strategy on historical data

        Args:
            markets: List of HistoricalMarket objects
            strategy_func: Function that takes (market, time) and returns bet_size
            entry_time: What time period to enter trades
        """
        print(f"Running backtest with ${self.initial_bankroll:,.0f} starting capital...")
        print(f"Testing on {len(markets)} markets")
        print("=" * 70)

        # Enter trades at entry_time
        for market in markets:
            price_at_entry = market.prices[entry_time]

            # Call strategy function to get bet size
            bet_size = strategy_func(market, entry_time, price_at_entry)

            if bet_size > 0 and bet_size <= self.bankroll:
                # Create trade
                trade = Trade(
                    market_name=market.name,
                    entry_time=entry_time,
                    entry_price=price_at_entry,
                    position_size=bet_size
                )

                # Execute trade
                self.bankroll -= bet_size
                self.trades.append(trade)

        # Close all trades at the end
        for trade in self.trades:
            for market in markets:
                if market.name == trade.market_name:
                    trade.close_trade(market.outcome, len(market.prices) - 1)
                    self.bankroll += trade.position_size + trade.pnl
                    break

        # Update equity curve
        self.equity_curve.append(self.bankroll)

        # Calculate and display metrics
        self._display_results()

    def _display_results(self):
        """Display backtest results"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        # Calculate metrics
        total_return = self.bankroll - self.initial_bankroll
        return_pct = (total_return / self.initial_bankroll) * 100

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Display
        print(f"\nInitial Bankroll:    ${self.initial_bankroll:>10,.2f}")
        print(f"Final Bankroll:      ${self.bankroll:>10,.2f}")
        print(f"Total Return:        ${total_return:>10,.2f} ({return_pct:+.2f}%)")
        print(f"\nTotal Trades:        {len(self.trades):>10}")
        print(f"Winning Trades:      {len(winning_trades):>10}")
        print(f"Losing Trades:       {len(losing_trades):>10}")
        print(f"Win Rate:            {win_rate:>10.1f}%")
        print(f"\nAverage Win:         ${avg_win:>10,.2f}")
        print(f"Average Loss:        ${avg_loss:>10,.2f}")

        # Best and worst trades
        if self.trades:
            best_trade = max(self.trades, key=lambda t: t.pnl)
            worst_trade = min(self.trades, key=lambda t: t.pnl)
            print(f"\nBest Trade:          ${best_trade.pnl:>10,.2f} ({best_trade.market_name})")
            print(f"Worst Trade:         ${worst_trade.pnl:>10,.2f} ({worst_trade.market_name})")

        print("=" * 70)

    def plot_results(self):
        """Plot backtest results"""
        if len(self.trades) == 0:
            print("No trades to plot!")
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        # TODO: Plot equity over time
        ax1.plot([self.initial_bankroll, self.bankroll], 'b-o', linewidth=3, markersize=10)
        ax1.axhline(self.initial_bankroll, color='gray', linestyle='--',
                    label='Initial Bankroll')
        ax1.fill_between([0, 1], self.initial_bankroll,
                         [self.initial_bankroll, self.bankroll],
                         alpha=0.3, color='green' if self.bankroll > self.initial_bankroll else 'red')

        # Formatting
        ax1.set_ylabel('Bankroll ($)', fontsize=12)
        ax1.set_title('Equity Curve', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Start', 'End'])

        # BOTTOM PLOT: P&L distribution
        # Create histogram of trade P&Ls
        pnls = [t.pnl for t in self.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, edgecolor='black', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)

        # Formatting
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax2.set_title('Trade P&L Distribution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


# Example strategies


# Function: _display_results
def _display_results(self):
        """Display backtest results"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        # Calculate metrics
        total_return = self.bankroll - self.initial_bankroll
        return_pct = (total_return / self.initial_bankroll) * 100

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Display
        print(f"\nInitial Bankroll:    ${self.initial_bankroll:>10,.2f}")
        print(f"Final Bankroll:      ${self.bankroll:>10,.2f}")
        print(f"Total Return:        ${total_return:>10,.2f} ({return_pct:+.2f}%)")
        print(f"\nTotal Trades:        {len(self.trades):>10}")
        print(f"Winning Trades:      {len(winning_trades):>10}")
        print(f"Losing Trades:       {len(losing_trades):>10}")
        print(f"Win Rate:            {win_rate:>10.1f}%")
        print(f"\nAverage Win:         ${avg_win:>10,.2f}")
        print(f"Average Loss:        ${avg_loss:>10,.2f}")

        # Best and worst trades
        if self.trades:
            best_trade = max(self.trades, key=lambda t: t.pnl)
            worst_trade = min(self.trades, key=lambda t: t.pnl)
            print(f"\nBest Trade:          ${best_trade.pnl:>10,.2f} ({best_trade.market_name})")
            print(f"Worst Trade:         ${worst_trade.pnl:>10,.2f} ({worst_trade.market_name})")

        print("=" * 70)

    def plot_results(self):
        """Plot backtest results"""
        if len(self.trades) == 0:
            print("No trades to plot!")
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        # TODO: Plot equity over time
        ax1.plot([self.initial_bankroll, self.bankroll], 'b-o', linewidth=3, markersize=10)
        ax1.axhline(self.initial_bankroll, color='gray', linestyle='--',
                    label='Initial Bankroll')
        ax1.fill_between([0, 1], self.initial_bankroll,
                         [self.initial_bankroll, self.bankroll],
                         alpha=0.3, color='green' if self.bankroll > self.initial_bankroll else 'red')

        # Formatting
        ax1.set_ylabel('Bankroll ($)', fontsize=12)
        ax1.set_title('Equity Curve', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Start', 'End'])

        # BOTTOM PLOT: P&L distribution
        # Create histogram of trade P&Ls
        pnls = [t.pnl for t in self.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, edgecolor='black', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)

        # Formatting
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax2.set_title('Trade P&L Distribution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


# Example strategies


# Function: plot_results
def plot_results(self):
        """Plot backtest results"""
        if len(self.trades) == 0:
            print("No trades to plot!")
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        # TODO: Plot equity over time
        ax1.plot([self.initial_bankroll, self.bankroll], 'b-o', linewidth=3, markersize=10)
        ax1.axhline(self.initial_bankroll, color='gray', linestyle='--',
                    label='Initial Bankroll')
        ax1.fill_between([0, 1], self.initial_bankroll,
                         [self.initial_bankroll, self.bankroll],
                         alpha=0.3, color='green' if self.bankroll > self.initial_bankroll else 'red')

        # Formatting
        ax1.set_ylabel('Bankroll ($)', fontsize=12)
        ax1.set_title('Equity Curve', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Start', 'End'])

        # BOTTOM PLOT: P&L distribution
        # Create histogram of trade P&Ls
        pnls = [t.pnl for t in self.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, edgecolor='black', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)

        # Formatting
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12)
        ax2.set_title('Trade P&L Distribution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


# Example strategies


# Function: simple_kelly_strategy
def simple_kelly_strategy(market, time, price):
    """
    Simple Kelly strategy: bet if we think price is too low

    Args:
        market: HistoricalMarket object
        time: Current time period
        price: Current market price

    Returns:
        Bet size in dollars
    """
    # Estimate true probability (naive: use future price as proxy)
    # In real life, you'd use your own models!
    future_prices = market.prices[time:]
    estimated_prob = np.mean(future_prices)

    # Calculate edge
    edge = estimated_prob - price

    # Only bet if edge > 0.05 (5%)
    if edge > 0.05:
        # Simple fixed bet
        return 500  # Bet $500

    return 0




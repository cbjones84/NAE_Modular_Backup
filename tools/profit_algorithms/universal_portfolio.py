# NAE/tools/profit_algorithms/universal_portfolio.py
"""
Universal Portfolio Algorithm
Adaptive portfolio rebalancing that maximizes log-optimal growth rate
Based on Thomas M. Cover's Universal Portfolio Algorithm
"""

import numpy as np
from typing import Dict, Any, List, Optional
import math

class UniversalPortfolio:
    """
    Universal Portfolio Algorithm implementation
    Maximizes long-term geometric mean return through adaptive rebalancing
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio_weights = {}  # Asset -> weight (0.0 to 1.0)
        self.price_history = {}  # Asset -> list of historical prices
        self.returns_history = {}  # Asset -> list of historical returns
        self.rebalance_count = 0
        
    def add_asset(self, symbol: str, initial_price: float, initial_weight: Optional[float] = None):
        """Add an asset to the portfolio"""
        if initial_weight is None:
            # Equal weight if not specified
            initial_weight = 1.0 / max(1, len(self.portfolio_weights) + 1)
        
        self.portfolio_weights[symbol] = initial_weight
        self.price_history[symbol] = [initial_price]
        self.returns_history[symbol] = []
    
    def update_prices(self, prices: Dict[str, float]):
        """Update asset prices and calculate returns"""
        for symbol, price in prices.items():
            if symbol in self.price_history:
                old_price = self.price_history[symbol][-1]
                self.price_history[symbol].append(price)
                
                # Calculate return
                if old_price > 0:
                    returns = (price - old_price) / old_price
                    self.returns_history[symbol].append(returns)
    
    def calculate_portfolio_return(self, returns: Dict[str, float]) -> float:
        """
        Calculate portfolio return given asset returns
        """
        portfolio_return = 0.0
        for symbol, weight in self.portfolio_weights.items():
            if symbol in returns:
                portfolio_return += weight * returns[symbol]
        return portfolio_return
    
    def rebalance_universal(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Rebalance portfolio using Universal Portfolio Algorithm
        Returns new target weights for each asset
        """
        self.update_prices(current_prices)
        
        # Calculate recent returns for all assets
        asset_returns = {}
        for symbol in self.portfolio_weights.keys():
            if symbol in self.returns_history and len(self.returns_history[symbol]) > 0:
                # Use recent returns (last N periods)
                recent_returns = self.returns_history[symbol][-10:] if len(self.returns_history[symbol]) >= 10 else self.returns_history[symbol]
                asset_returns[symbol] = np.mean(recent_returns)
            else:
                asset_returns[symbol] = 0.0
        
        # Universal Portfolio: Weight by exponential of cumulative log returns
        # Simplified version: weight by geometric mean of returns
        log_returns = {}
        for symbol, ret in asset_returns.items():
            if ret > -1.0:  # Avoid log of negative
                log_returns[symbol] = math.log(1.0 + ret)
            else:
                log_returns[symbol] = -10.0  # Very negative
        
        # Calculate cumulative log returns
        cumulative_log_returns = {}
        for symbol in self.portfolio_weights.keys():
            if symbol in self.returns_history:
                cumulative = sum(log_returns.get(symbol, 0.0) for _ in self.returns_history[symbol])
            else:
                cumulative = log_returns.get(symbol, 0.0)
            cumulative_log_returns[symbol] = cumulative
        
        # Exponential weighting: w_i = exp(cumulative_log_return_i) / sum(exp(cumulative_log_return_j))
        exp_weights = {}
        total_exp = 0.0
        for symbol in self.portfolio_weights.keys():
            exp_weight = math.exp(cumulative_log_returns.get(symbol, 0.0))
            exp_weights[symbol] = exp_weight
            total_exp += exp_weight
        
        # Normalize weights
        new_weights = {}
        if total_exp > 0:
            for symbol in self.portfolio_weights.keys():
                new_weights[symbol] = exp_weights[symbol] / total_exp
        else:
            # Fallback to equal weights
            equal_weight = 1.0 / len(self.portfolio_weights)
            for symbol in self.portfolio_weights.keys():
                new_weights[symbol] = equal_weight
        
        # Update portfolio weights
        self.portfolio_weights = new_weights
        self.rebalance_count += 1
        
        return new_weights
    
    def get_target_allocation(self, total_capital: float) -> Dict[str, float]:
        """
        Get target dollar allocation for each asset based on current weights
        """
        allocation = {}
        for symbol, weight in self.portfolio_weights.items():
            allocation[symbol] = total_capital * weight
        return allocation
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate portfolio Sharpe ratio"""
        if not self.returns_history:
            return 0.0
        
        # Calculate portfolio returns over time
        portfolio_returns = []
        for i in range(max(len(r) for r in self.returns_history.values() if r)):
            period_return = 0.0
            for symbol, weight in self.portfolio_weights.items():
                if symbol in self.returns_history and i < len(self.returns_history[symbol]):
                    period_return += weight * self.returns_history[symbol][i]
            portfolio_returns.append(period_return)
        
        if len(portfolio_returns) < 2:
            return 0.0
        
        # Calculate Sharpe ratio
        excess_returns = [r - risk_free_rate/252 for r in portfolio_returns]  # Daily risk-free rate
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * math.sqrt(252)  # Annualized
        return float(sharpe)
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        total_value = sum(
            self.price_history.get(symbol, [0])[-1] * (self.current_capital * weight)
            if symbol in self.price_history and len(self.price_history[symbol]) > 0
            else self.current_capital * weight
            for symbol, weight in self.portfolio_weights.items()
        )
        
        return {
            "total_value": total_value,
            "weights": self.portfolio_weights.copy(),
            "rebalance_count": self.rebalance_count,
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "number_of_assets": len(self.portfolio_weights)
        }



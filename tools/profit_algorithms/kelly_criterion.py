# NAE/tools/profit_algorithms/kelly_criterion.py
"""
Kelly Criterion: Optimal position sizing algorithm
Maximizes long-term geometric mean return by calculating optimal bet size
Formula: f = (bp - q) / b
where:
  f = fraction of capital to bet
  b = odds received (net profit if win / initial bet)
  p = probability of winning
  q = probability of losing (1 - p)
"""

import math
from typing import Dict, Any, Optional

class KellyCriterion:
    """
    Kelly Criterion calculator for optimal position sizing
    """
    
    @staticmethod
    def calculate_kelly_fraction(win_probability: float, win_odds: float, 
                                 loss_ratio: float = 1.0) -> float:
        """
        Calculate optimal Kelly fraction
        
        Args:
            win_probability: Probability of winning (0.0 to 1.0)
            win_odds: Net profit if win / initial bet (e.g., 2.0 means 2x return)
            loss_ratio: Loss amount if lose / initial bet (default 1.0 for 1:1 loss)
        
        Returns:
            Optimal fraction of capital to bet (0.0 to 1.0)
        """
        p = win_probability
        q = 1.0 - p
        b = win_odds
        
        # Kelly formula: f = (bp - q) / b
        # Adjusted for asymmetric losses: f = (p * b - q * loss_ratio) / b
        if b <= 0:
            return 0.0
        
        kelly_f = (p * b - q * loss_ratio) / b
        
        # Kelly fraction must be non-negative and typically capped at 0.25 (1/4 Kelly)
        # VERY_AGGRESSIVE MODE: Allow up to 50% Kelly for faster growth
        # Standard Kelly caps at 25%, but for $5M→$15.7M goal we need more
        return max(0.0, min(kelly_f, 0.50))
    
    @staticmethod
    def calculate_position_size(account_value: float, win_probability: float,
                                win_odds: float, loss_ratio: float = 1.0,
                                kelly_fraction: float = 1.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            account_value: Total account value
            win_probability: Probability of winning trade (0.0 to 1.0)
            win_odds: Net profit if win / initial bet
            loss_ratio: Loss amount if lose / initial bet
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly, safer)
        
        Returns:
            Optimal position size in dollars
        """
        # Calculate full Kelly fraction
        full_kelly = KellyCriterion.calculate_kelly_fraction(
            win_probability, win_odds, loss_ratio
        )
        
        # Apply fractional Kelly (safer, more conservative)
        adjusted_kelly = full_kelly * kelly_fraction
        
        # Calculate position size
        position_size = account_value * adjusted_kelly
        
        return max(0.0, position_size)
    
    @staticmethod
    def calculate_from_strategy(strategy: Dict[str, Any], 
                                account_value: float,
                                kelly_fraction: float = 0.50) -> Dict[str, Any]:
        # VERY_AGGRESSIVE: Default to half Kelly (was quarter Kelly)
        """
        Calculate optimal position size from strategy metrics
        
        Args:
            strategy: Strategy dictionary with trust_score, backtest_score, etc.
            account_value: Total account value
            kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        
        Returns:
            Dict with position_size, kelly_f, win_prob, etc.
        """
        # Estimate win probability from strategy metrics
        trust_score = strategy.get("trust_score", 55) / 100.0  # Normalize
        backtest_score = strategy.get("backtest_score", 50) / 100.0  # Normalize
        
        # Weighted average of trust and backtest scores
        win_probability = (trust_score * 0.6 + backtest_score * 0.4)
        
        # Estimate win odds from historical performance
        # If backtest shows 20% return, win_odds = 1.2 (20% profit)
        # If backtest shows loss, use conservative estimate
        expected_return = strategy.get("expected_return", 0.1)
        win_odds = max(1.0, 1.0 + expected_return)  # Minimum 1:1
        
        # Estimate loss ratio (typically 1:1 unless strategy specifies stop loss)
        stop_loss_pct = strategy.get("stop_loss_pct", 0.02)  # 2% default
        loss_ratio = stop_loss_pct / expected_return if expected_return > 0 else 1.0
        
        # Calculate Kelly fraction
        kelly_f = KellyCriterion.calculate_kelly_fraction(
            win_probability, win_odds, loss_ratio
        )
        
        # Calculate position size
        position_size = KellyCriterion.calculate_position_size(
            account_value, win_probability, win_odds, loss_ratio, kelly_fraction
        )
        
        return {
            "position_size": position_size,
            "kelly_fraction": kelly_f * kelly_fraction,
            "win_probability": win_probability,
            "win_odds": win_odds,
            "loss_ratio": loss_ratio,
            "account_value": account_value,
            "position_pct": (position_size / account_value) * 100 if account_value > 0 else 0
        }
    
    @staticmethod
    def adjust_for_risk(account_value: float, base_position_size: float,
                       risk_tolerance: float = 0.02) -> float:
        """
        Adjust position size based on risk tolerance
        
        Args:
            account_value: Total account value
            base_position_size: Base position size from Kelly
            risk_tolerance: Maximum risk per trade as % of account (default 2%)
        
        Returns:
            Risk-adjusted position size
        """
        max_risk = account_value * risk_tolerance
        # Cap position size so risk doesn't exceed tolerance
        # Assuming stop loss, adjust position size accordingly
        return min(base_position_size, max_risk * 50)  # Assuming 2% stop loss = 50x position



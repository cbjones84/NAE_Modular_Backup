#!/usr/bin/env python3
"""
Aggressive Day Trading Strategies for Optimus

Goal: Maximize returns through intelligent day trading
Target: $5M in 8 years requires aggressive but smart day trading

Strategies:
1. Momentum Scalping - Quick in/out on momentum
2. Volatility Breakouts - Trade volatility spikes
3. Mean Reversion - Quick reversals
4. Gap Trading - Trade gap fills/continuations
5. News Trading - React to news events quickly
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class DayTradingStrategies:
    """Aggressive day trading strategies for maximum returns"""
    
    def __init__(self, nav: float):
        self.nav = nav
        # ULTRA AGGRESSIVE DAY TRADING - Maximum risk for maximum returns
        # Target: $5M in 8 years requires extreme aggression
        self.strategies = {
            "momentum_scalp": {
                "enabled": True,
                "min_profit_pct": 0.002,  # 0.2% minimum profit target (was 0.5%)
                "max_hold_minutes": 15,  # Exit within 15 minutes (was 30)
                "risk_reward": 1.0,  # 1:1 minimum (was 1.5:1) - accept break-even
                "position_size_pct": 0.50  # 50% of NAV per trade (was 15%) - EXTREME
            },
            "volatility_breakout": {
                "enabled": True,
                "min_profit_pct": 0.003,  # 0.3% minimum profit target (was 1%)
                "max_hold_minutes": 30,  # Exit within 30 minutes (was 60)
                "risk_reward": 1.2,  # 1.2:1 minimum (was 2:1)
                "position_size_pct": 0.60  # 60% of NAV per trade (was 20%) - EXTREME
            },
            "mean_reversion": {
                "enabled": True,
                "min_profit_pct": 0.002,  # 0.2% minimum profit target (was 0.8%)
                "max_hold_minutes": 20,  # Exit within 20 minutes (was 45)
                "risk_reward": 1.0,  # 1:1 minimum (was 1.8:1)
                "position_size_pct": 0.45  # 45% of NAV per trade (was 18%) - EXTREME
            },
            "gap_trading": {
                "enabled": True,
                "min_profit_pct": 0.005,  # 0.5% minimum profit target (was 1.5%)
                "max_hold_minutes": 60,  # Exit within 1 hour (was 2 hours)
                "risk_reward": 1.5,  # 1.5:1 minimum (was 2.5:1)
                "position_size_pct": 0.70  # 70% of NAV per trade (was 25%) - EXTREME
            },
            "news_trading": {
                "enabled": True,
                "min_profit_pct": 0.005,  # 0.5% minimum profit target (was 2%)
                "max_hold_minutes": 45,  # Exit within 45 minutes (was 90)
                "risk_reward": 1.5,  # 1.5:1 minimum (was 3:1)
                "position_size_pct": 0.80  # 80% of NAV per trade (was 30%) - EXTREME
            }
        }
    
    def find_day_trade_opportunity(self, market_data: Dict) -> Optional[Dict]:
        """
        Find best day trading opportunity
        
        Args:
            market_data: Market data including price, volume, volatility
        
        Returns:
            Trade opportunity dict or None
        """
        opportunities = []
        
        # Check each strategy
        for strategy_name, config in self.strategies.items():
            if not config["enabled"]:
                continue
            
            opportunity = self._check_strategy(strategy_name, market_data, config)
            if opportunity:
                opportunities.append(opportunity)
        
        # Return best opportunity
        if opportunities:
            # Sort by expected return
            opportunities.sort(key=lambda x: x.get("expected_return", 0), reverse=True)
            return opportunities[0]
        
        # ULTRA AGGRESSIVE: ALWAYS return a default momentum scalp opportunity
        # This ensures we're ALWAYS looking for trades during market hours
        # NO EXCEPTIONS - trade on ANY price movement
        symbol = market_data.get("symbol", "")
        current_price = market_data.get("price", 0)
        momentum = market_data.get("momentum", 0)
        
        if current_price > 0 and symbol:
            # Default momentum scalp opportunity (EXTREMELY aggressive - always trade)
            momentum_config = self.strategies.get("momentum_scalp", {})
            # Use maximum position size for default trades
            position_pct = momentum_config.get("position_size_pct", 0.50)
            return {
                "strategy": "momentum_scalp",
                "symbol": symbol,
                "side": "buy" if momentum >= -0.001 else "sell",  # Trade on ANY movement, even tiny
                "entry_price": current_price,
                "target_price": current_price * (1 + momentum_config.get("min_profit_pct", 0.002)),
                "stop_loss": current_price * (1 - momentum_config.get("min_profit_pct", 0.002) / max(momentum_config.get("risk_reward", 1.0), 1.0)),
                "quantity": max(1, int((self.nav * position_pct) / current_price)),  # Minimum 1 share, use aggressive position size
                "expected_return": momentum_config.get("min_profit_pct", 0.002),
                "max_hold_minutes": momentum_config.get("max_hold_minutes", 15),
                "risk_reward": momentum_config.get("risk_reward", 1.0),
                "aggressive": True,  # Flag as aggressive default trade
                "always_trade": True  # Flag: ALWAYS execute this trade - NO EXCEPTIONS
            }
        
        return None
    
    def _check_strategy(self, strategy_name: str, market_data: Dict, config: Dict) -> Optional[Dict]:
        """Check if strategy conditions are met"""
        symbol = market_data.get("symbol", "")
        current_price = market_data.get("price", 0)
        volume = market_data.get("volume", 0)
        volatility = market_data.get("volatility", 0)
        momentum = market_data.get("momentum", 0)
        
        if current_price == 0:
            return None
        
        opportunity = {
            "strategy": strategy_name,
            "symbol": symbol,
            "side": "buy",  # Default
            "entry_price": current_price,
            "target_price": current_price * (1 + config["min_profit_pct"]),
            "stop_loss": current_price * (1 - config["min_profit_pct"] / config["risk_reward"]),
            "quantity": max(1, int((self.nav * config["position_size_pct"]) / current_price)),  # Minimum 1 share
            "expected_return": config["min_profit_pct"],
            "max_hold_minutes": config["max_hold_minutes"],
            "risk_reward": config["risk_reward"],
            "always_trade": True  # ULTRA AGGRESSIVE: Always execute all opportunities
        }
        
        # Strategy-specific logic - EXTREMELY AGGRESSIVE THRESHOLDS
        # Goal: Trade on ANY market movement, no matter how small
        if strategy_name == "momentum_scalp":
            # EXTREME: Trade on ANY momentum, even 0.01% (essentially always)
            if abs(momentum) >= 0:  # ALWAYS TRUE - trade on any movement
                opportunity["side"] = "buy" if momentum >= -0.0001 else "sell"  # Tiny threshold
                opportunity["always_trade"] = True  # Flag as always-execute
                return opportunity
        
        elif strategy_name == "volatility_breakout":
            # EXTREME: Trade on ANY volatility, even 0.01%
            if volatility >= 0.0001:  # Essentially always true
                opportunity["side"] = "buy"  # Trade breakouts
                opportunity["always_trade"] = True
                return opportunity
        
        elif strategy_name == "mean_reversion":
            # EXTREME: Trade on ANY move
            if abs(momentum) >= 0:  # Always true
                # CASH ACCOUNT FIX: Only allow BUY orders (no short selling)
                is_cash_account = market_data.get("account_type", "").lower() == "cash"
                if is_cash_account:
                    opportunity["side"] = "buy"  # Force buy for cash accounts
                else:
                    opportunity["side"] = "sell" if momentum > 0 else "buy"  # Fade the move
                opportunity["always_trade"] = True
                return opportunity
        
        elif strategy_name == "gap_trading":
            gap_pct = market_data.get("gap_pct", momentum)  # Use momentum as gap proxy
            # EXTREME: Trade on ANY gap, even 0.01%
            if abs(gap_pct) >= 0:  # Always true
                # CASH ACCOUNT FIX: Only allow BUY orders (no short selling)
                # For cash accounts, we can only buy, so always use "buy" side
                # Check if this is a cash account (passed via market_data or default to buy)
                is_cash_account = market_data.get("account_type", "").lower() == "cash"
                if is_cash_account:
                    opportunity["side"] = "buy"  # Force buy for cash accounts
                else:
                    opportunity["side"] = "sell" if gap_pct > 0 else "buy"
                opportunity["always_trade"] = True
                return opportunity
        
        elif strategy_name == "news_trading":
            news_score = market_data.get("news_score", 0)
            # EXTREME: Trade on ANY news score
            if abs(news_score) >= 0:  # Always true
                # CASH ACCOUNT FIX: Only allow BUY orders (no short selling)
                is_cash_account = market_data.get("account_type", "").lower() == "cash"
                if is_cash_account:
                    opportunity["side"] = "buy"  # Force buy for cash accounts
                else:
                    opportunity["side"] = "buy" if news_score >= 0 else "sell"
                opportunity["always_trade"] = True
                return opportunity
        
        # ULTRA AGGRESSIVE: ALWAYS return opportunity for momentum scalping
        # This ensures we're ALWAYS looking for trades - NO EXCEPTIONS
        if strategy_name == "momentum_scalp":
            # Default: ALWAYS trade
            # CASH ACCOUNT FIX: Only allow BUY orders (no short selling)
            is_cash_account = market_data.get("account_type", "").lower() == "cash"
            opportunity["side"] = "buy"  # Always buy for cash accounts (or default)
            opportunity["always_trade"] = True
            return opportunity
        
        return None
    
    def should_exit_day_trade(self, position: Dict, entry_time: datetime) -> Tuple[bool, str]:
        """
        Check if day trade should be exited
        
        Args:
            position: Position details
            entry_time: When position was entered
        
        Returns:
            (should_exit, reason)
        """
        current_time = datetime.now()
        hold_minutes = (current_time - entry_time).total_seconds() / 60
        
        strategy = position.get("strategy", "")
        config = self.strategies.get(strategy, {})
        
        if not config:
            return False, "Unknown strategy"
        
        max_hold = config.get("max_hold_minutes", 60)
        
        # Time-based exit
        if hold_minutes >= max_hold:
            return True, f"Max hold time reached ({max_hold} minutes)"
        
        # Profit target check
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        if entry_price > 0 and current_price > 0:
            profit_pct = (current_price - entry_price) / entry_price
            target_pct = config.get("min_profit_pct", 0.01)
            
            if profit_pct >= target_pct:
                return True, f"Profit target reached ({profit_pct:.2%})"
            
            # Stop loss check
            stop_loss_pct = -target_pct / config.get("risk_reward", 2.0)
            if profit_pct <= stop_loss_pct:
                return True, f"Stop loss hit ({profit_pct:.2%})"
        
        return False, "Hold"


#!/usr/bin/env python3
"""
Day Trading Prevention and Compliance System
Ensures NAE never violates Pattern Day Trader (PDT) rules
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DayTrade:
    """Represents a day trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # "buy" or "sell"
    pnl: float


@dataclass
class TradingDay:
    """Trading day record"""
    date: str  # YYYY-MM-DD
    day_trades: int
    total_trades: int
    trades: List[Dict[str, any]]


class DayTradingPrevention:
    """
    Prevents Pattern Day Trader violations
    
    PDT Rule: No more than 3 day trades in any rolling 5-business-day period
    Day Trade: Opening and closing a position in the same security on the same day
    """
    
    def __init__(self, data_file: str = "data/day_trading_compliance.json"):
        """
        Initialize day trading prevention system
        
        Args:
            data_file: File to store trading history
        """
        self.data_file = data_file
        self.trading_history: List[TradingDay] = []
        self.open_positions: Dict[str, Dict] = {}  # symbol -> position info
        
        # PDT Rule: Max 3 day trades in 5 business days
        self.max_day_trades_per_period = 3
        self.rolling_period_days = 5
        
        # Load existing history
        self._load_history()
        
        # Logging
        self.log_file = "logs/day_trading_compliance.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logger.info("Day Trading Prevention system initialized")
        self.log_action("Day Trading Prevention system initialized - MAX 3 DAY TRADES PER 5 BUSINESS DAYS")
    
    def log_action(self, message: str):
        """Log compliance action"""
        ts = datetime.now().isoformat()
        log_msg = f"[{ts}] {message}"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
        
        logger.info(message)
        print(f"[Day Trading Compliance] {message}")
    
    def _load_history(self):
        """Load trading history from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                    self.trading_history = [
                        TradingDay(**day) for day in data.get("trading_history", [])
                    ]
                    self.open_positions = data.get("open_positions", {})
        except Exception as e:
            logger.warning(f"Failed to load trading history: {e}")
            self.trading_history = []
            self.open_positions = {}
    
    def _save_history(self):
        """Save trading history to file"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            data = {
                "trading_history": [asdict(day) for day in self.trading_history],
                "open_positions": self.open_positions,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trading history: {e}")
    
    def _get_trading_day(self, date: datetime) -> TradingDay:
        """Get or create trading day record"""
        date_str = date.strftime("%Y-%m-%d")
        
        for day in self.trading_history:
            if day.date == date_str:
                return day
        
        # Create new trading day
        new_day = TradingDay(
            date=date_str,
            day_trades=0,
            total_trades=0,
            trades=[]
        )
        self.trading_history.append(new_day)
        return new_day
    
    def _is_business_day(self, date: datetime) -> bool:
        """Check if date is a business day (Monday-Friday)"""
        return date.weekday() < 5  # 0-4 = Monday-Friday
    
    def _get_business_days_back(self, days: int) -> List[datetime]:
        """Get list of business days going back"""
        business_days = []
        current_date = datetime.now().date()
        days_back = 0
        
        while len(business_days) < days:
            check_date = current_date - timedelta(days=days_back)
            if self._is_business_day(datetime.combine(check_date, datetime.min.time())):
                business_days.append(datetime.combine(check_date, datetime.min.time()))
            days_back += 1
            
            # Safety limit
            if days_back > 30:
                break
        
        return business_days
    
    def record_trade(self, symbol: str, side: str, quantity: int, price: float) -> Dict[str, any]:
        """
        Record a trade and check for day trading
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            price: Trade price
            
        Returns:
            Trade record with compliance status
        """
        now = datetime.now()
        trade_record = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "timestamp": now.isoformat(),
            "is_day_trade": False,
            "compliance_status": "allowed"
        }
        
        position_key = symbol.lower()
        
        if side.lower() == "buy":
            # Opening a position
            self.open_positions[position_key] = {
                "symbol": symbol,
                "entry_time": now.isoformat(),
                "entry_price": price,
                "quantity": quantity,
                "side": "long"
            }
            trade_record["action"] = "opened_position"
        
        elif side.lower() == "sell":
            # Check if closing a position opened today
            if position_key in self.open_positions:
                position = self.open_positions[position_key]
                entry_time = datetime.fromisoformat(position["entry_time"])
                
                # Check if same day
                if entry_time.date() == now.date():
                    # This is a day trade!
                    trade_record["is_day_trade"] = True
                    trade_record["action"] = "day_trade_closed"
                    
                    # Record day trade
                    trading_day = self._get_trading_day(now)
                    trading_day.day_trades += 1
                    trading_day.total_trades += 1
                    trading_day.trades.append(trade_record)
                    
                    self.log_action(
                        f"⚠️ DAY TRADE DETECTED: {symbol} - Opened and closed on {now.date()}"
                    )
                    
                    # Remove position
                    del self.open_positions[position_key]
                else:
                    # Normal trade (position opened on different day)
                    trade_record["action"] = "closed_position"
                    del self.open_positions[position_key]
            else:
                # Opening a short position (sell without existing position)
                self.open_positions[position_key] = {
                    "symbol": symbol,
                    "entry_time": now.isoformat(),
                    "entry_price": price,
                    "quantity": quantity,
                    "side": "short"
                }
                trade_record["action"] = "opened_short_position"
        
        # Update trading day
        trading_day = self._get_trading_day(now)
        if not trade_record.get("is_day_trade"):
            trading_day.total_trades += 1
            trading_day.trades.append(trade_record)
        
        # Save history
        self._save_history()
        
        return trade_record
    
    def check_day_trade_allowed(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Check if a day trade would be allowed
        
        Args:
            symbol: Stock symbol
            side: "sell" (checking if closing would create day trade)
            
        Returns:
            (allowed, reason)
        """
        if side.lower() != "sell":
            return True, "Buy orders don't create day trades"
        
        position_key = symbol.lower()
        
        # Check if we have an open position
        if position_key not in self.open_positions:
            return True, "No open position to close"
        
        position = self.open_positions[position_key]
        entry_time = datetime.fromisoformat(position["entry_time"])
        now = datetime.now()
        
        # Check if same day
        if entry_time.date() == now.date():
            # This would be a day trade - check if allowed
            day_trades_in_period = self.count_day_trades_in_period()
            
            if day_trades_in_period >= self.max_day_trades_per_period:
                reason = (
                    f"Day trade BLOCKED: Already {day_trades_in_period} day trades "
                    f"in last {self.rolling_period_days} business days "
                    f"(limit: {self.max_day_trades_per_period})"
                )
                self.log_action(f"🚫 {reason}")
                return False, reason
            else:
                remaining = self.max_day_trades_per_period - day_trades_in_period
                reason = (
                    f"Day trade ALLOWED: {day_trades_in_period}/{self.max_day_trades_per_period} "
                    f"used ({remaining} remaining)"
                )
                return True, reason
        
        return True, "Position opened on different day - not a day trade"
    
    def count_day_trades_in_period(self) -> int:
        """Count day trades in rolling 5-business-day period"""
        business_days = self._get_business_days_back(self.rolling_period_days)
        business_day_dates = {day.date() for day in business_days}
        
        total_day_trades = 0
        for day in self.trading_history:
            day_date = datetime.strptime(day.date, "%Y-%m-%d").date()
            if day_date in business_day_dates:
                total_day_trades += day.day_trades
        
        return total_day_trades
    
    def get_compliance_status(self) -> Dict[str, any]:
        """Get current compliance status"""
        day_trades_in_period = self.count_day_trades_in_period()
        remaining_day_trades = max(0, self.max_day_trades_per_period - day_trades_in_period)
        
        return {
            "day_trades_in_period": day_trades_in_period,
            "max_allowed": self.max_day_trades_per_period,
            "remaining_day_trades": remaining_day_trades,
            "rolling_period_days": self.rolling_period_days,
            "compliance_status": "compliant" if day_trades_in_period < self.max_day_trades_per_period else "at_limit",
            "open_positions": len(self.open_positions),
            "can_day_trade": remaining_day_trades > 0
        }
    
    def validate_order(self, symbol: str, side: str, quantity: int) -> Tuple[bool, str]:
        """
        Validate if order is compliant with day trading rules
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            
        Returns:
            (is_allowed, reason)
        """
        # Check if this would create a day trade
        allowed, reason = self.check_day_trade_allowed(symbol, side)
        
        if not allowed:
            return False, reason
        
        return True, "Order compliant with day trading rules"


if __name__ == "__main__":
    # Test the system
    prevention = DayTradingPrevention()
    
    # Test compliance check
    status = prevention.get_compliance_status()
    print("\nDay Trading Compliance Status:")
    print(f"  Day Trades in Period: {status['day_trades_in_period']}/{status['max_allowed']}")
    print(f"  Remaining Day Trades: {status['remaining_day_trades']}")
    print(f"  Can Day Trade: {status['can_day_trade']}")
    print(f"  Open Positions: {status['open_positions']}")


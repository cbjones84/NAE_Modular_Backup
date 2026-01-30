#!/usr/bin/env python3
"""
Day Trading Compliance for Cash Accounts

Cash Account Day Trading Rules:
1. NO Pattern Day Trader (PDT) restrictions - PDT only applies to margin accounts
2. Good Faith Violation (GFV) prevention:
   - Cannot buy and sell same security on same day using unsettled funds
   - Funds settle T+2 (trade date + 2 business days)
   - Must track settled vs unsettled funds
3. Free Riding Violation prevention:
   - Cannot sell before paying for purchase
4. Cash account can day trade unlimited times IF using settled funds

This module enables aggressive day trading for cash accounts while preventing violations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeSettlement:
    """Track trade settlement dates"""
    trade_date: datetime
    settlement_date: datetime
    amount: float
    symbol: str
    side: str  # 'buy' or 'sell'
    settled: bool = False


@dataclass
class DayTrade:
    """Track day trades"""
    symbol: str
    buy_time: datetime
    sell_time: datetime
    buy_price: float
    sell_price: float
    quantity: int
    profit: float


class CashAccountDayTradingManager:
    """
    Manages day trading for cash accounts with GFV prevention
    
    Key Features:
    - Unlimited day trades (no PDT restrictions)
    - GFV prevention (track settled funds)
    - Free riding prevention
    - Aggressive day trading enabled
    """
    
    def __init__(self, account_type: str = "cash"):
        self.account_type = account_type.lower()
        self.day_trades: List[DayTrade] = []
        self.settlements: List[TradeSettlement] = []
        self.pending_buys: Dict[str, List[TradeSettlement]] = defaultdict(list)
        self.pending_sells: Dict[str, List[TradeSettlement]] = defaultdict(list)
        
        # Track same-day round trips per symbol
        self.same_day_trades: Dict[str, List[datetime]] = defaultdict(list)
        
        # GFV tracking
        self.gfv_count = 0
        self.free_riding_violations = 0
        
        # Settlement is T+2 (trade date + 2 business days)
        self.settlement_days = 2
    
    def can_day_trade(self, symbol: str, side: str, amount: float, 
                     settled_cash: float) -> Tuple[bool, str]:
        """
        Check if day trade is allowed
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            amount: Trade amount
            settled_cash: Available settled cash
        
        Returns:
            (allowed, reason)
        """
        # Cash accounts: No PDT restrictions!
        # Only need to check GFV and free riding
        
        if side == "buy":
            # Check if we have settled funds
            if amount > settled_cash:
                return False, f"Insufficient settled funds. Need ${amount:.2f}, have ${settled_cash:.2f}"
            
            # Check for free riding: Can't buy if we have pending sells of same symbol
            if symbol in self.pending_sells:
                pending_sells = sum(s.amount for s in self.pending_sells[symbol] if not s.settled)
                if pending_sells > 0:
                    return False, f"Free riding violation: Cannot buy {symbol} with pending unsettled sells"
        
        elif side == "sell":
            # Check if we own the stock (can't sell what we don't have)
            # This is handled by position tracking, but we check here too
            
            # Check for GFV: Can't sell same day if we bought with unsettled funds
            today = datetime.now().date()
            if symbol in self.pending_buys:
                for buy_settlement in self.pending_buys[symbol]:
                    if buy_settlement.trade_date.date() == today and not buy_settlement.settled:
                        return False, f"GFV prevention: Cannot sell {symbol} same day as buy using unsettled funds"
        
        return True, "Day trade allowed"
    
    def record_trade(self, symbol: str, side: str, quantity: int, 
                    price: float, trade_time: Optional[datetime] = None) -> TradeSettlement:
        """
        Record a trade for settlement tracking
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Price per share
            trade_time: Trade timestamp (defaults to now)
        
        Returns:
            TradeSettlement object
        """
        if trade_time is None:
            trade_time = datetime.now()
        
        amount = quantity * price
        settlement_date = self._calculate_settlement_date(trade_time)
        
        settlement = TradeSettlement(
            trade_date=trade_time,
            settlement_date=settlement_date,
            amount=amount,
            symbol=symbol,
            side=side
        )
        
        self.settlements.append(settlement)
        
        if side == "buy":
            self.pending_buys[symbol].append(settlement)
        else:
            self.pending_sells[symbol].append(settlement)
        
        # Track same-day round trips
        self.same_day_trades[symbol].append(trade_time)
        
        logger.info(f"Recorded {side} trade: {symbol} {quantity} @ ${price:.2f}, settles {settlement_date.date()}")
        
        return settlement
    
    def record_day_trade(self, symbol: str, buy_time: datetime, sell_time: datetime,
                        buy_price: float, sell_price: float, quantity: int):
        """Record a completed day trade"""
        profit = (sell_price - buy_price) * quantity
        day_trade = DayTrade(
            symbol=symbol,
            buy_time=buy_time,
            sell_time=sell_time,
            buy_price=buy_price,
            sell_price=sell_price,
            quantity=quantity,
            profit=profit
        )
        self.day_trades.append(day_trade)
        logger.info(f"Day trade recorded: {symbol} profit=${profit:.2f}")
    
    def update_settlements(self, current_date: Optional[datetime] = None):
        """Update settlement status for all pending trades"""
        if current_date is None:
            current_date = datetime.now()
        
        for settlement in self.settlements:
            if not settlement.settled and current_date >= settlement.settlement_date:
                settlement.settled = True
                logger.debug(f"Settlement complete: {settlement.symbol} {settlement.side} ${settlement.amount:.2f}")
    
    def get_settled_cash(self, total_cash: float) -> float:
        """
        Calculate available settled cash
        
        Args:
            total_cash: Total cash in account
        
        Returns:
            Settled cash available
        """
        self.update_settlements()
        
        # Subtract pending unsettled buys
        pending_buy_amount = sum(
            s.amount for s in self.settlements 
            if s.side == "buy" and not s.settled
        )
        
        # Add pending unsettled sells (money coming in)
        pending_sell_amount = sum(
            s.amount for s in self.settlements 
            if s.side == "sell" and not s.settled
        )
        
        # Settled cash = total - pending buys + pending sells (that will settle)
        # But we can only use what's actually settled
        settled_cash = total_cash - pending_buy_amount
        
        return max(0, settled_cash)
    
    def get_day_trade_count(self, days: int = 5) -> int:
        """Get number of day trades in last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return len([dt for dt in self.day_trades if dt.buy_time >= cutoff])
    
    def _calculate_settlement_date(self, trade_date: datetime) -> datetime:
        """Calculate settlement date (T+2 business days)"""
        settlement = trade_date
        business_days_added = 0
        
        while business_days_added < self.settlement_days:
            settlement += timedelta(days=1)
            # Skip weekends (Saturday=5, Sunday=6)
            if settlement.weekday() < 5:
                business_days_added += 1
        
        return settlement
    
    def is_pattern_day_trader(self) -> bool:
        """
        Check if account would be classified as Pattern Day Trader
        
        Note: This only applies to margin accounts. Cash accounts are NOT
        subject to PDT rules, but we track it for informational purposes.
        """
        if self.account_type == "cash":
            return False  # Cash accounts don't have PDT restrictions
        
        # For margin accounts: 4+ day trades in 5 business days
        day_trade_count = self.get_day_trade_count(days=5)
        return day_trade_count >= 4
    
    def get_compliance_status(self) -> Dict:
        """Get current compliance status"""
        self.update_settlements()
        
        return {
            "account_type": self.account_type,
            "day_trades_today": len([dt for dt in self.day_trades 
                                    if dt.buy_time.date() == datetime.now().date()]),
            "day_trades_5_days": self.get_day_trade_count(days=5),
            "gfv_count": self.gfv_count,
            "free_riding_violations": self.free_riding_violations,
            "is_pattern_day_trader": self.is_pattern_day_trader(),
            "pending_settlements": len([s for s in self.settlements if not s.settled]),
            "can_day_trade": self.account_type == "cash"  # Cash accounts can day trade!
        }


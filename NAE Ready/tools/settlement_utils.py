"""
Settlement Ledger - Tracks settled cash and prevents free-riding violations

This module tracks reserved/unsettled cash for free-riding protection.
It ensures that only settled funds are used for new trades, preventing
violations of SEC free-riding rules.

For Tradier cash accounts:
- Options settle T+1 (next business day)
- Stocks settle T+2 (two business days)
- Conservative default: 24 hours for options, 48 hours for stocks
"""

import datetime
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SettlementLedger:
    """
    Tracks reserved/unsettled cash for free-riding protection.
    
    Use broker.get_settled_cash() where available to reconcile.
    For Tradier cash accounts, options settle T+1, stocks settle T+2.
    """

    def __init__(self, broker):
        """
        Initialize settlement ledger
        
        Args:
            broker: Broker adapter instance (must have get_account_balance, 
                   get_settled_cash, get_unsettled_cash methods or equivalents)
        """
        self.broker = broker
        
        # List of dicts: {"amount": float, "reserved_at": datetime, 
        #                "expires_at": datetime, "reason": str, "cleared": bool}
        self.unsettled_reservations: List[Dict[str, Any]] = []
        
        # Settlement times (in seconds)
        # Options: T+1 = next business day (conservative: 24 hours)
        # Stocks: T+2 = two business days (conservative: 48 hours)
        self.options_settlement_seconds = 24 * 3600  # 24 hours
        self.stocks_settlement_seconds = 48 * 3600   # 48 hours
        
        logger.info("SettlementLedger initialized")

    def get_settled_cash(self) -> float:
        """
        Get settled cash available from broker.
        
        Prefers exact broker API if available, otherwise falls back
        to account balance minus unsettled cash.
        
        Returns:
            Settled cash amount (float)
        """
        try:
            # Prefer exact broker API if available
            if hasattr(self.broker, "get_settled_cash"):
                settled = self.broker.get_settled_cash()
                if settled is not None:
                    return float(settled)
            
            # Fallback: account balance - unsettled cash
            bal = self.broker.get_account_balance()
            if bal is None:
                return 0.0
            
            if hasattr(self.broker, "get_unsettled_cash"):
                unsettled = self.broker.get_unsettled_cash()
                if unsettled is not None:
                    return max(0.0, float(bal) - float(unsettled))
            
            # Last resort: use buying_power if that indicates available settled cash
            if hasattr(self.broker, "get_buying_power"):
                buying_power = self.broker.get_buying_power()
                if buying_power is not None:
                    return float(buying_power)
            
            # If no broker methods available, return account balance
            # (conservative - will be filtered by available_settled_cash)
            return float(bal) if bal else 0.0
            
        except Exception as e:
            logger.error(f"Error getting settled cash: {e}")
            return 0.0

    def reserve_for_order(self, amount: float, settle_after_seconds: Optional[int] = None, 
                         reason: str = "trade", security_type: str = "option") -> None:
        """
        Reserve 'amount' of cash immediately when placing order.
        
        Args:
            amount: Amount to reserve
            settle_after_seconds: Settlement time in seconds (defaults based on security_type)
            reason: Reason for reservation (e.g., "option_buy", "stock_sell")
            security_type: "option" or "stock" (determines default settlement time)
        """
        if amount <= 0:
            return
        
        now = datetime.datetime.utcnow()
        
        # Determine settlement time
        if settle_after_seconds is None:
            if security_type.lower() == "option":
                settle_after_seconds = self.options_settlement_seconds
            else:
                settle_after_seconds = self.stocks_settlement_seconds
        
        expires_at = now + datetime.timedelta(seconds=settle_after_seconds)
        
        reservation = {
            "amount": float(amount),
            "reserved_at": now,
            "expires_at": expires_at,
            "reason": reason,
            "security_type": security_type,
            "cleared": False
        }
        
        self.unsettled_reservations.append(reservation)
        
        logger.debug(f"Reserved ${amount:.2f} for {reason} (settles at {expires_at.isoformat()})")

    def release_settled(self) -> int:
        """
        Release reservations that are settled according to broker or expired.
        
        Prefers broker confirmation if available, otherwise uses time-based expiration.
        
        Returns:
            Number of reservations cleared
        """
        now = datetime.datetime.utcnow()
        before = len(self.unsettled_reservations)
        
        # Prefer broker confirmation if available
        if hasattr(self.broker, "reconcile_settled_reservations"):
            try:
                # Optional broker-provided helper
                self.broker.reconcile_settled_reservations(self.unsettled_reservations)
                # Assume broker updates or returns which reservations cleared
                self.unsettled_reservations = [
                    r for r in self.unsettled_reservations 
                    if not r.get("cleared", False)
                ]
                after = len(self.unsettled_reservations)
                cleared = before - after
                if cleared > 0:
                    logger.debug(f"Cleared {cleared} reservations via broker reconciliation")
                return cleared
            except Exception as e:
                logger.warning(f"Broker reconciliation failed, using time-based: {e}")
        
        # Fallback: expire by time (conservative)
        self.unsettled_reservations = [
            r for r in self.unsettled_reservations 
            if r["expires_at"] > now and not r.get("cleared", False)
        ]
        
        after = len(self.unsettled_reservations)
        cleared = before - after
        if cleared > 0:
            logger.debug(f"Cleared {cleared} expired reservations")
        
        return cleared

    def total_unsettled(self) -> float:
        """
        Get total amount of unsettled/reserved cash.
        
        Returns:
            Total unsettled amount (float)
        """
        return sum(
            r["amount"] for r in self.unsettled_reservations 
            if not r.get("cleared", False)
        )

    def available_settled_cash(self) -> float:
        """
        Get settled cash available net of reserved unsettled amounts.
        
        This is the amount that can be safely used for new trades.
        
        Returns:
            Available settled cash (float)
        """
        settled = self.get_settled_cash()
        reserved = self.total_unsettled()
        available = max(0.0, settled - reserved)
        
        return available

    def ensure_funds_available(self, amount: float) -> bool:
        """
        Check if specified amount is available in settled cash.
        
        Args:
            amount: Amount to check
            
        Returns:
            True if amount is available, False otherwise
        """
        available = self.available_settled_cash()
        return available >= amount

    def get_settlement_status(self) -> Dict[str, Any]:
        """
        Get current settlement status summary.
        
        Returns:
            Dictionary with settlement status information
        """
        self.release_settled()  # Update cleared reservations
        
        return {
            "settled_cash": self.get_settled_cash(),
            "total_unsettled": self.total_unsettled(),
            "available_settled_cash": self.available_settled_cash(),
            "active_reservations": len(self.unsettled_reservations),
            "reservations": [
                {
                    "amount": r["amount"],
                    "reason": r["reason"],
                    "reserved_at": r["reserved_at"].isoformat(),
                    "expires_at": r["expires_at"].isoformat(),
                    "security_type": r.get("security_type", "unknown")
                }
                for r in self.unsettled_reservations
                if not r.get("cleared", False)
            ]
        }


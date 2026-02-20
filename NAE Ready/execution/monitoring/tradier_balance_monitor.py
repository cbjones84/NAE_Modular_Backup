#!/usr/bin/env python3
"""
Tradier Account Balance Monitor
Monitors Tradier account for available funds and triggers trading activation
"""

import os
import sys
import time
import logging
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from broker_adapters.tradier_adapter import TradierBrokerAdapter
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../execution'))
    from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter

logger = logging.getLogger(__name__)


@dataclass
class AccountBalance:
    """Account balance snapshot"""
    account_id: str
    timestamp: str
    total_equity: float
    cash: float  # Direct cash field (if available)
    cash_available: float  # Available cash for trading
    buying_power: float
    day_trading_buying_power: float
    has_funds: bool
    funds_threshold: float = 100.0  # Minimum funds to trigger trading
    # Pending funds fields
    pending_cash: float = 0.0  # Pending deposits/transfers
    unsettled_funds: float = 0.0  # Unsettled from trades
    uncleared_funds: float = 0.0  # Uncleared deposits
    total_available: float = 0.0  # Cash + pending + unsettled
    # Margin account fields
    margin_balance: float = 0.0  # Margin account balance


class TradierBalanceMonitor:
    """
    Monitors Tradier account balance and triggers trading when funds are available
    """
    
    def __init__(
        self,
        account_id: Optional[str] = None,
        check_interval: int = 300,  # Check every 5 minutes
        funds_threshold: float = 100.0,  # Minimum $100 to start trading
        sandbox: bool = False
    ):
        """
        Initialize balance monitor
        
        Args:
            account_id: Tradier account ID (auto-detected if None)
            check_interval: Seconds between balance checks
            funds_threshold: Minimum funds required to trigger trading
            sandbox: Use sandbox environment
        """
        self.account_id = account_id or os.getenv("TRADIER_ACCOUNT_ID")
        self.check_interval = check_interval
        self.funds_threshold = funds_threshold
        # Fix: Ensure correct endpoint
        sandbox_env = os.getenv("TRADIER_SANDBOX", "").lower()
        # Use environment variable if set, otherwise use parameter
        self.sandbox = sandbox_env == "true" if sandbox_env else sandbox
        
        # Initialize Tradier adapter
        self.tradier_adapter = TradierBrokerAdapter(sandbox=sandbox)
        
        # Balance history
        self.balance_history: list[AccountBalance] = []
        self.last_balance: Optional[AccountBalance] = None
        self.funds_detected = False
        self.trading_activated = False
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Logging
        self.log_file = "logs/tradier_balance_monitor.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Redis for trading activation signal
        try:
            import redis
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True
            )
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            self.redis_available = False
        
        self.log_action("Tradier Balance Monitor initialized")
    
    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.now().isoformat()
        log_msg = f"[{ts}] {message}"
        
        # Write to log file
        try:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
        
        # Also log via logger
        logger.info(message)
        print(f"[Balance Monitor] {message}")
    
    def check_balance(self) -> Optional[AccountBalance]:
        """
        Check current account balance
        
        Returns:
            AccountBalance object or None if check failed
        """
        try:
            # Ensure valid token (for API key or OAuth)
            if not self.tradier_adapter.oauth.ensure_valid_token():
                self.log_action("Failed to authenticate with Tradier")
                return None
            
            # Get account details
            if not self.account_id:
                # Get accounts first
                accounts = self.tradier_adapter.rest_client.get_accounts()
                if accounts and len(accounts) > 0:
                    account = accounts[0]
                    self.account_id = account.get("account_number") or account.get("id")
                    if not self.account_id:
                        self.log_action("Could not determine account ID")
                        return None
            
            # Get balance information from /accounts/{id}/balances endpoint
            # This endpoint provides detailed balance info including pending funds
            balances = None
            try:
                response = self.tradier_adapter.rest_client.session.get(
                    f"{self.tradier_adapter.rest_client.api_base}/accounts/{self.account_id}/balances",
                    headers=self.tradier_adapter.rest_client._get_headers()
                )
                response.raise_for_status()
                balances_data = response.json()
                balances = balances_data.get("balances", {})
            except Exception as e:
                self.log_action(f"Failed to get balances from /balances endpoint: {e}")
                # Fallback to account details
                account_details = self.tradier_adapter.rest_client.get_account_details(self.account_id)
                if not account_details:
                    self.log_action("Failed to retrieve account details")
                    return None
                balances = account_details
            
            # Extract balance information
            total_equity = 0.0
            cash = 0.0  # Direct cash field
            cash_available = 0.0  # Available cash for trading
            buying_power = 0.0
            day_trading_buying_power = 0.0
            pending_cash = 0.0
            unsettled_funds = 0.0
            uncleared_funds = 0.0
            margin_balance = 0.0
            
            if isinstance(balances, dict):
                # Total equity
                total_equity = float(balances.get("total_equity", balances.get("equity", 0)))
                
                # Cash fields - check multiple locations
                # 1. Direct "cash" field (if it's a number)
                if "cash" in balances and isinstance(balances.get("cash"), (int, float)):
                    cash = float(balances.get("cash", 0))
                
                # 2. Cash object with cash_available
                cash_obj = balances.get("cash", {})
                if isinstance(cash_obj, dict):
                    # cash_available is the primary field we want
                    cash_available = float(cash_obj.get("cash_available", cash_obj.get("cash", 0)))
                    unsettled_funds = float(cash_obj.get("unsettled_funds", 0))
                    # If cash wasn't set from direct field, use cash_available
                    if cash == 0:
                        cash = cash_available
                elif isinstance(cash_obj, (int, float)):
                    # If cash is a number, use it for both
                    cash = float(cash_obj)
                    cash_available = cash
                
                # Total cash (includes all cash)
                total_cash = float(balances.get("total_cash", cash_available))
                
                # Pending funds
                pending_cash = float(balances.get("pending_cash", 0))
                uncleared_funds = float(balances.get("uncleared_funds", 0))
                
                # Margin balance - check multiple locations
                margin_obj = balances.get("margin", {})
                if isinstance(margin_obj, dict):
                    margin_balance = float(margin_obj.get("margin_balance", margin_obj.get("balance", 0)))
                    buying_power = float(margin_obj.get("stock_buying_power", margin_obj.get("buying_power", 0)))
                    day_trading_buying_power = float(margin_obj.get("day_trading_buying_power", 0))
                elif "buying_power" in balances:
                    buying_power = float(balances.get("buying_power", 0))
                
                # Also check for margin_balance as direct field
                if "margin_balance" in balances:
                    margin_balance = float(balances.get("margin_balance", 0))
            
            # Calculate total available (cash_available + pending + unsettled)
            total_available = cash_available + pending_cash + unsettled_funds
            
            # Use total_equity if available, otherwise use total_available
            if total_equity == 0:
                total_equity = total_available
            
            # Determine if funds are available (check both settled and pending)
            has_funds = (
                (cash_available >= self.funds_threshold) or 
                (cash >= self.funds_threshold) or
                (buying_power >= self.funds_threshold) or
                (total_available >= self.funds_threshold) or
                (pending_cash >= self.funds_threshold)
            )
            
            balance = AccountBalance(
                account_id=self.account_id,
                timestamp=datetime.now().isoformat(),
                total_equity=total_equity,
                cash=cash,
                cash_available=cash_available,
                buying_power=buying_power,
                day_trading_buying_power=day_trading_buying_power,
                has_funds=has_funds,
                funds_threshold=self.funds_threshold,
                pending_cash=pending_cash,
                unsettled_funds=unsettled_funds,
                uncleared_funds=uncleared_funds,
                total_available=total_available,
                margin_balance=margin_balance
            )
            
            self.last_balance = balance
            self.balance_history.append(balance)
            
            # Keep only last 1000 balance records
            if len(self.balance_history) > 1000:
                self.balance_history = self.balance_history[-1000:]
            
            return balance
        
        except Exception as e:
            self.log_action(f"Error checking balance: {e}")
            logger.exception("Balance check failed")
            return None
    
    def detect_funds_available(self, balance: AccountBalance) -> bool:
        """
        Detect if funds have become available (transition from no funds to funds)
        
        Args:
            balance: Current balance snapshot
            
        Returns:
            True if funds just became available
        """
        if not balance.has_funds:
            return False
        
        # Check if we previously had no funds
        if self.last_balance and not self.last_balance.has_funds:
            # Funds just became available!
            pending_info = ""
            if balance.pending_cash > 0 or balance.unsettled_funds > 0:
                pending_info = f" (Pending: ${balance.pending_cash:.2f}, Unsettled: ${balance.unsettled_funds:.2f})"
            margin_info = ""
            if balance.margin_balance != 0:
                margin_info = f", Margin: ${balance.margin_balance:.2f}"
            self.log_action(
                f"💰 FUNDS DETECTED! Cash: ${balance.cash:.2f}, "
                f"Cash Available: ${balance.cash_available:.2f}, "
                f"Buying Power: ${balance.buying_power:.2f}, "
                f"Total Available: ${balance.total_available:.2f}{pending_info}{margin_info}"
            )
            return True
        
        # If this is first check and funds are available
        if not self.funds_detected and balance.has_funds:
            pending_info = ""
            if balance.pending_cash > 0 or balance.unsettled_funds > 0:
                pending_info = f" (Pending: ${balance.pending_cash:.2f}, Unsettled: ${balance.unsettled_funds:.2f})"
            margin_info = ""
            if balance.margin_balance != 0:
                margin_info = f", Margin: ${balance.margin_balance:.2f}"
            self.log_action(
                f"💰 FUNDS AVAILABLE! Cash: ${balance.cash:.2f}, "
                f"Cash Available: ${balance.cash_available:.2f}, "
                f"Buying Power: ${balance.buying_power:.2f}, "
                f"Total Available: ${balance.total_available:.2f}{pending_info}{margin_info}"
            )
            return True
        
        return False
    
    def activate_trading(self, balance: AccountBalance):
        """
        Activate trading when funds are detected
        
        Args:
            balance: Current balance snapshot
        """
        if self.trading_activated:
            return  # Already activated
        
        self.log_action("🚀 ACTIVATING TRADING - Funds available for intelligent trading")
        
        # Set Redis flag for trading activation
        if self.redis_available:
            try:
                activation_signal = {
                    "event": "funds_available",
                    "account_id": balance.account_id,
                    "cash": balance.cash,
                    "cash_available": balance.cash_available,
                    "pending_cash": balance.pending_cash,
                    "margin_balance": balance.margin_balance,
                    "buying_power": balance.buying_power,
                    "total_available": balance.total_available,
                    "timestamp": balance.timestamp,
                    "action": "activate_trading"
                }
                self.redis_client.set("nae:trading:funds_available", json.dumps(activation_signal))
                self.redis_client.publish("nae:trading:activation", json.dumps(activation_signal))
                self.log_action("✅ Trading activation signal sent via Redis")
            except Exception as e:
                self.log_action(f"⚠️ Failed to send Redis signal: {e}")
        
        # Write activation file
        activation_file = "logs/trading_activated.json"
        try:
            activation_data = {
                "activated": True,
                "timestamp": datetime.now().isoformat(),
                "balance": asdict(balance),
                "reason": "Funds available in Tradier account"
            }
            with open(activation_file, "w") as f:
                json.dump(activation_data, f, indent=2)
            self.log_action(f"✅ Trading activation logged to {activation_file}")
        except Exception as e:
            self.log_action(f"⚠️ Failed to write activation file: {e}")
        
        self.trading_activated = True
        self.funds_detected = True
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.log_action("Starting balance monitoring loop")
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                balance = self.check_balance()
                
                if balance:
                    # Log balance status with all requested fields
                    if balance.has_funds:
                        pending_info = ""
                        if balance.pending_cash > 0 or balance.unsettled_funds > 0:
                            pending_info = f", Pending=${balance.pending_cash:.2f}, Unsettled=${balance.unsettled_funds:.2f}"
                        margin_info = ""
                        if balance.margin_balance != 0:
                            margin_info = f", Margin Balance=${balance.margin_balance:.2f}"
                        self.log_action(
                            f"Balance check: Cash=${balance.cash:.2f}, "
                            f"Cash Available=${balance.cash_available:.2f}, "
                            f"Buying Power=${balance.buying_power:.2f}, "
                            f"Equity=${balance.total_equity:.2f}, "
                            f"Total Available=${balance.total_available:.2f}{pending_info}{margin_info}"
                        )
                    else:
                        pending_info = ""
                        if balance.pending_cash > 0 or balance.unsettled_funds > 0:
                            pending_info = f" (Pending: ${balance.pending_cash:.2f}, Unsettled: ${balance.unsettled_funds:.2f})"
                        margin_info = ""
                        if balance.margin_balance != 0:
                            margin_info = f", Margin: ${balance.margin_balance:.2f}"
                        self.log_action(
                            f"Balance check: Insufficient funds (Cash Available: ${balance.cash_available:.2f} < ${self.funds_threshold:.2f}){pending_info}{margin_info}"
                        )
                    
                    # Check if funds just became available
                    if self.detect_funds_available(balance):
                        self.activate_trading(balance)
                
                # Sleep until next check
                time.sleep(self.check_interval)
            
            except KeyboardInterrupt:
                self.log_action("Monitoring stopped by user")
                break
            except Exception as e:
                self.log_action(f"Error in monitoring loop: {e}")
                logger.exception("Monitoring loop error")
                time.sleep(self.check_interval)  # Wait before retrying
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.monitoring_active:
            self.log_action("Monitoring already active")
            return
        
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.log_action("Balance monitoring started in background thread")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.log_action("Balance monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "monitoring_active": self.monitoring_active,
            "account_id": self.account_id,
            "funds_detected": self.funds_detected,
            "trading_activated": self.trading_activated,
            "last_balance": asdict(self.last_balance) if self.last_balance else None,
            "check_interval": self.check_interval,
            "funds_threshold": self.funds_threshold
        }


if __name__ == "__main__":
    # Test the monitor
    import argparse
    
    parser = argparse.ArgumentParser(description="Tradier Balance Monitor")
    parser.add_argument("--account-id", type=str, help="Tradier account ID")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--threshold", type=float, default=100.0, help="Minimum funds threshold")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox environment")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    
    args = parser.parse_args()
    
    monitor = TradierBalanceMonitor(
        account_id=args.account_id,
        check_interval=args.interval,
        funds_threshold=args.threshold,
        sandbox=args.sandbox
    )
    
    if args.once:
        balance = monitor.check_balance()
        if balance:
            print(f"\nAccount Balance:")
            print(f"  Account ID: {balance.account_id}")
            print(f"  Cash: ${balance.cash:,.2f}")
            print(f"  Buying Power: ${balance.buying_power:,.2f}")
            print(f"  Total Equity: ${balance.total_equity:,.2f}")
            print(f"  Has Funds: {balance.has_funds}")
            
            if monitor.detect_funds_available(balance):
                monitor.activate_trading(balance)
    else:
        monitor.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()


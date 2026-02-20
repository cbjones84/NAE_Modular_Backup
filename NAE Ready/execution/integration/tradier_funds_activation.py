#!/usr/bin/env python3
"""
Tradier Funds Activation Integration
Integrates balance monitoring with NAE trading activation
"""

import os
import sys
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from monitoring.tradier_balance_monitor import TradierBalanceMonitor
from compliance.day_trading_prevention import DayTradingPrevention

logger = logging.getLogger(__name__)


class TradierFundsActivation:
    """
    Main integration class that monitors Tradier funds and activates NAE trading
    """
    
    def __init__(self):
        """Initialize funds activation system"""
        self.balance_monitor = TradierBalanceMonitor(
            check_interval=300,  # Check every 5 minutes
            funds_threshold=100.0,  # Minimum $100 to activate
            sandbox=False  # Live trading
        )
        
        self.day_trading_prevention = DayTradingPrevention()
        
        # Load goals
        self.goals = self._load_goals()
        
        # Activation state
        self.trading_active = False
        self.activation_thread: Optional[threading.Thread] = None
        
        # Logging
        self.log_file = "logs/tradier_funds_activation.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        self.log_action("Tradier Funds Activation System initialized")
        self.log_action(f"Primary Goal: Generate ${self.goals.get('primary_goal', {}).get('target_amount', 0):,.2f} within {self.goals.get('primary_goal', {}).get('timeframe_years', 8)} years")
        self.log_action("Day Trading Prevention: ENABLED (MAX 3 day trades per 5 business days)")
    
    def _load_goals(self) -> Dict[str, Any]:
        """Load goals from goal_manager.json"""
        try:
            goals_file = os.path.join(os.path.dirname(__file__), '../../config/goal_manager.json')
            with open(goals_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load goals: {e}")
            return {}
    
    def log_action(self, message: str):
        """Log action"""
        ts = datetime.now().isoformat()
        log_msg = f"[{ts}] {message}"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log: {e}")
        
        logger.info(message)
        print(f"[Funds Activation] {message}")
    
    def check_funds_and_activate(self):
        """Check for funds and activate trading if available"""
        balance = self.balance_monitor.check_balance()
        
        if not balance:
            return False
        
        # Check if funds are available
        if balance.has_funds and not self.trading_active:
            self.log_action("=" * 60)
            self.log_action("💰 FUNDS DETECTED - ACTIVATING TRADING")
            self.log_action("=" * 60)
            self.log_action(f"Account: {balance.account_id}")
            self.log_action(f"Cash: ${balance.cash:,.2f}")
            self.log_action(f"Buying Power: ${balance.buying_power:,.2f}")
            self.log_action(f"Total Equity: ${balance.total_equity:,.2f}")
            
            # Activate trading
            self.activate_trading(balance)
            return True
        
        elif balance.has_funds and self.trading_active:
            # Funds still available, log status
            self.log_action(
                f"Trading Active - Balance: Cash=${balance.cash:,.2f}, "
                f"Buying Power=${balance.buying_power:,.2f}"
            )
        
        return False
    
    def activate_trading(self, balance):
        """Activate NAE trading system"""
        self.log_action("🚀 ACTIVATING NAE TRADING SYSTEM")
        
        # Activate via balance monitor (sends Redis signals)
        self.balance_monitor.activate_trading(balance)
        
        # Set trading active flag
        self.trading_active = True
        
        # Log goal reinforcement
        primary_goal = self.goals.get("primary_goal", {})
        target_amount = primary_goal.get("target_amount", 5000000.00)
        timeframe_years = primary_goal.get("timeframe_years", 8)
        
        self.log_action("=" * 60)
        self.log_action("GOAL REINFORCEMENT")
        self.log_action("=" * 60)
        self.log_action(f"PRIMARY GOAL: Generate ${target_amount:,.2f} within {timeframe_years} years")
        self.log_action("RECURRING: Every 8 years until owner commands stop")
        self.log_action("ULTIMATE GOAL: Consistent Generational Wealth")
        self.log_action("COMPLIANCE: Absolutely NO day trading - Full legal compliance")
        self.log_action("=" * 60)
        
        # Create activation manifest
        activation_manifest = {
            "activated": True,
            "timestamp": datetime.now().isoformat(),
            "account_id": balance.account_id,
            "balance": {
                "cash": balance.cash,
                "buying_power": balance.buying_power,
                "total_equity": balance.total_equity
            },
            "goals": {
                "primary_goal": primary_goal,
                "compliance": self.goals.get("compliance", {})
            },
            "day_trading_compliance": self.day_trading_prevention.get_compliance_status()
        }
        
        # Save activation manifest
        manifest_file = "logs/trading_activation_manifest.json"
        try:
            with open(manifest_file, "w") as f:
                json.dump(activation_manifest, f, indent=2)
            self.log_action(f"✅ Activation manifest saved to {manifest_file}")
        except Exception as e:
            self.log_action(f"⚠️ Failed to save manifest: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.log_action("Starting funds monitoring and activation loop")
        
        while True:
            try:
                # Check for funds and activate if needed
                self.check_funds_and_activate()
                
                # Log compliance status periodically
                compliance_status = self.day_trading_prevention.get_compliance_status()
                if compliance_status["day_trades_in_period"] > 0:
                    self.log_action(
                        f"Day Trading Compliance: {compliance_status['day_trades_in_period']}/"
                        f"{compliance_status['max_allowed']} used, "
                        f"{compliance_status['remaining_day_trades']} remaining"
                    )
                
                # Sleep until next check
                time.sleep(300)  # 5 minutes
            
            except KeyboardInterrupt:
                self.log_action("Monitoring stopped by user")
                break
            except Exception as e:
                self.log_action(f"Error in monitoring loop: {e}")
                logger.exception("Monitoring loop error")
                time.sleep(300)
    
    def start(self):
        """Start the funds activation system"""
        self.log_action("Starting Tradier Funds Activation System")
        
        # Start balance monitor
        self.balance_monitor.start_monitoring()
        
        # Start activation monitoring thread
        self.activation_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.activation_thread.start()
        
        self.log_action("✅ Funds Activation System running")
    
    def stop(self):
        """Stop the system"""
        self.balance_monitor.stop_monitoring()
        self.log_action("Funds Activation System stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "trading_active": self.trading_active,
            "balance_monitor": self.balance_monitor.get_status(),
            "day_trading_compliance": self.day_trading_prevention.get_compliance_status(),
            "goals": {
                "primary_goal": self.goals.get("primary_goal", {}),
                "compliance": self.goals.get("compliance", {})
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tradier Funds Activation System")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    activation = TradierFundsActivation()
    
    if args.status:
        status = activation.get_status()
        print("\n" + "=" * 60)
        print("TRADIER FUNDS ACTIVATION STATUS")
        print("=" * 60)
        print(f"Trading Active: {status['trading_active']}")
        print(f"\nBalance Monitor:")
        print(f"  Monitoring: {status['balance_monitor']['monitoring_active']}")
        print(f"  Funds Detected: {status['balance_monitor']['funds_detected']}")
        if status['balance_monitor']['last_balance']:
            bal = status['balance_monitor']['last_balance']
            print(f"  Last Balance: Cash=${bal['cash']:,.2f}, Buying Power=${bal['buying_power']:,.2f}")
        print(f"\nDay Trading Compliance:")
        comp = status['day_trading_compliance']
        print(f"  Day Trades: {comp['day_trades_in_period']}/{comp['max_allowed']}")
        print(f"  Remaining: {comp['remaining_day_trades']}")
        print(f"  Can Day Trade: {comp['can_day_trade']}")
        print("=" * 60)
    
    elif args.once:
        activation.check_funds_and_activate()
    else:
        activation.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            activation.stop()


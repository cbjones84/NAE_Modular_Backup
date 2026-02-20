"""
Tradier-Specific Pre-Trade Validator

Handles Tradier-specific validation rules:
- Pre/post-market order constraints
- Order preview for warnings
- Session validation
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime, time
from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter

# Add compliance module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from compliance.day_trading_prevention import DayTradingPrevention
    DAY_TRADING_PREVENTION_AVAILABLE = True
except ImportError:
    DAY_TRADING_PREVENTION_AVAILABLE = False
    logger.warning("Day trading prevention module not available")

logger = logging.getLogger(__name__)


class TradierPreTradeValidator:
    """Tradier-specific pre-trade validation"""
    
    def __init__(self, tradier_adapter: TradierBrokerAdapter):
        """
        Initialize Tradier validator
        
        Args:
            tradier_adapter: TradierBrokerAdapter instance
        """
        self.tradier = tradier_adapter
        
        # Trading hours (ET)
        self.regular_market_open = time(9, 30)  # 9:30 AM ET
        self.regular_market_close = time(16, 0)  # 4:00 PM ET
        self.pre_market_open = time(4, 0)  # 4:00 AM ET
        self.post_market_close = time(20, 0)  # 8:00 PM ET
        
        # Day trading prevention
        if DAY_TRADING_PREVENTION_AVAILABLE:
            self.day_trading_prevention = DayTradingPrevention()
        else:
            self.day_trading_prevention = None
            logger.warning("Day trading prevention not available - compliance checks disabled")
    
    def validate_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate order for Tradier-specific rules
        
        Args:
            signal: Trade signal
        
        Returns:
            Validation result
        """
        result = {
            "passed": True,
            "checks": [],
            "errors": [],
            "warnings": []
        }
        
        order_type = signal.get("order_type", "MARKET")
        duration = signal.get("duration", "day")
        current_time = datetime.now().time()
        
        # Check trading hours
        trading_hours_check = self._check_trading_hours(order_type, duration, current_time)
        result["checks"].append(trading_hours_check)
        
        if not trading_hours_check["passed"]:
            result["passed"] = False
            result["errors"].append(trading_hours_check["message"])
        
        # Check pre/post-market constraints
        if duration in ["pre", "post"]:
            constraint_check = self._check_pre_post_constraints(order_type, duration)
            result["checks"].append(constraint_check)
            
            if not constraint_check["passed"]:
                result["passed"] = False
                result["errors"].append(constraint_check["message"])
        
        # Day trading compliance check (CRITICAL - NO DAY TRADES)
        if self.day_trading_prevention:
            day_trade_check = self._check_day_trading_compliance(signal)
            result["checks"].append(day_trade_check)
            
            if not day_trade_check["passed"]:
                result["passed"] = False
                result["errors"].append(day_trade_check["message"])
                logger.error(f"🚫 DAY TRADE BLOCKED: {day_trade_check['message']}")
        
        # Preview order if requested
        preview_result = signal.get("preview_order", False)
        if preview_result:
            preview_check = self._preview_order(signal)
            result["checks"].append(preview_check)
            
            if preview_check.get("warnings"):
                result["warnings"].extend(preview_check["warnings"])
        
        return result
    
    def _check_day_trading_compliance(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check day trading compliance - ABSOLUTELY NO DAY TRADES
        
        Returns:
            Compliance check result
        """
        if not self.day_trading_prevention:
            return {
                "check": "day_trading_compliance",
                "passed": True,
                "message": "Day trading prevention not available"
            }
        
        symbol = signal.get("symbol", "")
        side = signal.get("action", signal.get("side", "")).upper()
        quantity = signal.get("quantity", 0)
        
        # Check if order would violate day trading rules
        allowed, reason = self.day_trading_prevention.check_day_trade_allowed(symbol, side)
        
        if not allowed:
            return {
                "check": "day_trading_compliance",
                "passed": False,
                "message": f"DAY TRADE BLOCKED: {reason}",
                "compliance_status": "VIOLATION_PREVENTED"
            }
        
        return {
            "check": "day_trading_compliance",
            "passed": True,
            "message": reason,
            "compliance_status": "COMPLIANT"
        }
    
    def _check_trading_hours(
        self,
        order_type: str,
        duration: str,
        current_time: time
    ) -> Dict[str, Any]:
        """Check if order is within valid trading hours"""
        # Regular market hours
        if duration == "day":
            if self.regular_market_open <= current_time <= self.regular_market_close:
                return {
                    "check": "trading_hours",
                    "passed": True,
                    "message": "Within regular market hours"
                }
            else:
                return {
                    "check": "trading_hours",
                    "passed": False,
                    "message": "Outside regular market hours for 'day' duration"
                }
        
        # Pre-market
        elif duration == "pre":
            if self.pre_market_open <= current_time < self.regular_market_open:
                return {
                    "check": "trading_hours",
                    "passed": True,
                    "message": "Within pre-market hours"
                }
            else:
                return {
                    "check": "trading_hours",
                    "passed": False,
                    "message": "Outside pre-market hours"
                }
        
        # Post-market
        elif duration == "post":
            if self.regular_market_close < current_time <= self.post_market_close:
                return {
                    "check": "trading_hours",
                    "passed": True,
                    "message": "Within post-market hours"
                }
            else:
                return {
                    "check": "trading_hours",
                    "passed": False,
                    "message": "Outside post-market hours"
                }
        
        # GTC orders can be placed anytime
        elif duration == "gtc":
            return {
                "check": "trading_hours",
                "passed": True,
                "message": "GTC order, no time restriction"
            }
        
        return {
            "check": "trading_hours",
            "passed": True,
            "message": "Trading hours check passed"
        }
    
    def _check_pre_post_constraints(
        self,
        order_type: str,
        duration: str
    ) -> Dict[str, Any]:
        """
        Check pre/post-market order constraints
        
        Tradier requires limit orders for pre/post-market
        """
        if duration in ["pre", "post"]:
            if order_type.upper() not in ["LIMIT", "STOP_LIMIT"]:
                return {
                    "check": "pre_post_constraints",
                    "passed": False,
                    "message": f"Pre/post-market orders must be limit orders, got {order_type}"
                }
        
        return {
            "check": "pre_post_constraints",
            "passed": True,
            "message": "Pre/post-market constraints satisfied"
        }
    
    def _preview_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview order using Tradier preview API
        
        Returns warnings and cost information
        """
        try:
            preview_result = self.tradier.rest_client.preview_order(
                account_id=self.tradier.account_id,
                symbol=signal.get("symbol", ""),
                side=signal.get("side", "buy"),
                quantity=signal.get("quantity", 0),
                order_type=signal.get("order_type", "market").lower(),
                duration=signal.get("duration", "day"),
                price=signal.get("price"),
                stop=signal.get("stop"),
                option_symbol=signal.get("option_symbol"),
                tag=signal.get("tag")
            )
            
            warnings = []
            if "warnings" in preview_result:
                warnings = preview_result["warnings"]
            
            return {
                "check": "order_preview",
                "passed": True,
                "warnings": warnings,
                "preview_data": preview_result
            }
        
        except Exception as e:
            logger.error(f"Error previewing order: {e}")
            return {
                "check": "order_preview",
                "passed": False,
                "error": str(e)
            }


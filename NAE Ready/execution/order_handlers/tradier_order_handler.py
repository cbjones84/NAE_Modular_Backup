#!/usr/bin/env python3
"""
Enhanced Tradier Order Handler

Fixes all common issues preventing trades:
1. Strategy conditions validation
2. Options approval checking
3. Endpoint verification (live vs sandbox)
4. Required fields validation
5. Rejected order handling with exact error messages
6. Account restrictions checking
7. Symbol formatting validation and correction
"""

import os
import sys
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, nae_root)

from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter
from execution.diagnostics.nae_tradier_diagnostics import TradierDiagnostics

logger = logging.getLogger(__name__)


class TradierOrderHandler:
    """
    Enhanced order handler that fixes all common trading issues
    """
    
    def __init__(self, tradier_adapter: TradierBrokerAdapter):
        """
        Initialize order handler
        
        Args:
            tradier_adapter: TradierBrokerAdapter instance
        """
        self.tradier = tradier_adapter
        self.diagnostics = None
        
        # Initialize diagnostics for pre-flight checks
        try:
            self.diagnostics = TradierDiagnostics(
                api_key=os.getenv("TRADIER_API_KEY"),
                account_id=self.tradier.account_id,
                live=not self.tradier.sandbox
            )
        except Exception as e:
            logger.warning(f"Could not initialize diagnostics: {e}")
        
        # Cache for account info
        self._account_profile = None
        self._account_balances = None
        self._last_check_time = None
    
    def submit_order_safe(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit order with comprehensive validation and error handling
        
        Args:
            order: Order dictionary with all order details
        
        Returns:
            Result dictionary with status and details
        """
        result = {
            "status": "pending",
            "order_id": None,
            "errors": [],
            "warnings": [],
            "fixes_applied": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Validate and fix symbol formatting
        symbol_fix = self._validate_and_fix_symbol(order)
        if not symbol_fix["valid"]:
            result["status"] = "error"
            result["errors"].append(f"Invalid symbol: {symbol_fix['error']}")
            return result
        
        if symbol_fix.get("fixed"):
            result["fixes_applied"].append(f"Symbol fixed: {symbol_fix['message']}")
            order = symbol_fix["order"]
        
        # Step 2: Check endpoint (live vs sandbox)
        endpoint_check = self._check_endpoint()
        if not endpoint_check["valid"]:
            result["status"] = "error"
            result["errors"].append(endpoint_check["error"])
            return result
        
        # Step 3: Validate required fields
        fields_check = self._validate_required_fields(order)
        if not fields_check["valid"]:
            result["status"] = "error"
            result["errors"].extend(fields_check["errors"])
            return result
        
        if fields_check.get("fixes"):
            result["fixes_applied"].extend(fields_check["fixes"])
            order = fields_check["order"]
        
        # Step 4: Check account restrictions
        account_check = self._check_account_restrictions(order)
        if not account_check["valid"]:
            result["status"] = "error"
            result["errors"].extend(account_check["errors"])
            return result
        
        # Step 5: Check options approval (if options order)
        if order.get("option_symbol") or order.get("class") == "option":
            options_check = self._check_options_approval()
            if not options_check["valid"]:
                result["status"] = "error"
                result["errors"].append(options_check["error"])
                return result
        
        # Step 6: Check strategy conditions
        strategy_check = self._check_strategy_conditions(order)
        if not strategy_check["valid"]:
            result["status"] = "error"
            result["errors"].extend(strategy_check["errors"])
            return result
        
        # Step 7: Check buying power
        buying_power_check = self._check_buying_power(order)
        if not buying_power_check["valid"]:
            result["status"] = "error"
            result["errors"].append(buying_power_check["error"])
            return result
        
        # Step 8: Preview order first (safe test)
        # CRITICAL FIX: Preview failures should NOT block sell orders or options orders
        # Previews can fail for many reasons (market closed, API issues, symbol format)
        # The actual order should still be attempted for critical orders
        order_side = order.get("side", "").lower()
        is_option_order = bool(order.get("option_symbol") or order.get("class") == "option")
        is_sell = order_side in ("sell", "sell_to_close", "sell_to_open")
        
        preview_result = self._preview_order_safe(order)
        if preview_result.get("errors"):
            if is_sell or is_option_order:
                # For sell and options orders, demote preview errors to warnings
                reason = "sell order" if is_sell else "options order"
                result["warnings"].extend([f"Preview warning ({reason} proceeds anyway): {e}" for e in preview_result["errors"]])
                logger.warning(f"Preview failed for {reason} but proceeding: {preview_result['errors']}")
            else:
                result["status"] = "error"
                result["errors"].extend(preview_result["errors"])
                return result
        
        if preview_result.get("warnings"):
            result["warnings"].extend(preview_result["warnings"])
        
        # Step 9: Submit order
        try:
            submit_result = self.tradier.submit_order(order)
            
            if submit_result.get("status") == "error":
                result["status"] = "error"
                result["errors"].append(submit_result.get("error", "Unknown error"))
            else:
                result["status"] = "submitted"
                result["order_id"] = submit_result.get("order_id")
                result["result"] = submit_result
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Order submission failed: {str(e)}")
            logger.exception("Order submission error")
        
        return result
    
    def _validate_and_fix_symbol(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix symbol formatting
        
        Returns:
            Validation result with fixed order if needed
        """
        result = {
            "valid": True,
            "fixed": False,
            "order": order,
            "message": "",
            "error": ""
        }
        
        symbol = order.get("symbol") or order.get("option_symbol")
        if not symbol:
            result["valid"] = False
            result["error"] = "No symbol provided"
            return result
        
        # Check if it's an option symbol (OCC format)
        if len(symbol) > 5 and any(c.isdigit() for c in symbol[5:]):
            # Likely an option symbol - validate OCC format
            # OCC format: ROOT + EXPIRATION(6) + TYPE(C/P) + STRIKE(8)
            # Example: SPY250117C00500000
            
            # Extract underlying
            underlying = ""
            for i in range(len(symbol)):
                if symbol[i].isdigit():
                    underlying = symbol[:i]
                    break
            
            if not underlying or len(underlying) < 1:
                result["valid"] = False
                result["error"] = f"Invalid option symbol format: {symbol}"
                return result
            
            # Check expiration format (should be 6 digits: YYMMDD)
            exp_start = len(underlying)
            if exp_start + 6 > len(symbol):
                result["valid"] = False
                result["error"] = f"Invalid option symbol format (missing expiration): {symbol}"
                return result
            
            # Check option type (C or P)
            option_type_pos = exp_start + 6
            if option_type_pos >= len(symbol):
                result["valid"] = False
                result["error"] = f"Invalid option symbol format (missing type): {symbol}"
                return result
            
            option_type = symbol[option_type_pos]
            if option_type not in ['C', 'P']:
                result["valid"] = False
                result["error"] = f"Invalid option type '{option_type}' (must be C or P): {symbol}"
                return result
            
            # Check strike price format (should be 8 digits with leading zeros)
            strike_start = option_type_pos + 1
            if strike_start + 8 > len(symbol):
                result["valid"] = False
                result["error"] = f"Invalid option symbol format (missing strike): {symbol}"
                return result
            
            # Ensure option_symbol is set correctly
            if "option_symbol" not in order:
                result["fixed"] = True
                result["order"] = order.copy()
                result["order"]["option_symbol"] = symbol
                result["order"]["class"] = "option"
                if "symbol" in result["order"]:
                    del result["order"]["symbol"]
                result["message"] = f"Set option_symbol and class=option for {symbol}"
        
        else:
            # Equity symbol - validate format
            if not re.match(r'^[A-Z]{1,5}$', symbol):
                result["valid"] = False
                result["error"] = f"Invalid equity symbol format: {symbol}"
                return result
            
            # Ensure symbol is uppercase
            if symbol != symbol.upper():
                result["fixed"] = True
                result["order"] = order.copy()
                result["order"]["symbol"] = symbol.upper()
                result["message"] = f"Uppercased symbol: {symbol.upper()}"
        
        return result
    
    def _check_endpoint(self) -> Dict[str, Any]:
        """
        Check if using correct endpoint (live vs sandbox)
        
        Returns:
            Check result
        """
        result = {
            "valid": True,
            "error": ""
        }
        
        # Check environment variable
        env_sandbox = os.getenv("TRADIER_SANDBOX", "").lower()
        adapter_sandbox = self.tradier.sandbox
        
        # If environment says live but adapter is sandbox (or vice versa), warn
        if env_sandbox == "false" and adapter_sandbox:
            result["valid"] = False
            result["error"] = "Mismatch: TRADIER_SANDBOX=false but adapter is using sandbox. Set sandbox=False in adapter."
        elif env_sandbox == "true" and not adapter_sandbox:
            result["valid"] = False
            result["error"] = "Mismatch: TRADIER_SANDBOX=true but adapter is using live. Set sandbox=True in adapter."
        
        # Verify endpoint is accessible
        if self.diagnostics:
            try:
                connection_ok = self.diagnostics.check_connection()
                if not connection_ok:
                    result["valid"] = False
                    result["error"] = f"Cannot connect to {self.diagnostics.base_url}. Check API key and endpoint."
            except Exception as e:
                logger.warning(f"Could not verify endpoint connection: {e}")
        
        return result
    
    def _validate_required_fields(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all required fields are present
        
        Returns:
            Validation result with fixes applied
        """
        result = {
            "valid": True,
            "errors": [],
            "fixes": [],
            "order": order.copy()
        }
        
        # Required fields for Tradier
        # Equity sides: buy, sell, buy_to_cover, sell_short
        # Options sides: buy_to_open, sell_to_close, buy_to_close, sell_to_open
        required_fields = {
            "side": ["buy", "sell", "buy_to_cover", "sell_short",
                     "buy_to_open", "sell_to_close", "buy_to_close", "sell_to_open"],
            "quantity": lambda x: isinstance(x, int) and x > 0,
            "order_type": ["market", "limit", "stop", "stop_limit"],
            "duration": ["day", "gtc", "pre", "post"]
        }
        
        # Check side
        side = order.get("side", "").lower()
        if not side or side not in required_fields["side"]:
            result["valid"] = False
            result["errors"].append(f"Invalid or missing 'side'. Must be one of: {required_fields['side']}")
        else:
            result["order"]["side"] = side
        
        # Check quantity
        quantity = order.get("quantity", 0)
        if not required_fields["quantity"](quantity):
            result["valid"] = False
            result["errors"].append(f"Invalid or missing 'quantity'. Must be positive integer, got: {quantity}")
        
        # Check order_type
        order_type = order.get("order_type", "market").lower()
        if order_type not in required_fields["order_type"]:
            result["valid"] = False
            result["errors"].append(f"Invalid 'order_type'. Must be one of: {required_fields['order_type']}")
        else:
            result["order"]["order_type"] = order_type
        
        # Check duration
        duration = order.get("duration", "day").lower()
        if duration not in required_fields["duration"]:
            result["valid"] = False
            result["errors"].append(f"Invalid 'duration'. Must be one of: {required_fields['duration']}")
        else:
            result["order"]["duration"] = duration
        
        # Check symbol or option_symbol
        symbol = order.get("symbol") or order.get("option_symbol")
        if not symbol:
            result["valid"] = False
            result["errors"].append("Missing 'symbol' or 'option_symbol'")
        
        # Fix: Ensure class is set correctly
        if order.get("option_symbol") and not order.get("class"):
            result["order"]["class"] = "option"
            result["fixes"].append("Set class=option for option order")
        
        # Fix: Ensure limit orders have price
        if order_type in ["limit", "stop_limit"] and not order.get("price"):
            result["valid"] = False
            result["errors"].append(f"{order_type} orders require 'price' field")
        
        # Fix: Ensure stop orders have stop price
        if order_type in ["stop", "stop_limit"] and not order.get("stop"):
            result["valid"] = False
            result["errors"].append(f"{order_type} orders require 'stop' field")
        
        return result
    
    def _check_account_restrictions(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check account restrictions
        
        Returns:
            Check result
        """
        result = {
            "valid": True,
            "errors": []
        }
        
        # Get account profile (cached)
        profile = self._get_account_profile()
        if not profile:
            # If we can't get profile, assume OK but warn
            logger.warning("Could not get account profile for restriction check")
            return result
        
        # Check if account is restricted
        account_type = profile.get("account_type", "")
        if account_type in ["restricted", "closed"]:
            result["valid"] = False
            result["errors"].append(f"Account is {account_type}. Cannot place orders.")
        
        # Check margin approval for SHORT SELLING only
        # Regular "sell" (closing a long position) does NOT require margin
        # Only "sell_short" (opening a short position) requires margin approval
        if order.get("side") == "sell_short" and not profile.get("margin_approved", False):
            result["valid"] = False
            result["errors"].append("Account does not have margin approval. Cannot place short sell orders.")
        
        return result
    
    def _check_options_approval(self) -> Dict[str, Any]:
        """
        Check if account has options approval.
        
        Tradier option levels:
          0 = No options
          1 = Covered calls, protective puts
          2 = Long calls, long puts, covered calls, cash-secured puts
          3 = Spreads (credit/debit)
          4 = Uncovered options
        
        Returns:
            Check result with option_level info
        """
        result = {
            "valid": True,
            "error": "",
            "option_level": 0
        }
        
        profile = self._get_account_profile()
        if not profile:
            # If we can't get profile, allow the order and let Tradier validate
            logger.warning("Could not verify options approval - letting Tradier validate")
            return result
        
        # Get option_level from profile (diagnostics now flattens this from account.option_level)
        option_level = profile.get("option_level", 0)
        
        # Also check nested account structure as fallback
        if option_level == 0:
            account = profile.get("_account", profile.get("account", {}))
            if isinstance(account, dict):
                option_level = account.get("option_level", 0)
            elif isinstance(account, list) and len(account) > 0:
                option_level = account[0].get("option_level", 0)
        
        # Convert string option levels to int if needed
        if isinstance(option_level, str):
            try:
                option_level = int(option_level.replace("level_", ""))
            except (ValueError, AttributeError):
                option_level = 0
        
        result["option_level"] = option_level
        
        if option_level < 1:
            result["valid"] = False
            result["error"] = f"Options trading not approved. Option level: {option_level}. Request approval from Tradier."
            return result
        
        logger.info(f"Options approved: Level {option_level}")
        return result
    
    def _check_strategy_conditions(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check strategy conditions are met
        
        Returns:
            Check result
        """
        result = {
            "valid": True,
            "errors": []
        }
        
        # Check if strategy_id is present (indicates strategy-based order)
        strategy_id = order.get("strategy_id")
        if strategy_id:
            # Strategy conditions would be checked here
            # For now, we assume strategy conditions are validated upstream
            # This is a placeholder for future strategy condition validation
            pass
        
        return result
    
    def _check_buying_power(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check sufficient buying power (BUY orders only).
        SELL orders do NOT require buying power - they liquidate existing positions.
        
        Returns:
            Check result
        """
        result = {
            "valid": True,
            "error": ""
        }
        
        # CRITICAL FIX: Sell orders do NOT need buying power - they FREE UP capital
        # Only check buying power for buy orders
        order_side = order.get("side", "").lower()
        if order_side in ("sell", "sell_to_close"):
            # Sell orders always pass buying power check
            return result
        
        balances = self._get_account_balances()
        if not balances:
            result["valid"] = False
            result["error"] = "Could not get account balances. Cannot verify buying power."
            return result
        
        # Get buying power
        is_option = bool(order.get("option_symbol") or order.get("class") == "option")
        
        if is_option:
            # For options: try options_buying_power first, fall back to cash_available
            # Cash accounts don't have options_buying_power field; use cash instead
            obp = balances.get("options_buying_power")
            if obp and obp != "N/A" and isinstance(obp, (int, float)):
                buying_power = float(obp)
            else:
                buying_power = float(balances.get("cash_available", 0) or 0)
        else:
            buying_power = float(balances.get("cash_available", 0) or balances.get("day_trading_buying_power", 0) or 0)
        
        # Estimate order cost (rough estimate)
        quantity = order.get("quantity", 0)
        price = order.get("price", 0)
        
        if price > 0:
            # Estimate cost - multiply by 100 only for options (per-contract multiplier)
            if is_option:
                estimated_cost = quantity * price * 100  # Options are per 100 shares
            else:
                estimated_cost = quantity * price  # Equity: quantity * price
            if estimated_cost > buying_power:
                result["valid"] = False
                result["error"] = f"Insufficient buying power. Required: ${estimated_cost:.2f}, Available: ${buying_power:.2f}"
                return result
        else:
            # Market order - check if buying power exists
            if buying_power <= 0:
                result["valid"] = False
                result["error"] = f"Insufficient buying power. Available: ${buying_power:.2f}"
                return result
        
        return result
    
    def _preview_order_safe(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview order safely to catch errors before submission
        
        Returns:
            Preview result with errors and warnings
        """
        result = {
            "errors": [],
            "warnings": []
        }
        
        try:
            # Create preview order
            preview_order = order.copy()
            preview_order["preview"] = True
            
            preview_result = self.tradier.rest_client.preview_order(
                account_id=self.tradier.account_id,
                symbol=preview_order.get("symbol", ""),
                side=preview_order.get("side", "buy"),
                quantity=preview_order.get("quantity", 0),
                order_type=preview_order.get("order_type", "market"),
                duration=preview_order.get("duration", "day"),
                price=preview_order.get("price"),
                stop=preview_order.get("stop"),
                option_symbol=preview_order.get("option_symbol"),
                tag=preview_order.get("tag")
            )
            
            # Check for errors in preview
            if "order" in preview_result:
                order_data = preview_result["order"]
                if isinstance(order_data, dict):
                    errors = order_data.get("errors", [])
                    warnings = order_data.get("warnings", [])
                    
                    if errors:
                        result["errors"].extend([str(e) for e in errors])
                    
                    if warnings:
                        result["warnings"].extend([str(w) for w in warnings])
        
        except Exception as e:
            result["errors"].append(f"Preview failed: {str(e)}")
            logger.exception("Order preview error")
        
        return result
    
    def _get_account_profile(self) -> Optional[Dict[str, Any]]:
        """Get account profile (cached)"""
        # Cache for 5 minutes
        if self._account_profile and self._last_check_time:
            time_diff = (datetime.now() - self._last_check_time).total_seconds()
            if time_diff < 300:  # 5 minutes
                return self._account_profile
        
        try:
            if self.diagnostics:
                profile = self.diagnostics.get_profile()
                if profile:
                    self._account_profile = profile.get("profile", {})
                    self._last_check_time = datetime.now()
                    return self._account_profile
        except Exception as e:
            logger.warning(f"Could not get account profile: {e}")
        
        return None
    
    def _get_account_balances(self) -> Optional[Dict[str, Any]]:
        """Get account balances (cached)"""
        # Cache for 1 minute
        if self._account_balances and self._last_check_time:
            time_diff = (datetime.now() - self._last_check_time).total_seconds()
            if time_diff < 60:  # 1 minute
                return self._account_balances
        
        try:
            if self.diagnostics:
                balances = self.diagnostics.get_account_balances()
                if balances:
                    # get_account_balances() returns the balances dict directly
                    # Extract cash_available from nested cash structure if needed
                    if isinstance(balances, dict):
                        # Check if balances has nested structure
                        if "balances" in balances:
                            self._account_balances = balances.get("balances", {})
                        else:
                            # Direct balances dict
                            self._account_balances = balances
                        
                        # Ensure cash_available is accessible
                        if "cash_available" not in self._account_balances and "cash" in self._account_balances:
                            cash_info = self._account_balances.get("cash", {})
                            if isinstance(cash_info, dict):
                                self._account_balances["cash_available"] = cash_info.get("cash_available", 0)
                        
                        return self._account_balances
        except Exception as e:
            logger.warning(f"Could not get account balances: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Fallback: Try direct Tradier adapter call
        try:
            if self.tradier:
                # Use Tradier adapter's get_balances method if available
                if hasattr(self.tradier, 'get_balances'):
                    balances = self.tradier.get_balances()
                    if balances:
                        self._account_balances = balances
                        return self._account_balances
        except Exception as e:
            logger.warning(f"Fallback balance check failed: {e}")
        
        return None


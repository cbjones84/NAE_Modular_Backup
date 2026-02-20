#!/usr/bin/env python3
"""
Genny Tax Integration Hooks
Automatic integration with other agents for tax tracking
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class TaxIntegrationHooks:
    """
    Integration hooks to automatically track trades and transactions for tax purposes
    """
    
    def __init__(self, genny_agent):
        """
        Initialize tax integration hooks
        
        Args:
            genny_agent: GennyAgent instance
        """
        self.genny = genny_agent
        self.log_file = "logs/genny_tax_integration.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logger.info("Tax Integration Hooks initialized")
    
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
    
    def hook_optimus_trade(self, trade_data: Dict[str, Any]):
        """
        Hook to track Optimus trades for tax purposes
        
        Args:
            trade_data: Trade data from Optimus
        """
        try:
            if not self.genny.tax_tracking_enabled:
                return
            
            # Track trade
            self.genny._track_trade_for_taxes(trade_data, agent="optimus")
            
            self.log_action(f"Hooked Optimus trade: {trade_data.get('symbol', 'Unknown')}")
        
        except Exception as e:
            self.log_action(f"Error hooking Optimus trade: {e}")
    
    def hook_shredder_crypto(self, crypto_data: Dict[str, Any]):
        """
        Hook to track Shredder crypto transactions for tax purposes
        
        Args:
            crypto_data: Crypto transaction data from Shredder
        """
        try:
            if not self.genny.tax_tracking_enabled:
                return
            
            # Track crypto transaction
            self.genny.track_crypto_transaction(crypto_data, agent="shredder")
            
            self.log_action(f"Hooked Shredder crypto: {crypto_data.get('crypto_symbol', 'Unknown')}")
        
        except Exception as e:
            self.log_action(f"Error hooking Shredder crypto: {e}")
    
    def hook_april_crypto(self, crypto_data: Dict[str, Any]):
        """
        Hook to track April crypto transactions for tax purposes
        
        Args:
            crypto_data: Crypto transaction data from April
        """
        try:
            if not self.genny.tax_tracking_enabled:
                return
            
            # Track crypto transaction
            self.genny.track_crypto_transaction(crypto_data, agent="april")
            
            self.log_action(f"Hooked April crypto: {crypto_data.get('crypto_symbol', 'Unknown')}")
        
        except Exception as e:
            self.log_action(f"Error hooking April crypto: {e}")
    
    def hook_donnie_fiat(self, fiat_data: Dict[str, Any]):
        """
        Hook to track Donnie fiat flows for tax purposes
        
        Args:
            fiat_data: Fiat flow data from Donnie
        """
        try:
            if not self.genny.tax_tracking_enabled:
                return
            
            # Track fiat flow
            self.genny.track_fiat_flow(fiat_data, agent="donnie")
            
            self.log_action(f"Hooked Donnie fiat flow: ${fiat_data.get('amount', 0):.2f}")
        
        except Exception as e:
            self.log_action(f"Error hooking Donnie fiat: {e}")
    
    def hook_mikey_fiat(self, fiat_data: Dict[str, Any]):
        """
        Hook to track Mikey fiat flows for tax purposes
        
        Args:
            fiat_data: Fiat flow data from Mikey
        """
        try:
            if not self.genny.tax_tracking_enabled:
                return
            
            # Track fiat flow
            self.genny.track_fiat_flow(fiat_data, agent="mikey")
            
            self.log_action(f"Hooked Mikey fiat flow: ${fiat_data.get('amount', 0):.2f}")
        
        except Exception as e:
            self.log_action(f"Error hooking Mikey fiat: {e}")
    
    def auto_track_expenses(self):
        """
        Automatically track common deductible expenses
        """
        try:
            if not self.genny.tax_tracking_enabled:
                return
            
            # Common deductible expenses for trading business
            common_expenses = [
                {
                    "category": "software",
                    "description": "Trading platform subscription",
                    "amount": 0.0,  # Would be fetched from actual subscriptions
                    "deductible_pct": 100.0,
                    "business_use_pct": 100.0
                },
                {
                    "category": "software",
                    "description": "Market data feed subscription",
                    "amount": 0.0,
                    "deductible_pct": 100.0,
                    "business_use_pct": 100.0
                },
                {
                    "category": "hardware",
                    "description": "Trading computer/server",
                    "amount": 0.0,
                    "deductible_pct": 100.0,
                    "business_use_pct": 100.0
                },
                {
                    "category": "subscription",
                    "description": "Cloud hosting for NAE",
                    "amount": 0.0,
                    "deductible_pct": 100.0,
                    "business_use_pct": 100.0
                }
            ]
            
            # Would integrate with actual expense tracking
            self.log_action("Auto-expense tracking ready (requires expense data source)")
        
        except Exception as e:
            self.log_action(f"Error in auto expense tracking: {e}")


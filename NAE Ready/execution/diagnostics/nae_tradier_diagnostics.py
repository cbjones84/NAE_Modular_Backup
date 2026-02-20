#!/usr/bin/env python3
"""
NAE Tradier Diagnostics Module

This module performs comprehensive diagnostics to identify why NAE is not placing trades through Tradier.

The module performs all checks in the correct order:
- Confirms Live vs Sandbox endpoint
- Confirms your account ID is correct
- Confirms options approval level
- Shows buying power & settled cash
- Validates symbols & OCC formats
- Attempts a $0 "test order" (Tradier supports this via preview=true)
- Returns the exact error message from Tradier, if any
- Prints all logs clearly so Optimus, Donnie, or Ralph can read them

This will tell you in seconds whether the issue is:
- Strategy conditions
- Permissions
- Wrong endpoint
- Missing fields
- Rejected orders
- Account restrictions
- Bad symbol formatting
"""

import os
import sys
import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, nae_root)

# Try to import NAE's Tradier adapter for consistency
try:
    from execution.broker_adapters.tradier_adapter import TradierOAuth, TradierRESTClient
    NAE_ADAPTER_AVAILABLE = True
except ImportError:
    NAE_ADAPTER_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradierDiagnostics:
    """
    Comprehensive Tradier diagnostics to identify trading issues
    """
    
    def __init__(self, api_key: str = None, account_id: str = None, live: bool = True):
        """
        Initialize diagnostics
        
        Args:
            api_key: Tradier API key (auto-loaded from env/vault if None)
            account_id: Tradier account ID (auto-loaded from env if None)
            live: Use live endpoint (False for sandbox)
        """
        # Try to get credentials from environment or vault
        if not api_key:
            api_key = os.getenv("TRADIER_API_KEY")
            if not api_key and NAE_ADAPTER_AVAILABLE:
                try:
                    from secure_vault import get_vault
                    vault = get_vault()
                    api_key = vault.get_secret("tradier", "api_key")
                except:
                    pass
        
        if not account_id:
            account_id = os.getenv("TRADIER_ACCOUNT_ID")
        
        if not api_key:
            raise ValueError("TRADIER_API_KEY not found. Set environment variable or provide as parameter.")
        
        self.api_key = api_key
        self.account_id = account_id
        self.live = live
        self.base_url = "https://api.tradier.com/v1" if live else "https://sandbox.tradier.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        self.diagnostics_results = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": "live" if live else "sandbox",
            "account_id": account_id,
            "checks": {}
        }
    
    def log(self, label: str, data: Any):
        """
        Log diagnostic information in a clear format
        
        Args:
            label: Label for the diagnostic check
            data: Data to log
        """
        print(f"\n{'='*60}")
        print(f" {label}")
        print(f"{'='*60}")
        try:
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            elif isinstance(data, (list, tuple)):
                print(json.dumps(data, indent=2))
            else:
                print(str(data))
        except Exception as e:
            print(f"Error formatting data: {e}")
            print(str(data))
        
        # Store in results
        self.diagnostics_results["checks"][label] = data
    
    def check_connection(self) -> bool:
        """
        Check API connection and endpoint
        
        Returns:
            True if connection successful
        """
        try:
            url = f"{self.base_url}/markets/quotes"
            resp = requests.get(url, headers=self.headers, params={"symbols": "SPY"}, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                self.log("✅ Connection Test: SUCCESS", {
                    "endpoint": self.base_url,
                    "status_code": resp.status_code,
                    "response": data
                })
                return True
            else:
                self.log("❌ Connection Test: FAILED", {
                    "endpoint": self.base_url,
                    "status_code": resp.status_code,
                    "response": resp.text
                })
                return False
        except Exception as e:
            self.log("❌ Connection Test: ERROR", {
                "endpoint": self.base_url,
                "error": str(e)
            })
            return False
    
    def get_account_balances(self) -> Dict[str, Any]:
        """
        Get account balances and buying power
        
        Returns:
            Account balances data
        """
        if not self.account_id:
            self.log("⚠️ Account Balances: SKIPPED", {
                "reason": "No account_id provided"
            })
            return {}
        
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/balances"
            resp = requests.get(url, headers=self.headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                balances = data.get("balances", {})
                
                # Extract key fields
                key_fields = {
                    "cash": balances.get("cash", "N/A"),
                    "cash_available": balances.get("cash_available", "N/A"),
                    "margin_balance": balances.get("margin_balance", "N/A"),
                    "pending_cash": balances.get("pending_cash", "N/A"),
                    "unsettled_funds": balances.get("unsettled_funds", "N/A"),
                    "uncleared_funds": balances.get("uncleared_funds", "N/A"),
                    "total_equity": balances.get("total_equity", "N/A"),
                    "options_buying_power": balances.get("options_buying_power", "N/A"),
                    "day_trading_buying_power": balances.get("day_trading_buying_power", "N/A"),
                }
                
                self.log("✅ Account Balances", {
                    "status_code": resp.status_code,
                    "key_fields": key_fields,
                    "full_response": balances
                })
                return balances
            else:
                self.log("❌ Account Balances: FAILED", {
                    "status_code": resp.status_code,
                    "response": resp.text
                })
                return {}
        except Exception as e:
            self.log("❌ Account Balances: ERROR", {
                "error": str(e)
            })
            return {}
    
    def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile and options approval level
        
        Returns:
            User profile data
        """
        try:
            url = f"{self.base_url}/user/profile"
            resp = requests.get(url, headers=self.headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                profile = data.get("profile", {})
                
                # Tradier nests account info under profile.account
                # Handle both single account and multiple accounts
                account = profile.get("account", {})
                if isinstance(account, list) and len(account) > 0:
                    account = account[0]  # Use first account
                elif not isinstance(account, dict):
                    account = {}
                
                # Extract key fields from the correct nested location
                option_level = account.get("option_level", 0)
                key_fields = {
                    "account_type": account.get("type", profile.get("account_type", "N/A")),
                    "options_level": option_level,
                    "options_approved": option_level >= 1,  # Level 1+ means options approved
                    "day_trader": account.get("day_trader", profile.get("day_trader", False)),
                    "margin_approved": account.get("type", "") == "margin",
                    "account_status": account.get("status", "unknown"),
                    "classification": account.get("classification", "N/A"),
                }
                
                # Flatten: inject extracted fields into profile for downstream consumers
                profile["_account"] = account
                profile["option_level"] = option_level
                profile["options_approved"] = key_fields["options_approved"]
                profile["options_level"] = option_level
                profile["day_trader"] = key_fields["day_trader"]
                profile["margin_approved"] = key_fields["margin_approved"]
                profile["account_type"] = key_fields["account_type"]
                profile["account_status"] = key_fields["account_status"]
                
                self.log("✅ User Profile & Options Approval", {
                    "status_code": resp.status_code,
                    "key_fields": key_fields,
                    "full_response": profile
                })
                return profile
            else:
                self.log("❌ User Profile: FAILED", {
                    "status_code": resp.status_code,
                    "response": resp.text
                })
                return {}
        except Exception as e:
            self.log("❌ User Profile: ERROR", {
                "error": str(e)
            })
            return {}
    
    def get_accounts(self) -> Dict[str, Any]:
        """
        Get all accounts to verify account_id
        
        Returns:
            Accounts data
        """
        try:
            url = f"{self.base_url}/accounts"
            resp = requests.get(url, headers=self.headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                accounts = data.get("accounts", {}).get("account", [])
                
                # Normalize to list
                if isinstance(accounts, dict):
                    accounts = [accounts]
                
                # Extract account IDs
                account_ids = []
                for acc in accounts:
                    acc_id = acc.get("account_number") or acc.get("id")
                    if acc_id:
                        account_ids.append(acc_id)
                
                # Check if provided account_id matches
                account_found = False
                if self.account_id:
                    account_found = self.account_id in account_ids
                
                self.log("✅ Account Verification", {
                    "status_code": resp.status_code,
                    "provided_account_id": self.account_id,
                    "available_account_ids": account_ids,
                    "account_found": account_found,
                    "accounts": accounts
                })
                return {"accounts": accounts, "account_ids": account_ids}
            else:
                self.log("❌ Account Verification: FAILED", {
                    "status_code": resp.status_code,
                    "response": resp.text
                })
                return {}
        except Exception as e:
            self.log("❌ Account Verification: ERROR", {
                "error": str(e)
            })
            return {}
    
    def test_symbol(self, occ_symbol: str) -> Dict[str, Any]:
        """
        Validate option symbol and get option chain
        
        Args:
            occ_symbol: OCC format option symbol (e.g., SPY250117C00500000)
        
        Returns:
            Option chain data
        """
        try:
            # Extract underlying symbol (first 3-5 characters before expiration)
            # OCC format: ROOT + EXPIRATION + TYPE + STRIKE
            # For SPY250117C00500000: SPY is underlying
            underlying = ""
            for i in range(len(occ_symbol)):
                if occ_symbol[i].isdigit():
                    underlying = occ_symbol[:i]
                    break
            
            if not underlying:
                self.log("⚠️ Option Symbol Validation: INVALID FORMAT", {
                    "symbol": occ_symbol,
                    "reason": "Could not extract underlying symbol"
                })
                return {}
            
            url = f"{self.base_url}/markets/options/chains"
            resp = requests.get(url, headers=self.headers, params={"symbol": underlying}, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                chains = data.get("options", {})
                
                self.log("✅ Option Chain Validation", {
                    "underlying": underlying,
                    "occ_symbol": occ_symbol,
                    "status_code": resp.status_code,
                    "chains_available": bool(chains),
                    "response": data
                })
                return data
            else:
                self.log("❌ Option Chain Validation: FAILED", {
                    "underlying": underlying,
                    "occ_symbol": occ_symbol,
                    "status_code": resp.status_code,
                    "response": resp.text
                })
                return {}
        except Exception as e:
            self.log("❌ Option Chain Validation: ERROR", {
                "occ_symbol": occ_symbol,
                "error": str(e)
            })
            return {}
    
    def place_test_order(self, occ_symbol: str = "SPY250117C00500000") -> Dict[str, Any]:
        """
        Place a test order with preview=true (doesn't execute)
        
        Tradier supports preview=true to test orders safely without executing.
        
        Args:
            occ_symbol: Option symbol to test (default: SPY250117C00500000)
        
        Returns:
            Test order response
        """
        if not self.account_id:
            self.log("⚠️ Test Order: SKIPPED", {
                "reason": "No account_id provided"
            })
            return {}
        
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/orders"
            
            payload = {
                "class": "option",
                "symbol": occ_symbol,
                "side": "buy",
                "quantity": 1,
                "type": "market",
                "duration": "day",
                "preview": "true"  # This makes it a test order
            }
            
            resp = requests.post(url, headers=self.headers, data=payload, timeout=10)
            
            data = resp.json() if resp.status_code in [200, 400, 422] else {"error": resp.text}
            
            # Check for errors
            if resp.status_code == 200:
                order = data.get("order", {})
                warnings = order.get("warnings", [])
                errors = order.get("errors", [])
                
                if warnings:
                    self.log("⚠️ Test Order: WARNINGS", {
                        "status_code": resp.status_code,
                        "warnings": warnings,
                        "full_response": data
                    })
                elif errors:
                    self.log("❌ Test Order: ERRORS", {
                        "status_code": resp.status_code,
                        "errors": errors,
                        "full_response": data
                    })
                else:
                    self.log("✅ Test Order: SUCCESS", {
                        "status_code": resp.status_code,
                        "response": data
                    })
            else:
                self.log("❌ Test Order: FAILED", {
                    "status_code": resp.status_code,
                    "response": data
                })
            
            return data
        except Exception as e:
            self.log("❌ Test Order: ERROR", {
                "error": str(e),
                "occ_symbol": occ_symbol
            })
            return {}
    
    def run_full_diagnostics(self, test_symbol: str = "SPY250117C00500000") -> Dict[str, Any]:
        """
        Run complete diagnostic suite
        
        Args:
            test_symbol: Option symbol to test (default: SPY250117C00500000)
        
        Returns:
            Complete diagnostics results
        """
        print("\n" + "="*60)
        print(" NAE → TRADIER DIAGNOSTIC RUN")
        print("="*60 + "\n")
        
        print("1️⃣  Checking API connection...")
        connection_ok = self.check_connection()
        
        print("\n2️⃣  Checking available accounts...")
        accounts_data = self.get_accounts()
        
        print("\n3️⃣  Checking user profile & options approval...")
        profile = self.get_profile()
        
        print("\n4️⃣  Checking cash & buying power...")
        balances = self.get_account_balances()
        
        print("\n5️⃣  Validating option symbol...")
        symbol_data = self.test_symbol(test_symbol)
        
        print("\n6️⃣  Attempting test order (preview=true)...")
        test_order = self.place_test_order(test_symbol)
        
        # Generate summary
        print("\n" + "="*60)
        print(" DIAGNOSTIC SUMMARY")
        print("="*60)
        
        summary = {
            "connection": "✅ OK" if connection_ok else "❌ FAILED",
            "endpoint": "LIVE" if self.live else "SANDBOX",
            "account_id": self.account_id or "NOT PROVIDED",
            "options_approved": profile.get("options_approved", False) if profile else "UNKNOWN",
            "options_level": profile.get("options_level", "N/A") if profile else "UNKNOWN",
            "has_buying_power": bool(balances.get("cash_available") or balances.get("options_buying_power")) if balances else False,
            "test_order_status": "SUCCESS" if test_order and not test_order.get("order", {}).get("errors") else "FAILED"
        }
        
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Diagnostics complete.\n")
        
        # Store summary
        self.diagnostics_results["summary"] = summary
        
        return self.diagnostics_results


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NAE Tradier Diagnostics")
    parser.add_argument("--api-key", help="Tradier API key (or set TRADIER_API_KEY env var)")
    parser.add_argument("--account-id", help="Tradier account ID (or set TRADIER_ACCOUNT_ID env var)")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox endpoint")
    parser.add_argument("--test-symbol", default="SPY250117C00500000", help="Option symbol to test")
    
    args = parser.parse_args()
    
    diag = TradierDiagnostics(
        api_key=args.api_key,
        account_id=args.account_id,
        live=not args.sandbox
    )
    
    results = diag.run_full_diagnostics(test_symbol=args.test_symbol)
    
    # Save results to file
    output_file = f"logs/tradier_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full results saved to: {output_file}\n")


if __name__ == "__main__":
    main()


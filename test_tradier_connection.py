#!/usr/bin/env python3
"""
Comprehensive Tradier API Connection Test
Tests connection, account access, funds availability, and trading capabilities
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add execution directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'execution'))

from broker_adapters.tradier_adapter import TradierBrokerAdapter, TradierRESTClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradierConnectionTest:
    """Comprehensive Tradier connection and functionality test"""
    
    def __init__(self, sandbox: bool = False):
        """
        Initialize test
        
        Args:
            sandbox: Use sandbox environment (default: False for live)
        """
        self.sandbox = sandbox
        self.adapter: Optional[TradierBrokerAdapter] = None
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "sandbox": sandbox,
            "tests": {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        logger.info("=" * 60)
        logger.info("TRADIER CONNECTION TEST SUITE")
        logger.info("=" * 60)
        logger.info(f"Environment: {'SANDBOX' if self.sandbox else 'LIVE'}")
        logger.info("")
        
        try:
            # Test 1: Initialize adapter
            self.test_initialize_adapter()
            
            # Test 2: Authenticate
            if not self.test_authenticate():
                logger.error("Authentication failed. Cannot proceed with further tests.")
                return self.test_results
            
            # Test 3: Get accounts
            accounts = self.test_get_accounts()
            
            # Test 4: Get account details (if accounts available)
            if accounts:
                account_id = accounts[0].get("account_number") if isinstance(accounts[0], dict) else None
                if account_id:
                    self.test_get_account_details(account_id)
                    self.test_get_account_balance(account_id)
                    self.test_get_positions(account_id)
                    self.test_get_orders(account_id)
                    
                    # Test 5: Test order preview (safe, no actual order)
                    self.test_order_preview(account_id)
            
            # Test 6: Test adapter status
            self.test_adapter_status()
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("TEST SUMMARY")
            logger.info("=" * 60)
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Test suite failed with error: {e}", exc_info=True)
            self.test_results["error"] = str(e)
        
        return self.test_results
    
    def test_initialize_adapter(self):
        """Test 1: Initialize Tradier adapter"""
        logger.info("Test 1: Initializing Tradier adapter...")
        try:
            self.adapter = TradierBrokerAdapter(sandbox=self.sandbox)
            logger.info("✅ Adapter initialized successfully")
            logger.info(f"   - Using API key auth: {self.adapter.oauth.use_api_key}")
            logger.info(f"   - Using OAuth auth: {self.adapter.oauth.use_oauth}")
            self.test_results["tests"]["initialize"] = {
                "status": "PASS",
                "api_key_auth": self.adapter.oauth.use_api_key,
                "oauth_auth": self.adapter.oauth.use_oauth
            }
        except Exception as e:
            logger.error(f"❌ Failed to initialize adapter: {e}")
            self.test_results["tests"]["initialize"] = {
                "status": "FAIL",
                "error": str(e)
            }
            raise
    
    def test_authenticate(self) -> bool:
        """Test 2: Authenticate with Tradier"""
        logger.info("Test 2: Authenticating with Tradier...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            result = self.adapter.authenticate()
            if result:
                logger.info("✅ Authentication successful")
                logger.info(f"   - Access token available: {bool(self.adapter.oauth.access_token)}")
                logger.info(f"   - Token type: {'API Key' if self.adapter.oauth.use_api_key else 'OAuth'}")
                self.test_results["tests"]["authenticate"] = {
                    "status": "PASS",
                    "auth_method": "API_KEY" if self.adapter.oauth.use_api_key else "OAUTH"
                }
                return True
            else:
                logger.error("❌ Authentication failed")
                self.test_results["tests"]["authenticate"] = {
                    "status": "FAIL",
                    "error": "Authentication returned False"
                }
                return False
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            self.test_results["tests"]["authenticate"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_get_accounts(self) -> list:
        """Test 3: Get account list"""
        logger.info("Test 3: Retrieving account list...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            accounts = self.adapter.rest_client.get_accounts()
            logger.info(f"✅ Retrieved {len(accounts)} account(s)")
            
            for i, account in enumerate(accounts):
                if isinstance(account, dict):
                    account_num = account.get("account_number", "N/A")
                    account_type = account.get("type", "N/A")
                    logger.info(f"   Account {i+1}: {account_num} ({account_type})")
            
            self.test_results["tests"]["get_accounts"] = {
                "status": "PASS",
                "account_count": len(accounts),
                "accounts": accounts[:3]  # Limit to first 3 for privacy
            }
            return accounts
        except Exception as e:
            logger.error(f"❌ Failed to get accounts: {e}")
            self.test_results["tests"]["get_accounts"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return []
    
    def test_get_account_details(self, account_id: str):
        """Test 4: Get account details"""
        logger.info(f"Test 4: Retrieving account details for {account_id}...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            details = self.adapter.rest_client.get_account_details(account_id)
            logger.info("✅ Account details retrieved")
            
            # Log key information
            if isinstance(details, dict):
                account_type = details.get("type", "N/A")
                day_trading = details.get("day_trading", False)
                logger.info(f"   - Account type: {account_type}")
                logger.info(f"   - Day trading: {day_trading}")
            
            self.test_results["tests"]["get_account_details"] = {
                "status": "PASS",
                "account_id": account_id
            }
        except Exception as e:
            logger.error(f"❌ Failed to get account details: {e}")
            self.test_results["tests"]["get_account_details"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def test_get_account_balance(self, account_id: str):
        """Test 5: Get account balance and available funds"""
        logger.info(f"Test 5: Retrieving account balance for {account_id}...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            details = self.adapter.rest_client.get_account_details(account_id)
            logger.info("✅ Account balance retrieved")
            
            if isinstance(details, dict):
                # Extract balance information
                total_equity = details.get("total_equity", 0)
                cash = details.get("cash", {}).get("cash", 0) if isinstance(details.get("cash"), dict) else details.get("cash", 0)
                buying_power = details.get("margin", {}).get("stock_buying_power", 0) if isinstance(details.get("margin"), dict) else details.get("buying_power", 0)
                
                logger.info(f"   - Total Equity: ${total_equity:,.2f}")
                logger.info(f"   - Cash: ${cash:,.2f}")
                logger.info(f"   - Buying Power: ${buying_power:,.2f}")
                
                self.test_results["tests"]["get_account_balance"] = {
                    "status": "PASS",
                    "account_id": account_id,
                    "total_equity": total_equity,
                    "cash": cash,
                    "buying_power": buying_power
                }
            else:
                logger.warning("   - Balance details format unexpected")
                self.test_results["tests"]["get_account_balance"] = {
                    "status": "PASS",
                    "account_id": account_id,
                    "note": "Details retrieved but format unexpected"
                }
        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            self.test_results["tests"]["get_account_balance"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def test_get_positions(self, account_id: str):
        """Test 6: Get current positions"""
        logger.info(f"Test 6: Retrieving positions for {account_id}...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            positions = self.adapter.rest_client.get_positions(account_id)
            logger.info(f"✅ Retrieved {len(positions)} position(s)")
            
            for i, pos in enumerate(positions[:5]):  # Show first 5
                if isinstance(pos, dict):
                    symbol = pos.get("symbol", "N/A")
                    quantity = pos.get("quantity", 0)
                    logger.info(f"   Position {i+1}: {symbol} x {quantity}")
            
            self.test_results["tests"]["get_positions"] = {
                "status": "PASS",
                "position_count": len(positions)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            self.test_results["tests"]["get_positions"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def test_get_orders(self, account_id: str):
        """Test 7: Get recent orders"""
        logger.info(f"Test 7: Retrieving recent orders for {account_id}...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            orders = self.adapter.rest_client.get_orders(account_id)
            logger.info(f"✅ Retrieved {len(orders)} order(s)")
            
            # Show recent orders
            for i, order in enumerate(orders[:5]):  # Show first 5
                if isinstance(order, dict):
                    symbol = order.get("symbol", "N/A")
                    side = order.get("side", "N/A")
                    status = order.get("status", "N/A")
                    logger.info(f"   Order {i+1}: {side} {symbol} - {status}")
            
            self.test_results["tests"]["get_orders"] = {
                "status": "PASS",
                "order_count": len(orders)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get orders: {e}")
            self.test_results["tests"]["get_orders"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def test_order_preview(self, account_id: str):
        """Test 8: Preview order (safe, no actual order)"""
        logger.info("Test 8: Testing order preview (safe, no actual order)...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            # Preview a small market order for SPY (safe test)
            preview = self.adapter.rest_client.preview_order(
                account_id=account_id,
                symbol="SPY",
                side="buy",
                quantity=1,
                order_type="market",
                duration="day"
            )
            
            logger.info("✅ Order preview successful")
            
            if isinstance(preview, dict):
                if "warnings" in preview:
                    logger.info(f"   - Warnings: {preview['warnings']}")
                if "order" in preview:
                    order_info = preview["order"]
                    if isinstance(order_info, dict):
                        estimated_cost = order_info.get("estimated_commission", 0)
                        logger.info(f"   - Estimated commission: ${estimated_cost:.2f}")
            
            self.test_results["tests"]["order_preview"] = {
                "status": "PASS",
                "note": "Order preview completed successfully (no actual order placed)"
            }
        except Exception as e:
            logger.error(f"❌ Order preview failed: {e}")
            self.test_results["tests"]["order_preview"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def test_adapter_status(self):
        """Test 9: Get adapter status"""
        logger.info("Test 9: Checking adapter status...")
        try:
            if not self.adapter:
                raise Exception("Adapter not initialized")
            
            status = self.adapter.get_status()
            logger.info("✅ Adapter status retrieved")
            logger.info(f"   - Broker: {status.get('broker')}")
            logger.info(f"   - Sandbox: {status.get('sandbox')}")
            logger.info(f"   - Authenticated: {status.get('authenticated')}")
            logger.info(f"   - Token expired: {status.get('token_expired')}")
            logger.info(f"   - WebSocket connected: {status.get('websocket_connected')}")
            
            self.test_results["tests"]["adapter_status"] = {
                "status": "PASS",
                "details": status
            }
        except Exception as e:
            logger.error(f"❌ Failed to get adapter status: {e}")
            self.test_results["tests"]["adapter_status"] = {
                "status": "FAIL",
                "error": str(e)
            }
    
    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for test in self.test_results["tests"].values() if test.get("status") == "PASS")
        failed = sum(1 for test in self.test_results["tests"].values() if test.get("status") == "FAIL")
        total = len(self.test_results["tests"])
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info("")
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! Tradier connection is working correctly.")
        else:
            logger.warning("⚠️  Some tests failed. Review the errors above.")
        
        # Save results to file
        results_file = f"tradier_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"\nTest results saved to: {results_file}")


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Tradier API connection")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use sandbox environment (default: live)"
    )
    parser.add_argument(
        "--account-id",
        type=str,
        help="Specific account ID to test (optional)"
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = TradierConnectionTest(sandbox=args.sandbox)
    results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    failed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "FAIL")
    sys.exit(1 if failed_tests > 0 else 0)


if __name__ == "__main__":
    main()


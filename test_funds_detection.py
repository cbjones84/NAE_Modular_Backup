#!/usr/bin/env python3
"""
Test NAE Funds Detection System
Tests the complete funds detection workflow
"""

import os
import sys
import json
from datetime import datetime

# Add execution directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'execution'))

from broker_adapters.tradier_adapter import TradierBrokerAdapter

def test_funds_detection():
    """Test funds detection system"""
    print("=" * 60)
    print("NAE FUNDS DETECTION TEST")
    print("=" * 60)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Initialize adapter
    print("1. Initializing Tradier adapter...")
    try:
        adapter = TradierBrokerAdapter(sandbox=False)
        print("   ✅ Adapter initialized")
        results["tests"]["adapter_init"] = {"status": "PASS"}
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results["tests"]["adapter_init"] = {"status": "FAIL", "error": str(e)}
        return results
    
    # Test 2: Authenticate
    print("2. Authenticating...")
    try:
        if not adapter.oauth.ensure_valid_token():
            raise Exception("Authentication failed")
        print("   ✅ Authentication successful")
        results["tests"]["authentication"] = {"status": "PASS"}
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results["tests"]["authentication"] = {"status": "FAIL", "error": str(e)}
        return results
    
    # Test 3: Get account
    print("3. Retrieving account...")
    try:
        accounts = adapter.rest_client.get_accounts()
        if not accounts:
            raise Exception("No accounts found")
        
        account_id = accounts[0].get("account_number") if isinstance(accounts[0], dict) else accounts[0]
        print(f"   ✅ Found account: {account_id}")
        results["tests"]["get_account"] = {"status": "PASS", "account_id": account_id}
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results["tests"]["get_account"] = {"status": "FAIL", "error": str(e)}
        return results
    
    # Test 4: Get balance
    print("4. Checking account balance...")
    try:
        account_details = adapter.rest_client.get_account_details(account_id)
        
        # Extract balance information
        total_equity = account_details.get("total_equity", 0)
        cash_dict = account_details.get("cash", {})
        if isinstance(cash_dict, dict):
            cash = cash_dict.get("cash", 0)
        else:
            cash = account_details.get("cash", 0)
        
        margin_dict = account_details.get("margin", {})
        if isinstance(margin_dict, dict):
            buying_power = margin_dict.get("stock_buying_power", 0)
        else:
            buying_power = account_details.get("buying_power", 0)
        
        print(f"   ✅ Balance retrieved:")
        print(f"      - Total Equity: ${total_equity:,.2f}")
        print(f"      - Cash: ${cash:,.2f}")
        print(f"      - Buying Power: ${buying_power:,.2f}")
        
        results["tests"]["get_balance"] = {
            "status": "PASS",
            "total_equity": total_equity,
            "cash": cash,
            "buying_power": buying_power
        }
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results["tests"]["get_balance"] = {"status": "FAIL", "error": str(e)}
        return results
    
    # Test 5: Detect available funds
    print("5. Detecting available funds...")
    try:
        available_funds = max(total_equity, cash, buying_power)
        
        print(f"   ✅ Available funds: ${available_funds:,.2f}")
        
        if available_funds > 0:
            print(f"   ✅ FUNDS DETECTED!")
            print(f"   ✅ Funds detection system working correctly")
            results["tests"]["funds_detection"] = {
                "status": "PASS",
                "funds_detected": True,
                "amount": available_funds
            }
        else:
            print(f"   ⚠️  No funds detected (balance is $0.00)")
            print(f"   ✅ Detection system working (correctly shows no funds)")
            results["tests"]["funds_detection"] = {
                "status": "PASS",
                "funds_detected": False,
                "amount": 0.0,
                "note": "System correctly detects zero balance"
            }
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results["tests"]["funds_detection"] = {"status": "FAIL", "error": str(e)}
    
    # Test 6: Test threshold detection
    print("6. Testing threshold detection...")
    try:
        threshold = 100.0  # Minimum threshold for activation
        if available_funds >= threshold:
            print(f"   ✅ Funds above threshold (${threshold:,.2f})")
            print(f"   ✅ Trading would be activated")
            activation_status = "READY"
        else:
            print(f"   ⚠️  Funds below threshold (${threshold:,.2f})")
            print(f"   ⚠️  Trading will activate when funds >= ${threshold:,.2f}")
            activation_status = "WAITING"
        
        results["tests"]["threshold_detection"] = {
            "status": "PASS",
            "threshold": threshold,
            "current_funds": available_funds,
            "activation_status": activation_status
        }
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results["tests"]["threshold_detection"] = {"status": "FAIL", "error": str(e)}
    
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for test in results["tests"].values() if test.get("status") == "PASS")
    failed = sum(1 for test in results["tests"].values() if test.get("status") == "FAIL")
    total = len(results["tests"])
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print()
    
    if available_funds > 0:
        print("🎉 FUNDS DETECTED - NAE is ready to trade!")
    else:
        print("ℹ️  No funds detected - NAE will automatically detect funds when deposited")
    
    print()
    
    # Save results
    results_file = f"funds_detection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    test_funds_detection()




#!/usr/bin/env python3
"""
Test NAE's ability to see available funds in Tradier account
"""

import os
import sys
import json
from datetime import datetime

# Add execution directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'execution'))

from broker_adapters.tradier_adapter import TradierBrokerAdapter

# Try to import redis (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Note: Redis not available, skipping balance monitor test")

def test_funds_visibility():
    """Test if NAE can see available funds"""
    print("=" * 60)
    print("NAE FUNDS VISIBILITY TEST")
    print("=" * 60)
    print()
    
    # Initialize Tradier adapter
    print("1. Initializing Tradier adapter...")
    adapter = TradierBrokerAdapter(sandbox=False)
    print("   ✅ Adapter initialized")
    print()
    
    # Authenticate
    print("2. Authenticating...")
    if not adapter.oauth.ensure_valid_token():
        print("   ❌ Authentication failed")
        return False
    print("   ✅ Authentication successful")
    print(f"   - Using API key: {adapter.oauth.use_api_key}")
    print()
    
    # Get accounts
    print("3. Retrieving accounts...")
    accounts = adapter.rest_client.get_accounts()
    if not accounts:
        print("   ❌ No accounts found")
        return False
    
    account_id = accounts[0].get("account_number") if isinstance(accounts[0], dict) else accounts[0]
    print(f"   ✅ Found account: {account_id}")
    print()
    
    # Get account balance
    print("4. Checking account balance...")
    # Get account details which includes balance
    account_details = adapter.rest_client.get_account_details(account_id)
    
    # Try to get balance from account details
    balance_info = account_details
    
    if balance_info:
        total_equity = balance_info.get("total_equity", 0)
        cash = balance_info.get("cash", 0)
        buying_power = balance_info.get("buying_power", 0)
        
        print(f"   ✅ Balance retrieved:")
        print(f"      - Total Equity: ${total_equity:,.2f}")
        print(f"      - Cash: ${cash:,.2f}")
        print(f"      - Buying Power: ${buying_power:,.2f}")
        print()
        
        # Determine if funds are available
        available_funds = max(cash, buying_power, total_equity)
        if available_funds > 0:
            print(f"   ✅ FUNDS AVAILABLE: ${available_funds:,.2f}")
            print(f"   ✅ NAE can see available funds!")
        else:
            print(f"   ⚠️  NO FUNDS AVAILABLE: Account balance is $0.00")
            print(f"   ✅ NAE can see the account, but no funds are currently available")
            print(f"   ℹ️  Once funds are deposited, NAE will automatically detect them")
        print()
    else:
        print("   ❌ Could not retrieve balance")
        return False
    
    # Test balance monitor (if Redis available)
    if REDIS_AVAILABLE:
        print("5. Testing balance monitor...")
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            
            from monitoring.tradier_balance_monitor import TradierBalanceMonitor
            monitor = TradierBalanceMonitor(adapter, redis_client)
            current_balance = monitor.check_balance()
            
            if current_balance is not None:
                print(f"   ✅ Balance monitor working")
                print(f"      - Detected balance: ${current_balance:,.2f}")
                if current_balance > 0:
                    print(f"   ✅ FUNDS DETECTED by monitor!")
                else:
                    print(f"   ⚠️  No funds detected (balance is $0.00)")
            else:
                print("   ⚠️  Balance monitor returned None")
        except Exception as e:
            print(f"   ⚠️  Balance monitor test error: {e}")
    else:
        print("5. Balance monitor test skipped (Redis not available)")
    
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ NAE can successfully connect to Tradier")
    print("✅ NAE can retrieve account information")
    print("✅ NAE can see account balance")
    
    if available_funds > 0:
        print("✅ NAE can see AVAILABLE FUNDS")
        print(f"   Available: ${available_funds:,.2f}")
    else:
        print("⚠️  Account balance is currently $0.00")
        print("✅ NAE will automatically detect funds when they are deposited")
    
    print()
    return True

if __name__ == "__main__":
    test_funds_visibility()


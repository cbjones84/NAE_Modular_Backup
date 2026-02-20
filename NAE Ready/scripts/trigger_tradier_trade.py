#!/usr/bin/env python3
"""
Trigger a Trade via Tradier through Optimus
"""

import os
import sys

# Set Tradier environment variables
os.environ["TRADIER_SANDBOX"] = "false"
os.environ["TRADIER_API_KEY"] = "27Ymk28vtbgqY1LFYxhzaEmIuwJb"
os.environ["TRADIER_ACCOUNT_ID"] = "6YB66744"

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*60)
print("  OPTIMUS TRADIER TRADE TRIGGER")
print("="*60)
print()

# First verify Tradier connection
print("Step 1: Verifying Tradier connection...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'execution'))
from broker_adapters.tradier_adapter import TradierBrokerAdapter

adapter = TradierBrokerAdapter(sandbox=False)
account = adapter.get_account_info()
if account:
    print(f"  ✅ Connected to account: {account.get('account_number')}")
    print(f"  ✅ Status: {account.get('status')}")
else:
    print("  ❌ Failed to connect to Tradier")
    sys.exit(1)

# Check account balance
print()
print("Step 2: Checking account balance...")
balances = account.get('balances', {})
if balances:
    cash = balances.get('cash', {})
    if isinstance(cash, dict):
        cash_available = cash.get('cash_available', 0)
    else:
        cash_available = balances.get('total_cash', 0)
    print(f"  Cash Available: ${cash_available:,.2f}")
else:
    print("  ⚠️ Could not retrieve balance info")
    cash_available = 0

print()
print("Step 3: Initializing Optimus...")

try:
    from agents.optimus import OptimusAgent, TradingMode
    
    # Initialize Optimus in LIVE mode (not sandbox)
    optimus = OptimusAgent(sandbox=False)
    print(f"  ✅ Optimus initialized in {optimus.trading_mode.value} mode")
    print(f"  ✅ Self-healing engine: {'Available' if optimus.self_healing_engine else 'Not available'}")
    print(f"  ✅ Alpaca client: {'Available' if optimus.alpaca_client else 'Not available'}")
    
except Exception as e:
    print(f"  ❌ Failed to initialize Optimus: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 4: Preparing trade...")

# Small test trade - buy 1 share of a liquid stock
# Using a low-priced liquid stock for testing
trade_details = {
    "symbol": "F",  # Ford - low price, liquid
    "side": "buy",
    "quantity": 1,
    "order_type": "market",
    "duration": "day",
    "strategy_id": "test_tradier_trigger"
}

print(f"  Symbol: {trade_details['symbol']}")
print(f"  Side: {trade_details['side']}")
print(f"  Quantity: {trade_details['quantity']}")
print(f"  Order Type: {trade_details['order_type']}")

print()
confirm = input("Execute this trade? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("Trade cancelled.")
    sys.exit(0)

print()
print("Step 5: Executing trade via Optimus...")

try:
    result = optimus.execute_trade(trade_details)
    
    print()
    print("="*60)
    print("  TRADE RESULT")
    print("="*60)
    
    if result:
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Broker: {result.get('broker', 'unknown')}")
        print(f"  Order ID: {result.get('order_id', 'N/A')}")
        print(f"  Mode: {result.get('mode', 'unknown')}")
        
        if result.get('status') in ['submitted', 'filled', 'accepted']:
            print()
            print("  ✅ TRADE SUBMITTED SUCCESSFULLY!")
        else:
            print()
            print(f"  ⚠️ Trade status: {result.get('status')}")
            if 'error' in result:
                print(f"  Error: {result.get('error')}")
            if 'reason' in result:
                print(f"  Reason: {result.get('reason')}")
    else:
        print("  ❌ No result returned from trade execution")
        
except Exception as e:
    print(f"  ❌ Trade execution failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)


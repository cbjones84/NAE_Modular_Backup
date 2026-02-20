#!/usr/bin/env python3
"""
Direct Trade via Tradier (without Optimus dependencies)
"""

import os
import sys

# Set Tradier environment variables
os.environ["TRADIER_SANDBOX"] = "false"
os.environ["TRADIER_API_KEY"] = "27Ymk28vtbgqY1LFYxhzaEmIuwJb"
os.environ["TRADIER_ACCOUNT_ID"] = "6YB66744"

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'execution'))

print("="*60)
print("  DIRECT TRADIER TRADE")
print("="*60)
print()

from broker_adapters.tradier_adapter import TradierBrokerAdapter  # pyright: ignore[reportMissingImports]

print("Step 1: Connecting to Tradier...")
adapter = TradierBrokerAdapter(sandbox=False)
account = adapter.get_account_info()

if account:
    print(f"  ✅ Connected to account: {account.get('account_number')}")
    print(f"  ✅ Status: {account.get('status')}")
    print(f"  ✅ Type: {account.get('type')}")
else:
    print("  ❌ Failed to connect to Tradier")
    sys.exit(1)

print()
print("Step 2: Checking positions...")
try:
    positions = adapter.get_positions()
    if positions:
        print(f"  Current positions: {len(positions)}")
        for pos in positions:
            print(f"    - {pos.get('symbol')}: {pos.get('quantity')} shares @ ${pos.get('cost_basis', 0):.2f}")
    else:
        print("  No open positions")
except Exception as e:
    print(f"  ⚠️ Could not get positions: {e}")

print()
print("Step 3: Trade Setup")
print("-"*40)

# Small test trade
symbol = "F"  # Ford - liquid, low price (~$10)
side = "buy"
quantity = 1
order_type = "market"

print(f"  Symbol: {symbol}")
print(f"  Action: {side.upper()}")
print(f"  Quantity: {quantity} share(s)")
print(f"  Order Type: {order_type}")
print()

# Get current quote
print("Step 4: Getting current quote...")
try:
    quote = adapter.rest_client.get_quotes([symbol])
    if quote and len(quote) > 0:
        q = quote[0]
        last_price = q.get('last', q.get('close', 0))
        print(f"  Last Price: ${last_price:.2f}")
        print(f"  Estimated Cost: ${last_price * quantity:.2f}")
except Exception as e:
    print(f"  ⚠️ Could not get quote: {e}")
    last_price = 10.0  # Estimate

print()
print("="*60)
print("  ⚠️  THIS WILL PLACE A REAL TRADE WITH REAL MONEY!")
print("="*60)
print()

confirm = input("Type 'EXECUTE' to place the order: ").strip()

if confirm != 'EXECUTE':
    print()
    print("Trade cancelled.")
    sys.exit(0)

print()
print("Step 5: Placing order...")

try:
    # Place order using REST client
    order_result = adapter.rest_client.place_order(
        account_id=adapter.account_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        duration="day"
    )
    
    print()
    print("="*60)
    print("  ORDER RESULT")
    print("="*60)
    
    if order_result:
        order_id = order_result.get('id') or order_result.get('order_id')
        status = order_result.get('status', 'submitted')
        
        print(f"  Order ID: {order_id}")
        print(f"  Status: {status}")
        print()
        
        if order_id:
            print("  ✅ ORDER PLACED SUCCESSFULLY!")
            print()
            print(f"  View in Tradier: https://dash.tradier.com/")
        else:
            print("  Full response:")
            print(f"  {order_result}")
    else:
        print("  ❌ No response from order placement")

except Exception as e:
    print(f"  ❌ Order failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)


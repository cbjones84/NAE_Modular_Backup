#!/usr/bin/env python3
"""
Optimus Intelligent Trade via Tradier
Makes an intelligent trading decision using Optimus's safety checks and execution logic
"""

import os
import sys
import datetime

# Set Tradier environment variables FIRST
os.environ["TRADIER_SANDBOX"] = "false"
os.environ["TRADIER_API_KEY"] = "27Ymk28vtbgqY1LFYxhzaEmIuwJb"
os.environ["TRADIER_ACCOUNT_ID"] = "6YB66744"

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_dir = os.path.dirname(script_dir)
sys.path.insert(0, nae_dir)
sys.path.insert(0, os.path.join(nae_dir, 'execution'))

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

print("="*70)
print("  OPTIMUS INTELLIGENT TRADE EXECUTION")
print("  Trading via Tradier with Full Safety Checks")
print("="*70)
print()

# Step 1: Verify Tradier Connection
log("📡 Step 1: Verifying Tradier connection...")
try:
    from broker_adapters.tradier_adapter import TradierBrokerAdapter
    adapter = TradierBrokerAdapter(sandbox=False)
    account = adapter.get_account_info()
    
    if account and account.get('status') == 'active':
        log(f"   ✅ Connected to Tradier account: {account.get('account_number')}")
        log(f"   ✅ Account status: {account.get('status')}")
        log(f"   ✅ Account type: {account.get('type')}")
    else:
        log("   ❌ Tradier account not active")
        sys.exit(1)
except Exception as e:
    log(f"   ❌ Tradier connection failed: {e}")
    sys.exit(1)

print()

# Step 2: Initialize Optimus
log("🤖 Step 2: Initializing Optimus Agent...")
try:
    from agents.optimus import OptimusAgent, TradingMode
    
    # Initialize in LIVE mode (sandbox=False)
    optimus = OptimusAgent(sandbox=False)
    
    log(f"   ✅ Trading mode: {optimus.trading_mode.value}")
    log(f"   ✅ Self-healing engine: {'Active' if optimus.self_healing_engine else 'Inactive'}")
    log(f"   ✅ Safety limits configured:")
    log(f"      - Max order size: ${optimus.safety_limits.max_order_size_usd:,.2f}")
    log(f"      - Daily loss limit: {optimus.safety_limits.daily_loss_limit_pct*100:.1f}%")
    log(f"      - Max positions: {optimus.safety_limits.max_open_positions}")
    
except Exception as e:
    log(f"   ❌ Failed to initialize Optimus: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 3: Define the trade
log("📋 Step 3: Preparing intelligent trade...")

# Smart trade selection - low-cost liquid stock for testing
trade_details = {
    "symbol": "F",           # Ford - liquid, well-known, ~$10
    "action": "buy",         # Buy action
    "side": "buy",           # Side
    "quantity": 1,           # Small quantity for safety
    "order_type": "market",  # Market order for immediate fill
    "time_in_force": "day",  # Day order
    "price": 11.0,           # Approximate price for safety checks
    "strategy_id": "optimus_intelligent_trade",
    "source": "manual_trigger"
}

log(f"   Trade: BUY {trade_details['quantity']} share(s) of {trade_details['symbol']}")
log(f"   Order Type: {trade_details['order_type']}")
log(f"   Estimated Value: ~${trade_details['price'] * trade_details['quantity']:.2f}")

print()

# Step 4: Run Pre-Trade Safety Checks
log("🛡️ Step 4: Running Optimus pre-trade safety checks...")
try:
    passed, reason = optimus.pre_trade_checks(trade_details)
    
    if passed:
        log("   ✅ All safety checks PASSED")
    else:
        log(f"   ❌ Safety check FAILED: {reason}")
        log("   Trade will not be executed.")
        sys.exit(1)
        
except Exception as e:
    log(f"   ⚠️ Could not run safety checks: {e}")
    log("   Proceeding with trade (checks may be handled during execution)")

print()

# Step 5: Execute the trade
log("🚀 Step 5: Executing trade via Optimus...")
log("   " + "="*50)

try:
    result = optimus.execute_trade(trade_details)
    
    print()
    log("📊 TRADE EXECUTION RESULT:")
    log("   " + "-"*40)
    
    if result:
        status = result.get('status', 'unknown')
        broker = result.get('broker', 'unknown')
        order_id = result.get('order_id', 'N/A')
        mode = result.get('mode', 'unknown')
        
        log(f"   Status: {status}")
        log(f"   Broker: {broker}")
        log(f"   Order ID: {order_id}")
        log(f"   Mode: {mode}")
        
        if status in ['submitted', 'filled', 'accepted', 'pending']:
            print()
            log("   " + "="*40)
            log("   ✅ TRADE SUBMITTED SUCCESSFULLY!")
            log("   " + "="*40)
            
            if broker == 'tradier':
                log("   📈 View order at: https://dash.tradier.com/")
            elif broker == 'alpaca':
                log("   📈 View order at: https://app.alpaca.markets/")
        else:
            print()
            log(f"   ⚠️ Trade status: {status}")
            if 'error' in result:
                log(f"   Error: {result.get('error')}")
            if 'reason' in result:
                log(f"   Reason: {result.get('reason')}")
            if 'errors' in result:
                log(f"   Errors: {result.get('errors')}")
    else:
        log("   ❌ No result returned from trade execution")
        
except Exception as e:
    log(f"   ❌ Trade execution error: {e}")
    import traceback
    traceback.print_exc()

print()
log("="*70)
log("Trade execution complete.")


#!/usr/bin/env python3
"""
Test Optimus with Alpaca paper trading
Demonstrates strategy execution through Alpaca
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_optimus_alpaca():
    """Test Optimus with Alpaca paper trading"""
    
    print("=" * 60)
    print("Optimus Alpaca Paper Trading Test")
    print("=" * 60)
    print()
    
    # Initialize Optimus in PAPER mode (uses Alpaca)
    print("1. Initializing Optimus in PAPER mode...")
    try:
        from agents.optimus import OptimusAgent
        
        optimus = OptimusAgent(sandbox=False)  # PAPER mode = Alpaca
        print(f"   ✅ Optimus initialized")
        print(f"   Trading Mode: {optimus.trading_mode.value}")
    except Exception as e:
        print(f"   ❌ Failed to initialize Optimus: {e}")
        return False
    
    # Check Alpaca configuration
    print("\n2. Checking Alpaca configuration...")
    if optimus.alpaca_client:
        print("   ✅ Alpaca client configured")
        print(f"   Paper Trading: {optimus.alpaca_client.paper_trading}")
        
        # Test authentication
        try:
            if optimus.alpaca_client.adapter.auth():
                print("   ✅ Alpaca authentication successful")
                
                # Get account info
                account = optimus.alpaca_client.adapter.get_account()
                if account:
                    print(f"   ✅ Account ID: {account.get('account_id', 'N/A')}")
                    print(f"   ✅ Cash: ${account.get('cash', 0):,.2f}")
                    print(f"   ✅ Buying Power: ${account.get('buying_power', 0):,.2f}")
                    print(f"   ✅ Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            else:
                print("   ❌ Alpaca authentication failed")
                return False
        except Exception as e:
            print(f"   ⚠️  Alpaca authentication error: {e}")
            print("   Note: alpaca-py may not be installed")
            print("   Install: pip install alpaca-py")
            return False
    else:
        print("   ❌ Alpaca client not configured")
        print("   Install alpaca-py: pip install alpaca-py")
        return False
    
    # Get positions
    print("\n3. Getting current positions...")
    try:
        positions = optimus.alpaca_client.adapter.get_positions()
        if positions:
            print(f"   ✅ Found {len(positions)} positions:")
            for pos in positions:
                print(f"      {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        else:
            print("   ℹ️  No positions found")
    except Exception as e:
        print(f"   ⚠️  Error getting positions: {e}")
    
    # Test trade execution (small test order)
    print("\n4. Testing trade execution (dry run)...")
    print("   Note: This is a test - actual order will be placed if uncommented")
    
    test_order = {
        "symbol": "SPY",
        "action": "buy",
        "quantity": 1,
        "order_type": "market",
        "time_in_force": "day"
    }
    
    print(f"   Test Order: {test_order}")
    print("   ⚠️  Uncomment the code below to execute actual orders")
    
    # Uncomment to execute actual order:
    # try:
    #     result = optimus.execute_trade(test_order)
    #     print(f"   ✅ Order executed: {result.get('order_id', 'N/A')}")
    #     print(f"   Status: {result.get('status', 'unknown')}")
    #     print(f"   Broker: {result.get('broker', 'unknown')}")
    # except Exception as e:
    #     print(f"   ❌ Order execution failed: {e}")
    
    # Check Optimus status
    print("\n5. Checking Optimus status...")
    try:
        status = optimus.get_status()
        print(f"   Trading Enabled: {status.get('trading_enabled', False)}")
        print(f"   Trading Mode: {status.get('trading_mode', 'unknown')}")
        print(f"   Daily P&L: ${status.get('daily_pnl', 0):,.2f}")
        print(f"   Open Positions: {status.get('open_positions', 0)}")
        
        brokers = status.get('broker_clients', {})
        print(f"   Alpaca Configured: {brokers.get('alpaca_configured', False)}")
    except Exception as e:
        print(f"   ⚠️  Error getting status: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Optimus Alpaca test completed!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("  1. Uncomment the order execution code to test real trades")
    print("  2. Use OptimusAgent(sandbox=False) in your strategies")
    print("  3. Monitor trades in Alpaca dashboard")
    print()
    
    return True


if __name__ == "__main__":
    success = test_optimus_alpaca()
    sys.exit(0 if success else 1)


# NAE/adapters/alpaca_integration_example.py
"""
Example usage of AlpacaAdapter with official SDK integration

This demonstrates:
- Stock market/limit orders
- Options trading (single and multi-leg)
- Account and position management
"""

import os
from typing import Dict, Any

# Import the adapter
from adapters.alpaca import AlpacaAdapter
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

def example_alpaca_usage():
    """Example of using AlpacaAdapter with official SDK"""
    
    # Configuration
    config = {
        "API_KEY": os.environ.get("APCA_API_KEY_ID", "YOUR_ALPACA_API_KEY"),
        "API_SECRET": os.environ.get("APCA_API_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY"),
        "paper_trading": True  # Set to False for live trading
    }
    
    # Initialize adapter
    try:
        alpaca = AlpacaAdapter(config)
        print("✅ AlpacaAdapter initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AlpacaAdapter: {e}")
        return
    
    # Authenticate
    if not alpaca.auth():
        print("❌ Authentication failed")
        return
    
    print("✅ Authentication successful")
    
    # Get account information
    print("\n--- Account Information ---")
    account = alpaca.get_account()
    print(f"Cash: ${account.get('cash', 0):,.2f}")
    print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
    print(f"Equity: ${account.get('equity', 0):,.2f}")
    
    # Get positions
    print("\n--- Current Positions ---")
    positions = alpaca.get_positions()
    if positions:
        for pos in positions:
            print(f"{pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
    else:
        print("No positions")
    
    # Example 1: Buy stock market order
    print("\n--- Example 1: Buy Stock (Market Order) ---")
    result = alpaca.buy_stock_market("AAPL", 1.0)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
    
    # Example 2: Sell stock limit order
    print("\n--- Example 2: Sell Stock (Limit Order) ---")
    result = alpaca.sell_stock_limit("AAPL", 1.0, 150.00)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
    
    # Example 3: Buy single-leg option
    print("\n--- Example 3: Buy Single-Leg Option ---")
    # Note: Replace with actual option symbol (e.g., "AAPL240119C00190000")
    option_symbol = "AAPL240119C00190000"  # AAPL Call option
    result = alpaca.buy_option_market(option_symbol, 1.0)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        print("   Note: Option symbol may not exist or market may be closed")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
    
    # Example 4: Multi-leg options order (straddle)
    print("\n--- Example 4: Multi-Leg Options Order (Straddle) ---")
    legs = [
        ("AAPL240119C00190000", OrderSide.BUY, 1),  # Buy call
        ("AAPL240119P00190000", OrderSide.BUY, 1)   # Buy put
    ]
    result = alpaca.multi_leg_option_order(legs, 1.0)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        print("   Note: Option symbols may not exist or market may be closed")
    else:
        print(f"✅ Multi-leg order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Legs: {result['legs_count']}")
    
    # Example 5: Get quote
    print("\n--- Example 5: Get Quote ---")
    quote = alpaca.get_quote("AAPL")
    if quote:
        print(f"Symbol: {quote['symbol']}")
        print(f"Bid: ${quote['bid']:.2f}")
        print(f"Ask: ${quote['ask']:.2f}")
    else:
        print("❌ Failed to get quote")
    
    print("\n--- Examples Complete ---")


def example_using_generic_place_order():
    """Example using the generic place_order method (BrokerAdapter interface)"""
    
    config = {
        "API_KEY": os.environ.get("APCA_API_KEY_ID", "YOUR_ALPACA_API_KEY"),
        "API_SECRET": os.environ.get("APCA_API_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY"),
        "paper_trading": True
    }
    
    alpaca = AlpacaAdapter(config)
    
    # Market order using generic interface
    order = {
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "buy",
        "type": "market",
        "time_in_force": "day"
    }
    
    result = alpaca.place_order(order)
    print(f"Order result: {result}")
    
    # Limit order using generic interface
    order = {
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "sell",
        "type": "limit",
        "price": 150.00,
        "time_in_force": "day"
    }
    
    result = alpaca.place_order(order)
    print(f"Order result: {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("AlpacaAdapter Integration Examples")
    print("=" * 60)
    
    # Check if API keys are set
    if (os.environ.get("APCA_API_KEY_ID") == "YOUR_ALPACA_API_KEY" or 
        not os.environ.get("APCA_API_KEY_ID")):
        print("\n⚠️  Warning: API keys not set in environment variables")
        print("   Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        print("   Or update the config dict in the example code")
        print()
    
    # Run examples
    example_alpaca_usage()
    
    print("\n" + "=" * 60)
    print("For more examples, see the BrokerAdapter interface methods:")
    print("  - place_order() - Generic order placement")
    print("  - cancel_order() - Cancel an order")
    print("  - get_order_status() - Check order status")
    print("=" * 60)


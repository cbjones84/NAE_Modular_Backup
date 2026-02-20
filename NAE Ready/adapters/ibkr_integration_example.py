# NAE/adapters/ibkr_integration_example.py
"""
Example usage of IBKRAdapter with TWS API integration

Prerequisites:
- TWS or IB Gateway must be running
- API access must be enabled in TWS settings
- Default ports: 7497 (paper trading), 7496 (live trading)

This demonstrates:
- Stock market/limit orders
- Options trading (single-leg)
- Account and position management
- Market data retrieval
"""

import time
from adapters.ibkr import IBKRAdapter


def example_ibkr_usage():
    """Example of using IBKRAdapter with TWS API"""
    
    # Configuration
    config = {
        "host": "127.0.0.1",
        "port": 7497,  # 7497 for paper trading, 7496 for live trading
        "client_id": 1,  # Must be unique (1-100)
        "paper_trading": True
    }
    
    print("=" * 60)
    print("IBKR TWS API Integration Example")
    print("=" * 60)
    print("\n⚠️  Prerequisites:")
    print("   1. TWS or IB Gateway must be running")
    print("   2. API access must be enabled in TWS settings")
    print("   3. Port 7497 (paper) or 7496 (live) must be open")
    print("   4. This IP must be in TWS trusted IPs")
    print()
    
    # Initialize adapter
    try:
        print("Connecting to TWS/Gateway...")
        ibkr = IBKRAdapter(config)
        print("✅ Connected to TWS/Gateway")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  - Is TWS/Gateway running?")
        print("  - Is API access enabled in TWS settings?")
        print("  - Is the port correct? (7497 for paper, 7496 for live)")
        print("  - Is this IP in trusted IPs?")
        return
    
    # Authenticate
    if not ibkr.auth():
        print("❌ Authentication failed")
        return
    
    print("✅ Authentication successful\n")
    
    # Get account information
    print("--- Account Information ---")
    account = ibkr.get_account()
    if account:
        print(f"Account ID: {account.get('account_id', 'N/A')}")
        print(f"Cash: ${account.get('cash', 0):,.2f}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"Equity: ${account.get('equity', 0):,.2f}")
        print(f"Currency: {account.get('currency', 'USD')}")
    else:
        print("❌ Failed to get account information")
    print()
    
    # Get positions
    print("--- Current Positions ---")
    positions = ibkr.get_positions()
    if positions:
        for pos in positions:
            print(f"{pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
            print(f"  Market Value: ${pos['market_value']:,.2f}")
            print(f"  Unrealized P&L: ${pos['unrealized_pl']:,.2f}")
    else:
        print("No positions")
    print()
    
    # Example 1: Buy stock market order
    print("--- Example 1: Buy Stock (Market Order) ---")
    result = ibkr.buy_stock_market("AAPL", 1.0)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Filled: {result.get('filled', 0)}")
    print()
    
    # Example 2: Sell stock limit order
    print("--- Example 2: Sell Stock (Limit Order) ---")
    result = ibkr.sell_stock_limit("AAPL", 1.0, 150.00)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Limit Price: $150.00")
    print()
    
    # Example 3: Buy single-leg option
    print("--- Example 3: Buy Single-Leg Option ---")
    # Note: Replace with actual option contract details
    # Format: YYYYMMDD for expiration date
    expiration_date = "20241220"  # Example: Dec 20, 2024
    strike_price = 150.0
    option_type = "C"  # 'C' for call, 'P' for put
    
    result = ibkr.buy_option_market(
        symbol="AAPL",
        lastTradeDate=expiration_date,
        strike=strike_price,
        right=option_type,
        qty=1.0
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        print("   Note: Option contract may not exist or market may be closed")
        print("   Verify expiration date format: YYYYMMDD")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Contract: AAPL {expiration_date} {option_type} {strike_price}")
    print()
    
    # Example 4: Get quote
    print("--- Example 4: Get Quote ---")
    quote = ibkr.get_quote("AAPL")
    if quote:
        print(f"Symbol: {quote['symbol']}")
        print(f"Bid: ${quote['bid']:.2f}")
        print(f"Ask: ${quote['ask']:.2f}")
        print(f"Last: ${quote['last']:.2f}")
    else:
        print("❌ Failed to get quote")
    print()
    
    # Example 5: Using generic place_order interface
    print("--- Example 5: Generic Order Interface ---")
    order = {
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "buy",
        "type": "market",
        "secType": "STK"
    }
    
    result = ibkr.place_order(order)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Order placed: {result['order_id']}")
        print(f"   Status: {result['status']}")
    print()
    
    print("=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nNote: Orders may take a few seconds to appear in TWS")
    print("      Check TWS order log for detailed status")


def example_connection_test():
    """Simple connection test"""
    
    config = {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1,
        "paper_trading": True
    }
    
    print("Testing IBKR connection...")
    
    try:
        ibkr = IBKRAdapter(config)
        
        if ibkr.auth():
            print("✅ Connection successful!")
            
            # Get account
            account = ibkr.get_account()
            if account:
                print(f"   Account: {account.get('account_id', 'N/A')}")
                print(f"   Cash: ${account.get('cash', 0):,.2f}")
            
            return True
        else:
            print("❌ Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IBKRAdapter Integration Examples")
    print("=" * 60)
    print()
    
    # Run connection test first
    print("Step 1: Connection Test")
    print("-" * 60)
    if not example_connection_test():
        print("\n⚠️  Connection test failed. Please check:")
        print("   1. TWS/Gateway is running")
        print("   2. API access is enabled")
        print("   3. Port is correct (7497 for paper, 7496 for live)")
        print("   4. IP is in trusted IPs")
        exit(1)
    
    print("\n" + "=" * 60)
    print("Step 2: Full Examples")
    print("-" * 60)
    print()
    
    # Run full examples
    example_ibkr_usage()
    
    print("\n" + "=" * 60)
    print("For more examples, see:")
    print("  - adapters/ibkr.py - Full adapter implementation")
    print("  - docs/IBKR_INTEGRATION.md - Complete documentation")
    print("=" * 60)


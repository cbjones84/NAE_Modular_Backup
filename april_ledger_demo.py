# april_ledger_demo.py
"""
April + Ledger Live Demo (Non-Interactive)

This demonstrates how April connects to Ledger Live without requiring user input.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.april import April

def demo_setup():
    print("🔐 April + Ledger Live Demo")
    print("=" * 50)
    
    # Initialize April agent
    april = April()
    
    print("\n📋 Setup Instructions:")
    print(april.get_setup_instructions())
    
    print("\n🚀 Demo Setup with Sample Addresses:")
    
    # Demo wallet addresses (these are example addresses, not real ones)
    wallet_addresses = {
        "bitcoin": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block (example)
        "ethereum": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",  # Example
        "litecoin": "LTC1qT3Q9vDRwH6fDt9R4fpV5K2JNGXhF7H"  # Example
    }
    
    print("\n📝 Sample wallet addresses:")
    for coin, address in wallet_addresses.items():
        print(f"  {coin.upper()}: {address}")
    
    # Connect to Ledger Live
    print("\n🔗 Connecting to Ledger Live...")
    success = april.connect_to_ledger_live(wallet_addresses)
    if success:
        print("✅ Successfully connected to Ledger Live!")
    else:
        print("❌ Failed to connect to Ledger Live")
    
    # Demo exchange API setup (with placeholder credentials)
    print("\n🔑 Exchange API Setup Demo:")
    exchanges = ["binance", "coinbase"]
    for exchange in exchanges:
        print(f"Setting up {exchange.upper()} API...")
        # Using placeholder credentials for demo
        april.setup_exchange_api(exchange, f"demo_{exchange}_key", f"demo_{exchange}_secret")
    
    # Test the setup
    print("\n🧪 Testing Setup:")
    april.run()
    
    # Show portfolio summary
    print("\n📊 Portfolio Summary:")
    portfolio = april.get_portfolio_summary()
    print(f"Last updated: {portfolio.get('last_updated', 'N/A')}")
    
    if 'portfolio' in portfolio:
        for coin, data in portfolio['portfolio'].items():
            print(f"{coin.upper()}: {data['balance']} (Address: {data['wallet_address'][:10]}...)")
    
    # Test Bitcoin migration
    print("\n💰 Testing Bitcoin Migration:")
    result = april.migrate_bitcoin_strategy(100.0)
    if result['success']:
        print(f"✅ Migration successful!")
        print(f"   Amount: ${result['allocation_amount']}")
        print(f"   Bitcoin: {result['bitcoin_amount']:.8f} BTC")
        print(f"   Price: ${result['bitcoin_price']:,.2f}")
        print(f"   Wallet: {result['wallet_address'][:10]}...")
    else:
        print(f"❌ Migration failed: {result.get('error', 'Unknown error')}")
    
    print("\n✅ Demo complete! April is ready for Bitcoin operations.")
    print("\n📝 To use with your real Ledger Live:")
    print("1. Replace sample addresses with your actual wallet addresses")
    print("2. Add your real exchange API credentials")
    print("3. Test with small amounts first")
    print("4. Monitor your Ledger Live app for transactions")

if __name__ == "__main__":
    demo_setup()



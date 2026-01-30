# april_ledger_setup.py
"""
April + Ledger Live Setup Script

This script helps you connect April agent to your Ledger Live wallet.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.april import April

def main():
    print("🔐 April + Ledger Live Setup")
    print("=" * 50)
    
    # Initialize April agent
    april = April()
    
    print("\n📋 Setup Instructions:")
    print(april.get_setup_instructions())
    
    print("\n🚀 Quick Setup:")
    
    # Get wallet addresses from user
    print("\nEnter your Ledger Live wallet addresses:")
    wallet_addresses = {}
    
    coins = ["bitcoin", "ethereum", "litecoin", "bitcoin_cash"]
    for coin in coins:
        address = input(f"{coin.upper()} address (or press Enter to skip): ").strip()
        if address:
            wallet_addresses[coin] = address
    
    if wallet_addresses:
        # Connect to Ledger Live
        success = april.connect_to_ledger_live(wallet_addresses)
        if success:
            print("✅ Successfully connected to Ledger Live!")
        else:
            print("❌ Failed to connect to Ledger Live")
    else:
        print("⚠️ No wallet addresses provided")
    
    # Optional: Setup exchange APIs
    print("\n🔑 Exchange API Setup (Optional):")
    setup_exchanges = input("Do you want to setup exchange APIs? (y/n): ").lower().strip()
    
    if setup_exchanges == 'y':
        exchanges = ["binance", "coinbase", "kraken"]
        for exchange in exchanges:
            setup = input(f"Setup {exchange.upper()} API? (y/n): ").lower().strip()
            if setup == 'y':
                api_key = input(f"Enter {exchange.upper()} API key: ").strip()
                secret = input(f"Enter {exchange.upper()} secret: ").strip()
                if api_key and secret:
                    april.setup_exchange_api(exchange, api_key, secret)
    
    # Test the setup
    print("\n🧪 Testing Setup:")
    april.run()
    
    # Show portfolio summary
    portfolio = april.get_portfolio_summary()
    print(f"\n📊 Portfolio Summary:")
    print(f"Last updated: {portfolio.get('last_updated', 'N/A')}")
    
    if 'portfolio' in portfolio:
        for coin, data in portfolio['portfolio'].items():
            print(f"{coin.upper()}: {data['balance']} (Address: {data['wallet_address'][:10]}...)")
    
    print("\n✅ Setup complete! April is ready for Bitcoin operations.")
    print("\nNext steps:")
    print("1. Test with small amounts first")
    print("2. Monitor your Ledger Live app")
    print("3. Use april.migrate_bitcoin_strategy(amount) to convert profits")

if __name__ == "__main__":
    main()



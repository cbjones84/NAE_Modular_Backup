#!/usr/bin/env python3
"""
Verify Alpaca API Keys
Tests both paper and live trading connections
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from adapters.alpaca import AlpacaAdapter

def test_keys():
    """Test Alpaca keys"""
    # Load config
    with open('config/api_keys.json', 'r') as f:
        api_keys = json.load(f)
    
    alpaca_config = api_keys.get('alpaca', {})
    api_key = alpaca_config.get('api_key')
    api_secret = alpaca_config.get('api_secret')
    
    print("="*70)
    print("Alpaca API Key Verification")
    print("="*70)
    print(f"\nAPI Key: {api_key[:20]}...{api_key[-5:]}")
    print(f"Secret: {'*' * 20}...{api_secret[-5:]}")
    print(f"\nKey Type: {'Paper (PK)' if api_key.startswith('PK') else 'Live (AK)'}")
    
    # Test LIVE trading first
    print("\n" + "="*70)
    print("Testing LIVE Trading Connection")
    print("="*70)
    
    try:
        config_live = {
            'API_KEY': api_key,
            'API_SECRET': api_secret,
            'paper_trading': False  # LIVE
        }
        
        adapter_live = AlpacaAdapter(config_live)
        print("✅ Adapter created for LIVE trading")
        
        if adapter_live.auth():
            print("✅ LIVE trading authentication SUCCESSFUL")
            account = adapter_live.get_account()
            print(f"\nAccount Info:")
            print(f"  Cash: ${account.get('cash', 0):,.2f}")
            print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
            print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"  Trading Blocked: {account.get('trading_blocked', False)}")
            return True
        else:
            print("❌ LIVE trading authentication FAILED")
            print("\nPossible reasons:")
            print("  1. Keys are incorrect")
            print("  2. Account not activated for API access")
            print("  3. Keys are for paper trading, not live")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LIVE connection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    success = test_keys()
    sys.exit(0 if success else 1)


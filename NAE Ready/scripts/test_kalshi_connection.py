#!/usr/bin/env python3
"""
Kalshi Connection Test Script
Tests the connection to Kalshi's CFTC-regulated prediction market API
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_credentials():
    """Load Kalshi credentials from config files"""
    credentials = {
        "api_key_id": None,
        "private_key": None
    }
    
    # Try api_keys.json first
    api_keys_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "api_keys.json")
    if os.path.exists(api_keys_path):
        try:
            with open(api_keys_path, 'r') as f:
                api_keys = json.load(f)
                kalshi_config = api_keys.get("kalshi", {})
                credentials["api_key_id"] = kalshi_config.get("api_key_id")
                credentials["private_key"] = kalshi_config.get("private_key")
                print(f"✅ Loaded credentials from api_keys.json")
        except Exception as e:
            print(f"⚠️ Error loading api_keys.json: {e}")
    
    # Try kalshi_config.json as backup for api_key_id
    kalshi_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "kalshi_config.json")
    if os.path.exists(kalshi_config_path) and not credentials["api_key_id"]:
        try:
            with open(kalshi_config_path, 'r') as f:
                config = json.load(f)
                credentials["api_key_id"] = config.get("api_key_id")
                print(f"✅ Loaded API key ID from kalshi_config.json")
        except Exception as e:
            print(f"⚠️ Error loading kalshi_config.json: {e}")
    
    # Try secure vault
    try:
        from secure_vault import get_vault
        vault = get_vault()
        if not credentials["api_key_id"]:
            credentials["api_key_id"] = vault.get_secret("kalshi", "api_key_id")
        if not credentials["private_key"]:
            credentials["private_key"] = vault.get_secret("kalshi", "private_key")
        if credentials["api_key_id"] or credentials["private_key"]:
            print(f"✅ Loaded credentials from secure vault")
    except Exception as e:
        print(f"ℹ️ Vault not available: {e}")
    
    return credentials


def test_kalshi_connection():
    """Test connection to Kalshi API"""
    print("=" * 60)
    print("🔐 KALSHI CONNECTION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load credentials
    print("📋 Loading credentials...")
    credentials = load_credentials()
    
    api_key_id = credentials.get("api_key_id")
    private_key = credentials.get("private_key")
    
    if not api_key_id:
        print("❌ API Key ID not found!")
        return False
    
    if not private_key:
        print("❌ Private key not found!")
        return False
    
    print(f"✅ API Key ID: {api_key_id[:12]}...{api_key_id[-4:]}")
    print(f"✅ Private Key: {'*' * 20} (loaded, {len(private_key)} chars)")
    print()
    
    # Try to connect using the adapter
    print("🔌 Testing Kalshi adapter connection...")
    try:
        from adapters.kalshi import KalshiAdapter, get_kalshi_adapter
        
        adapter = get_kalshi_adapter(demo=False)
        
        if adapter:
            print(f"✅ Kalshi adapter initialized")
            print(f"   API Key ID: {adapter.api_key_id[:12] if adapter.api_key_id else 'None'}...")
            
            # Get base URL from internal client
            if adapter.api_client:
                print(f"   Base URL: {adapter.api_client.base_url}")
                print(f"   Mode: {'DEMO' if adapter.demo else 'PRODUCTION'}")
            
            # Test authentication
            print()
            print("🔑 Testing authentication...")
            
            if hasattr(adapter, 'authenticate') and callable(adapter.authenticate):
                auth_result = adapter.authenticate()
                if auth_result:
                    print("✅ Authentication successful!")
                else:
                    print("⚠️ Authentication returned False")
            
            # Try to get balance
            print()
            print("💰 Testing API access (get balance)...")
            try:
                balance = adapter.get_balance()
                if balance:
                    print(f"✅ Balance retrieved successfully!")
                    if isinstance(balance, dict):
                        print(f"   Available: ${balance.get('available_balance', 0) / 100:.2f}")
                        print(f"   Portfolio: ${balance.get('portfolio_value', 0) / 100:.2f}")
                    else:
                        print(f"   Balance data: {balance}")
                else:
                    print("⚠️ No balance data returned")
            except Exception as e:
                print(f"⚠️ Balance check error: {e}")
            
            # Try to get markets
            print()
            print("📊 Testing market access...")
            try:
                markets = adapter.get_markets(limit=5)
                if markets:
                    print(f"✅ Retrieved {len(markets)} markets!")
                    for i, market in enumerate(markets[:3]):
                        if hasattr(market, 'ticker'):
                            ticker = market.ticker
                            title = market.title[:50] if hasattr(market, 'title') else 'N/A'
                            yes_price = market.yes_price if hasattr(market, 'yes_price') else 'N/A'
                        elif isinstance(market, dict):
                            ticker = market.get('ticker', 'N/A')
                            title = market.get('title', 'N/A')[:50]
                            yes_price = market.get('yes_price', 'N/A')
                        else:
                            ticker = str(market)
                            title = ''
                            yes_price = 'N/A'
                        print(f"   {i+1}. {ticker}")
                        print(f"      {title}...")
                        print(f"      YES: {yes_price:.1%}" if isinstance(yes_price, float) else f"      YES: {yes_price}")
                else:
                    print("⚠️ No markets returned")
            except Exception as e:
                print(f"⚠️ Market fetch error: {e}")
            
            print()
            print("=" * 60)
            print("✅ KALSHI CONNECTION TEST COMPLETE")
            print("=" * 60)
            return True
            
        else:
            print("❌ Failed to initialize Kalshi adapter")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure kalshi-python is installed: pip install kalshi-python")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kalshi_trader():
    """Test the Kalshi trader agent"""
    print()
    print("=" * 60)
    print("🤖 KALSHI TRADER AGENT TEST")
    print("=" * 60)
    
    try:
        from agents.kalshi_trader import KalshiTrader, get_kalshi_trader
        
        trader = get_kalshi_trader(demo=False)
        
        if trader:
            print("✅ Kalshi trader initialized")
            
            # Check adapter
            if trader.adapter:
                print(f"✅ Trader has adapter connected")
            else:
                print("⚠️ Trader adapter not available")
            
            # Check LLM
            if trader.llm:
                print(f"✅ LLM available for superforecasting")
            else:
                print("ℹ️ LLM not configured (optional)")
            
            # Get status
            if hasattr(trader, 'get_status'):
                status = trader.get_status()
                print(f"   Status: {status}")
            
            return True
        else:
            print("❌ Failed to initialize Kalshi trader")
            return False
            
    except Exception as e:
        print(f"❌ Trader error: {e}")
        return False


if __name__ == "__main__":
    success = test_kalshi_connection()
    
    if success:
        test_kalshi_trader()
    
    print()
    print("🏁 Test complete!")


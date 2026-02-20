#!/usr/bin/env python3
"""
Securely setup Alpaca API credentials
Stores credentials in secure vault and optionally updates config files
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from secure_vault import get_vault

def setup_alpaca_credentials():
    """Securely store Alpaca API credentials"""
    
    # Alpaca API credentials (paper trading)
    API_KEY = "PKQIXYQPWDKTGGQG7PQZ36JWGF"
    API_SECRET = "EMPH6gEs5tSinsfb1BjB8ZD3p1HSugPq69rZMzUt942P"
    ENDPOINT = "https://paper-api.alpaca.markets/v2"
    
    print("=" * 60)
    print("Alpaca API Credentials Setup")
    print("=" * 60)
    print()
    
    # Initialize vault
    try:
        vault = get_vault()
        print("✅ Secure vault initialized")
    except Exception as e:
        print(f"❌ Failed to initialize vault: {e}")
        return False
    
    # Store in secure vault
    print("\nStoring credentials in secure vault...")
    try:
        vault.set_secret("alpaca", "api_key", API_KEY)
        vault.set_secret("alpaca", "api_secret", API_SECRET)
        vault.set_secret("alpaca", "endpoint", ENDPOINT)
        vault.set_secret("alpaca", "paper_trading", "true")
        print("✅ Credentials stored in secure vault")
    except Exception as e:
        print(f"❌ Failed to store in vault: {e}")
        return False
    
    # Verify storage
    print("\nVerifying credentials...")
    stored_key = vault.get_secret("alpaca", "api_key")
    stored_secret = vault.get_secret("alpaca", "api_secret")
    
    if stored_key == API_KEY and stored_secret == API_SECRET:
        print("✅ Credentials verified in vault")
    else:
        print("❌ Credential verification failed")
        return False
    
    # Update api_keys.json (already in .gitignore)
    print("\nUpdating config/api_keys.json...")
    try:
        import json
        config_path = Path(__file__).parent.parent / "config" / "api_keys.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        if "alpaca" not in config:
            config["alpaca"] = {}
        
        config["alpaca"]["api_key"] = API_KEY
        config["alpaca"]["api_secret"] = API_SECRET
        config["alpaca"]["paper_trading_url"] = "https://paper-api.alpaca.markets"
        config["alpaca"]["live_trading_url"] = "https://api.alpaca.markets"
        config["alpaca"]["endpoint"] = ENDPOINT
        config["alpaca"]["rate_limit"] = 200
        config["alpaca"]["description"] = "Alpaca API for paper and live trading"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Updated config/api_keys.json")
    except Exception as e:
        print(f"⚠️  Warning: Could not update api_keys.json: {e}")
    
    # Update broker_adapters.json
    print("\nUpdating config/broker_adapters.json...")
    try:
        broker_config_path = Path(__file__).parent.parent / "config" / "broker_adapters.json"
        
        if broker_config_path.exists():
            with open(broker_config_path, 'r') as f:
                broker_config = json.load(f)
        else:
            broker_config = {"default": "mock", "adapters": {}}
        
        if "alpaca" not in broker_config.get("adapters", {}):
            broker_config.setdefault("adapters", {})["alpaca"] = {
                "module": "adapters.alpaca",
                "class": "AlpacaAdapter",
                "config": {}
            }
        
        # Store reference to vault (not actual keys)
        broker_config["adapters"]["alpaca"]["config"]["paper_trading"] = True
        broker_config["adapters"]["alpaca"]["config"]["API_KEY"] = "FROM_VAULT"
        broker_config["adapters"]["alpaca"]["config"]["API_SECRET"] = "FROM_VAULT"
        
        with open(broker_config_path, 'w') as f:
            json.dump(broker_config, f, indent=2)
        
        print("✅ Updated config/broker_adapters.json")
    except Exception as e:
        print(f"⚠️  Warning: Could not update broker_adapters.json: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Alpaca credentials setup complete!")
    print("=" * 60)
    print()
    print("Storage locations:")
    print("  🔒 Secure Vault: config/.vault.encrypted (encrypted)")
    print("  📄 Config File: config/api_keys.json (in .gitignore)")
    print()
    print("Usage:")
    print("  The AlpacaAdapter will automatically:")
    print("  1. Check environment variables (APCA_API_KEY_ID, APCA_API_SECRET_KEY)")
    print("  2. Check secure vault (alpaca.api_key, alpaca.api_secret)")
    print("  3. Check config file (api_keys.json)")
    print()
    print("Optional: Set environment variables for convenience:")
    print("  export APCA_API_KEY_ID='PKQIXYQPWDKTGGQG7PQZ36JWGF'")
    print("  export APCA_API_SECRET_KEY='EMPH6gEs5tSinsfb1BjB8ZD3p1HSugPq69rZMzUt942P'")
    print()
    
    return True


if __name__ == "__main__":
    success = setup_alpaca_credentials()
    sys.exit(0 if success else 1)


#!/usr/bin/env python3
"""
Securely setup Kalshi API credentials
Stores credentials in secure vault for encrypted storage
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from secure_vault import get_vault

def setup_kalshi_credentials():
    """Securely store Kalshi API credentials"""
    
    # Kalshi API credentials
    API_KEY_ID = "72c42c1a-4ded-4d9f-91ae-3bed33952355"
    
    # Note: Kalshi uses RSA key authentication
    # The private key should be stored separately and securely
    # For now, we store the API Key ID
    
    print("=" * 60)
    print("Kalshi API Credentials Setup")
    print("=" * 60)
    print()
    print("REGULATORY INFO:")
    print("  - Kalshi is CFTC-regulated (legal in US)")
    print("  - Tax forms (1099) issued automatically")
    print("  - Funds held at FDIC-insured banks")
    print()
    
    # Initialize vault
    try:
        vault = get_vault()
        print("[OK] Secure vault initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize vault: {e}")
        return False
    
    # Store in secure vault
    print("\nStoring credentials in secure vault...")
    try:
        vault.set_secret("kalshi", "api_key_id", API_KEY_ID)
        vault.set_secret("kalshi", "environment", "production")
        vault.set_secret("kalshi", "endpoint", "https://api.elections.kalshi.com/trade-api/v2")
        print("[OK] API Key ID stored in secure vault")
    except Exception as e:
        print(f"[ERROR] Failed to store in vault: {e}")
        return False
    
    # Verify storage
    print("\nVerifying credentials...")
    stored_key_id = vault.get_secret("kalshi", "api_key_id")
    
    if stored_key_id == API_KEY_ID:
        print("[OK] API Key ID verified in vault")
    else:
        print("[ERROR] Credential verification failed")
        return False
    
    # Instructions for private key
    print("\n" + "=" * 60)
    print("IMPORTANT: RSA Private Key Required")
    print("=" * 60)
    print("""
Kalshi uses RSA-PSS authentication. You need to add your private key:

1. Log into Kalshi and go to Account Settings -> API Keys
2. Download or copy your RSA private key (PEM format)
3. Store it securely using one of these methods:

   Method A - Environment Variable (recommended):
   export KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
   ... your key content ...
   -----END RSA PRIVATE KEY-----"

   Method B - File:
   Save to: config/kalshi_private_key.pem
   (This file is gitignored for security)

   Method C - Vault:
   Run this Python code:
   from secure_vault import get_vault
   vault = get_vault()
   vault.set_secret("kalshi", "private_key", "your-key-content")

The adapter will automatically check these locations.
""")
    
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nKalshi API Key ID: {API_KEY_ID[:8]}...{API_KEY_ID[-4:]}")
    print("Environment: Production")
    print("Endpoint: https://api.elections.kalshi.com/trade-api/v2")
    
    return True


if __name__ == "__main__":
    success = setup_kalshi_credentials()
    sys.exit(0 if success else 1)


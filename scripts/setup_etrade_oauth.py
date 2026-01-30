#!/usr/bin/env python3
"""
E*Trade OAuth Setup Script
Completes OAuth 1.0a flow for E*Trade API authentication
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

def setup_etrade_oauth(sandbox: bool = True):
    """Setup E*Trade OAuth authentication"""
    
    print("="*80)
    print(f"E*TRADE OAuth Setup ({'SANDBOX' if sandbox else 'PRODUCTION'})")
    print("="*80)
    
    # Get credentials from vault
    vault = get_vault()
    
    if sandbox:
        consumer_key = vault.get_secret('etrade', 'sandbox_api_key')
        consumer_secret = vault.get_secret('etrade', 'sandbox_api_secret')
    else:
        consumer_key = vault.get_secret('etrade', 'prod_api_key')
        consumer_secret = vault.get_secret('etrade', 'prod_api_secret')
    
    if not consumer_key or not consumer_secret:
        print(f"❌ Error: {'Sandbox' if sandbox else 'Production'} credentials not found in vault")
        return False
    
    print(f"✅ Retrieved credentials from vault")
    print(f"   Consumer Key: {consumer_key[:10]}...")
    
    # Initialize OAuth handler
    oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox)
    
    # Step 1: Get request token
    print("\n" + "-"*80)
    print("Step 1: Getting request token...")
    print("-"*80)
    
    request_token, request_token_secret, auth_url = oauth.get_request_token()
    
    if not request_token:
        print("❌ Failed to get request token")
        return False
    
    print("✅ Request token obtained")
    print(f"\n🔗 Authorization URL:")
    print(f"   {auth_url}")
    print("\n⚠️  ACTION REQUIRED:")
    print("   1. Open the URL above in your browser")
    print("   2. Log in to your E*Trade account")
    print("   3. Authorize the application")
    print("   4. Copy the verification code from the page")
    
    # Step 2: Get verification code from user
    print("\n" + "-"*80)
    print("Step 2: Enter verification code")
    print("-"*80)
    
    verification_code = input("\nEnter verification code: ").strip()
    
    if not verification_code:
        print("❌ Verification code is required")
        return False
    
    # Step 3: Exchange for access token
    print("\n" + "-"*80)
    print("Step 3: Exchanging for access token...")
    print("-"*80)
    
    access_token, access_token_secret = oauth.get_access_token(verification_code)
    
    if not access_token:
        print("❌ Failed to get access token")
        return False
    
    print("✅ Access token obtained")
    
    # Step 4: Save tokens
    print("\n" + "-"*80)
    print("Step 4: Saving tokens...")
    print("-"*80)
    
    token_file = f"config/etrade_tokens_{'sandbox' if sandbox else 'prod'}.json"
    oauth.save_tokens(token_file)
    
    # Create authenticated session
    session = oauth.create_authenticated_session()
    if session:
        print("✅ Authenticated session created")
    
    # Test API call
    print("\n" + "-"*80)
    print("Step 5: Testing API connection...")
    print("-"*80)
    
    try:
        url = f"{oauth.base_url}/v1/accounts/list"
        response = session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            accounts = data.get('AccountListResponse', {}).get('Accounts', {}).get('Account', [])
            print(f"✅ API connection successful!")
            print(f"   Found {len(accounts)} account(s)")
            
            for i, account in enumerate(accounts, 1):
                account_id = account.get('accountId', {})
                print(f"   Account {i}: {account_id.get('accountId', 'N/A')}")
        else:
            print(f"⚠️  API test returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"⚠️  API test error: {e}")
    
    print("\n" + "="*80)
    print("✅ OAuth setup complete!")
    print("="*80)
    print(f"\nTokens saved to: {token_file}")
    print("You can now use E*Trade API with Optimus Agent")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup E*Trade OAuth authentication')
    parser.add_argument('--prod', action='store_true', help='Use production (default: sandbox)')
    args = parser.parse_args()
    
    sandbox = not args.prod
    
    try:
        success = setup_etrade_oauth(sandbox)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


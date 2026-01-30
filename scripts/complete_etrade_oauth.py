#!/usr/bin/env python3
"""
Complete E*Trade OAuth Setup (Part 2)
Use this script after you have the verification code from E*Trade
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

def complete_oauth(verification_code: str, sandbox: bool = True):
    """Complete OAuth flow with verification code"""
    
    print("="*80)
    print(f"COMPLETING E*TRADE OAUTH ({'SANDBOX' if sandbox else 'PRODUCTION'})")
    print("="*80)
    
    # Get credentials
    vault = get_vault()
    
    if sandbox:
        consumer_key = vault.get_secret('etrade', 'sandbox_api_key')
        consumer_secret = vault.get_secret('etrade', 'sandbox_api_secret')
    else:
        consumer_key = vault.get_secret('etrade', 'prod_api_key')
        consumer_secret = vault.get_secret('etrade', 'prod_api_secret')
    
    if not consumer_key or not consumer_secret:
        print(f"❌ Error: {'Sandbox' if sandbox else 'Production'} credentials not found")
        return False
    
    # Initialize OAuth
    oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox)
    
    # Get request token again (need fresh request token)
    print("\nStep 1: Getting request token...")
    request_token, request_token_secret, auth_url = oauth.get_request_token()
    
    if not request_token:
        print("❌ Failed to get request token")
        return False
    
    print("✅ Request token obtained")
    print(f"\n⚠️  NOTE: Make sure you authorized this token:")
    print(f"   {auth_url}")
    
    # Exchange for access token
    print("\nStep 2: Exchanging for access token...")
    access_token, access_token_secret = oauth.get_access_token(verification_code)
    
    if not access_token:
        print("❌ Failed to get access token")
        print("   Please verify:")
        print("   1. You authorized the correct request token")
        print("   2. The verification code is correct")
        print("   3. The verification code hasn't expired")
        return False
    
    print("✅ Access token obtained")
    
    # Save tokens
    print("\nStep 3: Saving tokens...")
    token_file = f"config/etrade_tokens_{'sandbox' if sandbox else 'prod'}.json"
    oauth.save_tokens(token_file)
    
    # Create authenticated session
    session = oauth.create_authenticated_session()
    if session:
        print("✅ Authenticated session created")
    
    # Test API call
    print("\nStep 4: Testing API connection...")
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
    print("✅ OAUTH SETUP COMPLETE!")
    print("="*80)
    print(f"\nTokens saved to: {token_file}")
    print("You can now test with: python3 test_etrade_sandbox_real.py")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete E*Trade OAuth setup')
    parser.add_argument('verification_code', help='Verification code from E*Trade authorization')
    parser.add_argument('--prod', action='store_true', help='Use production (default: sandbox)')
    args = parser.parse_args()
    
    try:
        success = complete_oauth(args.verification_code, sandbox=not args.prod)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


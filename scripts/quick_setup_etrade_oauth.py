#!/usr/bin/env python3
"""
E*Trade OAuth Quick Setup Guide
Shows instructions for completing OAuth flow
"""

import sys
import os
import webbrowser
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

def quick_setup():
    """Quick setup with guided instructions"""
    
    print("="*80)
    print("E*TRADE OAUTH SETUP - QUICK START")
    print("="*80)
    
    # Get credentials
    vault = get_vault()
    consumer_key = vault.get_secret('etrade', 'sandbox_api_key')
    consumer_secret = vault.get_secret('etrade', 'sandbox_api_secret')
    
    if not consumer_key or not consumer_secret:
        print("❌ Error: Sandbox credentials not found in vault")
        return False
    
    print("\n✅ Credentials retrieved from vault")
    
    # Initialize OAuth
    oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox=True)
    
    # Get request token
    print("\n" + "-"*80)
    print("Step 1: Getting request token...")
    print("-"*80)
    
    request_token, request_token_secret, auth_url = oauth.get_request_token()
    
    if not request_token:
        print("❌ Failed to get request token")
        print("   Please check your API credentials in the vault")
        return False
    
    print("✅ Request token obtained")
    print(f"\n🔗 Authorization URL:")
    print(f"   {auth_url}")
    print("\n" + "="*80)
    print("ACTION REQUIRED:")
    print("="*80)
    print("1. Open the URL above in your browser")
    print("2. Log in to your E*Trade account")
    print("3. Authorize the application")
    print("4. Copy the verification code from the page")
    print("\nThe code will look like: 12345678")
    print("\n" + "="*80)
    
    # Try to open browser
    try:
        print("\n🌐 Attempting to open browser...")
        webbrowser.open(auth_url)
        print("✅ Browser opened")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please copy and paste the URL manually")
    
    # Wait for verification code
    print("\n" + "-"*80)
    print("Step 2: Enter verification code")
    print("-"*80)
    
    verification_code = input("\n📝 Enter verification code: ").strip()
    
    if not verification_code:
        print("❌ Verification code is required")
        return False
    
    # Exchange for access token
    print("\n" + "-"*80)
    print("Step 3: Exchanging for access token...")
    print("-"*80)
    
    access_token, access_token_secret = oauth.get_access_token(verification_code)
    
    if not access_token:
        print("❌ Failed to get access token")
        print("   Please verify your verification code is correct")
        return False
    
    print("✅ Access token obtained")
    
    # Save tokens
    print("\n" + "-"*80)
    print("Step 4: Saving tokens...")
    print("-"*80)
    
    token_file = "config/etrade_tokens_sandbox.json"
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
    print("✅ OAUTH SETUP COMPLETE!")
    print("="*80)
    print(f"\nTokens saved to: {token_file}")
    print("You can now run: python3 test_etrade_sandbox_real.py")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


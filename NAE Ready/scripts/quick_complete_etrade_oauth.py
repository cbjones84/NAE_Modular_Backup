#!/usr/bin/env python3
"""
Quick Complete E*TRADE OAuth
Complete OAuth flow in one step - saves request tokens temporarily
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

def quick_complete_oauth(sandbox: bool = True):
    """Complete OAuth flow interactively"""
    
    print("="*80)
    print(f"E*TRADE OAUTH QUICK SETUP ({'SANDBOX' if sandbox else 'PRODUCTION'})")
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
    
    # Step 1: Start OAuth (get authorization URL)
    print("\n" + "-"*80)
    print("Step 1: Getting Authorization URL")
    print("-"*80)
    
    result = oauth.start_oauth()
    
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False
    
    authorize_url = result["authorize_url"]
    resource_owner_key = result["resource_owner_key"]
    resource_owner_secret = result["resource_owner_secret"]
    
    print("✅ Authorization URL obtained")
    print(f"\n🔗 Open this URL in your browser:")
    print(f"\n{authorize_url}\n")
    print("📋 Steps:")
    print("   1. Log in to your E*TRADE sandbox account")
    print("   2. Authorize the application")
    print("   3. Copy the verification code from the page")
    
    # Step 2: Get verification code
    print("\n" + "-"*80)
    print("Step 2: Enter Verification Code")
    print("-"*80)
    
    verification_code = input("\nEnter verification code: ").strip()
    
    if not verification_code:
        print("❌ Verification code is required")
        return False
    
    # Step 3: Finish OAuth (exchange for access token)
    print("\n" + "-"*80)
    print("Step 3: Completing OAuth")
    print("-"*80)
    
    tokens = oauth.finish_oauth(
        resource_owner_key,
        resource_owner_secret,
        verification_code
    )
    
    if tokens.get("error"):
        print(f"❌ Error: {tokens['error']}")
        return False
    
    if not tokens.get("oauth_token"):
        print("❌ Failed to get access token")
        print("   Please verify:")
        print("   1. The verification code is correct")
        print("   2. The verification code hasn't expired")
        print("   3. You authorized the application")
        return False
    
    print("✅ Access token obtained")
    
    # Step 4: Save tokens
    print("\n" + "-"*80)
    print("Step 4: Saving Tokens")
    print("-"*80)
    
    token_file = f"config/etrade_tokens_{'sandbox' if sandbox else 'prod'}.json"
    oauth.save_tokens(token_file)
    print(f"✅ Tokens saved to: {token_file}")
    
    # Step 5: Test API
    print("\n" + "-"*80)
    print("Step 5: Testing API Connection")
    print("-"*80)
    
    session = oauth.create_authenticated_session()
    if session:
        try:
            url = f"{oauth.base_url}/v1/accounts/list"
            response = session.get(url, timeout=10)
            
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
    print("\n💡 Next: Test the adapter")
    print("   python3 scripts/test_etrade_adapter.py")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick E*TRADE OAuth setup')
    parser.add_argument('--prod', action='store_true', help='Use production (default: sandbox)')
    args = parser.parse_args()
    
    sandbox = not args.prod
    
    try:
        success = quick_complete_oauth(sandbox)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



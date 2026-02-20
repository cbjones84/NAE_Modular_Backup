#!/usr/bin/env python3
"""
Finish E*TRADE OAuth Setup
Use this after you've authorized and received the verification code
Usage: python3 finish_etrade_oauth.py YOUR_VERIFICATION_CODE
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

def finish_oauth_with_code(verification_code: str, sandbox: bool = True):
    """Complete OAuth using saved request tokens and verification code"""
    
    print("="*80)
    print(f"COMPLETING E*TRADE OAUTH ({'SANDBOX' if sandbox else 'PRODUCTION'})")
    print("="*80)
    
    # Load saved request tokens
    temp_file = 'config/etrade_oauth_temp.json'
    
    if not os.path.exists(temp_file):
        print(f"❌ Error: Could not find temporary OAuth file: {temp_file}")
        print("\n💡 You need to run the OAuth setup first:")
        print("   python3 scripts/quick_complete_etrade_oauth.py")
        print("\n   Or get the authorization URL:")
        print("   python3 -c \"from agents.etrade_oauth import ETradeOAuth; from secure_vault import get_vault; v=get_vault(); o=ETradeOAuth(v.get_secret('etrade','sandbox_api_key'), v.get_secret('etrade','sandbox_api_secret'), sandbox=True); r=o.start_oauth(); print(r['authorize_url'])\"")
        return False
    
    try:
        with open(temp_file, 'r') as f:
            oauth_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading temporary OAuth file: {e}")
        return False
    
    resource_owner_key = oauth_data.get('resource_owner_key')
    resource_owner_secret = oauth_data.get('resource_owner_secret')
    
    if not resource_owner_key or not resource_owner_secret:
        print("❌ Error: Invalid temporary OAuth file")
        return False
    
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
    
    print(f"\n✅ Loaded request tokens from temporary file")
    print(f"✅ Using verification code: {verification_code}")
    
    # Initialize OAuth
    oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox)
    
    # Finish OAuth
    print("\n" + "-"*80)
    print("Step 1: Exchanging Verification Code for Access Tokens")
    print("-"*80)
    
    tokens = oauth.finish_oauth(
        resource_owner_key,
        resource_owner_secret,
        verification_code
    )
    
    if tokens.get("error"):
        print(f"❌ Error: {tokens['error']}")
        print("\n💡 Common issues:")
        print("   - Verification code expired (get a new one)")
        print("   - Wrong verification code (double-check)")
        print("   - Request token expired (start over)")
        return False
    
    if not tokens.get("oauth_token"):
        print("❌ Failed to get access token")
        return False
    
    print("✅ Access token obtained!")
    
    # Save tokens
    print("\n" + "-"*80)
    print("Step 2: Saving Access Tokens")
    print("-"*80)
    
    token_file = f"config/etrade_tokens_{'sandbox' if sandbox else 'prod'}.json"
    oauth.save_tokens(token_file)
    print(f"✅ Tokens saved to: {token_file}")
    
    # Clean up temporary file
    try:
        os.remove(temp_file)
        print(f"✅ Cleaned up temporary file: {temp_file}")
    except:
        pass
    
    # Test API
    print("\n" + "-"*80)
    print("Step 3: Testing API Connection")
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
                    account_id_str = account_id.get('accountId', 'N/A')
                    account_type = account.get('accountType', 'N/A')
                    print(f"   Account {i}: {account_id_str} ({account_type})")
                    
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
    if len(sys.argv) < 2:
        print("Usage: python3 finish_etrade_oauth.py VERIFICATION_CODE")
        print("\nExample: python3 finish_etrade_oauth.py 123456")
        print("\nGet the verification code by:")
        print("  1. Running: python3 scripts/quick_complete_etrade_oauth.py")
        print("  2. Opening the authorization URL in your browser")
        print("  3. Authorizing and copying the verification code")
        sys.exit(1)
    
    verification_code = sys.argv[1].strip()
    
    if not verification_code:
        print("❌ Error: Verification code cannot be empty")
        sys.exit(1)
    
    try:
        success = finish_oauth_with_code(verification_code, sandbox=True)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



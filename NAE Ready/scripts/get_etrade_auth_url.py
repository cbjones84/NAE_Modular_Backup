#!/usr/bin/env python3
"""
Get E*Trade Authorization URL
Quick script to get the authorization URL for OAuth setup
"""

import sys
import os
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

def get_authorization_url():
    """Get E*Trade authorization URL"""
    
    print("="*80)
    print("E*TRADE AUTHORIZATION URL GENERATOR")
    print("="*80)
    
    # Get credentials
    vault = get_vault()
    consumer_key = vault.get_secret('etrade', 'sandbox_api_key')
    consumer_secret = vault.get_secret('etrade', 'sandbox_api_secret')
    
    if not consumer_key or not consumer_secret:
        print("\n❌ Error: Sandbox credentials not found in vault")
        return None
    
    print("\n✅ Credentials found")
    
    # Get authorization URL
    print("\n📡 Requesting authorization URL from E*Trade...")
    oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox=True)
    
    request_token, request_token_secret, auth_url = oauth.get_request_token()
    
    if not auth_url:
        print("\n❌ Failed to get authorization URL")
        return None
    
    print("\n✅ Authorization URL obtained!")
    
    # Display URL
    print("\n" + "="*80)
    print("🔗 YOUR AUTHORIZATION URL:")
    print("="*80)
    print(f"\n{auth_url}\n")
    print("="*80)
    
    # Try to open in browser
    try:
        print("\n🌐 Attempting to open URL in browser...")
        webbrowser.open(auth_url)
        print("✅ Browser opened")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please copy and paste the URL manually")
    
    print("\n📋 INSTRUCTIONS:")
    print("-"*80)
    print("1. ✅ URL is displayed above (and opened in browser if possible)")
    print("2. Log in to your E*Trade account")
    print("3. Review and authorize the application")
    print("4. Copy the verification code shown on the page")
    print("5. Run: python3 complete_etrade_oauth.py YOUR_CODE")
    
    print("\n💡 Save this URL - you'll need it for authorization!")
    print("\n" + "="*80)
    
    return auth_url

if __name__ == "__main__":
    try:
        url = get_authorization_url()
        if url:
            print("\n✅ Authorization URL ready!")
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


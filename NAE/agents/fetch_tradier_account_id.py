#!/usr/bin/env python3
"""
Helper script to fetch Tradier Account ID from API
"""
import os
import sys
import requests
from typing import Optional

def fetch_account_id(api_key: str, sandbox: bool = True) -> Optional[str]:
    """
    Fetch account ID from Tradier API
    
    Args:
        api_key: Tradier API key
        sandbox: Whether to use sandbox environment
    
    Returns:
        Account ID if found, None otherwise
    """
    if sandbox:
        api_base = "https://sandbox.tradier.com/v1"
    else:
        api_base = "https://api.tradier.com/v1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    # Try /user/profile endpoint
    try:
        print(f"🔍 Fetching account info from {api_base}...")
        response = requests.get(
            f"{api_base}/user/profile",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            profile = data.get("profile", {})
            accounts = profile.get("account", [])
            
            if isinstance(accounts, list) and len(accounts) > 0:
                account = accounts[0]
                account_id = (
                    account.get("account_number") or 
                    account.get("id") or 
                    account.get("account_id")
                )
                if account_id:
                    print(f"✅ Found Account ID: {account_id}")
                    print(f"   Account Type: {account.get('type', 'unknown')}")
                    print(f"   Account Status: {account.get('status', 'unknown')}")
                    return account_id
            elif isinstance(accounts, dict):
                account_id = (
                    accounts.get("account_number") or 
                    accounts.get("id") or 
                    accounts.get("account_id")
                )
                if account_id:
                    print(f"✅ Found Account ID: {account_id}")
                    return account_id
                    
    except Exception as e:
        print(f"⚠️  Error fetching from /user/profile: {e}")
    
    # Try /accounts endpoint
    try:
        response = requests.get(
            f"{api_base}/accounts",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            accounts_data = data.get("accounts", {})
            
            if isinstance(accounts_data, dict):
                account_list = accounts_data.get("account", [])
                if isinstance(account_list, list) and len(account_list) > 0:
                    account = account_list[0]
                    account_id = (
                        account.get("account_number") or 
                        account.get("id") or 
                        account.get("account_id")
                    )
                    if account_id:
                        print(f"✅ Found Account ID via /accounts: {account_id}")
                        return account_id
                elif isinstance(accounts_data.get("account"), dict):
                    account = accounts_data["account"]
                    account_id = (
                        account.get("account_number") or 
                        account.get("id") or 
                        account.get("account_id")
                    )
                    if account_id:
                        print(f"✅ Found Account ID: {account_id}")
                        return account_id
                        
    except Exception as e:
        print(f"⚠️  Error fetching from /accounts: {e}")
    
    return None

if __name__ == "__main__":
    api_key = os.getenv("TRADIER_API_KEY", "27Ymk28vtbgqY1LFYxhzaEmIuwJb")
    sandbox = os.getenv("TRADIER_SANDBOX", "true").lower() == "true"
    
    print("="*70)
    print("Tradier Account ID Fetcher")
    print("="*70)
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"Environment: {'Sandbox' if sandbox else 'Production'}")
    print("")
    
    account_id = fetch_account_id(api_key, sandbox)
    
    if account_id:
        print("")
        print("="*70)
        print("✅ SUCCESS - Set this environment variable:")
        print("="*70)
        print(f"export TRADIER_ACCOUNT_ID={account_id}")
        print("")
        print("Or add to your shell profile (~/.bashrc or ~/.zshrc):")
        print(f"export TRADIER_ACCOUNT_ID={account_id}")
        print("="*70)
    else:
        print("")
        print("="*70)
        print("⚠️  Could not automatically fetch Account ID")
        print("="*70)
        print("Please set TRADIER_ACCOUNT_ID manually:")
        print("1. Log into your Tradier account")
        print("2. Find your account number")
        print("3. Set: export TRADIER_ACCOUNT_ID=your_account_number")
        print("="*70)
        sys.exit(1)


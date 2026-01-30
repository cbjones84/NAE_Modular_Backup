#!/usr/bin/env python3
"""Quick test of Tradier API connection"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'execution'))

from broker_adapters.tradier_adapter import TradierBrokerAdapter

print('='*50)
print('Testing Tradier API Connection')
print('='*50)

# Get env vars
api_key = os.getenv('TRADIER_API_KEY')
account_id = os.getenv('TRADIER_ACCOUNT_ID')
sandbox = os.getenv('TRADIER_SANDBOX', 'true').lower() == 'true'

print(f'API Key: {api_key[:15]}...' if api_key else 'API Key: NOT SET')
print(f'Account ID: {account_id}')
print(f'Sandbox: {sandbox}')
print()

if not api_key:
    print('ERROR: TRADIER_API_KEY not set')
    print('Run: $env:TRADIER_API_KEY = "your_key"')
    sys.exit(1)

print('Creating adapter...')
adapter = TradierBrokerAdapter(sandbox=sandbox)
print('  Adapter created successfully!')
print()

print('Testing API connection...')
try:
    account = adapter.get_account_info()
    if account:
        print('  Account Info:')
        print(f'    Account Number: {account.get("account_number", "N/A")}')
        print(f'    Type: {account.get("type", "N/A")}')
        print(f'    Status: {account.get("status", "N/A")}')
        
        # Get balances
        balances = account.get('balances', {})
        if balances:
            cash = balances.get('cash', {})
            if isinstance(cash, dict):
                print(f'    Cash Available: ${cash.get("cash_available", 0):,.2f}')
            print(f'    Market Value: ${balances.get("market_value", 0):,.2f}')
        
        print()
        print('SUCCESS: Tradier API is working!')
    else:
        print('  No account data returned')
        print('  This could mean:')
        print('    - API key is invalid')
        print('    - Account ID is wrong')
        print('    - Using sandbox=False but key is for sandbox')
except Exception as e:
    print(f'  Error: {e}')
    import traceback
    traceback.print_exc()
    print()
    print('Possible issues:')
    print('  - API key might be invalid or expired')
    print('  - Check if using correct endpoint (sandbox vs live)')


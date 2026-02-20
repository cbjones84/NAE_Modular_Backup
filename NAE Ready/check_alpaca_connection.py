#!/usr/bin/env python3
"""
Continuous Alpaca Connection Checker
Monitors Alpaca API connection status and alerts when ready
"""

import time
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from adapters.alpaca import AlpacaAdapter

def check_connection():
    """Check Alpaca connection status"""
    try:
        # Load config
        with open('config/api_keys.json', 'r') as f:
            api_keys = json.load(f)
        
        alpaca_config = api_keys.get('alpaca', {})
        
        # Test LIVE connection
        config = {
            'API_KEY': alpaca_config.get('api_key'),
            'API_SECRET': alpaca_config.get('api_secret'),
            'paper_trading': False  # LIVE
        }
        
        adapter = AlpacaAdapter(config)
        
        if adapter.auth():
            account = adapter.get_account()
            return True, account
        else:
            return False, None
            
    except Exception as e:
        return False, str(e)

def main():
    """Main monitoring loop"""
    print("="*70)
    print("Alpaca Connection Monitor")
    print("="*70)
    print("\nMonitoring Alpaca API connection...")
    print("Press Ctrl+C to stop\n")
    
    check_count = 0
    while True:
        try:
            check_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            success, result = check_connection()
            
            if success:
                account = result
                print(f"\n{'='*70}")
                print(f"✅ CONNECTION SUCCESSFUL at {timestamp}")
                print(f"{'='*70}")
                print(f"\nAccount Information:")
                print(f"  Cash: ${account.get('cash', 0):,.2f}")
                print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
                print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
                print(f"  Trading Blocked: {account.get('trading_blocked', False)}")
                print(f"\n✅ Alpaca is ready for live trading!")
                print(f"{'='*70}\n")
                break
            else:
                print(f"[{timestamp}] Check #{check_count}: ❌ Connection failed - {result if isinstance(result, str) else 'Unauthorized'}")
                print("   Waiting 30 seconds before next check...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()


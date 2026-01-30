#!/usr/bin/env python3
"""
E*TRADE Sandbox OAuth Status Monitor
Continuously checks if E*TRADE sandbox OAuth is back up and working
"""

import sys
import os
import time
import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault
import requests

class ETradeStatusMonitor:
    """Monitor E*TRADE sandbox OAuth status"""
    
    def __init__(self, sandbox=True, check_interval=60):
        """
        Initialize monitor
        
        Args:
            sandbox: Use sandbox (True) or production (False)
            check_interval: Seconds between checks (default: 60)
        """
        self.sandbox = sandbox
        self.check_interval = check_interval
        self.base_url = "https://apisb.etrade.com" if sandbox else "https://api.etrade.com"
        
        # Get credentials
        vault = get_vault()
        if sandbox:
            self.consumer_key = vault.get_secret('etrade', 'sandbox_api_key')
            self.consumer_secret = vault.get_secret('etrade', 'sandbox_api_secret')
        else:
            self.consumer_key = vault.get_secret('etrade', 'prod_api_key')
            self.consumer_secret = vault.get_secret('etrade', 'prod_api_secret')
        
        if not self.consumer_key or not self.consumer_secret:
            raise ValueError("E*TRADE credentials not found in vault")
        
        self.status_history = []
    
    def check_request_token_endpoint(self):
        """Check if request token endpoint is working"""
        try:
            oauth = ETradeOAuth(self.consumer_key, self.consumer_secret, self.sandbox)
            result = oauth.start_oauth()
            
            if result.get("error"):
                return {
                    "status": "error",
                    "error": result["error"],
                    "working": False
                }
            
            if result.get("authorize_url") and result.get("resource_owner_key"):
                return {
                    "status": "working",
                    "request_token": result["resource_owner_key"][:20] + "...",
                    "authorize_url": result["authorize_url"],
                    "working": True
                }
            
            return {
                "status": "unknown",
                "working": False
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "working": False
            }
    
    def check_authorization_url(self, auth_url):
        """Check if authorization URL returns valid response (not 500)"""
        try:
            # Use HEAD request first (lighter)
            response = requests.head(auth_url, timeout=10, allow_redirects=True)
            
            # If HEAD doesn't work, try GET
            if response.status_code == 405:  # Method not allowed
                response = requests.get(auth_url, timeout=10, allow_redirects=False)
            
            # Check status
            if response.status_code == 200:
                return {
                    "status": "working",
                    "status_code": response.status_code,
                    "working": True
                }
            elif response.status_code == 500:
                return {
                    "status": "server_error",
                    "status_code": 500,
                    "error": "E*TRADE server returning 500 error",
                    "working": False
                }
            elif response.status_code == 401 or response.status_code == 403:
                return {
                    "status": "auth_required",
                    "status_code": response.status_code,
                    "working": True,  # Server is up, just needs auth
                    "note": "Normal - requires user authorization"
                }
            elif response.status_code in [301, 302, 303, 307, 308]:
                return {
                    "status": "redirecting",
                    "status_code": response.status_code,
                    "working": True,  # Server is up, redirecting to login
                    "note": "Normal - redirecting to login page"
                }
            else:
                return {
                    "status": "unexpected",
                    "status_code": response.status_code,
                    "working": False,
                    "error": f"Unexpected status code: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "error": "Request timed out",
                "working": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "connection_error",
                "error": "Cannot connect to E*TRADE server",
                "working": False
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "working": False
            }
    
    def check_api_endpoint(self):
        """Check if E*TRADE API base endpoint is reachable"""
        try:
            # Try a simple endpoint that doesn't require auth
            test_url = f"{self.base_url}/v1/accounts/list"
            response = requests.get(test_url, timeout=5)
            
            # Even 401/403 means server is up
            if response.status_code in [401, 403]:
                return {
                    "status": "up",
                    "status_code": response.status_code,
                    "working": True,
                    "note": "Server is up (auth required)"
                }
            elif response.status_code == 200:
                return {
                    "status": "up",
                    "status_code": 200,
                    "working": True
                }
            else:
                return {
                    "status": "unexpected",
                    "status_code": response.status_code,
                    "working": False
                }
                
        except Exception as e:
            return {
                "status": "down",
                "error": str(e),
                "working": False
            }
    
    def run_check(self):
        """Run a complete status check"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*80}")
        print(f"E*TRADE {'SANDBOX' if self.sandbox else 'PRODUCTION'} STATUS CHECK - {timestamp}")
        print(f"{'='*80}\n")
        
        # Check 1: Request Token Endpoint
        print("1️⃣  Checking Request Token Endpoint...")
        token_check = self.check_request_token_endpoint()
        
        if token_check["working"]:
            print(f"   ✅ Request token endpoint: WORKING")
            print(f"      Request token: {token_check.get('request_token', 'N/A')}")
            
            # Check 2: Authorization URL
            print("\n2️⃣  Checking Authorization URL...")
            auth_url = token_check.get("authorize_url")
            if auth_url:
                url_check = self.check_authorization_url(auth_url)
                
                if url_check["working"]:
                    if url_check.get("status") == "server_error":
                        print(f"   ❌ Authorization URL: 500 SERVER ERROR")
                        print(f"      Status: {url_check.get('status_code')}")
                        print(f"      Error: {url_check.get('error')}")
                        overall_status = "down"
                    else:
                        print(f"   ✅ Authorization URL: WORKING")
                        print(f"      Status: {url_check.get('status_code')}")
                        if url_check.get("note"):
                            print(f"      Note: {url_check.get('note')}")
                        overall_status = "up"
                else:
                    print(f"   ❌ Authorization URL: NOT WORKING")
                    print(f"      Status: {url_check.get('status_code', 'N/A')}")
                    print(f"      Error: {url_check.get('error', 'Unknown')}")
                    overall_status = "down"
            else:
                print("   ⚠️  No authorization URL generated")
                overall_status = "unknown"
        else:
            print(f"   ❌ Request token endpoint: NOT WORKING")
            print(f"      Error: {token_check.get('error', 'Unknown')}")
            overall_status = "down"
        
        # Check 3: API Endpoint
        print("\n3️⃣  Checking API Base Endpoint...")
        api_check = self.check_api_endpoint()
        if api_check["working"]:
            print(f"   ✅ API endpoint: UP")
            print(f"      Status: {api_check.get('status_code')}")
            if api_check.get("note"):
                print(f"      Note: {api_check.get('note')}")
        else:
            print(f"   ❌ API endpoint: DOWN")
            print(f"      Error: {api_check.get('error', 'Unknown')}")
        
        # Overall Status
        print(f"\n{'='*80}")
        if overall_status == "up":
            print("✅ OVERALL STATUS: E*TRADE SANDBOX IS UP! 🎉")
            print(f"{'='*80}")
            if auth_url:
                print(f"\n📋 You can now complete OAuth:")
                print(f"   {auth_url}")
                print(f"\n   Then run:")
                print(f"   python3 scripts/finish_etrade_oauth.py YOUR_VERIFICATION_CODE")
        elif overall_status == "down":
            print("❌ OVERALL STATUS: E*TRADE SANDBOX IS DOWN")
            print(f"{'='*80}")
        else:
            print("⚠️  OVERALL STATUS: UNKNOWN")
            print(f"{'='*80}")
        
        # Save to history
        self.status_history.append({
            "timestamp": timestamp,
            "overall_status": overall_status,
            "token_check": token_check,
            "url_check": url_check if token_check.get("working") else None,
            "api_check": api_check
        })
        
        return overall_status == "up"
    
    def monitor_continuous(self, max_checks=None):
        """
        Monitor continuously
        
        Args:
            max_checks: Maximum number of checks (None for infinite)
        """
        print("="*80)
        print(f"E*TRADE {'SANDBOX' if self.sandbox else 'PRODUCTION'} STATUS MONITOR")
        print("="*80)
        print(f"\nChecking every {self.check_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        check_count = 0
        
        try:
            while True:
                is_up = self.run_check()
                check_count += 1
                
                if max_checks and check_count >= max_checks:
                    print(f"\n✅ Completed {max_checks} checks. Stopping.")
                    break
                
                if is_up:
                    print("\n🎉 E*TRADE IS BACK UP! Stopping monitor.")
                    break
                
                print(f"\n⏳ Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitor stopped by user")
            print(f"   Total checks: {check_count}")
    
    def save_history(self, filename="config/etrade_status_history.json"):
        """Save status history to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.status_history, f, indent=2)
        print(f"\n📝 Status history saved to: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor E*TRADE OAuth status')
    parser.add_argument('--once', action='store_true', help='Run check once and exit')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between checks (default: 60)')
    parser.add_argument('--max-checks', type=int, help='Maximum number of checks (default: infinite)')
    parser.add_argument('--prod', action='store_true', help='Monitor production (default: sandbox)')
    args = parser.parse_args()
    
    try:
        monitor = ETradeStatusMonitor(sandbox=not args.prod, check_interval=args.interval)
        
        if args.once:
            monitor.run_check()
        else:
            monitor.monitor_continuous(max_checks=args.max_checks)
        
        # Save history on exit
        if monitor.status_history:
            monitor.save_history()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitor stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



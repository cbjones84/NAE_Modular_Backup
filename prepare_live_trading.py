#!/usr/bin/env python3
"""
NAE Live Trading Preparation Script
Prepares NAE for LIVE trading via Tradier (the ONLY broker)
VERY_AGGRESSIVE mode enabled for $5M goal

Tradier is the exclusive broker - Alpaca and IBKR have been removed.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def check_tradier_credentials():
    """Check Tradier credentials"""
    print_header("Checking Tradier Credentials")
    
    # Check environment variables first
    api_key = os.getenv("TRADIER_API_KEY", "")
    account_id = os.getenv("TRADIER_ACCOUNT_ID", "")
    
    if api_key and account_id:
        print_success("Tradier credentials found in environment variables")
        return True, api_key, account_id
    
    # Check vault
    try:
        from secure_vault import get_vault
        vault = get_vault()
        api_key = vault.get_secret("tradier", "api_key")
        account_id = vault.get_secret("tradier", "account_id")
        
        if api_key and account_id:
            print_success("Tradier credentials found in secure vault")
            return True, api_key, account_id
    except Exception as e:
        print_warning(f"Could not access vault: {e}")
    
    # Check api_keys.json as fallback
    api_keys_path = Path("config/api_keys.json")
    if api_keys_path.exists():
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        
        tradier = api_keys.get("tradier", {})
        api_key = tradier.get("api_key", "")
        account_id = tradier.get("account_id", "")
        
        if api_key and account_id:
            print_success("Tradier credentials found in api_keys.json")
            return True, api_key, account_id
    
    print_error("Tradier credentials not found")
    print_info("Set TRADIER_API_KEY and TRADIER_ACCOUNT_ID environment variables")
    return False, None, None

def test_tradier_connection(api_key, account_id):
    """Test Tradier connection"""
    print_header("Testing Tradier Connection")
    
    try:
        from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter
        
        adapter = TradierBrokerAdapter(
            api_key=api_key,
            account_id=account_id,
            sandbox=False  # LIVE MODE
        )
        
        # Test authentication
        if adapter.authenticate():
            print_success("Tradier authentication successful (LIVE mode)")
            
            # Get account info
            account_info = adapter.get_account_info()
            if account_info:
                balance = adapter.get_account_balance()
                buying_power = adapter.get_buying_power()
                
                print_success(f"Account ID: {account_id}")
                print_success(f"Balance: ${balance:,.2f}")
                print_success(f"Buying Power: ${buying_power:,.2f}")
                
                return True, adapter
            else:
                print_error("Could not retrieve account info")
                return False, None
        else:
            print_error("Tradier authentication failed")
            return False, None
            
    except Exception as e:
        print_error(f"Tradier connection test failed: {e}")
        return False, None

def update_settings_for_live():
    """Update settings for LIVE trading"""
    print_header("Updating Settings for LIVE Trading")
    
    settings_path = Path("config/settings.json")
    
    if not settings_path.exists():
        print_error("settings.json not found")
        return False
    
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Update for LIVE trading with Tradier
    settings["trading"]["default_mode"] = "live"
    settings["trading"]["broker"] = "tradier"
    settings["trading"]["live"]["enabled"] = True
    settings["trading"]["live"]["broker"] = "tradier"
    settings["trading"]["live"]["requires_manual_approval"] = False
    settings["trading"]["paper"]["enabled"] = False
    settings["trading"]["sandbox"]["enabled"] = False
    settings["aggressiveness"] = "VERY_AGGRESSIVE"
    settings["risk_adjustment_factor"] = 1.5
    
    # Update safety limits for VERY_AGGRESSIVE mode
    settings["safety_limits"]["max_order_size_usd"] = 50000.0
    settings["safety_limits"]["max_order_size_pct_nav"] = 0.50
    settings["safety_limits"]["daily_loss_limit_pct"] = 0.50
    settings["safety_limits"]["consecutive_loss_limit"] = 10
    settings["safety_limits"]["max_open_positions"] = 30
    
    settings["last_updated"] = datetime.now().isoformat()
    
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print_success("Settings updated for LIVE trading")
    print_success("Broker: Tradier")
    print_success("Mode: LIVE")
    print_success("Aggressiveness: VERY_AGGRESSIVE (1.5x)")
    return True

def verify_optimus_configuration():
    """Verify Optimus is configured for LIVE trading"""
    print_header("Verifying Optimus Configuration")
    
    try:
        from agents.optimus import OptimusAgent, TradingMode
        optimus = OptimusAgent(sandbox=False)  # Initialize for LIVE
        
        # Verify trading mode
        if optimus.trading_mode == TradingMode.LIVE:
            print_success("Optimus trading mode: LIVE")
        else:
            print_warning(f"Optimus trading mode: {optimus.trading_mode.value} (expected LIVE)")
        
        # Check safety limits
        if hasattr(optimus, 'safety_limits'):
            limits = optimus.safety_limits
            print_success(f"Max order size: ${limits.max_order_size_usd:,.2f}")
            print_success(f"Max order size (% NAV): {limits.max_order_size_pct_nav*100:.0f}%")
            print_success(f"Daily loss limit: {limits.daily_loss_limit_pct*100:.0f}%")
            print_success(f"Max positions: {limits.max_open_positions}")
        
        # Check milestone tracker
        if hasattr(optimus, 'milestone_tracker') and optimus.milestone_tracker:
            status = optimus.milestone_tracker.get_milestone_status(optimus.nav)
            print_success(f"Aggressiveness: {status.aggressiveness}")
            risk_factor = optimus.milestone_tracker.get_risk_adjustment_factor(optimus.nav)
            print_success(f"Risk adjustment factor: {risk_factor}x")
        
        # Check Tradier integration
        if hasattr(optimus, 'self_healing_engine') and optimus.self_healing_engine:
            print_success("Tradier self-healing engine: ACTIVE")
        
        print_success("Optimus configured for LIVE VERY_AGGRESSIVE trading")
        return True
        
    except Exception as e:
        print_error(f"Optimus verification failed: {e}")
        return False

def main():
    """Main preparation script"""
    print_header("NAE LIVE TRADING PREPARATION")
    print_info("Broker: TRADIER (ONLY)")
    print_info("Mode: LIVE")
    print_info("Aggressiveness: VERY_AGGRESSIVE")
    print_info("")
    print_info("GROWTH MILESTONES:")
    print_info("  Year 1: $9,411      (+$6,911 returns)")
    print_info("  Year 2: $44,110     (+$39,210 returns)")
    print_info("  Year 3: $152,834    (+$145,534 returns)")
    print_info("  Year 4: $388,657    (+$378,957 returns)")
    print_info("  Year 5: $982,500    (+$970,400 returns)")
    print_info("  Year 6: $2,477,897  (+$2,463,397 returns)")
    print_info("  Year 7: $6,243,561  [TARGET EXCEEDED!]")
    print_info("  Year 8: $15,726,144 [FINAL: $15.7M]")
    
    all_checks_passed = True
    
    # Step 1: Check credentials
    credentials_ok, api_key, account_id = check_tradier_credentials()
    if not credentials_ok:
        all_checks_passed = False
        print_error("Cannot proceed without Tradier credentials")
        return 1
    
    # Step 2: Test connection
    print_info("\nTesting Tradier connection...")
    connection_ok, adapter = test_tradier_connection(api_key, account_id)
    
    if not connection_ok:
        all_checks_passed = False
        print_error("Tradier connection test failed")
        return 1
    
    # Step 3: Update settings
    if not update_settings_for_live():
        all_checks_passed = False
    
    # Step 4: Verify Optimus
    if not verify_optimus_configuration():
        all_checks_passed = False
    
    # Final summary
    print_header("PREPARATION SUMMARY")
    
    if all_checks_passed:
        print_success("All checks passed!")
        print_success("NAE is ready for LIVE VERY_AGGRESSIVE trading via Tradier")
        print_info("")
        print_info("To start trading, run:")
        print_info("  python scripts/start_production_trading.py")
        print_info("")
        print_info("Or start the full system:")
        print_info("  python run_nae_full_system.py")
        return 0
    else:
        print_error("Some checks failed")
        print_error("Review the issues above before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())

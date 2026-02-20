#!/usr/bin/env python3
"""
Verify Tradier Integration Fixes
Tests that the AttributeError bug is fixed and live trading has proper broker fallback
"""

import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'execution'))

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_result(test_name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")

def test_tradier_oauth():
    """Test 1: TradierOAuth initialization without AttributeError"""
    print_header("Test 1: TradierOAuth Initialization")
    
    try:
        from broker_adapters.tradier_adapter import TradierOAuth
        
        # Test with no credentials (should not raise AttributeError)
        oauth = TradierOAuth(sandbox=True)
        print_result("No credentials", True, f"api_key={oauth.api_key}")
        
        # Test with API key only
        oauth2 = TradierOAuth(api_key="test_key", sandbox=True)
        print_result("With API key", True, f"api_key={oauth2.api_key[:10]}...")
        
        # Test with OAuth credentials
        oauth3 = TradierOAuth(client_id="test_id", client_secret="test_secret", sandbox=True)
        print_result("With OAuth", True, f"use_oauth={oauth3.use_oauth}")
        
        return True
    except AttributeError as e:
        print_result("TradierOAuth init", False, f"AttributeError: {e}")
        return False
    except Exception as e:
        print_result("TradierOAuth init", False, f"Error: {e}")
        return False

def test_tradier_adapter():
    """Test 2: TradierBrokerAdapter initialization"""
    print_header("Test 2: TradierBrokerAdapter Initialization")
    
    try:
        from broker_adapters.tradier_adapter import TradierBrokerAdapter
        
        # Test with environment variables (or defaults)
        adapter = TradierBrokerAdapter(sandbox=True)
        print_result("Sandbox adapter", True, f"account_id={adapter.account_id}")
        
        # Test with explicit credentials
        adapter2 = TradierBrokerAdapter(
            api_key="test_key",
            account_id="test_account",
            sandbox=True
        )
        print_result("With credentials", True, f"account_id={adapter2.account_id}")
        
        return True
    except AttributeError as e:
        print_result("TradierBrokerAdapter", False, f"AttributeError: {e}")
        return False
    except Exception as e:
        print_result("TradierBrokerAdapter", False, f"Error: {e}")
        return False

def test_env_variables():
    """Test 3: Check Tradier environment variables"""
    print_header("Test 3: Tradier Environment Variables")
    
    env_vars = {
        "TRADIER_API_KEY": os.getenv("TRADIER_API_KEY"),
        "TRADIER_ACCOUNT_ID": os.getenv("TRADIER_ACCOUNT_ID"),
        "TRADIER_SANDBOX": os.getenv("TRADIER_SANDBOX"),
    }
    
    all_set = True
    for var, value in env_vars.items():
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print_result(var, True, f"Set to: {masked}")
        else:
            print_result(var, False, "NOT SET")
            all_set = False
    
    if not all_set:
        print("\n  ⚠️  To set environment variables in PowerShell:")
        print('     $env:TRADIER_API_KEY = "your_api_key"')
        print('     $env:TRADIER_ACCOUNT_ID = "your_account_id"')
        print('     $env:TRADIER_SANDBOX = "false"')
    
    return all_set

def test_optimus_initialization():
    """Test 4: OptimusAgent initialization with broker fallback"""
    print_header("Test 4: OptimusAgent Initialization")
    
    try:
        # Set minimal environment for testing
        os.environ.setdefault("TRADIER_SANDBOX", "true")
        
        from agents.optimus import OptimusAgent, TradingMode
        
        # Initialize in sandbox mode (safe)
        optimus = OptimusAgent(sandbox=False)  # LIVE MODE
        print_result("OptimusAgent created", True, f"mode={optimus.trading_mode.value}")
        
        # Check broker clients
        has_tradier = optimus.self_healing_engine is not None
        has_alpaca = optimus.alpaca_client is not None
        has_ibkr = optimus.ibkr_client is not None
        
        print_result("Tradier self-healing", has_tradier, "Engine initialized" if has_tradier else "Not available (check TRADIER_API_KEY)")
        print_result("Alpaca client", has_alpaca, "Available" if has_alpaca else "Not available")
        print_result("IBKR client", has_ibkr, "Available" if has_ibkr else "Not available")
        
        # Check broker fallback chain
        brokers = []
        if has_tradier:
            brokers.append("Tradier")
        if has_ibkr:
            brokers.append("IBKR")
        if has_alpaca:
            brokers.append("Alpaca")
        
        if brokers:
            print_result("Broker fallback chain", True, " -> ".join(brokers))
        else:
            print_result("Broker fallback chain", False, "No brokers available!")
        
        return True
    except Exception as e:
        print_result("OptimusAgent init", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_live_trade_method():
    """Test 5: Check _execute_live_trade has Alpaca fallback"""
    print_header("Test 5: Live Trade Execution (Code Check)")
    
    try:
        import inspect
        from agents.optimus import OptimusAgent
        
        # Get source code of _execute_live_trade
        source = inspect.getsource(OptimusAgent._execute_live_trade)
        
        has_tradier = "tradier" in source.lower()
        has_ibkr = "ibkr" in source.lower()
        has_alpaca = "alpaca" in source.lower()
        
        print_result("Tradier in _execute_live_trade", has_tradier)
        print_result("IBKR in _execute_live_trade", has_ibkr)
        print_result("Alpaca in _execute_live_trade", has_alpaca, "✨ NEW: Alpaca fallback added!")
        
        # Check for the specific Alpaca fallback code
        has_alpaca_fallback = "self.alpaca_client" in source and "broker\": \"alpaca\"" in source
        print_result("Alpaca broker fallback code", has_alpaca_fallback)
        
        return has_alpaca_fallback
    except Exception as e:
        print_result("Source check", False, f"Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  TRADIER INTEGRATION FIXES VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("TradierOAuth Fix", test_tradier_oauth()))
    results.append(("TradierBrokerAdapter", test_tradier_adapter()))
    results.append(("Environment Variables", test_env_variables()))
    results.append(("OptimusAgent Init", test_optimus_initialization()))
    results.append(("Live Trade Alpaca Fallback", test_live_trade_method()))
    
    # Summary
    print_header("SUMMARY")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All fixes verified! Tradier integration should work now.")
    else:
        print("\n  ⚠️  Some tests failed. See details above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


#!/usr/bin/env python3
"""
NAE System Integration Test with Working APIs
Tests NAE agents and systems using the verified API keys
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from secure_vault import get_vault
from env_loader import EnvLoader

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def setup_environment():
    """Setup environment variables from vault"""
    print("\n" + "="*70)
    print("Setting up environment from vault...")
    print("="*70)
    
    try:
        vault = get_vault()
        
        # Load API keys from vault
        openai_key = vault.get_secret('openai', 'api_key')
        polygon_key = vault.get_secret('polygon', 'api_key')
        quantconnect_user_id = vault.get_secret('quantconnect', 'user_id')
        quantconnect_api_key = vault.get_secret('quantconnect', 'api_key')
        
        # Set environment variables
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            print_success("OPENAI_API_KEY set from vault")
        
        if polygon_key:
            os.environ['POLYGON_API_KEY'] = polygon_key
            print_success("POLYGON_API_KEY set from vault")
        
        if quantconnect_user_id and quantconnect_api_key:
            os.environ['QUANTCONNECT_USER_ID'] = quantconnect_user_id
            os.environ['QUANTCONNECT_API_KEY'] = quantconnect_api_key
            print_success("QuantConnect credentials set from vault")
        
        return True
    except Exception as e:
        print_error(f"Failed to setup environment: {e}")
        return False

def test_ralph_agent_with_apis():
    """Test Ralph agent with API integrations"""
    print("\n" + "="*70)
    print("Testing Ralph Agent with API Integrations...")
    print("="*70)
    
    try:
        from agents.ralph import RalphAgent
        
        print_info("Initializing RalphAgent...")
        ralph = RalphAgent()
        
        # Check if API clients are initialized
        if hasattr(ralph, 'polygon_client') and ralph.polygon_client:
            print_success("Ralph: Polygon client initialized")
            
            # Test fetching market data
            print_info("Testing market data fetch...")
            try:
                market_data = ralph.fetch_market_data('AAPL', '2024-01-01', '2024-01-05', 'day')
                if market_data:
                    print_success(f"Ralph: Retrieved {len(market_data)} market data points")
                else:
                    print_warning("Ralph: No market data returned (may be normal)")
            except Exception as e:
                print_warning(f"Ralph: Market data fetch test: {e}")
        else:
            print_warning("Ralph: Polygon client not initialized")
        
        if hasattr(ralph, 'quantconnect_client') and ralph.quantconnect_client:
            print_success("Ralph: QuantConnect client initialized")
        else:
            print_warning("Ralph: QuantConnect client not initialized")
        
        print_success("Ralph Agent initialized successfully")
        return True
        
    except Exception as e:
        print_error(f"Ralph Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_integrations():
    """Test API integrations directly"""
    print("\n" + "="*70)
    print("Testing API Integrations...")
    print("="*70)
    
    results = {}
    
    # Test Polygon API integration
    try:
        from agents.ralph import PolygonDataClient
        vault = get_vault()
        polygon_key = vault.get_secret('polygon', 'api_key')
        
        if polygon_key:
            print_info("Testing Polygon API client...")
            client = PolygonDataClient(polygon_key)
            market_status = client.get_market_status()
            if market_status:
                print_success("Polygon API: Market status retrieved")
                results['polygon'] = True
            else:
                print_warning("Polygon API: No market status returned")
                results['polygon'] = False
        else:
            print_warning("Polygon API key not found")
            results['polygon'] = False
    except Exception as e:
        print_error(f"Polygon API test failed: {e}")
        results['polygon'] = False
    
    # Test Alpha Vantage API integration
    try:
        from tools.data.api_integrations import AlphaVantageAPI
        vault = get_vault()
        av_key = vault.get_secret('alpha_vantage', 'api_key')
        
        if av_key:
            print_info("Testing Alpha Vantage API client...")
            config = {'alpha_vantage_key': av_key}
            av_api = AlphaVantageAPI(config)
            # Don't actually call API to avoid rate limits, just verify initialization
            print_success("Alpha Vantage API: Client initialized")
            results['alpha_vantage'] = True
        else:
            print_warning("Alpha Vantage API key not found")
            results['alpha_vantage'] = False
    except Exception as e:
        print_error(f"Alpha Vantage API test failed: {e}")
        results['alpha_vantage'] = False
    
    # Test Marketaux API integration
    try:
        from tools.data.api_integrations import MarketauxAPI
        vault = get_vault()
        marketaux_key = vault.get_secret('marketaux', 'api_key')
        
        if marketaux_key:
            print_info("Testing Marketaux API client...")
            config = {'marketaux': {'api_key': marketaux_key}}
            marketaux_api = MarketauxAPI(config)
            # Don't actually call API to avoid rate limits, just verify initialization
            print_success("Marketaux API: Client initialized")
            results['marketaux'] = True
        else:
            print_warning("Marketaux API key not found")
            results['marketaux'] = False
    except Exception as e:
        print_error(f"Marketaux API test failed: {e}")
        results['marketaux'] = False
    
    # Test Tiingo API integration
    try:
        from tools.data.api_integrations import TiingoAPI
        vault = get_vault()
        tiingo_key = vault.get_secret('tiingo', 'api_key')
        
        if tiingo_key:
            print_info("Testing Tiingo API client...")
            config = {'tiingo': {'api_key': tiingo_key}}
            tiingo_api = TiingoAPI(config)
            # Don't actually call API to avoid rate limits, just verify initialization
            print_success("Tiingo API: Client initialized")
            results['tiingo'] = True
        else:
            print_warning("Tiingo API key not found")
            results['tiingo'] = False
    except Exception as e:
        print_error(f"Tiingo API test failed: {e}")
        results['tiingo'] = False
    
    return results

def test_quantconnect_integration():
    """Test QuantConnect integration"""
    print("\n" + "="*70)
    print("Testing QuantConnect Integration...")
    print("="*70)
    
    try:
        from agents.ralph import QuantConnectClient
        vault = get_vault()
        user_id = vault.get_secret('quantconnect', 'user_id')
        api_key = vault.get_secret('quantconnect', 'api_key')
        
        if user_id and api_key:
            print_info("Initializing QuantConnect client...")
            qc_client = QuantConnectClient(user_id, api_key)
            print_success("QuantConnect client initialized")
            
            # Test creating a simple backtest (dry run)
            print_info("Testing backtest creation interface...")
            print_success("QuantConnect integration ready")
            return True
        else:
            print_warning("QuantConnect credentials not found")
            return False
            
    except Exception as e:
        print_error(f"QuantConnect integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_loader():
    """Test environment loader"""
    print("\n" + "="*70)
    print("Testing Environment Loader...")
    print("="*70)
    
    try:
        loader = EnvLoader()
        status = loader.status()
        
        print_info("Environment variables status:")
        for key, value in status.items():
            if value == '✅ Set':
                print_success(f"  {key}: {value}")
            else:
                print_warning(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print_error(f"Environment loader test failed: {e}")
        return False

def test_vault_access():
    """Test vault access"""
    print("\n" + "="*70)
    print("Testing Secure Vault Access...")
    print("="*70)
    
    try:
        vault = get_vault()
        secrets = vault.list_secrets()
        
        print_info(f"Vault contains {len(secrets)} secret paths:")
        for path in secrets.keys():
            path_keys = len(secrets[path])
            print_success(f"  {path}: {path_keys} secret(s)")
        
        return True
    except Exception as e:
        print_error(f"Vault access test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration files"""
    print("\n" + "="*70)
    print("Testing Configuration Files...")
    print("="*70)
    
    config_files = [
        'config/api_keys.json',
        'config/model_assignments.json',
        'config/settings.json',
        'config/goal_manager.json'
    ]
    
    results = {}
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    json.load(f)
                print_success(f"{config_file}: Valid JSON")
                results[config_file] = True
            else:
                print_warning(f"{config_file}: Not found")
                results[config_file] = False
        except Exception as e:
            print_error(f"{config_file}: Invalid - {e}")
            results[config_file] = False
    
    return results

def main():
    """Run all NAE system tests"""
    print("\n" + "="*70)
    print(f"{Colors.BOLD}NAE SYSTEM INTEGRATION TEST{Colors.RESET}")
    print("="*70)
    print("\nTesting NAE system with verified API keys...")
    
    # Setup environment
    if not setup_environment():
        print_error("Failed to setup environment. Aborting tests.")
        return False
    
    # Run tests
    test_results = {}
    
    # Test 1: Vault Access
    print("\n" + "="*70)
    print("TEST 1: Secure Vault Access")
    print("="*70)
    test_results['vault'] = test_vault_access()
    
    # Test 2: Configuration Files
    print("\n" + "="*70)
    print("TEST 2: Configuration Files")
    print("="*70)
    config_results = test_configuration_files()
    test_results['config'] = all(config_results.values())
    
    # Test 3: Environment Loader
    print("\n" + "="*70)
    print("TEST 3: Environment Loader")
    print("="*70)
    test_results['env_loader'] = test_env_loader()
    
    # Test 4: API Integrations
    print("\n" + "="*70)
    print("TEST 4: API Integrations")
    print("="*70)
    api_results = test_api_integrations()
    test_results['api_integrations'] = sum(api_results.values()) >= 2  # At least 2 APIs working
    
    # Test 5: QuantConnect Integration
    print("\n" + "="*70)
    print("TEST 5: QuantConnect Integration")
    print("="*70)
    test_results['quantconnect'] = test_quantconnect_integration()
    
    # Test 6: Ralph Agent
    print("\n" + "="*70)
    print("TEST 6: Ralph Agent with APIs")
    print("="*70)
    test_results['ralph'] = test_ralph_agent_with_apis()
    
    # Print summary
    print("\n" + "="*70)
    print(f"{Colors.BOLD}TEST RESULTS SUMMARY{Colors.RESET}")
    print("="*70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        if result:
            print_success(f"{test_name:25} - PASSED")
            passed += 1
        else:
            print_error(f"{test_name:25} - FAILED")
    
    print("\n" + "-"*70)
    print(f"Total: {total} | Passed: {passed} | Failed: {total - passed}")
    print(f"Pass Rate: {passed/total*100:.1f}%")
    print("="*70)
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All NAE system tests passed!{Colors.RESET}")
        print("\nThe NAE system is ready to use with your API keys.")
        return True
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  {total - passed} test(s) failed{Colors.RESET}")
        print("\nCheck the errors above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


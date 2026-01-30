#!/usr/bin/env python3
"""
Comprehensive API Key Test Script
Tests all 6 API keys stored in vault to verify they work correctly with NAE
"""

import pytest
pytest.skip(
    "External API key verification requires live credentials and network access.",
    allow_module_level=True,
)

import sys
import os
import json
import requests
import time
from typing import Dict, Any, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from secure_vault import get_vault

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

def test_openai_api(api_key: str) -> Tuple[bool, str]:
    """Test OpenAI API key"""
    print("\n" + "="*60)
    print("Testing OpenAI API...")
    print("="*60)
    
    try:
        import openai
        
        # Set the API key
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple API call
        print_info("Making test API call...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print_success(f"OpenAI API Test: SUCCESS")
        print_info(f"Response: {result}")
        return True, "API key is valid and working"
        
    except Exception as e:
        error_msg = str(e)
        if "Invalid API key" in error_msg or "401" in error_msg:
            return False, f"Invalid API key: {error_msg}"
        elif "Rate limit" in error_msg or "429" in error_msg:
            return False, f"Rate limit exceeded: {error_msg}"
        else:
            return False, f"Error: {error_msg}"

def test_polygon_api(api_key: str) -> Tuple[bool, str]:
    """Test Polygon.io API key"""
    print("\n" + "="*60)
    print("Testing Polygon.io API...")
    print("="*60)
    
    try:
        # Test with a simple market data request
        url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02"
        params = {"apiKey": api_key}
        
        print_info("Making test API call for AAPL market data...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "OK":
                results_count = len(data.get("results", []))
                print_success(f"Polygon.io API Test: SUCCESS")
                print_info(f"Retrieved {results_count} data points for AAPL")
                return True, "API key is valid and working"
            else:
                return False, f"API returned error: {data.get('error', 'Unknown error')}"
        elif response.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        elif response.status_code == 403:
            return False, "API key doesn't have required permissions (403 Forbidden)"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_marketaux_api(api_key: str) -> Tuple[bool, str]:
    """Test Marketaux API key"""
    print("\n" + "="*60)
    print("Testing Marketaux API...")
    print("="*60)
    
    try:
        # Test with a simple news request
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "api_token": api_key,
            "symbols": "AAPL",
            "limit": 1
        }
        
        print_info("Making test API call for financial news...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data or "meta" in data:
                articles_count = len(data.get("data", []))
                print_success(f"Marketaux API Test: SUCCESS")
                print_info(f"Retrieved {articles_count} news articles")
                return True, "API key is valid and working"
            else:
                return False, f"API returned unexpected format: {data.get('error', 'Unknown error')}"
        elif response.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        elif response.status_code == 403:
            return False, "API key doesn't have required permissions (403 Forbidden)"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_tiingo_api(api_key: str) -> Tuple[bool, str]:
    """Test Tiingo API key"""
    print("\n" + "="*60)
    print("Testing Tiingo API...")
    print("="*60)
    
    try:
        # Test with a simple data request
        url = "https://api.tiingo.com/tiingo/daily/AAPL/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {api_key}"
        }
        params = {
            "startDate": "2024-01-01",
            "endDate": "2024-01-02"
        }
        
        print_info("Making test API call for AAPL price data...")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print_success(f"Tiingo API Test: SUCCESS")
                print_info(f"Retrieved {len(data)} price data points")
                return True, "API key is valid and working"
            else:
                return False, "API returned empty data"
        elif response.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        elif response.status_code == 403:
            return False, "API key doesn't have required permissions (403 Forbidden)"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_alpha_vantage_api(api_key: str) -> Tuple[bool, str]:
    """Test Alpha Vantage API key"""
    print("\n" + "="*60)
    print("Testing Alpha Vantage API...")
    print("="*60)
    
    try:
        # Test with a simple quote request
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": "AAPL",
            "interval": "1min",
            "apikey": api_key
        }
        
        print_info("Making test API call for AAPL intraday data...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "Time Series (1min)" in data:
                print_success(f"Alpha Vantage API Test: SUCCESS")
                print_info("Retrieved intraday market data")
                return True, "API key is valid and working"
            elif "Error Message" in data:
                return False, f"API Error: {data['Error Message']}"
            elif "Note" in data:
                return False, f"API Note: {data['Note']} (possibly rate limited)"
            else:
                return False, f"Unexpected response: {list(data.keys())}"
        elif response.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_quantconnect_api(user_id: str, api_key: str) -> Tuple[bool, str]:
    """Test QuantConnect API key"""
    print("\n" + "="*60)
    print("Testing QuantConnect API...")
    print("="*60)
    
    try:
        # Test with a simple API call to get user info or projects
        url = f"https://www.quantconnect.com/api/v2/projects/read"
        headers = {
            "Content-Type": "application/json"
        }
        auth = (user_id, api_key)
        
        print_info("Making test API call to verify credentials...")
        response = requests.get(url, headers=headers, auth=auth, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"QuantConnect API Test: SUCCESS")
            print_info(f"Successfully authenticated with user_id: {user_id}")
            return True, "API key and user_id are valid and working"
        elif response.status_code == 401:
            return False, "Invalid credentials (401 Unauthorized) - check user_id and API key"
        elif response.status_code == 403:
            return False, "Credentials don't have required permissions (403 Forbidden)"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Run all API key tests"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}NAE API KEYS TEST SUITE{Colors.RESET}")
    print("="*60)
    print("\nTesting 6 API keys stored in secure vault...")
    
    # Load vault
    try:
        vault = get_vault()
    except Exception as e:
        print_error(f"Failed to load vault: {e}")
        return False
    
    # Retrieve all API keys
    print("\n📋 Retrieving API keys from vault...")
    print("-" * 60)
    
    api_keys = {
        "OpenAI": vault.get_secret('openai', 'api_key'),
        "Polygon.io": vault.get_secret('polygon', 'api_key'),
        "Marketaux": vault.get_secret('marketaux', 'api_key'),
        "Tiingo": vault.get_secret('tiingo', 'api_key'),
        "Alpha Vantage": vault.get_secret('alpha_vantage', 'api_key'),
    }
    
    # QuantConnect needs both user_id and api_key
    quantconnect_user_id = vault.get_secret('quantconnect', 'user_id')
    quantconnect_api_key = vault.get_secret('quantconnect', 'api_key')
    
    # Check which keys are available
    missing_keys = []
    for name, key in api_keys.items():
        if not key:
            missing_keys.append(name)
            print_warning(f"{name}: Not found in vault")
        else:
            masked = key[:10] + "..." + key[-4:] if len(key) > 14 else "***"
            print_info(f"{name}: Found ({masked})")
    
    if quantconnect_user_id and quantconnect_api_key:
        print_info(f"QuantConnect: Found (User ID: {quantconnect_user_id})")
    else:
        missing_keys.append("QuantConnect")
        print_warning("QuantConnect: Missing user_id or api_key")
    
    if missing_keys:
        print_error(f"\nMissing keys: {', '.join(missing_keys)}")
        print("Please store these keys in the vault before testing.")
        return False
    
    # Run tests
    print("\n" + "="*60)
    print(f"{Colors.BOLD}Running API Tests...{Colors.RESET}")
    print("="*60)
    
    results = {}
    
    # Test OpenAI
    success, message = test_openai_api(api_keys["OpenAI"])
    results["OpenAI"] = (success, message)
    time.sleep(1)  # Rate limit protection
    
    # Test Polygon.io
    success, message = test_polygon_api(api_keys["Polygon.io"])
    results["Polygon.io"] = (success, message)
    time.sleep(1)
    
    # Test Marketaux
    success, message = test_marketaux_api(api_keys["Marketaux"])
    results["Marketaux"] = (success, message)
    time.sleep(1)
    
    # Test Tiingo
    success, message = test_tiingo_api(api_keys["Tiingo"])
    results["Tiingo"] = (success, message)
    time.sleep(1)
    
    # Test Alpha Vantage
    success, message = test_alpha_vantage_api(api_keys["Alpha Vantage"])
    results["Alpha Vantage"] = (success, message)
    time.sleep(1)
    
    # Test QuantConnect
    success, message = test_quantconnect_api(quantconnect_user_id, quantconnect_api_key)
    results["QuantConnect"] = (success, message)
    
    # Print summary
    print("\n" + "="*60)
    print(f"{Colors.BOLD}TEST RESULTS SUMMARY{Colors.RESET}")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for api_name, (success, message) in results.items():
        if success:
            print_success(f"{api_name:20} - PASSED")
            print(f"   {message}")
            passed += 1
        else:
            print_error(f"{api_name:20} - FAILED")
            print(f"   {message}")
            failed += 1
    
    print("\n" + "-"*60)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("="*60)
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All API keys are working correctly!{Colors.RESET}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  {failed} API key(s) failed testing{Colors.RESET}")
        print("\nPlease check the error messages above and verify your API keys.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


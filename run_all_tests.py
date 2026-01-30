#!/usr/bin/env python3
"""
NAE Comprehensive Test Suite Runner
Runs all tests and prepares system for live trading
"""

import os
import sys
import subprocess
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

def run_test_file(test_file):
    """Run a single test file"""
    test_path = Path("tests") / test_file
    if not test_path.exists():
        print_warning(f"Test file not found: {test_file}")
        return False
    
    print_info(f"Running {test_file}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_success(f"{test_file} passed")
            return True
        else:
            print_error(f"{test_file} failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print_error(f"{test_file} timed out")
        return False
    except Exception as e:
        print_error(f"Error running {test_file}: {e}")
        return False

def check_alpaca_config():
    """Check Alpaca configuration for live trading"""
    print_header("Checking Alpaca Configuration")
    
    # Check api_keys.json
    api_keys_path = Path("config/api_keys.json")
    if api_keys_path.exists():
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
            alpaca = api_keys.get("alpaca", {})
            
            api_key = alpaca.get("api_key", "")
            api_secret = alpaca.get("api_secret", "")
            
            if api_key and api_secret:
                if "YOUR_" in api_key or "YOUR_" in api_secret:
                    print_warning("Alpaca credentials contain placeholders")
                    return False
                else:
                    print_success("Alpaca credentials found in api_keys.json")
                    
                    # Check if paper trading keys
                    if "PK" in api_key:  # Paper trading keys start with PK
                        print_warning("Paper trading keys detected (PK prefix)")
                        print_info("For live trading, you need LIVE keys (not PK)")
                        return False
                    else:
                        print_success("Live trading keys detected")
                        return True
            else:
                print_error("Alpaca credentials missing")
                return False
    else:
        print_error("api_keys.json not found")
        return False

def check_environment():
    """Check environment setup"""
    print_header("Checking Environment")
    
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 7:
        print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print_error(f"Python 3.7+ required, found {python_version.major}.{python_version.minor}")
        checks.append(False)
    
    # Check required packages
    package_checks = [
        ("alpaca-py", "alpaca"),
        ("redis", "redis"),
        ("pytest", "pytest"),
    ]
    
    for package_name, import_name in package_checks:
        try:
            __import__(import_name)
            print_success(f"Package {package_name} installed")
            checks.append(True)
        except ImportError:
            print_warning(f"Package {package_name} not installed (may be optional)")
            checks.append(True)  # Don't fail on optional packages
    
    return all(checks)

def prepare_live_trading():
    """Prepare system for live trading"""
    print_header("Preparing for Live Trading")
    
    # Update settings.json
    settings_path = Path("config/settings.json")
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Update trading settings
        settings["trading"]["live"]["enabled"] = True
        settings["trading"]["live"]["broker"] = "alpaca"
        settings["trading"]["live"]["requires_manual_approval"] = True
        settings["environment"] = "production"
        
        # Save updated settings
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        print_success("Updated settings.json for live trading")
    else:
        print_error("settings.json not found")
        return False
    
    # Verify Alpaca adapter
    try:
        from adapters.alpaca import AlpacaAdapter
        
        # Test with live credentials
        config = {
            "paper_trading": False  # Live trading
        }
        
        try:
            adapter = AlpacaAdapter(config)
            if adapter.auth():
                print_success("Alpaca adapter authenticated for live trading")
                return True
            else:
                print_error("Alpaca adapter authentication failed")
                return False
        except Exception as e:
            print_error(f"Alpaca adapter error: {e}")
            return False
    except ImportError as e:
        print_error(f"Could not import AlpacaAdapter: {e}")
        return False

def main():
    """Main test runner"""
    print_header("NAE Comprehensive Test Suite")
    
    # Change to NAE directory
    os.chdir(Path(__file__).parent)
    
    # Check environment
    if not check_environment():
        print_error("Environment checks failed")
        return 1
    
    # Check Alpaca config
    alpaca_ready = check_alpaca_config()
    
    # Run tests
    print_header("Running Test Suite")
    
    test_files = [
        "test_api_keys.py",
        "test_nae_system.py",
        "test_optimus_alpaca_paper.py",
        "test_feedback_loops.py",
        "test_timing_strategies.py",
        "test_pnl_tracking.py",
        "test_security_alerting.py",
        "test_legal_compliance_integration.py",
    ]
    
    results = {}
    for test_file in test_files:
        results[test_file] = run_test_file(test_file)
    
    # Summary
    print_header("Test Results Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_file, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status}: {test_file}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}\n")
    
    # Prepare for live trading if requested
    if alpaca_ready and passed == total:
        print_header("Preparing for Live Trading")
        
        response = input(f"{Colors.YELLOW}⚠️  Ready to enable live trading? (yes/no): {Colors.RESET}")
        if response.lower() == "yes":
            if prepare_live_trading():
                print_success("System prepared for live trading")
                print_warning("⚠️  Live trading is now ENABLED")
                print_warning("⚠️  Manual approval required for all trades")
                print_info("Monitor logs/optimus.log for trading activity")
            else:
                print_error("Failed to prepare for live trading")
        else:
            print_info("Live trading preparation skipped")
    else:
        print_warning("Cannot prepare for live trading:")
        if not alpaca_ready:
            print_warning("  - Alpaca configuration incomplete")
        if passed != total:
            print_warning("  - Some tests failed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())


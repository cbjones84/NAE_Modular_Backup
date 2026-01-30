#!/usr/bin/env python3
"""
NAE System Verification Script
Tests all components to ensure proper configuration
"""

import os
import json
import sys
import redis
import requests
from datetime import datetime

def test_redis_connection():
    """Test Redis connection and kill switch functionality"""
    print("🔍 Testing Redis connection...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
        
        # Test kill switch
        r.set('TRADING_ENABLED', 'true')
        state = r.get('TRADING_ENABLED')
        print(f"✅ Kill switch test: {state}")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_configuration_files():
    """Test configuration file structure"""
    print("\n🔍 Testing configuration files...")
    
    config_files = [
        'config/api_keys.json',
        'config/settings.json',
        'config/goal_manager.json',
        'config/sandbox_phases.json'
    ]
    
    all_valid = True
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                json.load(f)
            print(f"✅ {config_file} - Valid JSON")
        except FileNotFoundError:
            print(f"❌ {config_file} - File not found")
            all_valid = False
        except json.JSONDecodeError as e:
            print(f"❌ {config_file} - Invalid JSON: {e}")
            all_valid = False
    
    return all_valid

def test_python_dependencies():
    """Test Python package dependencies"""
    print("\n🔍 Testing Python dependencies...")
    
    required_packages = [
        'redis',
        'requests',
        'numpy',
        'pandas',
        'pyautogen'
    ]
    
    all_available = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_available = False
    
    return all_available

def test_agent_files():
    """Test agent file availability"""
    print("\n🔍 Testing agent files...")
    
    agent_files = [
        'agents/optimus.py',
        'agents/ralph.py',
        'redis_kill_switch.py',
        'production_deploy.py'
    ]
    
    all_available = True
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            print(f"✅ {agent_file} - Available")
        else:
            print(f"❌ {agent_file} - Not found")
            all_available = False
    
    return all_available

def test_docker_setup():
    """Test Docker and docker-compose availability"""
    print("\n🔍 Testing Docker setup...")
    
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker not available")
            return False
        
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose: {result.stdout.strip()}")
        else:
            print("❌ Docker Compose not available")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False

def test_api_key_placeholders():
    """Check if API keys need to be configured"""
    print("\n🔍 Checking API key configuration...")
    
    try:
        with open('config/api_keys.json', 'r') as f:
            api_config = json.load(f)
        
        needs_config = []
        for platform, config in api_config.items():
            if platform in ['polygon', 'quantconnect', 'interactive_brokers', 'alpaca']:
                for key, value in config.items():
                    if isinstance(value, str) and 'YOUR_' in value and 'HERE' in value:
                        needs_config.append(f"{platform}.{key}")
        
        if needs_config:
            print("⚠️  API keys need configuration:")
            for key in needs_config:
                print(f"   - {key}")
            print("\nPlease edit config/api_keys.json with your actual API keys")
            return False
        else:
            print("✅ API keys appear to be configured")
            return True
            
    except Exception as e:
        print(f"❌ Error checking API keys: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚀 NAE System Verification")
    print("=" * 50)
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Configuration Files", test_configuration_files),
        ("Python Dependencies", test_python_dependencies),
        ("Agent Files", test_agent_files),
        ("Docker Setup", test_docker_setup),
        ("API Key Configuration", test_api_key_placeholders)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! NAE system is ready for deployment.")
        print("\nNext steps:")
        print("1. Configure API keys in config/api_keys.json")
        print("2. Run: ./setup.sh")
        print("3. Test: docker-compose up -d")
        print("4. Deploy: python3 production_deploy.py --deploy")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("- Install missing Python packages: pip3 install -r requirements.txt")
        print("- Start Redis: sudo systemctl start redis-server")
        print("- Install Docker: https://docs.docker.com/get-docker/")
        print("- Configure API keys: Edit config/api_keys.json")

if __name__ == "__main__":
    main()

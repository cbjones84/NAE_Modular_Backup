#!/usr/bin/env python3
"""
Test Real Data Sources for Ralph
Tests web scrapers, API integrations, and data validation
"""

import os
import sys
import json
import time
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

def test_web_scrapers():
    """Test web scraping functionality"""
    print("🔍 Testing Web Scrapers...")
    
    try:
        from web_scrapers import TradingForumScraper
        
        scraper = TradingForumScraper()
        
        # Test Reddit scraping
        print("\n1. Testing Reddit scraping...")
        reddit_strategies = scraper.scrape_reddit_options()
        print(f"   Found {len(reddit_strategies)} Reddit strategies")
        
        if reddit_strategies:
            strategy = reddit_strategies[0]
            print(f"   Sample: {strategy.title[:50]}...")
            print(f"   Source: {strategy.source}")
            print(f"   Confidence: {strategy.confidence_score:.1f}")
            print(f"   Strategy Type: {strategy.strategy_type}")
        
        # Test TradingView scraping
        print("\n2. Testing TradingView scraping...")
        tv_strategies = scraper.scrape_tradingview_ideas()
        print(f"   Found {len(tv_strategies)} TradingView strategies")
        
        if tv_strategies:
            strategy = tv_strategies[0]
            print(f"   Sample: {strategy.title[:50]}...")
            print(f"   Source: {strategy.source}")
            print(f"   Confidence: {strategy.confidence_score:.1f}")
        
        # Test Seeking Alpha scraping
        print("\n3. Testing Seeking Alpha scraping...")
        sa_strategies = scraper.scrape_seeking_alpha()
        print(f"   Found {len(sa_strategies)} Seeking Alpha strategies")
        
        if sa_strategies:
            strategy = sa_strategies[0]
            print(f"   Sample: {strategy.title[:50]}...")
            print(f"   Source: {strategy.source}")
            print(f"   Confidence: {strategy.confidence_score:.1f}")
        
        print("✅ Web scrapers test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Web scrapers test failed: {e}")
        return False

def test_api_integrations():
    """Test API integrations"""
    print("\n🔗 Testing API Integrations...")
    
    try:
        from api_integrations import APIManager
        
        # Test with empty config (should handle gracefully)
        config = {
            'twitter': {'bearer_token': ''},
            'reddit': {'client_id': '', 'client_secret': ''},
            'discord': {'bot_token': ''},
            'alpha_vantage_key': ''
        }
        
        api_manager = APIManager(config)
        
        # Test getting strategies (should return empty list with no API keys)
        strategies = api_manager.get_all_strategies()
        print(f"   Retrieved {len(strategies)} strategies from APIs")
        
        # Test individual sources
        twitter_strategies = api_manager.get_strategies_by_source('twitter')
        reddit_strategies = api_manager.get_strategies_by_source('reddit')
        
        print(f"   Twitter strategies: {len(twitter_strategies)}")
        print(f"   Reddit strategies: {len(reddit_strategies)}")
        
        print("✅ API integrations test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ API integrations test failed: {e}")
        return False

def test_data_validation():
    """Test data validation"""
    print("\n📊 Testing Data Validation...")
    
    try:
        from strategy_validator import StrategyValidator, StrategyScorer
        
        validator = StrategyValidator()
        scorer = StrategyScorer()
        
        # Test strategy data
        test_strategies = [
            {
                'title': 'Iron Condor Strategy on SPY',
                'content': 'Sell call spread and put spread on SPY. Use 30 DTE options. Set stop loss at 50% of max profit. Monitor delta and theta.',
                'source': 'seeking_alpha',
                'author': 'trading_expert',
                'upvotes': 25,
                'comments': 8
            },
            {
                'title': 'YOLO AAPL Calls',
                'content': 'Buy calls on AAPL. No risk management. All in!',
                'source': 'reddit',
                'author': 'unknown',
                'upvotes': 5,
                'comments': 2
            },
            {
                'title': 'Covered Call Strategy',
                'content': 'Buy 100 shares of stock and sell call options. Use delta 0.3 calls. Set stop loss. Monitor earnings dates.',
                'source': 'tradingview',
                'author': 'options_trader',
                'upvotes': 15,
                'comments': 5
            }
        ]
        
        # Test validation
        print("   Testing strategy validation...")
        scored_strategies = scorer.score_strategies(test_strategies)
        
        print(f"   Validated {len(scored_strategies)} strategies:")
        for i, strategy in enumerate(scored_strategies, 1):
            validation = strategy['validation_result']
            print(f"     {i}. {strategy['title'][:30]}...")
            print(f"        Valid: {validation.is_valid}")
            print(f"        Confidence: {validation.confidence_score:.1f}")
            print(f"        Risk Score: {validation.risk_score:.1f}")
            print(f"        Quality: {validation.quality_score:.1f}")
            print(f"        Strategy Type: {validation.strategy_type}")
        
        print("✅ Data validation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False

def test_enhanced_ralph():
    """Test enhanced Ralph agent"""
    print("\n🤖 Testing Enhanced Ralph Agent...")
    
    try:
        from enhanced_ralph import EnhancedRalphAgent
        
        # Initialize enhanced Ralph
        enhanced_ralph = EnhancedRalphAgent()
        
        # Check status
        status = enhanced_ralph.get_real_data_status()
        print(f"   Real Data Enabled: {status['real_data_enabled']}")
        print(f"   Sources Available: {status['sources_available']}")
        
        # Test enhanced cycle
        print("   Running enhanced cycle...")
        result = enhanced_ralph.run_enhanced_cycle()
        
        print(f"   Results:")
        print(f"     Total Candidates: {result['total_candidates']}")
        print(f"     Validated: {result['validated_count']}")
        print(f"     Approved: {result['approved_count']}")
        print(f"     Real Data Enabled: {result['real_data_enabled']}")
        
        # Show top strategies
        if enhanced_ralph.strategy_database:
            print(f"   Top 3 Approved Strategies:")
            for i, strategy in enumerate(enhanced_ralph.strategy_database[:3], 1):
                print(f"     {i}. {strategy['name'][:40]}...")
                print(f"        Source: {strategy['source']}")
                print(f"        Trust Score: {strategy.get('trust_score', 0):.1f}")
        
        print("✅ Enhanced Ralph test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Ralph test failed: {e}")
        return False

def test_configuration():
    """Test configuration files"""
    print("\n⚙️ Testing Configuration...")
    
    try:
        # Test API keys config
        api_keys_path = os.path.join(os.path.dirname(__file__), 'config', 'api_keys.json')
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        
        print(f"   API Keys Config: ✅ Loaded")
        print(f"   Trading APIs: {len([k for k in api_keys.keys() if k in ['polygon', 'quantconnect', 'interactive_brokers', 'alpaca']])}")
        print(f"   Data APIs: {len([k for k in api_keys.keys() if k in ['twitter', 'reddit', 'discord', 'news_api', 'alpha_vantage']])}")
        
        # Test real data config
        real_data_path = os.path.join(os.path.dirname(__file__), 'config', 'real_data_config.json')
        with open(real_data_path, 'r') as f:
            real_data = json.load(f)
        
        print(f"   Real Data Config: ✅ Loaded")
        print(f"   Web Scraping Enabled: {real_data.get('web_scraping', {}).get('enabled', False)}")
        print(f"   API Integrations Enabled: {real_data.get('api_integrations', {}).get('enabled', False)}")
        print(f"   Data Validation Enabled: {real_data.get('data_validation', {}).get('enabled', False)}")
        
        print("✅ Configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Real Data Sources for Ralph")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Web Scrapers", test_web_scrapers),
        ("API Integrations", test_api_integrations),
        ("Data Validation", test_data_validation),
        ("Enhanced Ralph", test_enhanced_ralph)
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
        print("\n🎉 All tests passed! Ralph is ready for real data ingestion.")
        print("\nNext steps:")
        print("1. Configure API keys in config/api_keys.json")
        print("2. Configure real data sources in config/real_data_config.json")
        print("3. Run: python3 agents/enhanced_ralph.py")
        print("4. Monitor logs for real strategy ingestion")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip3 install -r requirements.txt")
        print("- Check configuration files")
        print("- Verify file paths and permissions")

if __name__ == "__main__":
    main()

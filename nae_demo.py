#!/usr/bin/env python3
"""
NAE Practical Demo - Using Working APIs
Demonstrates real-world usage of NAE agents with verified API keys
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Setup environment from vault
from secure_vault import get_vault
from env_loader import EnvLoader

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def demo_ralph_market_data():
    """Demonstrate Ralph agent fetching market data"""
    print_header("DEMO 1: Fetching Market Data with Ralph Agent")
    
    try:
        from agents.ralph import RalphAgent
        
        print_info("Initializing Ralph Agent...")
        ralph = RalphAgent()
        
        # Fetch market data for AAPL
        symbol = "AAPL"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        print_info(f"Fetching market data for {symbol} from {start_date} to {end_date}...")
        market_data = ralph.fetch_market_data(symbol, start_date, end_date, "day")
        
        if market_data:
            print_success(f"Retrieved {len(market_data)} data points")
            print_info("\nSample data points:")
            for i, data_point in enumerate(market_data[:5]):  # Show first 5
                print(f"  {i+1}. Date: {data_point.timestamp[:10]}, Close: ${data_point.close_price:.2f}, Volume: {data_point.volume:,}")
            
            # Calculate basic stats
            if len(market_data) > 0:
                closes = [d.close_price for d in market_data]
                avg_price = sum(closes) / len(closes)
                max_price = max(closes)
                min_price = min(closes)
                
                print_info("\n📊 Market Statistics:")
                print(f"  Average Close Price: ${avg_price:.2f}")
                print(f"  Highest Price: ${max_price:.2f}")
                print(f"  Lowest Price: ${min_price:.2f}")
                print(f"  Price Range: ${max_price - min_price:.2f}")
        else:
            print_warning("No market data returned")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False

def demo_real_time_price():
    """Demonstrate real-time price fetching"""
    print_header("DEMO 2: Real-Time Price Lookup")
    
    try:
        from agents.ralph import RalphAgent
        
        ralph = RalphAgent()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        print_info("Fetching real-time prices...")
        for symbol in symbols:
            price = ralph.get_real_time_price(symbol)
            if price:
                print_success(f"{symbol}: ${price:.2f}")
            else:
                print_warning(f"{symbol}: Price not available")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        return False

def demo_market_news():
    """Demonstrate market news fetching"""
    print_header("DEMO 3: Fetching Financial News")
    
    try:
        from tools.data.api_integrations import MarketauxAPI
        from secure_vault import get_vault
        
        vault = get_vault()
        marketaux_key = vault.get_secret('marketaux', 'api_key')
        
        if not marketaux_key:
            print_warning("Marketaux API key not found")
            return False
        
        config = {'marketaux_api_key': marketaux_key}
        marketaux = MarketauxAPI(config)
        
        print_info("Fetching latest financial news for AAPL...")
        news = marketaux.get_financial_news(['AAPL'], limit=5)
        
        if news:
            print_success(f"Retrieved {len(news)} news articles")
            print_info("\n📰 Latest News Headlines:")
            for i, article in enumerate(news[:5], 1):
                title = article.get('title', 'No title')
                source = article.get('source', 'Unknown')
                published = article.get('published_at', '')[:10] if article.get('published_at') else 'Unknown date'
                print(f"\n  {i}. {title}")
                print(f"     Source: {source} | Published: {published}")
        else:
            print_warning("No news articles returned")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False

def demo_market_analysis():
    """Demonstrate market analysis with multiple APIs"""
    print_header("DEMO 4: Comprehensive Market Analysis")
    
    try:
        from tools.data.api_integrations import AlphaVantageAPI, TiingoAPI
        from secure_vault import get_vault
        
        vault = get_vault()
        
        # Alpha Vantage analysis
        print_info("Analyzing market sentiment with Alpha Vantage...")
        av_key = vault.get_secret('alpha_vantage', 'api_key')
        if av_key:
            av_config = {'alpha_vantage_key': av_key}
            av_api = AlphaVantageAPI(av_config)
            print_success("Alpha Vantage API initialized")
            print_info("(Note: Full sentiment analysis requires API calls with rate limits)")
        else:
            print_warning("Alpha Vantage API key not found")
        
        # Tiingo analysis
        print_info("\nAnalyzing price data with Tiingo...")
        tiingo_key = vault.get_secret('tiingo', 'api_key')
        if tiingo_key:
            tiingo_config = {'tiingo': {'api_key': tiingo_key}}
            tiingo_api = TiingoAPI(tiingo_config)
            print_success("Tiingo API initialized")
        else:
            print_warning("Tiingo API key not found")
        
        print_info("\n✅ Market analysis tools ready for use")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        return False

def demo_quantconnect_readiness():
    """Demonstrate QuantConnect integration readiness"""
    print_header("DEMO 5: QuantConnect Backtesting Readiness")
    
    try:
        from agents.ralph import QuantConnectClient
        from secure_vault import get_vault
        
        vault = get_vault()
        user_id = vault.get_secret('quantconnect', 'user_id')
        api_key = vault.get_secret('quantconnect', 'api_key')
        
        if user_id and api_key:
            print_info("Initializing QuantConnect client...")
            qc_client = QuantConnectClient(user_id, api_key)
            print_success("QuantConnect client initialized")
            print_info(f"User ID: {user_id}")
            print_info("✅ Ready for backtest creation and execution")
            print_info("\nYou can now:")
            print("  - Create backtests")
            print("  - Run strategy simulations")
            print("  - Deploy strategies")
        else:
            print_warning("QuantConnect credentials not found")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        return False

def demo_interactive_menu():
    """Interactive menu for using NAE"""
    print_header("NAE INTERACTIVE DEMO MENU")
    
    demos = {
        '1': ('Fetch Market Data', demo_ralph_market_data),
        '2': ('Real-Time Prices', demo_real_time_price),
        '3': ('Financial News', demo_market_news),
        '4': ('Market Analysis', demo_market_analysis),
        '5': ('QuantConnect Status', demo_quantconnect_readiness),
        '6': ('Run All Demos', None),
        '0': ('Exit', None)
    }
    
    while True:
        print(f"\n{Colors.BOLD}Available Demos:{Colors.RESET}")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        
        choice = input(f"\n{Colors.CYAN}Select a demo (0-6): {Colors.RESET}").strip()
        
        if choice == '0':
            print_info("Exiting demo. Thank you for using NAE!")
            break
        elif choice == '6':
            print_info("Running all demos...")
            demo_ralph_market_data()
            demo_real_time_price()
            demo_market_news()
            demo_market_analysis()
            demo_quantconnect_readiness()
        elif choice in demos:
            name, func = demos[choice]
            if func:
                func()
            else:
                print_warning("Invalid selection")
        else:
            print_warning("Invalid choice. Please select 0-6.")

def main():
    """Main demo function"""
    print_header("NAE PRACTICAL DEMO - USING WORKING APIs")
    
    print_info("Setting up environment from vault...")
    try:
        # Setup environment
        loader = EnvLoader()
        vault = get_vault()
        
        # Verify API keys are available
        print_info("Verifying API keys...")
        keys_status = {
            'Polygon': vault.get_secret('polygon', 'api_key'),
            'Marketaux': vault.get_secret('marketaux', 'api_key'),
            'Tiingo': vault.get_secret('tiingo', 'api_key'),
            'Alpha Vantage': vault.get_secret('alpha_vantage', 'api_key'),
            'QuantConnect': vault.get_secret('quantconnect', 'api_key'),
        }
        
        available_keys = [name for name, key in keys_status.items() if key]
        print_success(f"Available APIs: {', '.join(available_keys)}")
        
        # Check if we should run interactively or all demos
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--all':
            # Run all demos automatically
            print_info("Running all demos automatically...")
            demo_ralph_market_data()
            demo_real_time_price()
            demo_market_news()
            demo_market_analysis()
            demo_quantconnect_readiness()
        else:
            # Run interactive menu
            demo_interactive_menu()
        
    except Exception as e:
        print(f"{Colors.RED}❌ Setup failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()


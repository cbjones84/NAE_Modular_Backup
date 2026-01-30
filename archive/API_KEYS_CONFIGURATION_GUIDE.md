# 🔑 Complete API Keys Configuration Guide

## **Step-by-Step API Key Setup**

### **📋 Overview**
You need to configure API keys in `config/api_keys.json` to enable Ralph's full learning capabilities. Here's how to get each API key and configure them.

---

## **🐦 Step 1: Twitter API Setup**

### **Get Twitter API Access:**
1. **Go to**: https://developer.twitter.com/
2. **Sign up** with your Twitter account
3. **Apply for developer access** (usually approved in 1-3 days)
4. **Create a new app**:
   - App Name: "NAE Trading Bot"
   - Description: "Neural Agency Engine trading strategy learning bot"
   - Use Case: "Educational/Research"
5. **Get your credentials**:
   - Bearer Token (for API v2)
   - API Key and Secret (for user context)

### **Configure Twitter Keys:**
```json
{
  "twitter": {
    "bearer_token": "AAAAAAAAAAAAAAAAAAAAAF7opgEAAAAA0%2BuSeid%2BULvsea4JtiGRiSDSJSI%3DEUifiRBkKG3E2XzMDjRfl76ZC9Ub0wnz4XsNiRVBChTYbJcE3F",
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here",
    "rate_limit": 300,
    "description": "Twitter API for trading insights and sentiment"
  }
}
```

---

## **🔴 Step 2: Reddit API Setup**

### **Get Reddit API Access:**
1. **Go to**: https://www.reddit.com/prefs/apps
2. **Click "Create App"** or "Create Another App"
3. **Fill in details**:
   - Name: "NAE Trading Bot"
   - App type: "script"
   - Description: "Neural Agency Engine trading strategy learning bot"
   - Redirect URI: `http://localhost:8080`
4. **Get credentials**:
   - Client ID (under the app name)
   - Secret (next to "secret")

### **Configure Reddit Keys:**
```json
{
  "reddit": {
    "client_id": "your_client_id_here",
    "client_secret": "your_client_secret_here",
    "user_agent": "NAE Trading Bot 1.0",
    "rate_limit": 60,
    "description": "Reddit API for trading discussions and strategies"
  }
}
```

---

## **📰 Step 3: News API Setup**

### **Get News API Access:**
1. **Go to**: https://newsapi.org/
2. **Click "Get API Key"**
3. **Sign up** with your email
4. **Verify your email**
5. **Copy your API key** from the dashboard

### **Configure News API Key:**
```json
{
  "news_api": {
    "api_key": "your_news_api_key_here",
    "rate_limit": 1000,
    "description": "News API for financial news and market insights"
  }
}
```

---

## **📊 Step 4: Alpha Vantage API Setup**

### **Get Alpha Vantage API Access:**
1. **Go to**: https://www.alphavantage.co/support/#api-key
2. **Click "Get Free API Key"**
3. **Fill in the form**:
   - First Name: Your first name
   - Last Name: Your last name
   - Email: Your email address
   - Purpose: "Educational/Research"
4. **Submit and check email** for your API key

### **Configure Alpha Vantage Key:**
```json
{
  "alpha_vantage": {
    "api_key": "your_alpha_vantage_key_here",
    "rate_limit": 5,
    "description": "Alpha Vantage API for market data and news sentiment"
  }
}
```

---

## **⚙️ Step 5: Update Configuration File**

### **Edit `config/api_keys.json`:**
```bash
# Open the file in your editor
nano config/api_keys.json
# or
code config/api_keys.json
# or
vim config/api_keys.json
```

### **Replace the placeholder values:**
```json
{
  "polygon": {
    "api_key": "YOUR_POLYGON_API_KEY_HERE",
    "base_url": "https://api.polygon.io",
    "rate_limit": 1000,
    "description": "Polygon.io market data API for real-time and historical data"
  },
  "quantconnect": {
    "user_id": "YOUR_QUANTCONNECT_USER_ID_HERE",
    "api_key": "YOUR_QUANTCONNECT_API_KEY_HERE",
    "base_url": "https://www.quantconnect.com/api/v2",
    "description": "QuantConnect API for backtesting and live deployment"
  },
  "interactive_brokers": {
    "api_key": "YOUR_IBKR_API_KEY_HERE",
    "api_secret": "YOUR_IBKR_API_SECRET_HERE",
    "paper_trading_url": "https://api.ibkr.com/v1/paper",
    "live_trading_url": "https://api.ibkr.com/v1",
    "rate_limit": 200,
    "description": "Interactive Brokers API for live trading execution"
  },
  "alpaca": {
    "api_key": "YOUR_ALPACA_API_KEY_HERE",
    "api_secret": "YOUR_ALPACA_API_SECRET_HERE",
    "paper_trading_url": "https://paper-api.alpaca.markets",
    "live_trading_url": "https://api.alpaca.markets",
    "rate_limit": 200,
    "description": "Alpaca API for paper and live trading"
  },
  "twitter": {
    "bearer_token": "YOUR_ACTUAL_TWITTER_BEARER_TOKEN",
    "api_key": "YOUR_ACTUAL_TWITTER_API_KEY",
    "api_secret": "YOUR_ACTUAL_TWITTER_API_SECRET",
    "rate_limit": 300,
    "description": "Twitter API for trading insights and sentiment"
  },
  "reddit": {
    "client_id": "YOUR_ACTUAL_REDDIT_CLIENT_ID",
    "client_secret": "YOUR_ACTUAL_REDDIT_CLIENT_SECRET",
    "user_agent": "NAE Trading Bot 1.0",
    "rate_limit": 60,
    "description": "Reddit API for trading discussions and strategies"
  },
  "discord": {
    "bot_token": "YOUR_DISCORD_BOT_TOKEN_HERE",
    "rate_limit": 50,
    "description": "Discord API for trading channel monitoring"
  },
  "news_api": {
    "api_key": "YOUR_ACTUAL_NEWS_API_KEY",
    "rate_limit": 1000,
    "description": "News API for financial news and market insights"
  },
  "alpha_vantage": {
    "api_key": "YOUR_ACTUAL_ALPHA_VANTAGE_KEY",
    "rate_limit": 5,
    "description": "Alpha Vantage API for market data and news sentiment"
  },
  "environment": "sandbox",
  "last_updated": "2025-01-27T00:00:00Z",
  "notes": "Replace placeholder values with actual API keys. Keep this file secure and never commit real keys to version control."
}
```

---

## **🧪 Step 6: Test Your Configuration**

### **Test Individual APIs:**
```bash
# Test Twitter API
python3 -c "
from tools.data.api_integrations import TwitterAPI, APISource
config = APISource(name='twitter', api_key='YOUR_BEARER_TOKEN')
api = TwitterAPI(config)
tweets = api.get_trading_tweets(['#options'], max_results=5)
print(f'Twitter: {len(tweets)} tweets')
"

# Test Reddit API
python3 -c "
from tools.data.api_integrations import RedditAPI, APISource
config = APISource(name='reddit', api_key='YOUR_CLIENT_ID', api_secret='YOUR_CLIENT_SECRET')
api = RedditAPI(config)
posts = api.get_trading_posts(['options'], limit=5)
print(f'Reddit: {len(posts)} posts')
"

# Test News API
python3 -c "
from tools.data.api_integrations import NewsAPI
api = NewsAPI({'news_api_key': 'YOUR_NEWS_API_KEY'})
news = api.get_financial_news(['SPY'], limit=5)
print(f'News: {len(news)} articles')
"

# Test Alpha Vantage API
python3 -c "
from tools.data.api_integrations import AlphaVantageAPI
api = AlphaVantageAPI({'alpha_vantage_key': 'YOUR_ALPHA_VANTAGE_KEY'})
data = api.get_market_sentiment(['SPY'])
print(f'Alpha Vantage: {len(data)} symbols')
"
```

### **Test All APIs Together:**
```bash
# Run comprehensive test
python3 test_all_api_integrations.py
```

### **Test Enhanced Ralph:**
```bash
# Test Ralph with all APIs
python3 agents/enhanced_ralph.py
```

---

## **🔒 Security Best Practices**

### **Keep API Keys Secure:**
1. **Never commit real keys** to version control
2. **Use environment variables** for production:
   ```bash
   export TWITTER_BEARER_TOKEN="your_token_here"
   export REDDIT_CLIENT_ID="your_client_id_here"
   # etc.
   ```
3. **Add to .gitignore**:
   ```bash
   echo "config/api_keys.json" >> .gitignore
   ```
4. **Use separate keys** for development and production

### **Rate Limiting:**
- **Twitter**: 300 requests per 15 minutes
- **Reddit**: 60 requests per minute
- **News API**: 1,000 requests per day (free tier)
- **Alpha Vantage**: 5 requests per minute (free tier)

---

## **🚀 Step 7: Run Ralph with Real Data**

### **Start Enhanced Learning:**
```bash
# Run Ralph with all APIs enabled
python3 agents/enhanced_ralph.py

# Or run continuously
python3 -c "
from agents.enhanced_ralph import EnhancedRalphAgent
import time
ralph = EnhancedRalphAgent()
print('🤖 Ralph is now learning from real data sources!')
while True:
    result = ralph.run_cycle()
    print(f'Cycle complete: {result.get(\"approved_candidates\", 0)} strategies approved')
    time.sleep(3600)  # Run every hour
"
```

---

## **📊 Expected Results**

### **With All APIs Configured:**
- **Twitter**: 10-50 trading tweets per cycle
- **Reddit**: 25-100 trading posts per cycle
- **News**: 20-50 financial articles per cycle
- **Alpha Vantage**: 4-10 market analyses per cycle
- **Total**: 50-200+ strategies per cycle

### **Monitoring:**
- Check `logs/ralph_enhanced_strategies_*.json` for approved strategies
- Monitor console output for real-time progress
- Review sentiment analysis and engagement scores

---

## **❓ Troubleshooting**

### **Common Issues:**
1. **401 Unauthorized**: Check API key format and permissions
2. **Rate Limit Exceeded**: Wait and retry, or upgrade API tier
3. **No Data Returned**: Verify API keys and check API status
4. **Import Errors**: Ensure all dependencies are installed

### **Get Help:**
- Check API documentation for each service
- Review error logs in console output
- Test individual APIs before running full system

---

## **🎉 Success!**

Once configured, Ralph will:
- ✅ Learn from real Twitter trading discussions
- ✅ Validate strategies through Reddit community engagement
- ✅ Analyze financial news for market sentiment
- ✅ Use technical indicators for market analysis
- ✅ Combine all sources for comprehensive strategy evaluation

**Ralph's learning capabilities will be maximized!** 🚀

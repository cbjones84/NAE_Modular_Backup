# News API Setup Guide for NAE Ralph

## **News API Integration**

### **Step 1: Get News API Access**

1. **Sign up for News API**
   - Go to: https://newsapi.org/
   - Click "Get API Key"
   - Sign up with your email
   - Verify your email address

2. **Get API Key**
   - Log into your News API dashboard
   - Copy your API key
   - Note: Free tier allows 1,000 requests per day

### **Step 2: Configure API Keys**

Update `config/api_keys.json`:

```json
{
  "news_api": {
    "api_key": "YOUR_NEWS_API_KEY_HERE",
    "rate_limit": 1000,
    "description": "News API for financial news and market insights"
  }
}
```

### **Step 3: Test News API Integration**

```bash
# Test News API connection
python3 -c "
from tools.data.api_integrations import NewsAPI
news = NewsAPI({'news_api_key': 'YOUR_API_KEY'})
articles = news.get_financial_news(['SPY', 'AAPL'], limit=5)
print(f'Fetched {len(articles)} news articles')
"
```

### **Step 4: Financial News Sources**

- **Bloomberg** - Professional financial news
- **Reuters** - Global financial coverage
- **MarketWatch** - Market analysis
- **CNBC** - Business news
- **Financial Times** - International finance
- **Wall Street Journal** - Premium financial news
- **Yahoo Finance** - Market updates
- **Seeking Alpha** - Investment analysis

### **Step 5: Trading Keywords to Monitor**

- `earnings`
- `options trading`
- `volatility`
- `market analysis`
- `stock market`
- `trading strategy`
- `SPY`
- `QQQ`
- `NASDAQ`
- `S&P 500`

### **Step 6: Rate Limits**

- **Free Tier**: 1,000 requests per day
- **Developer Tier**: 10,000 requests per day
- **Business Tier**: 50,000 requests per day

### **Step 7: Enhanced News Integration**

The system will automatically:
- Monitor financial news sources
- Extract trading-relevant articles
- Analyze market sentiment
- Identify trading opportunities
- Store market insights

---

## **Benefits for Ralph**

✅ **Real-time market news**  
✅ **Earnings announcements**  
✅ **Market sentiment analysis**  
✅ **Volatility triggers**  
✅ **Economic indicators**  

---

*Setup completed: News API integration ready for Ralph's learning*

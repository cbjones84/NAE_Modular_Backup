# Alpha Vantage API Setup Guide for NAE Ralph

## **Alpha Vantage API Integration**

### **Step 1: Get Alpha Vantage API Access**

1. **Sign up for Alpha Vantage**
   - Go to: https://www.alphavantage.co/support/#api-key
   - Click "Get Free API Key"
   - Fill in the form:
     - First Name: Your first name
     - Last Name: Your last name
     - Email: Your email address
     - Purpose: "Educational/Research"
   - Submit the form

2. **Get API Key**
   - Check your email for the API key
   - Copy the API key (starts with letters/numbers)
   - Note: Free tier allows 5 API calls per minute, 500 per day

### **Step 2: Configure API Keys**

Update `config/api_keys.json`:

```json
{
  "alpha_vantage": {
    "api_key": "YOUR_ALPHA_VANTAGE_KEY_HERE",
    "rate_limit": 5,
    "description": "Alpha Vantage API for market data and news sentiment"
  }
}
```

### **Step 3: Test Alpha Vantage Integration**

```bash
# Test Alpha Vantage API connection
python3 -c "
from tools.data.api_integrations import AlphaVantageAPI
av = AlphaVantageAPI({'alpha_vantage_key': 'YOUR_API_KEY'})
data = av.get_market_sentiment(['SPY', 'AAPL'])
print(f'Fetched market data for {len(data)} symbols')
"
```

### **Step 4: Available Data Types**

- **Stock Time Series**: Daily, weekly, monthly prices
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **News Sentiment**: News sentiment analysis
- **Economic Indicators**: GDP, inflation, unemployment
- **Crypto Data**: Bitcoin, Ethereum prices
- **Forex Data**: Currency exchange rates

### **Step 5: Trading Symbols to Monitor**

- **ETFs**: SPY, QQQ, IWM, VIX
- **Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **Sectors**: XLK, XLF, XLE, XLI, XLV
- **Crypto**: BTC, ETH, ADA, SOL

### **Step 6: Rate Limits**

- **Free Tier**: 5 calls per minute, 500 per day
- **Premium Tier**: 25 calls per minute, 1,200 per day
- **Best Practice**: Use 12-second delays between calls

### **Step 7: Enhanced Alpha Vantage Integration**

The system will automatically:
- Monitor market sentiment
- Track technical indicators
- Analyze news sentiment
- Identify volatility patterns
- Store market insights

---

## **Benefits for Ralph**

✅ **Real-time market data**  
✅ **Technical analysis indicators**  
✅ **News sentiment scoring**  
✅ **Volatility measurements**  
✅ **Economic indicators**  

---

*Setup completed: Alpha Vantage API integration ready for Ralph's learning*

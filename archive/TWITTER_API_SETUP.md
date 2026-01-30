# Twitter API Setup Guide for NAE Ralph

## **Twitter API v2 Integration**

### **Step 1: Get Twitter API Access**

1. **Apply for Twitter Developer Account**
   - Go to: https://developer.twitter.com/
   - Sign up with your Twitter account
   - Complete the application form
   - Wait for approval (usually 1-3 days)

2. **Create a Twitter App**
   - Go to: https://developer.twitter.com/en/portal/dashboard
   - Click "Create App"
   - Fill in app details:
     - App Name: "NAE Trading Bot"
     - Description: "Neural Agency Engine trading strategy learning bot"
     - Website: Your website or GitHub repo
     - Use Case: "Educational/Research"

3. **Get API Keys**
   - Go to your app's "Keys and Tokens" tab
   - Generate API Key and Secret
   - Generate Bearer Token
   - Generate Access Token and Secret (for user context)

### **Step 2: Configure API Keys**

Update `config/api_keys.json`:

```json
{
  "twitter": {
    "bearer_token": "YOUR_BEARER_TOKEN_HERE",
    "api_key": "YOUR_API_KEY_HERE", 
    "api_secret": "YOUR_API_SECRET_HERE",
    "access_token": "YOUR_ACCESS_TOKEN_HERE",
    "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET_HERE",
    "rate_limit": 300,
    "description": "Twitter API for trading insights and sentiment"
  }
}
```

### **Step 3: Test Twitter Integration**

```bash
# Test Twitter API connection
python3 -c "
from tools.data.api_integrations import APIIntegrations
api = APIIntegrations()
tweets = api.fetch_twitter_tweets('#options #trading', count=5)
print(f'Fetched {len(tweets)} tweets')
"
```

### **Step 4: Trading Hashtags to Monitor**

- `#options`
- `#trading`
- `#stockmarket`
- `#optionsTrading`
- `#daytrading`
- `#investing`
- `#SPY`
- `#QQQ`
- `#earnings`
- `#volatility`

### **Step 5: Rate Limits**

- **Tweet Lookup**: 300 requests per 15 minutes
- **User Lookup**: 300 requests per 15 minutes
- **Search**: 300 requests per 15 minutes

### **Step 6: Enhanced Twitter Integration**

The system will automatically:
- Monitor trading hashtags
- Extract strategy mentions
- Analyze sentiment
- Validate trading signals
- Store quality insights

---

## **Benefits for Ralph**

✅ **Real-time trading sentiment**  
✅ **Strategy mentions from traders**  
✅ **Market reaction analysis**  
✅ **Earnings call insights**  
✅ **Volatility discussions**  

---

*Setup completed: Twitter API integration ready for Ralph's learning*

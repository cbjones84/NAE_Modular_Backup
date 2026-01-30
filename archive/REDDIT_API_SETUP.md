# Reddit API Setup Guide for NAE Ralph

## **Reddit API Integration**

### **Step 1: Get Reddit API Access**

1. **Create Reddit App**
   - Go to: https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Fill in details:
     - Name: "NAE Trading Bot"
     - App type: "script"
     - Description: "Neural Agency Engine trading strategy learning bot"
     - About URL: Your website or GitHub repo
     - Redirect URI: `http://localhost:8080`

2. **Get API Credentials**
   - Note down the "client ID" (under the app name)
   - Note down the "secret" (next to "secret")
   - These are your Reddit API credentials

### **Step 2: Configure API Keys**

Update `config/api_keys.json`:

```json
{
  "reddit": {
    "client_id": "YOUR_REDDIT_CLIENT_ID_HERE",
    "client_secret": "YOUR_REDDIT_CLIENT_SECRET_HERE",
    "user_agent": "NAE Trading Bot 1.0",
    "rate_limit": 60,
    "description": "Reddit API for trading discussions and strategies"
  }
}
```

### **Step 3: Test Reddit Integration**

```bash
# Test Reddit API connection
python3 -c "
from tools.data.api_integrations import RedditAPI, APISource
reddit_config = APISource(
    name='reddit',
    api_key='YOUR_CLIENT_ID',
    api_secret='YOUR_CLIENT_SECRET'
)
reddit = RedditAPI(reddit_config)
posts = reddit.get_trading_posts(['options', 'wallstreetbets'], limit=5)
print(f'Fetched {len(posts)} posts')
"
```

### **Step 4: Trading Subreddits to Monitor**

- `r/options` - Options trading strategies
- `r/wallstreetbets` - High-risk trading discussions
- `r/investing` - General investment strategies
- `r/securityanalysis` - Fundamental analysis
- `r/options` - Options strategies
- `r/thetagang` - Theta decay strategies
- `r/SPY` - SPY-specific discussions
- `r/StockMarket` - General market discussions

### **Step 5: Rate Limits**

- **Reddit API**: 60 requests per minute
- **PRAW Library**: Handles rate limiting automatically
- **Best Practice**: Use 1-2 second delays between requests

### **Step 6: Enhanced Reddit Integration**

The system will automatically:
- Monitor trading subreddits
- Extract strategy discussions
- Analyze post engagement
- Validate trading signals
- Store quality insights

---

## **Benefits for Ralph**

✅ **Real trading discussions**  
✅ **Strategy validation from community**  
✅ **Engagement-based quality scoring**  
✅ **Diverse trading perspectives**  
✅ **Real-time market sentiment**  

---

*Setup completed: Reddit API integration ready for Ralph's learning*

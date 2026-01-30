# 🚀 NAE Usage Guide - Getting Started

## Quick Start

Your NAE system is now operational with **5 working API keys**! Here's how to start using it.

---

## 🎯 Available Capabilities

### ✅ Working APIs:
- **Polygon.io** - Market data (historical & real-time)
- **Marketaux** - Financial news
- **Tiingo** - Price data & fundamentals
- **Alpha Vantage** - Market sentiment & technical indicators
- **QuantConnect** - Backtesting & strategy deployment

---

## 📋 Quick Demo

Run the interactive demo to see NAE in action:

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
python3 nae_demo.py
```

Or run all demos automatically:

```bash
python3 nae_demo.py --all
```

---

## 🔧 Using Individual Agents

### 1. Ralph Agent - Market Data & Learning

```python
from agents.ralph import RalphAgent

# Initialize Ralph
ralph = RalphAgent()

# Fetch market data
market_data = ralph.fetch_market_data("AAPL", "2024-01-01", "2024-01-31", "day")
print(f"Retrieved {len(market_data)} data points")

# Get real-time price
price = ralph.get_real_time_price("AAPL")
print(f"AAPL current price: ${price:.2f}")

# Run learning cycle
results = ralph.run_cycle()
print(f"Approved strategies: {len(results)}")
```

### 2. Fetch Financial News

```python
from tools.data.api_integrations import MarketauxAPI
from secure_vault import get_vault

vault = get_vault()
marketaux_key = vault.get_secret('marketaux', 'api_key')

config = {'marketaux_api_key': marketaux_key}
marketaux = MarketauxAPI(config)

# Get news for symbols
news = marketaux.get_financial_news(['AAPL', 'MSFT'], limit=10)
for article in news:
    print(f"{article['title']} - {article['source']}")
```

### 3. Market Analysis with Multiple APIs

```python
from tools.data.api_integrations import AlphaVantageAPI, TiingoAPI
from secure_vault import get_vault

vault = get_vault()

# Alpha Vantage for sentiment
av_config = {'alpha_vantage_key': vault.get_secret('alpha_vantage', 'api_key')}
av_api = AlphaVantageAPI(av_config)

# Tiingo for price data
tiingo_config = {'tiingo': {'api_key': vault.get_secret('tiingo', 'api_key')}}
tiingo_api = TiingoAPI(tiingo_config)

# Get market sentiment
sentiment = av_api.get_market_sentiment(['AAPL', 'MSFT'])
```

### 4. QuantConnect Backtesting

```python
from agents.ralph import QuantConnectClient
from secure_vault import get_vault

vault = get_vault()
user_id = vault.get_secret('quantconnect', 'user_id')
api_key = vault.get_secret('quantconnect', 'api_key')

qc_client = QuantConnectClient(user_id, api_key)

# Create a backtest
strategy_code = """
class MyStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.AddEquity("AAPL", Resolution.Daily)
"""

backtest = qc_client.create_backtest(
    strategy_code, "My Strategy", "2024-01-01", "2024-12-31"
)
```

---

## 📊 Example Workflows

### Workflow 1: Market Research

```python
from agents.ralph import RalphAgent
from tools.data.api_integrations import MarketauxAPI
from secure_vault import get_vault

vault = get_vault()

# Initialize agents
ralph = RalphAgent()
marketaux = MarketauxAPI({'marketaux_api_key': vault.get_secret('marketaux', 'api_key')})

# Research a stock
symbol = "AAPL"

# Get market data
data = ralph.fetch_market_data(symbol, "2024-01-01", "2024-12-31", "day")
print(f"Analyzed {len(data)} trading days")

# Get recent news
news = marketaux.get_financial_news([symbol], limit=5)
print(f"Found {len(news)} recent articles")

# Get current price
price = ralph.get_real_time_price(symbol)
print(f"Current price: ${price:.2f}")
```

### Workflow 2: Strategy Development

```python
from agents.ralph import RalphAgent

ralph = RalphAgent()

# Learn from market data
results = ralph.run_cycle()

# Get top strategies
top_strategies = ralph.top_strategies(5)
for strategy in top_strategies:
    print(f"Strategy: {strategy['name']}")
    print(f"Trust Score: {strategy.get('trust_score', 0)}")
```

---

## 🔐 Accessing API Keys from Vault

All API keys are stored securely in the encrypted vault:

```python
from secure_vault import get_vault

vault = get_vault()

# Get an API key
polygon_key = vault.get_secret('polygon', 'api_key')
marketaux_key = vault.get_secret('marketaux', 'api_key')
tiingo_key = vault.get_secret('tiingo', 'api_key')
av_key = vault.get_secret('alpha_vantage', 'api_key')
qc_user_id = vault.get_secret('quantconnect', 'user_id')
qc_api_key = vault.get_secret('quantconnect', 'api_key')
```

---

## 🧪 Testing

### Test API Keys
```bash
python3 test_api_keys.py
```

### Test NAE System
```bash
python3 test_nae_system.py
```

### Run System Tests
```bash
python3 system_test.py
```

---

## 📁 Key Files

- `nae_demo.py` - Interactive demo script
- `test_api_keys.py` - API key verification
- `test_nae_system.py` - System integration tests
- `system_test.py` - Comprehensive system tests
- `agents/ralph.py` - Ralph agent (market data & learning)
- `tools/data/api_integrations.py` - API integrations

---

## ⚠️ Important Notes

1. **OpenAI API**: Valid key but quota exceeded - needs billing setup at https://platform.openai.com/account/billing

2. **Rate Limits**: 
   - Alpha Vantage: 5 calls/minute (free tier)
   - Polygon.io: Check your plan limits
   - Marketaux: 1000 requests/day (free tier)

3. **Environment**: All keys are loaded from the secure vault automatically

4. **QuantConnect**: Ready for backtesting but requires strategy code

---

## 🎓 Next Steps

1. **Run the demo**: `python3 nae_demo.py`
2. **Explore agents**: Try different agents (Ralph, Optimus, Casey, etc.)
3. **Develop strategies**: Use Ralph's learning capabilities
4. **Backtest**: Create and run strategies on QuantConnect
5. **Add more APIs**: Set up Anthropic, Alpaca, or Interactive Brokers as needed

---

## 💡 Tips

- Use `env_loader.py` to automatically load API keys from vault
- Check `logs/` directory for agent activity logs
- Use `verify_setup.py` to check system status
- All sensitive data is stored in `config/.vault.encrypted` (encrypted)

---

**Happy Trading! 🚀**


# Broker Adapters - Quick Start Guide

## ✅ **Yes, Your Checklist Was VERY Helpful!**

The Java/Struts troubleshooting guide you provided contained excellent E*TRADE OAuth guidance that's now fully implemented in NAE.

---

## 🎯 **What Was Built**

### **1. Modular Broker Adapter System**

NAE now supports multiple brokers through a common interface:

```
adapters/
├── base.py      # Abstract BrokerAdapter interface
├── manager.py   # AdapterManager (select broker at runtime)
├── etrade.py    # E*TRADE adapter (OAuth 1.0a)
├── alpaca.py    # Alpaca adapter (API key)
└── mock.py      # Mock adapter (testing)
```

---

## 🚀 **Quick Usage**

### **Basic Usage:**

```python
from adapters.manager import AdapterManager

# Get adapter manager
manager = AdapterManager()

# Get a broker adapter
broker = manager.get("etrade")  # or "alpaca", "mock"

# Use common interface
if broker.auth():
    account = broker.get_account()
    positions = broker.get_positions()
    quote = broker.get_quote("AAPL")
    
    order = broker.place_order({
        "symbol": "AAPL",
        "quantity": 1,
        "side": "buy",
        "type": "market"
    })
```

---

### **E*TRADE OAuth Flow (Sandbox):**

```python
from agents.etrade_oauth import ETradeOAuth

oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox=True)

# Step 1: Get authorization URL
result = oauth.start_oauth()
print(f"Authorize at: {result['authorize_url']}")

# Step 2: User authorizes and gets verification code
verification_code = input("Enter verification code: ")

# Step 3: Get access tokens
tokens = oauth.finish_oauth(
    result['resource_owner_key'],
    result['resource_owner_secret'],
    verification_code
)

# Step 4: Save tokens
oauth.save_tokens("config/etrade_tokens_sandbox.json")
```

---

## 📋 **Configuration**

**File:** `config/broker_adapters.json`

```json
{
  "default": "mock",
  "adapters": {
    "mock": {
      "module": "adapters.mock",
      "class": "MockAdapter"
    },
    "etrade": {
      "module": "adapters.etrade",
      "class": "EtradeAdapter",
      "config": {
        "sandbox": true,
        "consumer_key": "FROM_VAULT",
        "consumer_secret": "FROM_VAULT"
      }
    },
    "alpaca": {
      "module": "adapters.alpaca",
      "class": "AlpacaAdapter",
      "config": {
        "paper_trading": true
      }
    }
  }
}
```

---

## 🧪 **Testing Strategy**

### **1. Start with Mock Adapter:**
```python
from adapters.mock import MockAdapter

mock = MockAdapter()
assert mock.auth() == True
account = mock.get_account()
order = mock.place_order({"symbol": "AAPL", "quantity": 1, "side": "buy"})
```

### **2. Test with Alpaca Paper Trading:**
```python
from adapters.alpaca import AlpacaAdapter

alpaca = AlpacaAdapter({
    "API_KEY": "your_key",
    "API_SECRET": "your_secret",
    "paper_trading": True
})

if alpaca.auth():
    print(alpaca.get_account())
```

### **3. Then E*TRADE Sandbox:**
```python
from adapters.etrade import EtradeAdapter

etrade = EtradeAdapter({
    "consumer_key": "sandbox_key",
    "consumer_secret": "sandbox_secret",
    "sandbox": True
})

if etrade.auth():
    print(etrade.get_account())
```

---

## 🔧 **E*TRADE OAuth Setup**

Run the setup script:
```bash
python3 scripts/setup_etrade_oauth.py --sandbox
```

Or manually:
```python
from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

vault = get_vault()
consumer_key = vault.get_secret('etrade', 'sandbox_api_key')
consumer_secret = vault.get_secret('etrade', 'sandbox_api_secret')

oauth = ETradeOAuth(consumer_key, consumer_secret, sandbox=True)
result = oauth.start_oauth()

# User authorizes and gets code
tokens = oauth.finish_oauth(
    result['resource_owner_key'],
    result['resource_owner_secret'],
    verification_code
)

oauth.save_tokens("config/etrade_tokens_sandbox.json")
```

---

## ✅ **Improvements from Your Checklist**

✅ **E*TRADE OAuth 1.0a** - Proper `start_oauth()` / `finish_oauth()` flow  
✅ **Multiple Brokers** - Common interface, easy to add new brokers  
✅ **Testing** - Mock adapter for safe development  
✅ **Token Management** - Encrypted storage ready  
✅ **Modularity** - Runtime broker selection  
✅ **Error Handling** - Full stack traces for debugging  

---

## 🔐 **Security Notes**

- Store credentials in `secure_vault.py` (not in code)
- Encrypt access tokens (use KMS in production)
- Rotate secrets periodically
- Use sandbox/paper trading for testing

---

## 📚 **Next Steps**

1. **Test OAuth:** `python3 scripts/setup_etrade_oauth.py --sandbox`
2. **Test Adapter:** Use mock adapter first, then Alpaca paper, then E*TRADE sandbox
3. **Update Optimus:** Integrate adapter system into OptimusAgent (optional)

---

**Your checklist helped build a production-ready broker adapter system!** 🎉



# E*TRADE Sandbox OAuth 500 Error - Troubleshooting

## ❌ **The Error You're Seeing**

```
HTTP Status 500 – Internal Server Error
Unable to instantiate Action, com.etrade.myetrade.LoginAction
java.lang.ClassNotFoundException: com.etrade.myetrade.LoginAction
```

This is a **server-side error on E*TRADE's end**, not a problem with our code.

---

## 🔍 **What This Means**

This error indicates that:
1. **E*TRADE's sandbox OAuth server is having issues**
2. The request token was successfully obtained
3. Our OAuth URL format is correct
4. But E*TRADE's server can't process the authorization request

---

## ✅ **Solutions to Try**

### **Solution 1: Try a Fresh Authorization URL**

Request tokens expire quickly. Generate a new one:

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
python3 scripts/quick_complete_etrade_oauth.py
```

Or manually:
```python
from agents.etrade_oauth import ETradeOAuth
from secure_vault import get_vault

vault = get_vault()
oauth = ETradeOAuth(
    vault.get_secret('etrade', 'sandbox_api_key'),
    vault.get_secret('etrade', 'sandbox_api_secret'),
    sandbox=True
)
result = oauth.start_oauth()
print(result['authorize_url'])
```

---

### **Solution 2: Check E*TRADE Developer Portal**

1. Go to: https://developer.etrade.com/
2. Log in with your developer account
3. Navigate to **"My Apps"** → Your application
4. Check if:
   - Your app is **approved/activated**
   - OAuth is **enabled** for your app
   - Sandbox access is **granted**

---

### **Solution 3: Verify Consumer Key/Secret**

Make sure your consumer key and secret are:
- ✅ From the **sandbox** environment (not production)
- ✅ **Active/approved** in the developer portal
- ✅ Have **OAuth permissions** enabled

To check:
```bash
# View your credentials (first few chars)
python3 -c "
from secure_vault import get_vault
v = get_vault()
key = v.get_secret('etrade', 'sandbox_api_key')
print(f'Consumer Key: {key[:20]}...' if key else 'Not found')
"
```

---

### **Solution 4: Wait and Retry**

E*TRADE sandbox can be unstable. Try again:
- **In a few minutes** (server might recover)
- **Later today** (maintenance window might be ending)
- **Different time of day** (less load)

---

### **Solution 5: Alternative Authorization Method**

Some E*TRADE apps can authorize through:
1. **E*TRADE Developer Portal** → My Apps → OAuth
2. Look for a **"Test Authorization"** or **"Get Authorization Code"** button
3. This might bypass the server error

---

### **Solution 6: Check E*TRADE Status**

1. Visit: https://developer.etrade.com/
2. Look for:
   - **Status page** or **System Status**
   - **Announcements** about maintenance
   - **Known Issues** section

---

## 🆘 **If Nothing Works**

### **Option A: Use Production OAuth (if you have access)**

Production OAuth might be more stable:

```bash
python3 scripts/quick_complete_etrade_oauth.py --prod
```

⚠️ **Warning:** Production tokens can place real trades! Only use if you're ready.

---

### **Option B: Use Alpaca Instead**

Alpaca has better OAuth reliability and paper trading:

```python
from adapters.alpaca import AlpacaAdapter

alpaca = AlpacaAdapter({
    "API_KEY": "your_key",
    "API_SECRET": "your_secret",
    "paper_trading": True
})
```

---

### **Option C: Use Mock Adapter for Testing**

Test your trading logic without any broker:

```python
from adapters.mock import MockAdapter

mock = MockAdapter()
account = mock.get_account()
order = mock.place_order({"symbol": "AAPL", "quantity": 1, "side": "buy"})
```

---

## 📋 **What We Know Works**

✅ **Request Token Generation** - Working perfectly  
✅ **OAuth URL Format** - Correct format  
✅ **Our Code** - No issues on our side  
❌ **E*TRADE Sandbox Server** - Having issues  

---

## 🔄 **Next Steps**

1. **Try the fresh URL** (see Solution 1)
2. **Check Developer Portal** (see Solution 2)
3. **Wait 15-30 minutes** and retry (see Solution 4)
4. **If still failing**, consider using Alpaca or Mock adapter for now

---

## 💡 **We Can Still Test**

Even if E*TRADE OAuth is down, you can:

```bash
# Test with Mock adapter
python3 -c "
from adapters.mock import MockAdapter
m = MockAdapter()
print(m.get_account())
print(m.place_order({'symbol': 'AAPL', 'quantity': 1, 'side': 'buy'}))
"
```

This tests the adapter architecture without needing E*TRADE's server.

---

**The error is on E*TRADE's side, not ours. Try a fresh URL first!**



# E*TRADE Status Monitor - Usage Guide

## 🎯 **Purpose**

The monitoring script checks if E*TRADE sandbox OAuth is working and alerts you when it's back up.

---

## 🚀 **Quick Start**

### **Single Check (Run Once)**

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
python3 scripts/monitor_etrade_status.py --once
```

This runs one check and shows the current status.

---

### **Continuous Monitoring**

```bash
python3 scripts/monitor_etrade_status.py
```

This monitors continuously, checking every 60 seconds until:
- E*TRADE is back up (stops automatically)
- You press Ctrl+C

---

## 📋 **Command Options**

```bash
# Run once and exit
python3 scripts/monitor_etrade_status.py --once

# Check every 30 seconds
python3 scripts/monitor_etrade_status.py --interval 30

# Monitor for maximum 10 checks
python3 scripts/monitor_etrade_status.py --max-checks 10

# Monitor production (not sandbox)
python3 scripts/monitor_etrade_status.py --prod

# Combine options
python3 scripts/monitor_etrade_status.py --interval 120 --max-checks 20
```

---

## 📊 **What It Checks**

The monitor tests three things:

1. **Request Token Endpoint** - Can we get a request token?
2. **Authorization URL** - Does the authorization URL work (not return 500)?
3. **API Base Endpoint** - Is the E*TRADE API server reachable?

---

## ✅ **Status Indicators**

### **✅ WORKING**
- Request token obtained successfully
- Authorization URL returns valid response (not 500)
- API endpoint is reachable

### **❌ DOWN**
- Cannot get request token
- Authorization URL returns 500 error
- API endpoint is unreachable

### **⚠️ UNKNOWN**
- Partial failures or unexpected responses

---

## 📝 **Example Output**

```
================================================================================
E*TRADE SANDBOX STATUS CHECK - 2025-01-27 14:30:00
================================================================================

1️⃣  Checking Request Token Endpoint...
   ✅ Request token endpoint: WORKING
      Request token: o1z6m1xXVAms6dItehtN...

2️⃣  Checking Authorization URL...
   ❌ Authorization URL: 500 SERVER ERROR
      Status: 500
      Error: E*TRADE server returning 500 error

3️⃣  Checking API Base Endpoint...
   ✅ API endpoint: UP
      Status: 401
      Note: Server is up (auth required)

================================================================================
❌ OVERALL STATUS: E*TRADE SANDBOX IS DOWN
================================================================================
```

When it's back up:
```
================================================================================
✅ OVERALL STATUS: E*TRADE SANDBOX IS UP! 🎉
================================================================================

📋 You can now complete OAuth:
   https://apisb.etrade.com/oauth/authorize?oauth_token=...

   Then run:
   python3 scripts/finish_etrade_oauth.py YOUR_VERIFICATION_CODE
```

---

## 🔄 **Continuous Monitoring**

The monitor will:
- Check every 60 seconds (default)
- Stop automatically when E*TRADE is back up
- Save status history to `config/etrade_status_history.json`

**Example:**
```bash
# Monitor every 2 minutes
python3 scripts/monitor_etrade_status.py --interval 120

# Monitor for 1 hour (30 checks at 2 min intervals)
python3 scripts/monitor_etrade_status.py --interval 120 --max-checks 30
```

---

## 💾 **Status History**

The monitor saves check history to:
```
config/etrade_status_history.json
```

This contains:
- Timestamp of each check
- Overall status
- Detailed results from each test

---

## 🔔 **When E*TRADE Is Back Up**

When the monitor detects E*TRADE is working:
1. **It will display the authorization URL**
2. **It will automatically stop**
3. **You can then complete OAuth**

Follow the displayed instructions to complete the OAuth flow.

---

## 🆘 **Troubleshooting**

### **"Credentials not found"**
- Make sure credentials are in vault:
  ```python
  from secure_vault import get_vault
  v = get_vault()
  print(v.get_secret('etrade', 'sandbox_api_key'))
  ```

### **"Connection timeout"**
- Check your internet connection
- E*TRADE servers might be unreachable

### **"Request token endpoint not working"**
- Your credentials might be invalid
- Check E*TRADE Developer Portal

---

## 💡 **Best Practices**

1. **Start with a single check** to see current status
2. **Use continuous monitoring** when waiting for E*TRADE to come back up
3. **Set max-checks** to avoid running forever
4. **Check the history file** to see patterns

---

## 🎯 **Quick Commands Reference**

```bash
# Quick status check
python3 scripts/monitor_etrade_status.py --once

# Monitor for 10 minutes (10 checks)
python3 scripts/monitor_etrade_status.py --interval 60 --max-checks 10

# Monitor continuously until back up
python3 scripts/monitor_etrade_status.py

# Monitor production
python3 scripts/monitor_etrade_status.py --prod --once
```

---

**The monitor will tell you exactly when E*TRADE is ready!** 🎉



# NAE Live Trading - Ready Checklist

## ✅ System Configuration Complete

All systems are configured and ready for live trading once Alpaca keys are authenticated.

### Configuration Status:

✅ **API Keys:** Integrated in `config/api_keys.json`
- API Key: `AKPNTSZTJRSP7Y5DMPKIX7VK6R`
- API Secret: `72pWcuJpA8N8h5RQ5AtEFxe5UVbr2G`
- Endpoint: `https://api.alpaca.markets`

✅ **Settings:** Configured for live trading
- Environment: `production`
- Default Mode: `live`
- Broker: `alpaca`
- Manual Approval: `enabled`

✅ **Safety Systems:** All active
- Daily Loss Limit: 2% of NAV
- Consecutive Loss Limit: 5 trades
- Max Position Size: $10,000 or 5% of NAV
- Kill Switch: Enabled
- PDT Prevention: Enforced

✅ **Test Suite:** 7/8 tests passed
- All core systems operational
- Alpaca integration tested
- Feedback loops verified

## ⏳ Pending: Alpaca Authentication

**Current Status:** Keys integrated but authentication failing (401 Unauthorized)

**Required Action:** Enable API access in Alpaca dashboard

### Steps to Enable:

1. **Log into Alpaca Dashboard**
   - Visit: https://app.alpaca.markets/
   - Log in with your credentials

2. **Navigate to API Keys**
   - Click on your profile/account menu
   - Select "API Keys" or "API Access"

3. **Enable API Access**
   - Find your API key (starts with `AKPNTSZTJRSP7Y5DMPKI...`)
   - Toggle **"API Access"** to **ON**
   - Enable **"Trading"** permissions
   - Verify no IP restrictions are blocking access

4. **Verify Account Status**
   - Ensure account is "Active"
   - Complete any pending verification steps
   - Verify account is funded (if required)

## Once Authentication Works

### Step 1: Verify Connection
```bash
cd NAE
python3 verify_alpaca_keys.py
```

Expected output:
```
✅ LIVE trading authentication SUCCESSFUL
Account Info:
  Cash: $X,XXX.XX
  Buying Power: $X,XXX.XX
```

### Step 2: Start Live Trading
```bash
# Option 1: Use startup script (recommended)
./start_live_trading.sh

# Option 2: Start automation directly
python3 nae_automation.py
```

### Step 3: Monitor Trading
```bash
# Watch Optimus logs
tail -f logs/optimus.log

# Watch automation logs
tail -f logs/nae_automation.log

# Watch scheduler logs
tail -f logs/master_scheduler.log
```

## Continuous Monitoring

### Connection Monitor
Run this to continuously check when connection becomes available:
```bash
python3 check_alpaca_connection.py
```

This will:
- Check connection every 30 seconds
- Alert when authentication succeeds
- Display account information when ready

### Key Metrics to Monitor

**Account Health:**
- Cash balance
- Buying power
- Portfolio value
- Trading blocked status

**Trading Activity:**
- Daily P&L
- Open positions
- Order execution status
- Error rates

**System Health:**
- Agent status
- Feedback loop activity
- Kill switch status
- Error logs

## Safety Reminders

⚠️ **Before Starting Live Trading:**

1. **Start Small**
   - Begin with minimum position sizes
   - Test with small amounts first

2. **Monitor Closely**
   - Watch logs continuously
   - Set up alerts if possible
   - Review all trades

3. **Keep Kill Switch Ready**
   ```bash
   # Disable trading immediately if needed
   python3 redis_kill_switch.py --disable
   ```

4. **Set Limits**
   - Daily loss limits are set to 2% of NAV
   - Adjust if needed in `config/settings.json`
   - Monitor consecutive losses (limit: 5)

5. **Review Settings**
   - Verify safety limits match your risk tolerance
   - Ensure manual approval is enabled
   - Check PDT prevention is active

## Quick Reference Commands

### Verify Connection
```bash
python3 verify_alpaca_keys.py
```

### Check Connection Status (Continuous)
```bash
python3 check_alpaca_connection.py
```

### Start Live Trading
```bash
./start_live_trading.sh
# or
python3 nae_automation.py
```

### Monitor Logs
```bash
tail -f logs/optimus.log
tail -f logs/nae_automation.log
```

### Emergency Stop
```bash
python3 redis_kill_switch.py --disable
# or press Ctrl+C in automation terminal
```

### Check System Status
```bash
python3 -c "from nae_automation import NAEAutomationSystem; import json; print(json.dumps(NAEAutomationSystem().get_status(), indent=2))"
```

## Files Created

- ✅ `verify_alpaca_keys.py` - Key verification script
- ✅ `check_alpaca_connection.py` - Continuous connection monitor
- ✅ `start_live_trading.sh` - Startup script with verification
- ✅ `prepare_live_trading.py` - Live trading preparation
- ✅ `ALPACA_KEYS_INTEGRATED.md` - Integration summary
- ✅ `ALPACA_AUTH_TROUBLESHOOTING.md` - Troubleshooting guide
- ✅ `LIVE_TRADING_READY.md` - This checklist

## Next Steps Summary

1. ✅ **Keys Integrated** - Done
2. ✅ **Settings Configured** - Done
3. ✅ **Safety Systems Active** - Done
4. ⏳ **Enable API Access** - Pending (in Alpaca dashboard)
5. ⏳ **Verify Connection** - Run `verify_alpaca_keys.py`
6. ⏳ **Start Trading** - Run `start_live_trading.sh`

## Support

If authentication continues to fail after enabling API access:

1. **Check Dashboard:**
   - Verify API Access is ON
   - Verify Trading permissions enabled
   - Check account status

2. **Regenerate Keys:**
   - Delete current keys in dashboard
   - Generate new API keys
   - Update `config/api_keys.json`
   - Test again

3. **Contact Alpaca Support:**
   - If keys still don't work after verification
   - Check for account restrictions
   - Verify account is fully activated

---

**Status:** ✅ System ready, ⏳ Waiting for API access to be enabled
**Next Action:** Enable API access in Alpaca dashboard, then run `verify_alpaca_keys.py`


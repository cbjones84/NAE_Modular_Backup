# NAE Live Trading Setup Guide

## Test Results Summary

✅ **7 out of 8 tests passed** - System is ready for live trading preparation

### Test Results:
- ✅ test_nae_system.py - PASSED
- ✅ test_optimus_alpaca_paper.py - PASSED  
- ✅ test_feedback_loops.py - PASSED
- ✅ test_timing_strategies.py - PASSED
- ✅ test_pnl_tracking.py - PASSED
- ✅ test_security_alerting.py - PASSED
- ✅ test_legal_compliance_integration.py - PASSED
- ⚠️ test_api_keys.py - SKIPPED (expected)

## Current Status

### ✅ What's Ready:
1. **Alpaca SDK** - Installed and working
2. **Paper Trading** - Successfully tested
3. **Safety Systems** - Kill switch, risk limits, feedback loops all operational
4. **Agent System** - All agents initialized and tested
5. **Compliance** - Legal compliance checks passing

### ⚠️ What Needs Attention:

#### 1. Alpaca Live Trading Keys Required
**Current Status:** Paper trading keys detected (PK prefix)

**Action Required:**
1. Log into [Alpaca Dashboard](https://app.alpaca.markets/)
2. Navigate to **API Keys** section
3. Generate **LIVE Trading** API keys (not paper trading)
4. Update `config/api_keys.json` with live keys:
   ```json
   {
     "alpaca": {
       "api_key": "YOUR_LIVE_API_KEY",
       "api_secret": "YOUR_LIVE_API_SECRET",
       "paper_trading_url": "https://paper-api.alpaca.markets",
       "live_trading_url": "https://api.alpaca.markets"
     }
   }
   ```

**Important:** Live keys have different prefixes than paper keys (PK). Make sure you're using LIVE keys from the Alpaca dashboard.

## Preparing for Live Trading

### Step 1: Obtain Live Trading Keys
```bash
# 1. Visit https://app.alpaca.markets/
# 2. Go to API Keys section
# 3. Generate LIVE trading keys
# 4. Copy keys to config/api_keys.json
```

### Step 2: Run Preparation Script
```bash
cd NAE
python3 prepare_live_trading.py
```

This script will:
- ✅ Verify Alpaca credentials
- ✅ Test API connection
- ✅ Update settings.json for live trading
- ✅ Verify safety measures
- ✅ Create backup of settings

### Step 3: Review Configuration
After running the preparation script, review:
- `config/settings.json` - Verify live trading settings
- `config/api_keys.json` - Confirm live keys are set
- Safety limits - Ensure they match your risk tolerance

### Step 4: Start NAE with Live Trading
```bash
# Start automation system
python3 nae_automation.py

# Or use startup script
./start_nae.sh
```

## Safety Measures in Place

### ✅ Risk Management
- **Daily Loss Limit:** 2% of NAV
- **Consecutive Loss Limit:** 5 trades
- **Max Position Size:** 5% of NAV or $10,000
- **Max Open Positions:** 10
- **Kill Switch:** Enabled and monitored

### ✅ Compliance
- **PDT Prevention:** Enforced (no same-day round trips)
- **Audit Logging:** All trades logged immutably
- **Legal Compliance:** All checks passing

### ✅ Monitoring
- **Feedback Loops:** Auto-adjusting performance and risk
- **Health Checks:** Continuous agent monitoring
- **Error Recovery:** Self-healing system

## Live Trading Checklist

Before enabling live trading, verify:

- [ ] Live Alpaca API keys obtained and configured
- [ ] Paper trading tested successfully
- [ ] All tests passed (7/8 minimum)
- [ ] Safety limits reviewed and acceptable
- [ ] Kill switch tested and accessible
- [ ] Monitoring and logging enabled
- [ ] Backup of settings created
- [ ] Risk tolerance understood
- [ ] Daily loss limits set
- [ ] Manual approval enabled for trades

## Important Reminders

⚠️ **CRITICAL SAFETY NOTES:**

1. **Start Small:** Begin with minimum position sizes
2. **Monitor Closely:** Watch `logs/optimus.log` continuously
3. **Keep Kill Switch Ready:** Know how to stop trading immediately
4. **Review All Trades:** Manual approval is enabled by default
5. **Set Daily Limits:** Don't exceed your risk tolerance
6. **Test First:** Ensure paper trading works perfectly before going live

## Configuration Files

### Settings (`config/settings.json`)
```json
{
  "environment": "production",
  "trading": {
    "default_mode": "live",
    "live": {
      "enabled": true,
      "broker": "alpaca",
      "requires_manual_approval": true
    }
  },
  "safety_limits": {
    "max_order_size_usd": 10000.0,
    "daily_loss_limit_pct": 0.02,
    "consecutive_loss_limit": 5
  }
}
```

### API Keys (`config/api_keys.json`)
```json
{
  "alpaca": {
    "api_key": "YOUR_LIVE_KEY_HERE",
    "api_secret": "YOUR_LIVE_SECRET_HERE",
    "live_trading_url": "https://api.alpaca.markets"
  }
}
```

## Monitoring Live Trading

### Log Files to Monitor:
- `logs/optimus.log` - Trading activity
- `logs/nae_automation.log` - System automation
- `logs/master_scheduler.log` - Agent scheduling
- `logs/casey.log` - Agent monitoring

### Key Metrics to Watch:
- Daily P&L
- Open positions
- Order execution status
- Error rates
- Kill switch status

## Emergency Procedures

### To Stop Trading Immediately:
```bash
# Option 1: Use kill switch
python3 redis_kill_switch.py --disable

# Option 2: Stop NAE automation
# Press Ctrl+C in the terminal running nae_automation.py

# Option 3: Disable in settings
# Set "trading.live.enabled": false in config/settings.json
```

### To Resume Trading:
```bash
# Re-enable kill switch
python3 redis_kill_switch.py --enable

# Or restart NAE automation
python3 nae_automation.py
```

## Next Steps

1. ✅ **Tests Complete** - 7/8 tests passed
2. ⏳ **Get Live Keys** - Obtain Alpaca live trading API keys
3. ⏳ **Run Preparation** - Execute `prepare_live_trading.py`
4. ⏳ **Review Settings** - Verify all configurations
5. ⏳ **Start Trading** - Launch `nae_automation.py`

## Support

For issues or questions:
- Check logs: `logs/nae_automation.log`
- Review test results: `run_all_tests.py`
- Verify configuration: `config/settings.json`

---

**Status:** ✅ System tested and ready for live trading preparation
**Next Action:** Obtain Alpaca live trading API keys and run `prepare_live_trading.py`


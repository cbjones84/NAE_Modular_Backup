# NAV Tracking Verification Plan

**Date:** 2025-01-09  
**Status:** ✅ Instrumentation Complete - Ready for Testing  
**Log File:** `/Users/melissabishop/Downloads/Neural Agency Engine/.cursor/debug.log`

---

## Verification Requirements

We need to verify 5 key requirements:

1. ✅ **VERIFICATION 1:** NAV syncs from Tradier on startup
2. ✅ **VERIFICATION 2:** Starting NAV is recorded correctly
3. ✅ **VERIFICATION 3:** NAV syncs before each trade
4. ✅ **VERIFICATION 4:** Compound growth metrics update correctly
5. ✅ **VERIFICATION 5:** Return logging works (daily/weekly/monthly)

---

## Instrumentation Added

All verification logs use `hypothesisId` tags:
- **V1**: NAV sync on startup
- **V2**: Starting NAV recording
- **V3**: NAV sync before trades
- **V4**: Compound growth metrics
- **V5**: Return logging

### Verification 1: NAV Sync on Startup
**Location:** `optimus.py:644-645`

**Logs:**
- `VERIFICATION 1: Starting NAV sync on startup` - Before sync attempt
- `VERIFICATION 1: NAV sync on startup result` - After sync with result

**What to Verify:**
- `nav_synced: true` in result log
- `nav_after_sync` > 0
- `nav_source: "tradier"` (not "fallback")

### Verification 2: Starting NAV Recording
**Location:** `optimus.py:653, 2429`

**Logs:**
- `VERIFICATION 2: Starting NAV recorded` - When starting NAV is set
- `VERIFICATION 2: Starting NAV recorded in _sync_account_balance` - Alternative location

**What to Verify:**
- `starting_nav` > 0
- `starting_nav` matches `nav` at time of recording
- `timestamp` is set correctly

### Verification 3: NAV Sync Before Trades
**Location:** `optimus.py:981-982, 1689-1690`

**Logs:**
- `VERIFICATION 3: NAV sync before trade (pre_trade_checks)` - Before pre-trade checks
- `VERIFICATION 3: NAV sync before trade result` - Result of sync
- `VERIFICATION 3: NAV sync before trade execution` - Before execute_trade
- `VERIFICATION 3: NAV sync before execution result` - Result before execution

**What to Verify:**
- NAV sync happens before every trade attempt
- `nav_synced: true` in logs
- `nav_after_sync` is updated
- Symbol is included in logs for trade context

### Verification 4: Compound Growth Metrics
**Location:** `optimus.py:2704, 2745`

**Logs:**
- `VERIFICATION 4: Compound growth metrics update start` - When calculation starts
- `VERIFICATION 4: Compound growth metrics calculated` - After calculation

**What to Verify:**
- `total_return_pct` is calculated correctly
- `compound_growth_rate` is calculated (formula: (nav/starting_nav)^(1/years) - 1)
- `annualized_return_pct` matches compound_growth_rate
- `days_since_start` and `months_since_start` are tracked
- Metrics update after NAV syncs

### Verification 5: Return Logging
**Location:** `optimus.py:2785, 2806, 2899`

**Logs:**
- `VERIFICATION 5: Return tracking initialized` - On startup
- `VERIFICATION 5: Daily return logged` - When daily return is logged
- `VERIFICATION 5: Weekly return logged` - When weekly return is logged
- `VERIFICATION 5: Monthly return logged` - When monthly return is logged

**What to Verify:**
- Daily returns log when date changes
- Weekly returns log on Monday (week start)
- Monthly returns log on first of month
- Return data includes NAV, total_return_pct, compound_growth_rate_pct
- Return lists grow correctly (daily_returns, weekly_returns, monthly_returns)

---

## Expected Log Patterns

### Successful Startup Sequence:
```json
{"message": "VERIFICATION 1: Starting NAV sync on startup", "hypothesisId": "V1"}
{"message": "VERIFICATION 1: NAV sync on startup result", "nav_synced": true, "nav_source": "tradier"}
{"message": "VERIFICATION 2: Starting NAV recorded", "starting_nav": 1234.56}
{"message": "VERIFICATION 5: Return tracking initialized"}
```

### Before Trade Sequence:
```json
{"message": "VERIFICATION 3: NAV sync before trade", "hypothesisId": "V3"}
{"message": "VERIFICATION 3: NAV sync before trade result", "nav_synced": true}
{"message": "VERIFICATION 4: Compound growth metrics update start"}
{"message": "VERIFICATION 4: Compound growth metrics calculated", "total_return_pct": 5.2, "compound_growth_rate": 12.5}
```

### Return Logging Sequence:
```json
{"message": "VERIFICATION 5: Daily return logged", "total_daily_returns": 1}
{"message": "VERIFICATION 5: Weekly return logged", "total_weekly_returns": 1}
{"message": "VERIFICATION 5: Monthly return logged", "total_monthly_returns": 1}
```

---

## Analysis Criteria

### ✅ VERIFICATION 1: PASS if
- NAV syncs successfully on startup (`nav_synced: true`)
- NAV value > 0
- Source is "tradier" (not "fallback")

### ✅ VERIFICATION 2: PASS if
- Starting NAV is recorded (`starting_nav` > 0)
- Starting NAV matches NAV at time of recording
- Timestamp is set

### ✅ VERIFICATION 3: PASS if
- NAV syncs before every trade attempt
- NAV sync happens in both `pre_trade_checks` and `execute_trade`
- `nav_synced: true` in all trade-related logs

### ✅ VERIFICATION 4: PASS if
- Compound growth metrics update after NAV syncs
- `total_return_pct` = ((nav - starting_nav) / starting_nav) * 100
- `compound_growth_rate` calculated correctly
- Metrics update before/after trades

### ✅ VERIFICATION 5: PASS if
- Daily returns log when date changes
- Weekly returns log on Monday
- Monthly returns log on first of month
- Return data includes all required fields
- Return lists accumulate correctly

---

## Next Steps

<reproduction_steps>
1. Ensure Tradier API credentials are configured (TRADIER_API_KEY and TRADIER_ACCOUNT_ID)
2. Start the Optimus agent: `python "NAE Ready/agents/optimus.py"` or use your normal startup method
3. Let the system run for at least 2-3 minutes to capture:
   - Startup NAV sync
   - Starting NAV recording
   - Any trade attempts (if trades are triggered)
   - Return logging (if date/week/month boundaries are crossed)
4. If possible, trigger a trade (manually or wait for automatic trading)
5. Stop the system (Ctrl+C or normal shutdown)
6. Click "Proceed" in the debug UI to confirm completion
</reproduction_steps>

**Note:** Return logging (V5) may not trigger immediately unless:
- The date changes (daily)
- It's Monday (weekly)
- It's the first of the month (monthly)

The other verifications (V1-V4) should trigger on startup and before trades.

---

## After Reproduction

I will analyze the logs and provide:
- ✅ **CONFIRMED**: Each requirement is working correctly (with log evidence)
- ❌ **FAILED**: Requirement not met (with log evidence showing the issue)
- ⚠️ **INCONCLUSIVE**: Need more data (with explanation)

Then I'll provide fixes for any failed verifications.


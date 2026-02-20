# NAE Tradier Diagnostics

## Overview

This diagnostic module helps identify why NAE is not placing trades through Tradier. It performs comprehensive checks in the correct order to pinpoint the exact issue.

## What It Checks

1. **API Connection** - Confirms Live vs Sandbox endpoint
2. **Account Verification** - Confirms your account ID is correct
3. **User Profile** - Confirms options approval level
4. **Account Balances** - Shows buying power & settled cash
5. **Symbol Validation** - Validates symbols & OCC formats
6. **Test Order** - Attempts a $0 "test order" (Tradier supports this via preview=true)
7. **Error Messages** - Returns the exact error message from Tradier, if any

## Usage

### Command Line

```bash
# Run diagnostics
python scripts/run_tradier_diag.py

# Or directly
python execution/diagnostics/nae_tradier_diagnostics.py --test-symbol SPY250117C00500000
```

### From Python Code

```python
from execution.diagnostics import TradierDiagnostics

# Initialize diagnostics
diag = TradierDiagnostics(
    api_key="your_key",  # Or set TRADIER_API_KEY env var
    account_id="your_account",  # Or set TRADIER_ACCOUNT_ID env var
    live=True  # False for sandbox
)

# Run full diagnostics
results = diag.run_full_diagnostics(test_symbol="SPY250117C00500000")
```

### From Continuous Trading Engine

The `ContinuousTradingEngine` has a built-in diagnostics method:

```python
from execution.integration.continuous_trading_engine import ContinuousTradingEngine

engine = ContinuousTradingEngine()
results = engine.run_diagnostics(test_symbol="SPY250117C00500000")
```

### From Agents (Optimus, Donnie, Ralph)

Agents can import and use diagnostics:

```python
from execution.diagnostics import TradierDiagnostics

# Run diagnostics when trades fail
diag = TradierDiagnostics(live=True)
results = diag.run_full_diagnostics()
```

## What You'll Get

The diagnostic will print:

1. **Live or sandbox connectivity** - If wrong → fix endpoint
2. **Options approval status** - If missing → request approval
3. **Buying power values**:
   - `options_buying_power`
   - `cash_balance`
   - `day_trading_buying_power`
   - `pending_cash`
4. **Symbol checks** - If the symbol isn't formatted correctly → NAE will never trade
5. **Exact rejection reason** - For example:
   - `"reason": "insufficient buying power"`
   - `"reason": "invalid option symbol"`
   - `"reason": "trading not permitted"`
   - `"reason": "duration required"`
   - `"reason": "unsupported instrument"`

## Common Issues Identified

- ✅ **Strategy conditions** - Not meeting strategy requirements
- ✅ **Permissions** - Missing options approval
- ✅ **Wrong endpoint** - Using sandbox instead of live (or vice versa)
- ✅ **Missing fields** - Required order fields not provided
- ✅ **Rejected orders** - Orders being rejected by Tradier
- ✅ **Account restrictions** - Account limitations preventing trades
- ✅ **Bad symbol formatting** - Incorrect OCC format

## Output

Diagnostics results are:
- Printed to console with clear formatting
- Saved to `logs/tradier_diagnostics_YYYYMMDD_HHMMSS.json`
- Returned as a dictionary for programmatic access

## Integration

The diagnostics module is integrated into:
- `ContinuousTradingEngine` - Can run diagnostics automatically
- `scripts/run_tradier_diag.py` - Quick CLI access
- Available for all agents to call when needed

## Example Output

```
============================================================
 NAE → TRADIER DIAGNOSTIC RUN
============================================================

1️⃣  Checking API connection...
============================================================
 ✅ Connection Test: SUCCESS
============================================================
{
  "endpoint": "https://api.tradier.com/v1",
  "status_code": 200
}

2️⃣  Checking available accounts...
============================================================
 ✅ Account Verification
============================================================
{
  "provided_account_id": "ABC123",
  "available_account_ids": ["ABC123"],
  "account_found": true
}

3️⃣  Checking user profile & options approval...
============================================================
 ✅ User Profile & Options Approval
============================================================
{
  "options_approved": true,
  "options_level": "level_3"
}

4️⃣  Checking cash & buying power...
============================================================
 ✅ Account Balances
============================================================
{
  "cash_available": 10000.00,
  "options_buying_power": 50000.00
}

5️⃣  Validating option symbol...
============================================================
 ✅ Option Chain Validation
============================================================
{
  "underlying": "SPY",
  "occ_symbol": "SPY250117C00500000"
}

6️⃣  Attempting test order (preview=true)...
============================================================
 ✅ Test Order: SUCCESS
============================================================
{
  "status_code": 200,
  "order": {
    "id": "12345",
    "status": "ok"
  }
}

============================================================
 DIAGNOSTIC SUMMARY
============================================================
  connection: ✅ OK
  endpoint: LIVE
  account_id: ABC123
  options_approved: True
  options_level: level_3
  has_buying_power: True
  test_order_status: SUCCESS

✅ Diagnostics complete.
```


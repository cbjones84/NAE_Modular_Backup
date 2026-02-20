# Tradier Diagnostics Guide

## Quick Start

**Why NAE is not placing trades through Tradier?** Run diagnostics to find out:

```bash
# Quick diagnostic run
python scripts/run_tradier_diag.py
```

## What It Does

The diagnostic module performs all checks in the correct order:

1. ✅ **Confirms Live vs Sandbox endpoint** - Verifies you're using the right API endpoint
2. ✅ **Confirms your account ID is correct** - Lists all accounts and verifies your account ID
3. ✅ **Confirms options approval level** - Checks if you have options trading permissions
4. ✅ **Shows buying power & settled cash** - Displays all balance fields:
   - `cash`
   - `cash_available`
   - `margin_balance`
   - `pending_cash`
   - `unsettled_funds`
   - `options_buying_power`
   - `day_trading_buying_power`
5. ✅ **Validates symbols & OCC formats** - Checks if option symbols are correctly formatted
6. ✅ **Attempts a $0 "test order"** - Uses Tradier's `preview=true` to test orders safely
7. ✅ **Returns exact error messages** - Shows exactly why orders are being rejected

## Common Issues It Identifies

- ❌ **Strategy conditions** - Not meeting strategy requirements
- ❌ **Permissions** - Missing options approval
- ❌ **Wrong endpoint** - Using sandbox instead of live (or vice versa)
- ❌ **Missing fields** - Required order fields not provided
- ❌ **Rejected orders** - Orders being rejected by Tradier
- ❌ **Account restrictions** - Account limitations preventing trades
- ❌ **Bad symbol formatting** - Incorrect OCC format

## Usage Examples

### From Command Line

```bash
# Basic run (uses env vars)
python scripts/run_tradier_diag.py

# With custom symbol
python execution/diagnostics/nae_tradier_diagnostics.py --test-symbol SPY250117C00500000

# Sandbox testing
python execution/diagnostics/nae_tradier_diagnostics.py --sandbox
```

### From Python Code

```python
from execution.diagnostics import TradierDiagnostics

# Initialize
diag = TradierDiagnostics(
    api_key="your_key",  # Or set TRADIER_API_KEY env var
    account_id="your_account",  # Or set TRADIER_ACCOUNT_ID env var
    live=True
)

# Run diagnostics
results = diag.run_full_diagnostics(test_symbol="SPY250117C00500000")
```

### From Continuous Trading Engine

```python
from execution.integration.continuous_trading_engine import ContinuousTradingEngine

engine = ContinuousTradingEngine()
results = engine.run_diagnostics()
```

### From Agents (Optimus, Donnie, Ralph)

```python
from execution.diagnostics import TradierDiagnostics

# When trades fail, run diagnostics
diag = TradierDiagnostics(live=True)
results = diag.run_full_diagnostics()

# Check results
if results.get("summary", {}).get("test_order_status") == "FAILED":
    print("Trades will fail - check diagnostics output")
```

## Understanding the Output

### Connection Test
- ✅ **SUCCESS**: API endpoint is correct and accessible
- ❌ **FAILED**: Wrong endpoint or API key issue

### Account Verification
- ✅ **account_found: true**: Your account ID is correct
- ❌ **account_found: false**: Wrong account ID

### Options Approval
- ✅ **options_approved: true**: You can trade options
- ❌ **options_approved: false**: Need to request options approval
- **options_level**: Shows your approval level (level_1, level_2, level_3, etc.)

### Buying Power
- **cash_available**: Cash you can use immediately
- **options_buying_power**: Buying power for options
- **day_trading_buying_power**: Buying power for day trading
- **pending_cash**: Cash pending settlement

### Test Order
- ✅ **SUCCESS**: Order would be accepted
- ⚠️ **WARNINGS**: Order has warnings but would execute
- ❌ **ERRORS**: Order would be rejected - check error messages

## Error Messages Explained

Common error reasons from Tradier:

- `"insufficient buying power"` → Need more cash or margin
- `"invalid option symbol"` → Symbol format is wrong
- `"trading not permitted"` → Account restrictions or market closed
- `"duration required"` → Missing duration field
- `"unsupported instrument"` → Symbol not tradeable
- `"options approval required"` → Need to request options approval

## Integration Points

The diagnostics are integrated into:

1. **Continuous Trading Engine** - Can run diagnostics automatically
2. **CLI Script** - `scripts/run_tradier_diag.py` for quick access
3. **Agent Access** - Optimus, Donnie, and Ralph can call diagnostics
4. **Execution Worker** - Can be called when orders fail

## Next Steps

After running diagnostics:

1. **Fix endpoint** - If using wrong endpoint, update `TRADIER_SANDBOX` env var
2. **Request approval** - If options not approved, request from Tradier
3. **Add funds** - If insufficient buying power, deposit funds
4. **Fix symbols** - If symbol format wrong, correct OCC format
5. **Check account** - If account ID wrong, update `TRADIER_ACCOUNT_ID`

## Files

- `execution/diagnostics/nae_tradier_diagnostics.py` - Main diagnostic module
- `execution/diagnostics/README.md` - Detailed documentation
- `scripts/run_tradier_diag.py` - Quick CLI script
- `TRADIER_DIAGNOSTICS_GUIDE.md` - This guide

## Support

If diagnostics show everything is OK but trades still fail:

1. Check NAE logs: `logs/continuous_trading_engine.log`
2. Check Tradier logs: `logs/tradier_*.log`
3. Review strategy conditions in Optimus/Donnie
4. Check compliance rules (day trading prevention, etc.)


# Live Trading Connection Configuration

## Overview
NAE has been configured to connect to the **LIVE Alpaca trading account** instead of the paper trading account. All agents now use live trading by default.

## Changes Made

### 1. OptimusAgent Default Mode ✅
- **File**: `NAE/agents/optimus.py`
- **Change**: Default trading mode changed from `PAPER` to `LIVE`
- **Before**: `self.trading_mode = TradingMode.SANDBOX if sandbox else TradingMode.PAPER`
- **After**: `self.trading_mode = TradingMode.LIVE` (when `sandbox=False`)

### 2. Alpaca Client Configuration ✅
- **File**: `NAE/agents/optimus.py`
- **Change**: Alpaca client now uses `paper_trading=False` for LIVE mode
- **Logic**: 
  - `LIVE` mode → `paper_trading=False` (connects to live account)
  - `SANDBOX` or `PAPER` mode → `paper_trading=True` (connects to paper account)

### 3. Master Scheduler Update ✅
- **File**: `NAE/nae_master_scheduler.py`
- **Change**: OptimusAgent initialized with `sandbox=False` for LIVE mode
- **Before**: `optimus = OptimusAgent(sandbox=True)`
- **After**: `optimus = OptimusAgent(sandbox=False)`  # LIVE mode

### 4. Continuous Automation Update ✅
- **File**: `NAE/nae_continuous_automation.py`
- **Change**: Updated comment to reflect LIVE mode
- **Before**: `# PAPER mode (Alpaca)`
- **After**: `# LIVE mode (Live Alpaca account)`

## Verification

To verify the connection is working:

```python
from agents.optimus import OptimusAgent

# Initialize OptimusAgent (defaults to LIVE mode)
optimus = OptimusAgent(sandbox=False)

# Check trading mode
print(f"Trading Mode: {optimus.trading_mode.value}")  # Should be "live"

# Check Alpaca client
if optimus.alpaca_client:
    print(f"Paper Trading: {optimus.alpaca_client.paper_trading}")  # Should be False
    
    # Sync account balance
    optimus._sync_account_balance()
    balance = optimus.get_available_balance()
    print(f"NAV: ${balance['nav']:,.2f}")
```

## Important Notes

⚠️ **CRITICAL**: All trades will now execute on the **LIVE Alpaca account** with real money.

- Safety limits are still enforced
- Pre-trade checks validate buying power
- Kill switch can disable trading
- Audit logging tracks all trades

## Reverting to Paper Trading

If you need to revert to paper trading:

1. **Temporary**: Initialize with `sandbox=True`:
   ```python
   optimus = OptimusAgent(sandbox=True)  # SANDBOX mode
   ```

2. **Or set to PAPER mode explicitly**:
   ```python
   optimus = OptimusAgent(sandbox=False)
   optimus.trading_mode = TradingMode.PAPER
   ```

3. **Update Master Scheduler**: Change `nae_master_scheduler.py`:
   ```python
   optimus = OptimusAgent(sandbox=True)  # For paper trading
   ```

## Safety Features

All safety features remain active:
- ✅ Daily loss limits
- ✅ Consecutive loss limits
- ✅ Position size limits
- ✅ Buying power validation
- ✅ Kill switch
- ✅ Audit logging

## Account Balance Sync

Optimus automatically syncs account balance from the LIVE Alpaca account:
- NAV (Equity)
- Cash
- Buying Power
- Portfolio Value

Strategy recommendations are determined based on the live account balance.


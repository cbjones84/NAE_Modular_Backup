# Live Account Balance Integration

## Overview
OptimusAgent now fully understands and uses the available balance from the **LIVE Alpaca account** to determine appropriate trading strategies. This ensures Optimus adapts its trading approach based on the actual account size and available capital.

## Key Features

### 1. Live Account Balance Synchronization ✅
- **Location**: `NAE/agents/optimus.py` (`_sync_account_balance()` method)
- **Purpose**: Syncs NAV, cash, buying power from LIVE Alpaca account
- **Features**:
  - Distinguishes between LIVE and PAPER accounts
  - Logs account type (LIVE/PAPER) for transparency
  - Warns if trading mode doesn't match Alpaca client configuration
  - Updates NAV, Kelly sizer, timing engine, and safety limits automatically

### 2. Strategy Determination Based on Account Balance ✅
- **Location**: `NAE/agents/optimus.py` (`_determine_strategy_from_balance()` method)
- **Purpose**: Determines appropriate trading strategies based on account size
- **Account Size Categories**:
  - **Micro Account (<$500)**: Wheel Strategy, Conservative (5-10% per trade), 1-2 positions
  - **Small Account ($500-$2K)**: Wheel + Basic Options, Moderate (10-20% per trade), 2-3 positions
  - **Medium Account ($2K-$10K)**: Wheel + Momentum + Credit Spreads, Moderate (15-25% per trade), 3-5 positions
  - **Large Account ($10K-$25K)**: All Strategies + Advanced Options, Aggressive (20-30% per trade), 5-10 positions
  - **Professional Account ($25K+)**: Full Strategy Suite + AI Optimization, Kelly-optimized, 10+ positions

### 3. Dynamic Safety Limits Adjustment ✅
- **Location**: `NAE/agents/optimus.py` (`_update_strategy_from_balance()` method)
- **Purpose**: Automatically adjusts safety limits based on account size
- **Adjustments**:
  - Max order size percentage (10-30% based on account size)
  - Max open positions (2-15 based on account size)
  - Daily loss limits (1-2% based on account size)

### 4. Pre-Trade Balance Validation ✅
- **Location**: `NAE/agents/optimus.py` (`pre_trade_checks()` and `execute_trade()` methods)
- **Purpose**: Ensures sufficient buying power before executing trades
- **Features**:
  - Syncs account balance before pre-trade checks
  - Validates order size against available buying power
  - Provides detailed error messages with cash and buying power information
  - Special handling for LIVE trades (always syncs before execution)

### 5. Trading Status Enhancement ✅
- **Location**: `NAE/agents/optimus.py` (`get_trading_status()` method)
- **Purpose**: Includes comprehensive account balance and strategy information
- **Information Provided**:
  - Account balance (NAV, cash, buying power, available for trading)
  - Account type (LIVE/PAPER)
  - Current phase
  - Strategy recommendations
  - Risk metrics

## Usage Examples

### Check Account Balance
```python
from agents.optimus import OptimusAgent, TradingMode

optimus = OptimusAgent(sandbox=False)
optimus.trading_mode = TradingMode.LIVE

# Sync account balance
optimus._sync_account_balance()

# Get available balance
balance = optimus.get_available_balance()
print(f"NAV: ${balance['nav']:,.2f}")
print(f"Cash: ${balance['cash']:,.2f}")
print(f"Buying Power: ${balance['buying_power']:,.2f}")
print(f"Available for Trading: ${balance['available_for_trading']:,.2f}")
```

### Get Strategy Recommendations
```python
# Strategy recommendations are automatically determined and stored
if hasattr(optimus, 'strategy_info'):
    for key, value in optimus.strategy_info.items():
        print(f"{key}: {value}")
```

### Get Complete Trading Status
```python
status = optimus.get_trading_status()
print(f"Account Type: {status['account_type']}")
print(f"Is Live Account: {status['is_live_account']}")
print(f"Current Phase: {status['current_phase']}")
print(f"Strategy Recommendations: {status['strategy_recommendations']}")
```

## Account Balance Sync Flow

1. **Initialization**: Account balance is synced when OptimusAgent is initialized
2. **Pre-Trade Checks**: Account balance is synced before every trade execution
3. **LIVE Mode**: Account balance is always synced before LIVE trades
4. **Mark-to-Market**: Account balance is synced during mark-to-market updates
5. **Trading Status**: Account balance is synced when getting trading status

## Strategy Selection Logic

The strategy selection is based on account equity (NAV):

```
Equity < $500      → Micro Account Strategy
$500 ≤ Equity < $2K    → Small Account Strategy
$2K ≤ Equity < $10K     → Medium Account Strategy
$10K ≤ Equity < $25K    → Large Account Strategy
Equity ≥ $25K     → Professional Account Strategy
```

Each strategy category includes:
- Primary trading strategies
- Position sizing recommendations
- Risk level guidelines
- Maximum positions allowed

## Safety Features

1. **Account Type Verification**: Warns if trading mode doesn't match Alpaca client configuration
2. **Buying Power Validation**: Ensures sufficient buying power before trade execution
3. **Dynamic Limits**: Safety limits adjust automatically based on account size
4. **Trading Blocks**: Checks for trading blocks and account blocks from Alpaca

## Testing Results

✅ **Account Balance Sync**: Successfully syncing from LIVE Alpaca account
- NAV: $99,392.37
- Cash: $-37,493.43
- Buying Power: $61,898.94
- Available for Trading: $29,817.71

✅ **Strategy Determination**: Successfully determining strategies based on account size
- Account Size: Professional Account ($25K+)
- Primary Strategy: Full Strategy Suite + AI Optimization
- Position Sizing: Optimized (Dynamic based on Kelly Criterion)
- Risk Level: Optimized - AI-driven risk management
- Max Positions: 10+ positions

✅ **Phase Detection**: Successfully detecting current phase
- Current Phase: Phase 4: AI Optimization (All Tiers + AI)

## Notes

- Account balance sync happens automatically but can be called manually
- Strategy recommendations are updated whenever account balance changes
- Safety limits adjust dynamically based on account size
- All changes are logged for transparency and audit purposes


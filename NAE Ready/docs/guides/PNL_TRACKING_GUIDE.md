# P&L Tracking in Sandbox Mode

## Overview

The Optimus agent now includes comprehensive Profit & Loss (P&L) tracking for sandbox trades. This feature automatically calculates realized and unrealized P&L, tracks open positions, and marks positions to market using real-time prices.

## Features

### 1. Position Tracking
- Tracks all open positions with entry prices, quantities, and current market prices
- Supports adding to existing positions (averages entry price)
- Handles partial and full position closures

### 2. Realized P&L Calculation
- Automatically calculates realized P&L when positions are closed
- Tracks total realized P&L across all closed trades
- P&L is calculated as: `(exit_price - entry_price) × quantity` for long positions

### 3. Unrealized P&L Calculation
- Marks all open positions to market periodically
- Calculates unrealized P&L based on current market prices
- Updates every 10 seconds via monitoring thread
- Uses Polygon.io API for real-time prices when available

### 4. Daily P&L Aggregation
- Daily P&L = Realized P&L + Unrealized P&L
- Updated automatically after each trade and mark-to-market cycle
- Integrated into risk management and safety limits

## Usage

### Basic Trade Execution

```python
from agents.optimus import OptimusAgent

# Initialize Optimus in sandbox mode
optimus = OptimusAgent(sandbox=True)

# Open a position
buy_order = {
    'symbol': 'SPY',
    'side': 'buy',
    'quantity': 10,
    'price': 450.0  # Will use market price if Polygon client is available
}
result = optimus.execute_trade(buy_order)

# Close position (realizes P&L)
sell_order = {
    'symbol': 'SPY',
    'side': 'sell',
    'quantity': 10,
    'price': 455.0
}
result = optimus.execute_trade(sell_order)
```

### Getting Trading Status

```python
# Get comprehensive trading status including P&L
status = optimus.get_trading_status()

print(f"Daily P&L: ${status['daily_pnl']:.2f}")
print(f"Realized P&L: ${status['realized_pnl']:.2f}")
print(f"Unrealized P&L: ${status['unrealized_pnl']:.2f}")
print(f"Open Positions: {status['open_positions']}")

# View detailed position information
for symbol, pos in status['open_positions_detail'].items():
    print(f"{symbol}:")
    print(f"  Quantity: {pos['quantity']}")
    print(f"  Entry Price: ${pos['entry_price']:.2f}")
    print(f"  Current Price: ${pos['current_price']:.2f}")
    print(f"  Unrealized P&L: ${pos['unrealized_pnl']:.2f}")
```

### Manual Mark-to-Market Update

```python
# Force mark-to-market update
optimus._mark_to_market()

# Get updated status
status = optimus.get_trading_status()
```

## Position Management

### Adding to Positions
When you buy more of a symbol you already own, the system:
- Calculates a weighted average entry price
- Updates the position quantity
- Maintains the original entry time

### Partial Closes
When selling fewer shares than owned:
- Calculates realized P&L for the closed portion
- Reduces position quantity
- Keeps remaining position open with original entry price

### Full Closes
When selling all shares:
- Calculates final realized P&L
- Removes position from tracking
- Decrements open positions count

## Monitoring

The system automatically:
- Marks positions to market every 10 seconds
- Updates daily P&L continuously
- Checks risk limits using current P&L
- Logs all trades with P&L information

## Integration with Risk Management

P&L tracking is integrated with:
- Daily loss limits (based on daily P&L)
- Consecutive loss tracking (based on realized P&L per trade)
- NAV calculation (NAV + Daily P&L = Total Value)
- Position limit checks

## Testing

Run the test script to verify P&L tracking:

```bash
python3 test_pnl_tracking.py
```

This will test:
1. Opening positions
2. Adding to positions
3. Partial closes
4. Full closes
5. Multiple symbols
6. Mark-to-market updates

## Implementation Details

### Data Structures

- `open_positions_dict`: Dictionary tracking open positions
  - Key: Symbol (e.g., 'SPY')
  - Value: Position details (entry_price, quantity, side, entry_time, unrealized_pnl, current_price)

- `realized_pnl`: Accumulated P&L from closed positions
- `unrealized_pnl`: Current P&L from open positions
- `daily_pnl`: Total P&L (realized + unrealized)

### Methods

- `_execute_sandbox_trade()`: Enhanced to calculate P&L and manage positions
- `_mark_to_market()`: Updates unrealized P&L for all open positions
- `_update_risk_metrics()`: Updates risk metrics including P&L
- `get_trading_status()`: Returns comprehensive status including P&L breakdown

## Notes

- Market prices are fetched from Polygon.io API when available
- Falls back to order price or simulated prices if API unavailable
- Mark-to-market runs automatically but can be called manually
- All P&L calculations are logged for audit purposes


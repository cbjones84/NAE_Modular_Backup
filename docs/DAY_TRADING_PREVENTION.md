# Day Trading Prevention - Optimus Safety Feature

## Overview

Optimus has been configured to **prevent all day trading**. This means positions opened on the same day cannot be closed on that same day. Positions must be held overnight at minimum.

## How It Works

### Day Trading Check

Before executing any sell order, Optimus checks:
1. **Position exists**: Does the symbol have an open position?
2. **Entry date**: When was the position opened?
3. **Same day check**: Is the entry date the same as today's date?
4. **Block if same day**: If yes, the order is rejected

### Entry Time Tracking

- **New positions**: Entry time is recorded when a position is first opened
- **Position sync**: When Optimus starts, it syncs positions from Alpaca
- **Entry time preservation**: If a position already exists, original entry time is preserved

### Safety First

- **Conservative approach**: If entry time cannot be determined, the order is blocked
- **Error handling**: On any error, day trading is blocked (safer to prevent than allow)

## Implementation Details

### Pre-Trade Check

The day trading check is part of Optimus's `pre_trade_checks()` method:

```python
# DAY TRADING PREVENTION: Check if trying to close a position opened today
if self._would_close_same_day_position(order_data):
    return False, "Day trading not allowed: Cannot close position opened today"
```

### Method: `_would_close_same_day_position()`

This method:
1. Checks if the order is a sell order (closing position)
2. Verifies if a position exists for the symbol
3. Syncs with Alpaca if position not tracked locally
4. Compares entry date with current date
5. Returns `True` if same day (blocks order), `False` if different day (allows order)

## Testing

### Test Day Trading Prevention

```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=False)

# Try to sell a position opened today (should be blocked)
test_sell = {
    'symbol': 'SPY',
    'action': 'sell',
    'side': 'sell',
    'quantity': 1,
    'order_type': 'market'
}

checks_passed, message = optimus.pre_trade_checks(test_sell)
if not checks_passed:
    print(f"✅ Day trading blocked: {message}")
```

### Expected Behavior

**Same Day (Blocked):**
```
Order: SELL SPY (opened today)
Result: ❌ BLOCKED
Message: "Day trading not allowed: Cannot close position opened today"
```

**Different Day (Allowed):**
```
Order: SELL SPY (opened yesterday)
Result: ✅ ALLOWED
Message: "All pre-trade checks passed"
```

## Usage Examples

### Attempting to Day Trade (Will Fail)

```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=False)

# Buy SPY (opens position)
result1 = optimus.execute_trade({
    'symbol': 'SPY',
    'action': 'buy',
    'quantity': 1,
    'order_type': 'market'
})

# Try to sell immediately (same day) - WILL BE BLOCKED
result2 = optimus.execute_trade({
    'symbol': 'SPY',
    'action': 'sell',
    'quantity': 1,
    'order_type': 'market'
})

# Result2 will have status: "rejected"
# Reason: "Day trading not allowed: Cannot close position opened today"
```

### Allowed: Selling Position Opened Yesterday

```python
# If SPY position was opened yesterday, this will work:
result = optimus.execute_trade({
    'symbol': 'SPY',
    'action': 'sell',
    'quantity': 1,
    'order_type': 'market'
})

# Result: ✅ Order executed
```

### Allowed: Adding to Position (Same Day)

```python
# Buying more of a position you already own is allowed:
result = optimus.execute_trade({
    'symbol': 'SPY',
    'action': 'buy',
    'quantity': 1,
    'order_type': 'market'
})

# Result: ✅ Order executed (adding to position)
```

## Position Tracking

### Entry Time Storage

Positions are tracked with:
- `entry_time`: ISO format timestamp when position was opened
- `entry_price`: Average entry price
- `quantity`: Number of shares
- `side`: 'long' or 'short'

### Position Sync

On initialization, Optimus:
1. Syncs positions from Alpaca
2. Tracks entry times for day trading prevention
3. Updates position data periodically

## Logging

Day trading attempts are logged:
```
[Optimus LOG] ⚠️ DAY TRADING BLOCKED: Attempting to close SPY position opened today at 2025-11-04 11:58:35
```

## Configuration

### Enforcement Level

- **Strict**: Default - Blocks all same-day closes
- **Conservative**: If entry time unknown, blocks order (safer)
- **Error handling**: On any error, blocks order

### Cannot Be Disabled

Day trading prevention is **always active** and cannot be disabled. This is a safety feature to prevent:
- Pattern day trader restrictions
- Overtrading
- Impulsive decisions

## Compliance

This feature helps with:
- **FINRA Pattern Day Trader Rules**: Avoids PDT restrictions
- **Risk Management**: Prevents rapid trading
- **Strategy Discipline**: Encourages holding positions overnight

## FAQ

**Q: Can I close a position opened today if it's a loss?**
A: No. Day trading prevention blocks all same-day closes regardless of profit/loss.

**Q: Can I add to a position opened today?**
A: Yes. Adding to existing positions is allowed. Only closing is blocked.

**Q: When can I close a position?**
A: You can close a position the day after it was opened (or later).

**Q: What if I need to close urgently?**
A: The system is designed to prevent day trading. You would need to wait until the next trading day.

**Q: Does this apply to all trading modes?**
A: Yes. Day trading prevention is active in SANDBOX, PAPER, and LIVE modes.

## Status

✅ **Day Trading Prevention: ACTIVE**

- Same-day position closing: **BLOCKED**
- Entry time tracking: **ACTIVE**
- Position sync: **ACTIVE**
- Logging: **ACTIVE**

---

**Last Updated**: 2025-01-27


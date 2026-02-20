# NAE Trading System - Complete Process Documentation

## Overview

The NAE Trading System is an automated trading platform with extreme aggressive risk parameters designed for maximum returns. This document explains the complete workflow, intervals, timing, feedback loops, and learning mechanisms.

---

## System Architecture

### Core Components

1. **TradierClient** - Centralized API client with retries
2. **TradingSafetyManager** - Pre-trade checks and risk management
3. **TradingState** - State tracking for circuit breakers
4. **Fractional Kelly** - Position sizing algorithm

---

## Complete Process Flow

### Phase 1: Initialization (Startup)

```
Time: T+0 seconds
Duration: ~2-5 seconds

Steps:
1. Initialize TradierClient
   - Load API credentials (env vars or vault)
   - Configure retry strategy (3 attempts, exponential backoff)
   - Set up HTTP session with retries

2. Initialize TradingSafetyManager
   - Set extreme aggressive parameters:
     * Kelly fraction: 90%
     * Max position size: 25% of equity
     * Daily loss limit: 35%
     * Circuit breaker: 50% drawdown
     * Error tolerance: 10 consecutive errors

3. Initialize PDT Checker (if available)
   - Load trading history
   - Set up day trading compliance tracking

4. Log initialization status
   - Display risk parameters
   - Show warnings about extreme risk
```

**Output:**
```
TradierClient initialized (sandbox=True)
TradingSafetyManager initialized - EXTREME AGGRESSIVE MODE
⚠️  EXTREME RISK SETTINGS ENABLED - Maximum risk for maximum returns
  Position sizing: 90% Kelly, 25% max position size (EXTREME)
  Daily loss limit: 35% (EXTREME)
  Circuit breaker drawdown: 50% (EXTREME)
```

---

### Phase 2: Continuous Trading Loop

#### Cycle Interval: 30-60 seconds (randomized)

```
Time: T+5 seconds (first cycle)
Interval: Random between 30-60 seconds per cycle
Duration: Continuous (24/7 when market is open)
```

#### Cycle Steps:

**Step 1: Daily State Reset (9:30 AM ET)**
```
Time: 9:30 AM ET (once per day)
Action:
- Reset initial_equity = 0
- Reset current_equity = 0
- Reset daily_loss_pct = 0
- Reset trading_paused = False
- Log: "Daily state reset"
```

**Step 2: Market Hours Check**
```
Frequency: Every cycle (30-60s)
Checks:
- Is it a weekday? (Monday-Friday)
- Is time between 9:30 AM - 4:00 PM ET?
- Is time NOT in first 10 minutes? (9:30-9:40 AM)
- Is time NOT in last 20 minutes? (3:40-4:00 PM)

If FAILED:
- Log: "Outside trading hours: {reason}"
- Wait: 3600 seconds (1 hour)
- Return to Step 2

If PASSED:
- Continue to Step 3
```

**Step 3: Pre-Trade Safety Checks**
```
Frequency: Before EVERY order
Checks (in order):

3.1 Trading Paused Check
- If trading_paused == True:
  → BLOCK: "Trading paused (circuit breaker or daily loss limit)"
  → Exit cycle

3.2 Market Hours Check
- Call _is_market_hours()
- If FAILED: BLOCK with reason

3.3 Buying Power Check
- Call client.get_balances()
- Check: buying_power >= $25.00
- If FAILED: BLOCK "Buying power ${amount} below floor $25.00"

3.4 Daily Loss Check
- Get current equity from balances
- If initial_equity == 0:
  → Set initial_equity = current_equity
- Calculate: daily_loss_pct = (initial_equity - current_equity) / initial_equity
- Check: daily_loss_pct < 35%
- If FAILED:
  → Call pause_trading()
  → BLOCK: "Daily loss {pct}% exceeds limit 35%"
  → Send alert

3.5 PDT Compliance Check
- Get equity from balances
- If equity < $25,000:
  → Call pdt_checker.check_day_trade_allowed(symbol, side)
  → Check 5-day rolling day trade count
  → If FAILED: BLOCK "PDT violation: {reason}"

3.6 Circuit Breaker Check
- Check consecutive_errors < 10
- If FAILED:
  → Call pause_trading()
  → BLOCK: "Circuit breaker: {count} consecutive errors"
  → Send alert
- Check drawdown < 50%
- Calculate: drawdown_pct = (initial_equity - current_equity) / initial_equity
- If FAILED:
  → Call pause_trading()
  → BLOCK: "Circuit breaker: Drawdown {pct}% exceeds limit 50%"
  → Send alert

If ALL PASSED:
- Continue to Step 4
```

**Step 4: Position Sizing Calculation**
```
Frequency: Before every order
Algorithm: Fractional Kelly Criterion

Inputs:
- equity: Current account equity
- win_rate: Win probability (0.0 to 1.0)
- avg_win: Average win amount ($)
- avg_loss: Average loss amount ($)
- price: Current stock price

Calculation:
1. Calculate win_odds = avg_win / avg_loss
2. Calculate full_kelly = (p * b - q) / b
   where:
   - p = win_rate
   - q = 1 - win_rate
   - b = win_odds
3. Apply fractional Kelly: kelly_pct = full_kelly * 0.90
4. Cap at maximum: kelly_pct = min(kelly_pct, 0.25)
5. Calculate notional = equity * kelly_pct
6. Calculate quantity = floor(notional / price)

Output:
- quantity: Number of shares/contracts
- notional: Dollar value of position

Example:
- Equity: $10,000
- Win Rate: 65%
- Avg Win: $200
- Avg Loss: $100
- Price: $150
- Result: ~13 shares, $1,950 notional (19.5% of equity)
```

**Step 5: Order Submission**
```
Frequency: After all checks pass
Process:

5.1 Prepare Order Data
- symbol: Stock symbol
- side: "buy", "sell", "buy_to_cover", "sell_short"
- quantity: Calculated from Step 4
- order_type: "market" (default)
- duration: "day" (default)

5.2 Submit via TradierClient
- Call client.submit_order(...)
- TradierClient._request() handles:
  * Retry logic (3 attempts)
  * Exponential backoff
  * Error handling
  * Raises TradierError on failure

5.3 Handle Response
- If SUCCESS:
  → Record order ID
  → Call safety_manager.record_success()
  → Reset error counter
  → Log: "Order submitted successfully"
  
- If FAILURE:
  → Call safety_manager.record_error()
  → Increment consecutive_errors
  → If consecutive_errors >= 10:
    → Trigger circuit breaker
    → Pause trading
    → Send alert
```

**Step 6: Post-Order Monitoring**
```
Frequency: Continuous (every cycle)
Actions:

6.1 Update State
- Get latest balances
- Update current_equity
- Recalculate daily_loss_pct
- Check if circuit breaker should trigger

6.2 Error Recovery
- If error occurred:
  → Wait 3600 seconds (1 hour)
  → Retry cycle
- If success:
  → Reset error counter
  → Continue normal operation

6.3 Position Monitoring
- Get positions via client.get_positions()
- Track P&L
- Monitor for exit signals
```

**Step 7: Wait for Next Cycle**
```
Duration: Random 30-60 seconds
Process:
- Generate random interval: randint(30, 60)
- Sleep for interval
- Return to Step 2
```

---

## Timing & Intervals

### Market Hours
- **Open**: 9:30 AM ET
- **Close**: 4:00 PM ET
- **Trading Window**: 9:40 AM - 3:40 PM ET (excludes filtered periods)
- **Filtered Periods**:
  - First 10 minutes: 9:30-9:40 AM (skipped)
  - Last 20 minutes: 3:40-4:00 PM (skipped)

### Cycle Timing
- **Normal Cycle**: 30-60 seconds (randomized)
- **Outside Market Hours**: 3600 seconds (1 hour)
- **After Error**: 3600 seconds (1 hour)
- **Daily Reset**: 9:30 AM ET (once per day)

### API Call Timing
- **Retry Attempts**: 3 attempts
- **Backoff Factor**: 1 second (exponential)
- **Timeout**: 30 seconds per request
- **Total Max Time**: ~90 seconds (3 attempts × 30s timeout)

---

## Feedback Loops

### Loop 1: Error Recovery
```
Trigger: API error or exception
Process:
1. Record error → increment consecutive_errors
2. Check if consecutive_errors >= 10
3. If YES:
   → Trigger circuit breaker
   → Pause trading
   → Send alert
   → Wait 1 hour
4. If NO:
   → Wait 1 hour
   → Retry operation
5. On success:
   → Reset consecutive_errors = 0
   → Resume normal operation

Frequency: On-demand (when errors occur)
Learning: System becomes more resilient to temporary failures
```

### Loop 2: Daily Loss Monitoring
```
Trigger: Every pre-trade check
Process:
1. Get current equity
2. Calculate daily_loss_pct
3. If daily_loss_pct >= 35%:
   → Pause trading
   → Send alert
   → Stop all trading until next day
4. If daily_loss_pct < 35%:
   → Continue trading
   → Update state

Frequency: Every cycle (30-60s)
Learning: Prevents catastrophic daily losses
```

### Loop 3: Position Sizing Adaptation
```
Trigger: Before every order
Process:
1. Calculate win_rate from historical data
2. Calculate avg_win/avg_loss from trade history
3. Apply Kelly Criterion:
   - Higher win_rate → Larger positions
   - Better risk/reward → Larger positions
4. Cap at 25% of equity (extreme mode)
5. Execute order with calculated size

Frequency: Every order
Learning: Position sizes adapt to strategy performance
```

### Loop 4: Circuit Breaker Protection
```
Trigger: Every pre-trade check
Process:
1. Check consecutive_errors
2. Check current drawdown
3. If either threshold exceeded:
   → Pause trading immediately
   → Send alert
   → Wait for manual intervention or daily reset
4. If thresholds OK:
   → Continue trading

Frequency: Every cycle (30-60s)
Learning: Protects account from catastrophic losses
```

### Loop 5: PDT Compliance
```
Trigger: Before every sell order (if account < $25k)
Process:
1. Check if position was opened today
2. Count day trades in last 5 business days
3. If count >= 4:
   → BLOCK order
   → Log: "PDT violation"
4. If count < 4:
   → Allow order

Frequency: Before sell orders
Learning: Ensures regulatory compliance
```

---

## Learning & Adaptation Mechanisms

### 1. Position Sizing Learning
```
Data Sources:
- Historical win_rate
- Historical avg_win
- Historical avg_loss
- Current equity

Adaptation:
- Win rate improves → Larger positions
- Risk/reward improves → Larger positions
- Account grows → Absolute position sizes increase

Update Frequency: Every order
```

### 2. Risk Parameter Learning
```
Current Settings (Extreme Mode):
- Kelly Fraction: 90% (near full Kelly)
- Max Position: 25% of equity
- Daily Loss Limit: 35%
- Circuit Breaker: 50% drawdown

Adaptation:
- These are FIXED for extreme mode
- No automatic adjustment
- Manual intervention required to change

Future Enhancement:
- Could implement adaptive risk based on:
  * Recent performance
  * Volatility regime
  * Market conditions
```

### 3. Error Pattern Learning
```
Tracking:
- consecutive_errors counter
- last_error_time timestamp
- error types and frequencies

Learning:
- System learns which errors are transient
- Waits longer after repeated errors
- Resets counter on success

Adaptation:
- More resilient to temporary API issues
- Circuit breaker prevents infinite retry loops
```

### 4. Market Timing Learning
```
Current Behavior:
- Fixed time filters (first 10min, last 20min)
- Fixed market hours (9:30 AM - 4:00 PM ET)

Future Enhancement:
- Could learn optimal entry/exit times
- Could adapt filters based on volatility
- Could learn market regime patterns
```

---

## State Tracking

### TradingState Object
```python
@dataclass
class TradingState:
    consecutive_errors: int = 0          # Error counter
    initial_equity: float = 0.0          # Start-of-day equity
    current_equity: float = 0.0          # Current equity
    daily_loss_pct: float = 0.0         # Daily loss percentage
    trading_paused: bool = False         # Pause flag
    last_error_time: Optional[datetime] # Last error timestamp
```

### State Updates
```
Every Cycle:
- Update current_equity
- Recalculate daily_loss_pct
- Check circuit breaker conditions

On Error:
- Increment consecutive_errors
- Update last_error_time

On Success:
- Reset consecutive_errors = 0

Daily Reset (9:30 AM):
- Reset all daily tracking fields
- Reset trading_paused = False
```

---

## Risk Parameters Summary

### Extreme Aggressive Mode (Current)

| Parameter | Value | Original | Increase |
|-----------|-------|----------|----------|
| Kelly Fraction | 90% | 20% | 4.5x |
| Max Position Size | 25% | 2% | 12.5x |
| Daily Loss Limit | 35% | 5% | 7x |
| Circuit Breaker | 50% | 10% | 5x |
| Error Tolerance | 10 | 3 | 3.3x |
| Min Buying Power | $25 | $100 | 0.25x |

### Impact on Trading

**Position Sizing:**
- Can use up to 25% of equity per trade
- Near-full Kelly (90%) for maximum growth
- Very large positions possible

**Risk Tolerance:**
- Can withstand 35% daily losses
- Can tolerate 50% drawdowns
- Very high risk/reward profile

**Error Handling:**
- More resilient to temporary failures
- Allows 10 consecutive errors before circuit breaker
- Longer recovery periods

---

## Performance Metrics

### What NAE Tracks

1. **Account Metrics**
   - Equity (current, initial)
   - Buying power
   - Cash balance
   - Daily P&L

2. **Trade Metrics**
   - Win rate
   - Average win
   - Average loss
   - Position sizes
   - Notional values

3. **Risk Metrics**
   - Daily loss percentage
   - Drawdown percentage
   - Consecutive errors
   - Circuit breaker status

4. **Compliance Metrics**
   - PDT day trade count
   - Rolling 5-day count
   - Position holding periods

---

## Alert System

### Alert Triggers

1. **Circuit Breaker Triggered**
   - Message: "🚨 CIRCUIT BREAKER TRIGGERED: Trading paused - {reason}"
   - Action: Pause all trading
   - Frequency: Immediate

2. **Daily Loss Limit Exceeded**
   - Message: "Daily loss {pct}% exceeds limit 35%"
   - Action: Pause trading for rest of day
   - Frequency: Once per day (if triggered)

3. **PDT Violation**
   - Message: "PDT violation: {reason}"
   - Action: Block order
   - Frequency: Per blocked order

### Alert Channels (Future)
- Email notifications
- SMS alerts
- Slack/Discord webhooks
- Dashboard updates

---

## Example Complete Cycle

```
Time: 10:15:32 AM ET
Cycle #: 47

[Step 1] Daily Reset Check
  → Not 9:30 AM, skip

[Step 2] Market Hours Check
  → Current time: 10:15 AM
  → Weekday: Yes
  → After 9:40 AM: Yes
  → Before 3:40 PM: Yes
  → ✅ PASSED

[Step 3] Pre-Trade Checks
  [3.1] Trading Paused: No
  [3.2] Market Hours: ✅ PASSED
  [3.3] Buying Power: $8,000 >= $25 ✅ PASSED
  [3.4] Daily Loss: 2.3% < 35% ✅ PASSED
  [3.5] PDT: Account $10k < $25k, check day trades
         → Day trades in 5 days: 2 < 4 ✅ PASSED
  [3.6] Circuit Breaker: 0 errors < 10 ✅ PASSED
         → Drawdown: 2.3% < 50% ✅ PASSED
  → ✅ ALL CHECKS PASSED

[Step 4] Position Sizing
  → Equity: $10,000
  → Win Rate: 65%
  → Avg Win: $200
  → Avg Loss: $100
  → Price: $150
  → Calculate Kelly: 19.5% of equity
  → Notional: $1,950
  → Quantity: 13 shares

[Step 5] Order Submission
  → Symbol: TSLA
  → Side: buy
  → Quantity: 13
  → Submit via TradierClient
  → Retry logic: Success on attempt 1
  → Order ID: ORDER_1705324532
  → Status: filled
  → ✅ SUCCESS

[Step 6] Post-Order
  → Record success
  → Reset error counter: 0
  → Update state
  → Log: "Order executed successfully"

[Step 7] Wait
  → Random interval: 42 seconds
  → Sleep: 42 seconds
  → Next cycle: 10:16:14 AM ET
```

---

## Learning Summary

### What NAE Has Learned

1. **Position Sizing**
   - Uses Kelly Criterion for optimal sizing
   - Adapts to win rate and risk/reward
   - Caps at 25% for extreme mode

2. **Risk Management**
   - Pre-trade checks prevent bad trades
   - Circuit breakers prevent catastrophic losses
   - Daily limits prevent excessive drawdowns

3. **Error Handling**
   - Retry logic handles transient failures
   - Circuit breaker prevents infinite loops
   - Recovery mechanisms restore operation

4. **Market Timing**
   - Avoids volatile open/close periods
   - Respects market hours
   - Adapts to trading day structure

### What NAE Implements

1. **Automated Trading**
   - Continuous monitoring (30-60s cycles)
   - Automated order execution
   - Position management

2. **Risk Controls**
   - Pre-trade validation
   - Position sizing
   - Circuit breakers
   - Daily loss limits

3. **Compliance**
   - PDT rule enforcement
   - Regulatory compliance
   - Trade logging

4. **Resilience**
   - Error recovery
   - Retry mechanisms
   - State management

---

## Conclusion

The NAE Trading System operates as a continuous loop with:
- **30-60 second cycles** during market hours
- **Comprehensive pre-trade checks** before every order
- **Extreme aggressive risk parameters** for maximum returns
- **Multiple feedback loops** for learning and adaptation
- **Robust error handling** and recovery mechanisms

The system is designed to maximize returns while maintaining safety through circuit breakers and daily limits. All operations are logged and monitored for continuous improvement.


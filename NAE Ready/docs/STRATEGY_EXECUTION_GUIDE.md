# Strategy Execution Guide

## Quick Start: Execute More Strategies

### Option 1: Batch Execution (Recommended)
```bash
cd NAE
python3 scripts/batch_execute_strategies.py
```

This script:
- ✅ Creates multiple strategies aligned with current phase (Phase 1: Wheel Strategy)
- ✅ Generates additional strategies from Ralph
- ✅ Validates all strategies through Donnie
- ✅ Executes up to 10 strategies through Optimus
- ✅ Enforces PDT prevention (all positions hold overnight)
- ✅ Tracks progress toward $5M goal

### Option 2: Comprehensive Execution
```bash
cd NAE
python3 scripts/execute_strategies.py
```

This script provides more detailed output and analysis.

---

## Strategy Types Executed

### Phase 1: Tier 1 - Wheel Strategy
- **Cash-Secured Puts** on large-cap stocks (SPY, AAPL, MSFT, QQQ, TSLA)
- **Position Size:** 5% of NAV per position
- **Target Return:** 12-30% annual
- **PDT Compliant:** ✅ Yes (hold overnight minimum)
- **Execution:** Simulated as stock purchases (representing put assignment)

### Phase 2: Tier 2 - Momentum Plays
- **Long Calls** on high-volume stocks
- **Position Size:** 3% of NAV per position
- **Target Return:** 30-60% annual
- **PDT Compliant:** ✅ Yes (entry at close, exit next day or later)

---

## Execution Results

### Recent Execution (Latest Run)
- **Total Strategies Created:** 10
- **Validated by Donnie:** 10/10 (100%)
- **Executed through Optimus:** 10
- **Status:** Orders submitted to Alpaca (paper trading)
- **PDT Prevention:** ✅ Active (blocked same-day exits)

### Strategy Sources
1. **Predefined Strategies:** 8 Wheel Strategy strategies
2. **Ralph-Generated:** 3 additional strategies from learning system
3. **Total Available:** 10 strategies

---

## How to Execute More Strategies

### Method 1: Run Batch Script Again
Simply run the batch execution script again - it will:
- Generate new strategies from Ralph
- Create phase-appropriate strategies
- Execute them through the full flow

### Method 2: Manual Execution
```python
from agents.optimus import OptimusAgent
from agents.donnie import DonnieAgent

optimus = OptimusAgent(sandbox=False)  # Paper mode
donnie = DonnieAgent()

# Create strategy
strategy = {
    "symbol": "SPY",
    "side": "buy",
    "quantity": 1,
    "order_type": "market",
    "strategy_name": "My Strategy",
    "trust_score": 75,
    "backtest_score": 65,
    "pdt_compliant": True
}

# Validate
if donnie.validate_strategy(strategy):
    # Execute
    result = optimus.execute_trade(strategy)
    print(f"Status: {result.get('status')}")
```

### Method 3: Use Ralph to Generate Strategies
```python
from agents.ralph import RalphAgent

ralph = RalphAgent()
strategies = ralph.generate_strategies()  # Generates new strategies
# Then execute through Donnie → Optimus
```

---

## Strategy Execution Flow

```
1. Strategy Creation
   ├─ Predefined strategies (aligned with phase)
   └─ Ralph-generated strategies (from learning)

2. Validation (Donnie)
   ├─ Trust score ≥ 55
   ├─ Backtest score ≥ 50
   └─ PDT compliance check

3. Execution (Optimus)
   ├─ Entry timing analysis
   ├─ Kelly Criterion position sizing
   ├─ Smart Order Routing
   ├─ Pre-trade safety checks
   ├─ PDT prevention enforcement
   └─ Trade execution

4. Monitoring
   ├─ Exit timing analysis (continuous)
   ├─ PDT compliance monitoring
   ├─ NAV tracking (compound growth)
   └─ Progress toward $5M goal
```

---

## Key Features

### ✅ PDT Prevention
- All positions must hold overnight (minimum 1 day)
- Same-day round trips are BLOCKED
- Automatic detection and blocking

### ✅ Entry Timing
- Technical analysis (RSI, MACD, MAs, Bollinger Bands)
- Optimal entry price calculation
- Timing score (0-100) - rejects if < 40

### ✅ Exit Timing
- Profit targets (10%+)
- Trailing stops (activates at 5% profit)
- Stop losses (2%)
- Trend reversal detection

### ✅ Compound Growth
- NAV updates automatically
- Progress tracking toward $5M
- Phase-aware position sizing

---

## Troubleshooting

### Orders Show "PENDING_NEW"
- **Normal:** Orders are submitted to Alpaca
- **Action:** Wait for fill (usually within seconds)
- **Check:** Alpaca dashboard or Optimus status

### Orders Rejected: "PDT Prevention"
- **Expected:** System is blocking same-day round trips
- **Action:** This is correct behavior - positions must hold overnight
- **Note:** If you have an existing position opened today, you cannot close it same day

### No Strategies Generated from Ralph
- **Normal:** First run may have no strategies
- **Action:** Script uses predefined strategies
- **Future:** Ralph learns and generates more over time

---

## Monitoring Execution

### Check Optimus Status
```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=False)
status = optimus.get_trading_status()

print(f"NAV: ${status['nav']:.2f}")
print(f"Daily P&L: ${status['daily_pnl']:.2f}")
print(f"Open Positions: {status['open_positions']}")
print(f"Goal Progress: {(status['nav'] / 5000000) * 100:.4f}%")
```

### Check Logs
```bash
# Optimus logs
tail -f logs/optimus.log

# Donnie logs
tail -f logs/donnie.log

# Ralph logs
tail -f logs/ralph.log
```

---

## Next Steps

1. **Monitor Positions:** Check Alpaca dashboard for filled orders
2. **Review Logs:** Check entry/exit timing analysis in logs
3. **Track Progress:** Monitor NAV growth toward $5M goal
4. **Execute More:** Run batch script again for additional strategies
5. **Scale Up:** As NAV grows, system automatically transitions to Phase 2

---

## Alignment with Goals

✅ **Goal #1:** Achieve generational wealth - Compound growth enabled  
✅ **Goal #2:** Generate $5M in 8 years - Progress tracking active  
✅ **Goal #3:** Optimize options trading - Strategies aligned with long-term plan

---

**END OF GUIDE**


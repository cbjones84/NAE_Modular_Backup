# Micro-Scalp Accelerator Strategy - Implementation Guide

## Overview

The **Micro-Scalp Accelerator** is a temporary, ultra-aggressive strategy designed to bootstrap Optimus's account from $100 to $500-$1000 quickly. This strategy is **NOT** intended to replace Optimus's main long-term generational wealth strategy, but rather to provide initial capital growth before transitioning to more stable, long-term strategies.

## Key Features

### ✅ Tradier Cash Account Compliance

**IMPORTANT**: Tradier cash accounts are **NOT subject to Pattern Day Trading (PDT) rules**. The PDT rule only applies to margin accounts. However, cash accounts must still respect settlement rules:

- **Options settle T+1** (next business day)
- **Stocks settle T+2** (two business days)
- The accelerator includes **settled cash tracking** to prevent free-riding violations

### Strategy Parameters

- **Instrument**: SPY 0DTE (Zero Days to Expiration) options
- **Contract Price Range**: $3-$20 per contract
- **Position Size**: 1 contract initially (scales with account growth)
- **Signal Threshold**: >70% probability from Ralph
- **Hold Time**: 2-10 minutes (scalping)
- **Profit Target**: +25%
- **Stop Loss**: -15%
- **Max Trades/Day**: 2
- **Daily Drawdown Limit**: -25% (stops trading for the day)
- **Target Account Size**: $8000-$10000 (auto-disables when reached)
- **Weekly Return Target**: 4.3% (aligned with generational wealth goal)

### Risk Management Features

1. **Volatility Filters**
   - IV percentile: 20%-95% (avoids dead low IV and extreme tail risk)
   - ATR filter: Ensures underlying moves enough intraday

2. **Spread-Aware Exits**
   - Rejects contracts with >20% bid-ask spread
   - Adjusts profit/stop targets if spread widens during trade

3. **Time-of-Day Filters**
   - Trades only during high-probability windows:
     - 9:45 AM - 10:30 AM
     - 1:00 PM - 3:30 PM

4. **Kelly Criterion Position Sizing**
   - Computes optimal position size based on win probability and payout ratio
   - Caps risk at 12% of account per trade

5. **Settled Cash Enforcement**
   - Tracks reserved/unsettled cash
   - Prevents free-riding violations
   - Only uses settled funds for new trades

6. **Session-Based Retraining**
   - Feeds trade results back to Ralph for model adjustments
   - Retrains every 60 minutes or after 3+ trades

## Implementation Files

### 1. Settlement Ledger (`tools/settlement_utils.py`)

Tracks settled vs unsettled cash to prevent free-riding violations.

**Key Methods**:
- `get_settled_cash()`: Get available settled cash
- `reserve_for_order()`: Reserve cash when placing order
- `release_settled()`: Clear settled reservations
- `ensure_funds_available()`: Check if funds are available

### 2. Advanced Accelerator (`tools/profit_algorithms/advanced_micro_scalp.py`)

Main accelerator strategy module.

**Key Classes**:
- `AcceleratorConfig`: Configuration dataclass
- `AdvancedMicroScalpAccelerator`: Main strategy class

**Key Methods**:
- `execute()`: Run one accelerator cycle
- `pick_direction()`: Get direction from Ralph signals
- `pick_contract()`: Select optimal contract
- `execute_trade()`: Execute and monitor trade
- `get_status()`: Get current status

### 3. Ralph Signal Integration (`agents/ralph.py`)

Added methods to Ralph for signal generation:

- `get_intraday_direction_probability(symbol)`: Returns probability of up/down moves
- `retrain_hook(summary)`: Session-based retraining hook

### 4. Optimus Integration (`agents/optimus.py`)

Added accelerator mode to Optimus:

- `enable_accelerator_mode()`: Enable accelerator
- `disable_accelerator_mode()`: Disable accelerator
- `run_accelerator_cycle()`: Run one cycle

### 5. Tradier Adapter Enhancements (`execution/broker_adapters/tradier_adapter.py`)

Added methods for settled cash tracking:

- `get_account_balance()`: Get total account balance
- `get_buying_power()`: Get buying power
- `get_settled_cash()`: Get settled cash
- `get_unsettled_cash()`: Get unsettled cash

## Usage Instructions

### Step 1: Enable Accelerator Mode

```python
from agents.optimus import OptimusAgent
from agents.ralph import RalphAgent

# Initialize Optimus
optimus = OptimusAgent(sandbox=False)  # Use live trading

# Initialize Ralph (for signals)
ralph = RalphAgent()

# Enable accelerator mode
optimus.enable_accelerator_mode(ralph_agent=ralph)
```

### Step 2: Run Accelerator Cycles

```python
# Run accelerator cycle (call this periodically during trading hours)
result = optimus.run_accelerator_cycle()

# Check result
if result == "TARGET_REACHED":
    print("Account reached $500 - consider disabling accelerator")
elif result == "DAILY_DRAWDOWN_EXCEEDED":
    print("Daily drawdown limit hit - trading stopped for today")
```

### Step 3: Monitor Status

```python
# Get accelerator status
status = optimus.accelerator.get_status()
print(f"Daily P&L: ${status['daily_profit']:.2f}")
print(f"Trades today: {status['trades_today']}")
print(f"Account size: ${status['account_size']:.2f}")
```

### Step 4: Disable When Target Reached

```python
# Disable accelerator when account reaches target
if optimus.get_account_size() >= 8000:
    optimus.disable_accelerator_mode()
    print("Accelerator disabled - switching to main strategy")
```

## Dual-Mode Operation (Sandbox + Live)

The accelerator can run in **dual-mode** for simultaneous testing and production:

### Using Accelerator Controller

```bash
# Run both sandbox and live simultaneously
python3 -m execution.integration.accelerator_controller --sandbox --live --interval 60

# Or run separately:
# Sandbox only (testing)
python3 -m execution.integration.accelerator_controller --sandbox --no-live

# Live only (production)
python3 -m execution.integration.accelerator_controller --live --no-sandbox
```

### Deployment Script

Use the deployment script to push to GitHub and start both modes:

```bash
cd "NAE Ready"
./scripts/deploy_accelerator.sh
```

This script will:
1. Commit all changes to GitHub
2. Push to the `prod` branch
3. Start sandbox testing
4. Start live production
5. Start NAE master controller

## Integration with NAE Master Controller

To integrate into NAE's autonomous master controller, add a periodic task:

```python
# In nae_autonomous_master.py or similar
def run_accelerator_cycle():
    if optimus.accelerator_enabled:
        result = optimus.run_accelerator_cycle()
        logger.info(f"Accelerator cycle: {result}")

# Schedule to run every minute during trading hours
schedule.every(1).minutes.do(run_accelerator_cycle)
```

## Weekly Returns Tracking

The accelerator is configured with a **4.3% weekly return target** aligned with Optimus's generational wealth goal. Track progress:

```python
# Calculate weekly return
start_balance = 100.0  # Starting balance
current_balance = optimus.get_account_size()
weekly_return = (current_balance - start_balance) / start_balance

# Check against target
target_return = 0.043  # 4.3%
if weekly_return >= target_return:
    print(f"✅ Weekly target achieved: {weekly_return:.1%}")
```

## Safety Checklist

Before deploying:

- [ ] Confirm broker account type = Cash (not margin)
- [ ] Verify Tradier API has `get_settled_cash()` or equivalent
- [ ] Test in paper/sandbox mode for 3-7 days
- [ ] Confirm win-rate and average payout meet expectations
- [ ] Set hard manual kill-switch for unexpected behavior
- [ ] Monitor daily and review performance weekly
- [ ] Disable accelerator when account reaches $500-$1000

## When to Disable Accelerator

Disable the accelerator automatically when:

1. **Account reaches $8000-$10000** (target achieved)
2. **After 6 weeks** (maximum recommended duration)
3. **If daily drawdown limit hit 3+ times** (strategy not working)

After disabling, Optimus should transition to:

- Credit spreads
- High-probability swings
- Longer-term institutional setups
- Reinforcement learning optimization
- Ralph-fed strategy filtering

## Risk Warnings

⚠️ **This is an aggressive, high-risk strategy**:

- Small accounts ($100) are highly vulnerable to ruin
- 0DTE options are extremely volatile
- High probability of account blow-up if edge doesn't hold
- Only use as temporary bootstrap strategy
- **Never** replace main generational wealth strategy with this

## Performance Expectations

**Realistic expectations for $100 account**:

- **Best case**: 2-3 weeks to reach $500 (5x growth)
- **Typical case**: 4-6 weeks to reach $300-$500 (3-5x growth)
- **Worst case**: Account blow-up (high risk)

**Success factors**:

- High-quality signals from Ralph (>70% probability)
- Strict adherence to risk limits
- Good execution (minimal slippage)
- Favorable market conditions (volatility, trends)

## Troubleshooting

### "NO_SETTLED_FUNDS" Error

- Check settlement ledger status
- Wait for previous trades to settle (T+1 for options)
- Reduce position size if needed

### "NO_HIGH_CONFIDENCE_SIGNAL" Error

- Normal - strategy only trades when probability >70%
- Ralph may need more data or retraining
- Check market conditions (low volatility = fewer signals)

### "DAILY_DRAWDOWN_EXCEEDED" Error

- Daily loss limit (-25%) hit
- Trading stopped for the day (safety feature)
- Review strategy performance and adjust if needed

### "TARGET_REACHED" Status

- Account reached $500 target
- Consider disabling accelerator
- Transition to main strategy

## Support and Updates

For questions or issues:

1. Check logs: `logs/optimus.log` and `logs/ralph.log`
2. Review accelerator status: `optimus.accelerator.get_status()`
3. Check settlement status: `optimus.accelerator.ledger.get_settlement_status()`

## Conclusion

The Micro-Scalp Accelerator is a powerful tool for rapid account growth, but it must be used responsibly and temporarily. Always prioritize long-term generational wealth goals over short-term gains. Once the account reaches a stable size ($500-$1000), transition to Optimus's main strategies for sustainable, long-term growth.

---

**Remember**: This strategy is designed to bootstrap Optimus's account, not replace the main generational wealth strategy. Use it wisely and transition to long-term strategies as soon as practical.


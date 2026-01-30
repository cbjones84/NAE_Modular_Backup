# Optimus Alpaca Paper Trading Setup ✅

## Overview

Optimus has been updated to use Alpaca for paper trading. This allows NAE to test strategies with real market data and execution through Alpaca's paper trading environment.

## Changes Made

### 1. **AlpacaAdapter Integration**
- Replaced placeholder `AlpacaClient` with real `AlpacaAdapter` from `adapters.alpaca`
- AlpacaAdapter automatically loads credentials from secure vault
- Supports full Alpaca SDK functionality (market/limit orders, options, etc.)

### 2. **Trading Mode Configuration**
- **PAPER mode**: Uses Alpaca paper trading (default when `sandbox=False`)
- **SANDBOX mode**: Uses simulated trading (for testing without API)
- **LIVE mode**: Uses E*Trade or IBKR for live trading

### 3. **Execution Priority**
For paper trading, Optimus now prioritizes:
1. **Alpaca** (preferred) - Real paper trading with Alpaca
2. **E*Trade** (fallback) - If Alpaca unavailable
3. **Simulated** (fallback) - If no brokers available

## Setup

### Prerequisites

1. **Install Alpaca SDK**:
   ```bash
   pip install alpaca-py
   ```

2. **Credentials Already Configured** ✅
   - Credentials are stored in secure vault
   - AlpacaAdapter automatically loads them
   - No additional configuration needed

### Initialize Optimus

```python
from agents.optimus import OptimusAgent

# Initialize in PAPER mode (uses Alpaca)
optimus = OptimusAgent(sandbox=False)

# Verify Alpaca is configured
if optimus.alpaca_client:
    print("✅ Alpaca configured for paper trading")
    print(f"   Paper Trading: {optimus.alpaca_client.paper_trading}")
```

## Usage

### Execute Trades Through Alpaca

```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=False)

# Execute a trade (will use Alpaca in paper mode)
execution_details = {
    "symbol": "AAPL",
    "action": "buy",
    "quantity": 1,
    "order_type": "market",
    "time_in_force": "day"
}

result = optimus.execute_trade(execution_details)
print(f"Order ID: {result['order_id']}")
print(f"Broker: {result['broker']}")  # Should be "alpaca"
print(f"Status: {result['status']}")
```

### Check Account Status

```python
# Get account information from Alpaca
if optimus.alpaca_client:
    account = optimus.alpaca_client.adapter.get_account()
    print(f"Cash: ${account['cash']:,.2f}")
    print(f"Buying Power: ${account['buying_power']:,.2f}")
    print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
```

### Get Positions

```python
# Get current positions
if optimus.alpaca_client:
    positions = optimus.alpaca_client.adapter.get_positions()
    for pos in positions:
        print(f"{pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
```

## Integration with NAE Agents

### From Donnie (Strategy Execution)

```python
from agents.donnie import DonnieAgent
from agents.optimus import OptimusAgent

donnie = DonnieAgent()
optimus = OptimusAgent(sandbox=False)  # Paper trading with Alpaca

# Execute strategy
strategy = {
    "name": "Test Strategy",
    "trust_score": 75,
    "backtest_score": 60,
    "symbol": "AAPL",
    "action": "buy",
    "quantity": 1
}

donnie.execute_strategy(strategy, sandbox=False, optimus_agent=optimus)
```

### From Ralph (Strategy Generation)

```python
from agents.ralph import RalphAgent
from agents.optimus import OptimusAgent

ralph = RalphAgent()
optimus = OptimusAgent(sandbox=False)

# Generate and test strategies
strategies = ralph.generate_strategies()

# Execute top strategy
if strategies:
    top_strategy = strategies[0]
    result = optimus.execute_trade({
        "symbol": top_strategy.get("symbol", "SPY"),
        "action": "buy",
        "quantity": 1,
        "order_type": "market"
    })
```

## Testing

### Test Alpaca Connection

```python
from agents.optimus import OptimusAgent

optimus = OptimusAgent(sandbox=False)

# Check if Alpaca is configured
if optimus.alpaca_client:
    print("✅ Alpaca configured")
    
    # Test authentication
    if optimus.alpaca_client.adapter.auth():
        print("✅ Alpaca authentication successful")
        
        # Get account
        account = optimus.alpaca_client.adapter.get_account()
        print(f"✅ Account: {account.get('account_id', 'N/A')}")
        print(f"   Cash: ${account.get('cash', 0):,.2f}")
    else:
        print("❌ Alpaca authentication failed")
else:
    print("❌ Alpaca not configured")
    print("   Install: pip install alpaca-py")
```

### Test Trade Execution

```python
# Small test order
test_order = {
    "symbol": "SPY",
    "action": "buy",
    "quantity": 1,
    "order_type": "market",
    "time_in_force": "day"
}

result = optimus.execute_trade(test_order)
print(f"Result: {result}")
```

## Configuration

### Trading Mode Selection

```python
# Paper trading (uses Alpaca)
optimus = OptimusAgent(sandbox=False)  # TradingMode.PAPER

# Sandbox (simulated, no API)
optimus = OptimusAgent(sandbox=True)   # TradingMode.SANDBOX

# Live trading (uses E*Trade or IBKR)
optimus.trading_mode = TradingMode.LIVE
```

### Safety Limits

Safety limits are configured for paper trading:
- Max order size: Based on NAV (100% for growth)
- Daily loss limit: 2% of NAV
- Max open positions: 5
- Consecutive loss limit: 5

## Troubleshooting

### Alpaca Not Initialized

**Problem**: `optimus.alpaca_client` is `None`

**Solutions**:
1. Install alpaca-py: `pip install alpaca-py`
2. Check credentials in vault:
   ```python
   from secure_vault import get_vault
   v = get_vault()
   print(v.get_secret('alpaca', 'api_key'))
   ```
3. Check logs: `logs/optimus.log`

### Authentication Failed

**Problem**: Alpaca authentication fails

**Solutions**:
1. Verify credentials are correct
2. Check Alpaca dashboard for account status
3. Ensure paper trading is enabled on account
4. Verify endpoint: `https://paper-api.alpaca.markets/v2`

### Orders Not Executing

**Problem**: Orders are rejected or not executing

**Solutions**:
1. Check account balance (ensure sufficient buying power)
2. Verify market hours (Alpaca may reject orders outside trading hours)
3. Check order format (symbol, quantity, side)
4. Review logs: `logs/optimus.log`

## Status

✅ **Integration Complete**

- AlpacaAdapter integrated into Optimus
- Credentials automatically loaded from vault
- Paper trading mode configured
- Ready for strategy testing

**Next Steps**:
1. Install alpaca-py: `pip install alpaca-py`
2. Test connection: Run test scripts
3. Start testing strategies: Use Optimus in PAPER mode

---

**Last Updated**: 2025-01-27


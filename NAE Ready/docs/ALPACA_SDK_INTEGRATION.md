# Alpaca SDK Integration - Complete

## Overview

The Alpaca adapter has been upgraded to use the official Alpaca Python SDK (`alpaca-py`) instead of direct REST API calls. This provides:

- ✅ **Official SDK Support**: Uses `TradingClient` from `alpaca-py`
- ✅ **Stock Trading**: Market and limit orders
- ✅ **Options Trading**: Single-leg and multi-leg options orders
- ✅ **Enhanced Error Handling**: Proper exception handling with `APIError`
- ✅ **Backward Compatibility**: Still implements `BrokerAdapter` interface
- ✅ **Paper & Live Trading**: Supports both environments

## Installation

The SDK has been added to `requirements.txt`:

```bash
pip install alpaca-py>=0.26.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Configuration

Set your API credentials via environment variables or config:

```python
config = {
    "API_KEY": "YOUR_ALPACA_API_KEY",
    "API_SECRET": "YOUR_ALPACA_SECRET_KEY",
    "paper_trading": True  # Set to False for live trading
}
```

Or use environment variables:

```bash
export APCA_API_KEY_ID="your_api_key"
export APCA_API_SECRET_KEY="your_secret_key"
```

## Usage Examples

### Basic Usage (BrokerAdapter Interface)

```python
from adapters.alpaca import AlpacaAdapter

# Initialize
alpaca = AlpacaAdapter({
    "API_KEY": "your_key",
    "API_SECRET": "your_secret",
    "paper_trading": True
})

# Authenticate
if alpaca.auth():
    # Get account
    account = alpaca.get_account()
    
    # Place order
    order = {
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "buy",
        "type": "market",
        "time_in_force": "day"
    }
    result = alpaca.place_order(order)
```

### Stock Trading

```python
from adapters.alpaca import AlpacaAdapter

alpaca = AlpacaAdapter(config)

# Market buy order
result = alpaca.buy_stock_market("AAPL", 1.0)

# Limit sell order
result = alpaca.sell_stock_limit("AAPL", 1.0, 150.00)
```

### Options Trading

```python
from adapters.alpaca import AlpacaAdapter
from alpaca.trading.enums import OrderSide

alpaca = AlpacaAdapter(config)

# Single-leg option
option_symbol = "AAPL240119C00190000"  # Full option contract symbol
result = alpaca.buy_option_market(option_symbol, 1.0)

# Multi-leg options (straddle example)
legs = [
    ("AAPL240119C00190000", OrderSide.BUY, 1),   # Buy call
    ("AAPL240119P00190000", OrderSide.BUY, 1)    # Buy put
]
result = alpaca.multi_leg_option_order(legs, 1.0)
```

## Key Features

### 1. Stock Orders
- **Market Orders**: `buy_stock_market()`, `place_order()` with `type: "market"`
- **Limit Orders**: `sell_stock_limit()`, `place_order()` with `type: "limit"`

### 2. Options Trading
- **Single-Leg**: `buy_option_market()` or `place_order()` with option symbol
- **Multi-Leg**: `multi_leg_option_order()` for spreads, straddles, etc.

### 3. Account Management
- `get_account()` - Account balance, buying power, equity
- `get_positions()` - Current positions
- `get_quote()` - Real-time quotes

### 4. Order Management
- `place_order()` - Generic order placement (BrokerAdapter interface)
- `cancel_order()` - Cancel an order
- `get_order_status()` - Check order status

## Code Review Notes

### ✅ Correct Implementation

The user's provided code was **mostly correct** with the following improvements made:

1. **Error Handling**: Added proper `APIError` exception handling
2. **Type Safety**: Added type hints and proper enum usage
3. **Backward Compatibility**: Maintained `BrokerAdapter` interface
4. **Logging**: Enhanced error messages and status reporting
5. **Options Support**: Fixed multi-leg options to work with both `OrderSide` and `OptionSide`

### Fixed Issues

1. **OptionSide vs OrderSide**: The code correctly uses `OrderSide` for option legs in `OptionLegRequest`. The SDK accepts `OrderSide` directly for options legs.

2. **Multi-leg Options**: The implementation correctly uses `OrderClass.MLEG` and `OptionLegRequest` for multi-leg orders.

3. **Paper Trading**: Correctly implemented with `paper=False` for live trading (as per user's requirement).

## Testing

Run the example file to test the integration:

```bash
cd NAE
python adapters/alpaca_integration_example.py
```

Make sure to set your API keys:

```bash
export APCA_API_KEY_ID="your_key"
export APCA_API_SECRET_KEY="your_secret"
```

## Integration with NAE Agents

The adapter works seamlessly with existing NAE agents:

```python
from adapters.manager import BrokerManager

# Get Alpaca adapter
manager = BrokerManager()
alpaca = manager.get("alpaca")

# Use in Optimus or other agents
result = alpaca.place_order({
    "symbol": "AAPL",
    "quantity": 1.0,
    "side": "buy",
    "type": "market"
})
```

## Migration Notes

### From Old REST API to SDK

The old adapter used direct REST API calls. The new adapter:
- Uses `TradingClient` instead of `requests`
- Uses `MarketOrderRequest` and `LimitOrderRequest` instead of dict orders
- Uses SDK enums (`OrderSide`, `TimeInForce`, etc.)
- Provides better error messages via `APIError`

### Backward Compatibility

The `BrokerAdapter` interface is maintained, so existing code using:
- `alpaca.place_order(order_dict)`
- `alpaca.get_account()`
- `alpaca.get_positions()`

...will continue to work without changes.

## Security Notes

⚠️ **Important**: 
- Never commit API keys to version control
- Use environment variables or secure vault for credentials
- Always test with paper trading first (`paper_trading=True`)
- Set `paper_trading=False` only for production/live trading

## Resources

- [Alpaca Python SDK Documentation](https://alpaca.markets/sdks/python/)
- [Alpaca Trading API Docs](https://docs.alpaca.markets/docs)
- [Multi-leg Options Guide](https://docs.alpaca.markets/docs/multi-leg-orders)

## Status

✅ **Integration Complete** - Ready for use with paper and live trading.


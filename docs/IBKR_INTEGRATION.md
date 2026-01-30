# Interactive Brokers (IBKR) TWS API Integration - Complete

## Overview

The IBKR adapter has been implemented using the official IB API (`ibapi`) with EWrapper/EClient pattern. This provides:

- ✅ **TWS/Gateway Integration**: Connects to Trader Workstation or IB Gateway
- ✅ **Stock Trading**: Market and limit orders
- ✅ **Options Trading**: Single-leg and multi-leg options orders
- ✅ **Account Management**: Real-time account data and positions
- ✅ **Market Data**: Real-time quotes and market data
- ✅ **Backward Compatibility**: Implements `BrokerAdapter` interface

## Prerequisites

### 1. Install TWS or IB Gateway

**Option A: Trader Workstation (TWS)**
- Download from: https://www.interactivebrokers.com/en/index.php?f=16042
- Install and run TWS

**Option B: IB Gateway (Lighter, for API only)**
- Download from: https://www.interactivebrokers.com/en/index.php?f=16457
- Install and run IB Gateway

### 2. Enable API Access

1. In TWS/Gateway, go to: **File** > **Global Configuration** > **API** > **Settings**
2. Check: **"Enable ActiveX and Socket Clients"**
3. Uncheck: **"Read-Only API"** (if you want to place orders)
4. Set **"Socket Port"**: 
   - **7497** for paper trading
   - **7496** for live trading
5. Add your IP address to **"Trusted IPs"** (or use `127.0.0.1` for localhost)

### 3. Install ibapi Package

The `ibapi` package is included in the TWS API download, but you can also install it:

```bash
pip install ibapi
```

Or it's already in `requirements.txt`:
```
ibapi>=10.0.0
```

## Configuration

Configure the IBKR adapter in `config/broker_adapters.json`:

```json
{
  "ibkr": {
    "module": "adapters.ibkr",
    "class": "IBKRAdapter",
    "config": {
      "host": "127.0.0.1",
      "port": 7497,
      "client_id": 1,
      "paper_trading": true
    }
  }
}
```

### Configuration Options

- **host**: TWS/Gateway host (default: "127.0.0.1")
- **port**: Socket port (7497 for paper, 7496 for live)
- **client_id**: Unique client ID (1-100, must be unique per connection)
- **paper_trading**: `true` for paper trading, `false` for live

## Usage Examples

### Basic Usage (BrokerAdapter Interface)

```python
from adapters.ibkr import IBKRAdapter

# Initialize
ibkr = IBKRAdapter({
    "host": "127.0.0.1",
    "port": 7497,  # 7497 for paper, 7496 for live
    "client_id": 1,
    "paper_trading": True
})

# Authenticate (check connection)
if ibkr.auth():
    # Get account
    account = ibkr.get_account()
    print(f"Cash: ${account.get('cash', 0):,.2f}")
    
    # Place order
    order = {
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "buy",
        "type": "market",
        "secType": "STK"
    }
    result = ibkr.place_order(order)
    print(f"Order ID: {result['order_id']}")
```

### Stock Trading

```python
from adapters.ibkr import IBKRAdapter

ibkr = IBKRAdapter(config)

# Market buy order
result = ibkr.buy_stock_market("AAPL", 1.0)

# Limit sell order
result = ibkr.sell_stock_limit("AAPL", 1.0, 150.00)
```

### Options Trading

```python
from adapters.ibkr import IBKRAdapter

ibkr = IBKRAdapter(config)

# Single-leg option (call)
result = ibkr.buy_option_market(
    symbol="AAPL",
    lastTradeDate="20241220",  # Expiration: YYYYMMDD
    strike=150.0,
    right="C",  # 'C' for call, 'P' for put
    qty=1.0
)

# Multi-leg options (straddle example)
# First, resolve contract IDs for each leg
# (This requires additional contract resolution - see advanced usage)

# Create combo contract
legs = [
    (conId1, 1, "BUY", "SMART"),  # Buy call
    (conId2, 1, "BUY", "SMART")   # Buy put
]
combo = ibkr.create_combo_contract("AAPL", legs)
result = ibkr.place_multi_leg_order(combo, "BUY", 1.0)
```

### Account and Positions

```python
# Get account information
account = ibkr.get_account()
print(f"Cash: ${account['cash']:,.2f}")
print(f"Buying Power: ${account['buying_power']:,.2f}")
print(f"Equity: ${account['equity']:,.2f}")

# Get positions
positions = ibkr.get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
```

### Using with Adapter Manager

```python
from adapters.manager import AdapterManager

# Get IBKR adapter
manager = AdapterManager()
ibkr = manager.get("ibkr")

# Use in agents
result = ibkr.place_order({
    "symbol": "AAPL",
    "quantity": 1.0,
    "side": "buy",
    "type": "market",
    "secType": "STK"
})
```

## Code Review Notes

### ✅ Correct Implementation

The user's provided code was **correct** with the following enhancements made:

1. **Enhanced Error Handling**: Added comprehensive error callbacks and exception handling
2. **Account/Position Updates**: Implemented `updateAccountValue` and `updatePortfolio` callbacks
3. **Market Data**: Added `tickPrice` and `tickSize` callbacks for quotes
4. **Order Status Tracking**: Implemented `orderStatus` callback
5. **Connection Management**: Added timeout handling and connection state management
6. **BrokerAdapter Interface**: Full implementation of all required methods

### Key Improvements

1. **Connection Timeout**: Added 10-second timeout for connection establishment
2. **Account Data**: Automatic account data retrieval after connection
3. **Position Updates**: Real-time position tracking via callbacks
4. **Quote Data**: Real-time market data via tick callbacks
5. **Thread Safety**: Added locks for shared data structures

## Advanced Usage

### Contract Resolution for Multi-Leg Options

For multi-leg options, you need to resolve contract IDs (conId) first:

```python
# Request contract details
req_id = ibkr.client.reqContractDetails(1, contract)
time.sleep(1)  # Wait for callback

# Get conId from contract_details
conId = ibkr.client.contract_details[req_id]["conId"]

# Use conId in combo legs
legs = [
    (conId1, 1, "BUY", "SMART"),
    (conId2, 1, "SELL", "SMART")
]
combo = ibkr.create_combo_contract("AAPL", legs)
```

### Custom Order Types

```python
from ibapi.order import Order

# Create custom order
order = Order()
order.action = "BUY"
order.totalQuantity = 100
order.orderType = "LMT"
order.lmtPrice = 150.00
order.tif = "DAY"  # Time in force

# Place order
contract = ibkr._create_stock_contract("AAPL")
ibkr.client.placeOrder(ibkr.client.nextOrderId, contract, order)
```

## Troubleshooting

### Connection Issues

**Problem**: "Failed to connect to IB TWS/Gateway"
- **Solution**: Ensure TWS/Gateway is running and API access is enabled
- Check port: 7497 (paper) or 7496 (live)
- Verify IP address in trusted IPs

**Problem**: "Next valid order ID not received"
- **Solution**: Wait a few seconds and retry. Check TWS/Gateway logs.

### Order Issues

**Problem**: Orders not executing
- **Solution**: Check TWS/Gateway order log. Verify account permissions.
- For paper trading, ensure paper trading account is funded.

**Problem**: "Invalid contract specification"
- **Solution**: Verify contract details (symbol, exchange, currency)
- For options, ensure expiration date format is YYYYMMDD

### API Errors

**Error Codes**:
- **2104-2106**: Informational messages (market data subscriptions)
- **Other codes**: Check IBKR API documentation

## Security Notes

⚠️ **Important**: 
- TWS/Gateway must be running on the same machine or accessible network
- Use firewall to restrict access to TWS/Gateway port
- Set trusted IPs in TWS/Gateway settings
- Never expose TWS/Gateway to public internet
- Use paper trading for testing
- Verify orders in TWS before going live

## Integration with NAE Agents

The adapter works seamlessly with existing NAE agents:

```python
from adapters.manager import BrokerManager

# Get IBKR adapter
manager = BrokerManager()
ibkr = manager.get("ibkr")

# Use in Optimus or other agents
if ibkr.auth():
    result = ibkr.place_order({
        "symbol": "AAPL",
        "quantity": 1.0,
        "side": "buy",
        "type": "market",
        "secType": "STK"
    })
```

## Resources

- [IBKR TWS API Documentation](https://interactivebrokers.github.io/tws-api/)
- [IBKR Python API Guide](https://interactivebrokers.github.io/tws-api/introduction.html)
- [TWS API Download](https://www.interactivebrokers.com/en/index.php?f=16042)
- [IB Gateway Download](https://www.interactivebrokers.com/en/index.php?f=16457)

## Status

✅ **Integration Complete** - Ready for use with TWS/Gateway connection.

**Note**: TWS or IB Gateway must be running with API access enabled before using the adapter.


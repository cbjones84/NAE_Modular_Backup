# Tradier Integration Guide

## Overview

Tradier is integrated as an optional broker for NAE execution, supporting OAuth 2.0 authentication, REST API, and WebSocket streaming.

## Architecture

```
NAE Signal → Signal Middleware → Tradier Validator → Tradier Adapter
                                                              ↓
                                                    Tradier REST API
                                                    Tradier WebSocket
```

## Features

- ✅ OAuth 2.0 authentication with token refresh
- ✅ REST API for orders and account data
- ✅ WebSocket streaming for market data and account events
- ✅ Equity, options, and multileg order support
- ✅ Pre/post-market order handling
- ✅ Order preview for validation
- ✅ Automatic failover support

## Setup

### 1. Tradier Account Setup

1. Open Tradier brokerage account
2. Request API access (OAuth credentials)
3. Obtain client ID and client secret
4. Get account ID

### 2. OAuth 2.0 Flow

#### Authorization Code Flow

1. **Get Authorization URL**:
```python
from execution.broker_adapters.tradier_adapter import TradierOAuth

oauth = TradierOAuth(client_id="...", client_secret="...", sandbox=True)
auth_url = oauth.get_authorization_url(
    redirect_uri="https://your-app.com/callback",
    state="optional_state"
)
```

2. **User Authorizes**: Redirect user to `auth_url`

3. **Exchange Code for Token**:
```python
token_data = oauth.exchange_code_for_token(
    code="authorization_code",
    redirect_uri="https://your-app.com/callback"
)
```

4. **Token Refresh**:
```python
# Tokens expire in 24 hours
if oauth.is_token_expired():
    oauth.refresh_access_token()
```

### 3. Environment Configuration

```bash
# Tradier OAuth
export TRADIER_CLIENT_ID="your_client_id"
export TRADIER_CLIENT_SECRET="your_client_secret"
export TRADIER_ACCOUNT_ID="your_account_id"

# Environment
export TRADIER_SANDBOX="true"  # or "false" for live

# Broker Priority (include Tradier)
export BROKER_PRIORITY="schwab,ibkr,tradier"
```

## Usage

### Initialize Adapter

```python
from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter

tradier = TradierBrokerAdapter(
    client_id=os.getenv("TRADIER_CLIENT_ID"),
    client_secret=os.getenv("TRADIER_CLIENT_SECRET"),
    account_id=os.getenv("TRADIER_ACCOUNT_ID"),
    sandbox=True
)

# Authenticate
tradier.authenticate()

# Connect streaming
tradier.connect_streaming()
```

### Submit Order

```python
order = {
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 10,
    "order_type": "market",
    "duration": "day",
    "preview": True  # Preview first
}

result = tradier.submit_order(order)
```

### Options Order

```python
order = {
    "option_symbol": "AAPL240119C00150000",  # AAPL Jan 19 2024 $150 Call
    "side": "buy",
    "quantity": 1,
    "order_type": "limit",
    "price": 2.50,
    "duration": "day"
}

result = tradier.submit_order(order)
```

### Multileg Order

```python
from execution.broker_adapters.tradier_adapter import TradierRESTClient

legs = [
    {"symbol": "AAPL240119C00150000", "side": "buy", "quantity": 1},
    {"symbol": "AAPL240119P00150000", "side": "buy", "quantity": 1}
]

result = tradier.rest_client.submit_multileg_order(
    account_id=tradier.account_id,
    legs=legs,
    order_type="limit",
    price=5.00
)
```

## Tradier-Specific Features

### Pre/Post-Market Orders

Tradier supports pre/post-market orders with constraints:
- Must be limit orders (not market)
- Pre-market: 4:00 AM - 9:30 AM ET
- Post-market: 4:00 PM - 8:00 PM ET

```python
order = {
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 10,
    "order_type": "limit",  # Required for pre/post
    "price": 150.00,
    "duration": "pre"  # or "post"
}
```

### Order Preview

Always preview orders to check for warnings and commission:

```python
preview = tradier.rest_client.preview_order(
    account_id=tradier.account_id,
    symbol="AAPL",
    side="buy",
    quantity=10,
    order_type="market"
)

if "warnings" in preview:
    print(f"Warnings: {preview['warnings']}")
```

### WebSocket Streaming

Tradier WebSocket provides real-time:
- Market data (quotes)
- Account events (fills, order status)

```python
def handle_message(data):
    if data.get("event") == "fill":
        print(f"Fill: {data}")

tradier.ws_client.on_message = handle_message
tradier.connect_streaming()
```

## Risk Management

### Tradier-Specific Risks

1. **OAuth Token Expiry** (24 hours)
   - Mitigation: Automatic token refresh
   - Alert when < 48 hours remaining

2. **Single WebSocket Session**
   - Mitigation: Reconnection logic
   - Event deduplication

3. **Pre/Post-Market Constraints**
   - Mitigation: Pre-trade validation
   - Restrict unsupported order types

### Circuit Breakers

```python
# Stop trading if Tradier failures exceed threshold
if tradier_failures > 5:
    pause_trading("tradier")
```

## Monitoring

### Key Metrics

- OAuth token expiry time
- WebSocket connection status
- Order submission rate
- Fill rate
- API error rate

### Alerts

- Token expiry < 48 hours
- WebSocket disconnection
- Order failure rate > threshold
- Authentication failures

## Testing

### Sandbox Testing

```bash
export TRADIER_SANDBOX="true"
export TRADIER_CLIENT_ID="sandbox_client_id"
export TRADIER_CLIENT_SECRET="sandbox_secret"
```

### Paper Trading

1. Use sandbox environment
2. Test full cycle: signal → order → fill
3. Validate reconciliation
4. Test failover scenarios

## Failover

Tradier can be configured as:
- Primary broker
- Secondary broker (failover from Schwab/IBKR)
- Tertiary broker (last resort)

Configure in `BROKER_PRIORITY`:
```bash
export BROKER_PRIORITY="schwab,tradier,ibkr"
```

## Troubleshooting

### OAuth Token Expired

```bash
# Check token expiry
python -c "from execution.broker_adapters.tradier_adapter import TradierOAuth; oauth = TradierOAuth(); print(f'Expired: {oauth.is_token_expired()}')"

# Refresh token
oauth.refresh_access_token()
```

### WebSocket Disconnected

```python
# Check connection status
status = tradier.get_status()
print(f"WebSocket connected: {status['websocket_connected']}")

# Reconnect
tradier.connect_streaming()
```

### Order Rejected

Check Tradier-specific validation:
- Pre/post-market constraints
- Trading hours
- Order type restrictions

## Runbooks

See `docs/RUNBOOKS.md` for:
- OAuth token renewal
- WebSocket reconnection
- Order failure handling
- Failover procedures


# Tradier Risks, Challenges & Mitigations

## Overview

This document outlines Tradier-specific risks, challenges, and mitigation strategies for NAE execution.

## Risks & Mitigations

### 1. OAuth Token Expiry

**Risk**: Access tokens expire every 24 hours. If refresh token not available, requires daily manual reauthorization.

**Mitigation**:
- ✅ Apply for refresh token with Tradier (if eligible)
- ✅ Automatic token refresh when available
- ✅ Alert when token expires < 48 hours
- ✅ Automated reauth procedure if refresh token unavailable
- ✅ Monitoring: `tradier_oauth_token_expires_at` metric

**Implementation**:
```python
# Automatic refresh
if oauth.is_token_expired():
    if oauth.refresh_token:
        oauth.refresh_access_token()
    else:
        # Alert and pause trading
        alert("Tradier token expired, manual reauth required")
        pause_trading("tradier")
```

### 2. Single WebSocket Session Limit

**Risk**: Tradier only allows one WebSocket session at a time per account.

**Mitigation**:
- ✅ Reconnection logic with exponential backoff
- ✅ Event deduplication to handle reconnection duplicates
- ✅ Health checks to detect disconnections
- ✅ Automatic reconnection on disconnect
- ✅ Monitoring: `tradier_websocket_connected` metric

**Implementation**:
```python
# Reconnection logic
def _attempt_reconnect(self):
    if self.reconnect_attempts < self.max_reconnect_attempts:
        time.sleep(self.reconnect_delay * (2 ** self.reconnect_attempts))
        self.connect()
```

### 3. Pre/Post-Market Order Constraints

**Risk**: Tradier has strict rules for pre/post-market orders:
- Must be limit orders (not market)
- Pre-market: 4:00 AM - 9:30 AM ET
- Post-market: 4:00 PM - 8:00 PM ET

**Mitigation**:
- ✅ Pre-trade validation checks trading hours
- ✅ Restrict unsupported order types outside regular hours
- ✅ Validate order type matches session (pre/post requires limit)
- ✅ Automatic order type conversion if needed

**Implementation**:
```python
# Pre-trade validation
if duration in ["pre", "post"]:
    if order_type not in ["limit", "stop_limit"]:
        reject_order("Pre/post-market orders must be limit orders")
```

### 4. Order Preview Recommended

**Risk**: Orders may have warnings or unexpected costs that aren't visible until preview.

**Mitigation**:
- ✅ Always preview orders before submission (configurable)
- ✅ Check for warnings in preview response
- ✅ Alert on warnings
- ✅ Option to auto-reject orders with critical warnings
- ✅ Monitoring: `tradier_order_previews_total` metric

**Implementation**:
```python
# Preview before submit
preview = rest_client.preview_order(...)
if "warnings" in preview:
    logger.warning(f"Order preview warnings: {preview['warnings']}")
    # Optionally reject or alert
```

### 5. API Error Handling & Retries

**Risk**: API errors may occur (rate limits, network issues, etc.).

**Mitigation**:
- ✅ Robust retry logic with exponential backoff
- ✅ Rate limit detection and handling
- ✅ Circuit breakers for repeated failures
- ✅ Alert on high error rates
- ✅ Fallback to backup broker if Tradier fails

**Implementation**:
```python
# Retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        result = submit_order(...)
        break
    except RateLimitError:
        wait_time = 2 ** attempt
        time.sleep(wait_time)
    except Exception as e:
        if attempt == max_retries - 1:
            raise
```

### 6. Streaming Event Deduplication

**Risk**: WebSocket reconnection may cause duplicate events.

**Mitigation**:
- ✅ Event ID tracking
- ✅ Timestamp-based deduplication
- ✅ Order ID tracking to prevent duplicate processing
- ✅ State reconciliation

**Implementation**:
```python
# Deduplication
processed_events = set()

def handle_message(data):
    event_id = data.get("id")
    if event_id in processed_events:
        return  # Skip duplicate
    processed_events.add(event_id)
    process_event(data)
```

## Operational Considerations

### Security

- ✅ Store OAuth credentials in Vault
- ✅ Never log tokens or secrets
- ✅ Use HTTPS/TLS for all API calls
- ✅ Rotate credentials regularly
- ✅ Use least privilege access

### Audit

- ✅ Log all signal → order → fill events
- ✅ Include NAE strategy metadata
- ✅ Store order preview results
- ✅ Track OAuth token refreshes
- ✅ Log WebSocket reconnections

### Risk Control

- ✅ Maximum risk per strategy
- ✅ Max account drawdown thresholds
- ✅ Circuit breakers for Tradier failures
- ✅ Position size limits
- ✅ Pre/post-market order restrictions

### Alerting

- ✅ OAuth token expiry < 48 hours
- ✅ WebSocket disconnection
- ✅ Order failure rate > threshold
- ✅ Authentication failures
- ✅ High reconnection attempts

### Scaling

- ✅ Support multiple strategies via order tags
- ✅ Route orders through middleware
- ✅ Queue-based processing
- ✅ Horizontal scaling of execution workers

## Testing Checklist

- [ ] OAuth token refresh works
- [ ] WebSocket reconnection works
- [ ] Pre/post-market validation works
- [ ] Order preview catches warnings
- [ ] Retry logic handles errors
- [ ] Event deduplication works
- [ ] Failover to backup broker works
- [ ] Sandbox testing complete
- [ ] Paper trading validated
- [ ] Canary deployment successful

## Monitoring Dashboard

Key metrics to monitor:
- OAuth token expiry time
- WebSocket connection status
- Order submission rate
- Fill rate
- API error rate
- Reconnection attempts
- Pre/post-market order count

## Runbooks

See `docs/RUNBOOKS.md` for:
- OAuth token renewal procedures
- WebSocket reconnection procedures
- Order failure handling
- Failover procedures


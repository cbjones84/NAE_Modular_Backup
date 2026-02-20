# NAE Execution Runbooks

## OAuth Token Renewal

### Tradier

**Frequency**: Daily (tokens expire every 24 hours)

**Automatic Refresh** (if refresh token available):
1. System automatically refreshes token when expired
2. Check logs: `grep "token refreshed" logs/execution.log`
3. Verify token expiry updated: `curl http://localhost:8002/metrics | grep tradier_oauth_token_expires_at`

**Manual Refresh** (if no refresh token):
1. Get authorization URL:
   ```python
   from execution.broker_adapters.tradier_adapter import TradierOAuth
   oauth = TradierOAuth()
   auth_url = oauth.get_authorization_url(redirect_uri="...")
   ```
2. User authorizes and gets code
3. Exchange code for token:
   ```python
   oauth.exchange_code_for_token(code="...", redirect_uri="...")
   ```
4. Update token in Vault: `vault write secret/nae-execution/tradier token=<new_token>`
5. Restart execution engine

**Alerts**:
- Token expiry < 48 hours: Warning alert
- Token expired: Critical alert, trading paused

### Schwab via QuantConnect

**Frequency**: Weekly (QC sends reminder)

**Steps**:
1. Log into QuantConnect Cloud
2. Navigate to Brokerage settings
3. Click "Refresh Token" for Schwab
4. Complete OAuth flow
5. Verify token expiry date updated
6. Check monitoring alert clears

**Automation**: 
- Alert configured when token expires < 48 hours
- Runbook reminder sent weekly

### Self-Hosted LEAN

**If Schwab supports programmatic refresh**:
1. Check token expiry: `curl http://localhost:8002/metrics | grep oauth_token_expires_at`
2. Run refresh script: `python scripts/refresh_oauth_token.py --broker schwab`
3. Verify new token stored in Vault

**If manual refresh required**:
1. Follow Schwab OAuth flow
2. Update token in Vault: `vault write secret/nae-execution/schwab token=<new_token>`
3. Restart execution engine

## Failover Procedure

### Automatic Failover

System automatically fails over when:
- Primary broker failures > threshold (default: 5)
- Connectivity issues persist > 5 minutes

### Manual Failover

1. Check broker status:
   ```bash
   curl http://localhost:8001/health
   ```

2. Trigger failover:
   ```bash
   curl -X POST http://localhost:8001/admin/failover \
     -H "Authorization: Bearer <admin_token>"
   ```

3. Verify secondary broker active:
   ```bash
   curl http://localhost:8001/admin/broker-status
   ```

4. Monitor failover event in Grafana

### Failback to Primary

1. Verify primary broker recovered:
   ```bash
   curl http://localhost:8001/admin/broker-status
   ```

2. Trigger failback:
   ```bash
   curl -X POST http://localhost:8001/admin/failback \
     -H "Authorization: Bearer <admin_token>"
   ```

3. Monitor for 1 hour to ensure stability

## Emergency Halt

### Stop All Trading

1. Activate kill switch:
   ```bash
   curl -X POST http://localhost:8001/admin/kill-switch \
     -H "Authorization: Bearer <admin_token>" \
     -d '{"reason": "Emergency halt"}'
   ```

2. Verify all orders cancelled:
   ```bash
   curl http://localhost:8001/admin/pending-orders
   ```

3. Notify team via Slack/PagerDuty

### Resume Trading

1. Deactivate kill switch:
   ```bash
   curl -X POST http://localhost:8001/admin/kill-switch/resume \
     -H "Authorization: Bearer <admin_token>"
   ```

2. Verify system operational:
   ```bash
   curl http://localhost:8001/health
   ```

## Reconciliation Investigation

### Position Discrepancy

1. Check reconciliation results:
   ```sql
   SELECT * FROM reconciliation_results 
   WHERE status = 'DISCREPANCY' 
   ORDER BY timestamp DESC LIMIT 10;
   ```

2. Compare NAE vs Broker positions:
   ```bash
   python scripts/compare_positions.py --symbol AAPL
   ```

3. Investigate root cause:
   - Check execution ledger for missing fills
   - Verify broker API responses
   - Check for manual trades

4. Resolve discrepancy:
   - Update NAE ledger if broker correct
   - Contact broker if NAE correct
   - Document resolution

### PnL Discrepancy

1. Check PnL reconciliation:
   ```sql
   SELECT * FROM reconciliation_results 
   WHERE type = 'pnl' AND status = 'DISCREPANCY'
   ORDER BY timestamp DESC;
   ```

2. Calculate expected PnL:
   ```bash
   python scripts/calculate_pnl.py --period daily
   ```

3. Compare with broker statement
4. Investigate differences:
   - Fees not accounted
   - Timing differences
   - Corporate actions

## Circuit Breaker Reset

### Strategy Circuit Breaker

1. Check circuit breaker state:
   ```sql
   SELECT * FROM circuit_breaker_state WHERE name = 'strategy_<id>';
   ```

2. Reset if needed:
   ```bash
   curl -X POST http://localhost:8001/admin/circuit-breaker/reset \
     -H "Authorization: Bearer <admin_token>" \
     -d '{"name": "strategy_<id>"}'
   ```

### System Circuit Breaker

1. Investigate root cause of failures
2. Fix underlying issue
3. Reset circuit breaker:
   ```bash
   curl -X POST http://localhost:8001/admin/circuit-breaker/reset \
     -H "Authorization: Bearer <admin_token>" \
     -d '{"name": "system"}'
   ```

## Monitoring Alerts

### Tradier OAuth Token Expiry

**Alert**: `tradier_oauth_token_expires_at < 48 hours`

**Response**:
1. Check token status: `python scripts/check_tradier_token.py`
2. If refresh token available, refresh automatically
3. If no refresh token, follow manual OAuth flow
4. Verify token updated in Vault
5. Confirm alert clears

### Tradier WebSocket Disconnection

**Alert**: `tradier_websocket_connected == 0`

**Response**:
1. Check WebSocket status: `curl http://localhost:8001/admin/tradier-status`
2. Verify Tradier API status
3. Check network connectivity
4. Restart WebSocket connection: `python scripts/reconnect_tradier_ws.py`
5. Monitor reconnection attempts

### High Execution Failures

**Alert**: `execution_failures_total > 5/minute`

**Response**:
1. Check execution logs
2. Verify broker connectivity
3. Check OAuth token expiry (Tradier: daily, Schwab: weekly)
4. Review recent signal patterns
5. Check for Tradier-specific errors (pre/post-market constraints)

### PnL Drawdown

**Alert**: `strategy_drawdown_percent > X% in 24h`

**Response**:
1. Review strategy performance
2. Check market conditions
3. Review recent trades
4. Consider pausing strategy

### OAuth Expiry Warning

**Alert**: `oauth_token_expires_at < 48 hours`

**Response**:
1. Follow OAuth renewal runbook
2. Verify token refreshed
3. Confirm alert clears

### Pending Signals Queue

**Alert**: `pending_signals > threshold`

**Response**:
1. Check execution engine status
2. Verify queue consumer running
3. Check for execution errors
4. Scale execution engine if needed


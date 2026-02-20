# 🔥 Real-Time Self-Healing Diagnostic Engine

## Overview

Production-ready real-time self-healing diagnostic and remediation engine for Tradier (Live), designed to plug directly into Optimus. This makes Optimus act like a self-aware trading agent that knows when something is wrong and corrects itself automatically.

## Features

### ✅ Automatic Detection
- **Continuously monitors** account state and buying power (every 60 seconds)
- **Validates endpoints**, approvals, and symbol formats in real-time
- **Previews orders** (safe `preview=true`) and surfaces exact Tradier errors
- **Tracks failed orders** and diagnoses root causes

### ✅ Automatic Fixes
- **Applies safe, reversible auto-fixes** when possible
- **Validates and corrects** symbol formatting
- **Fills missing required fields** automatically
- **Logs all fixes** clearly for audit trail

### ✅ Clear Logging
- **Emits clear log messages** that Optimus can read
- **Categorizes issues** by severity (INFO, WARNING, ERROR, CRITICAL)
- **Tracks issue history** (last 1000 issues)
- **Provides statistics** on issues detected, fixed, and resolved

### ✅ Callback Hooks
- **Exposes callback hooks** so Optimus can react
- **Alerts** when issues are detected
- **Dashboards** can query health status
- **Escalations** for critical issues

## Architecture

### Components

1. **TradierSelfHealingEngine** (`execution/self_healing/tradier_self_healing_engine.py`)
   - Core monitoring and healing engine
   - Runs continuous diagnostic checks
   - Applies auto-fixes
   - Tracks issues and statistics

2. **OptimusSelfHealingIntegration** (`execution/self_healing/optimus_integration.py`)
   - Integration layer between Optimus and engine
   - Provides callbacks for Optimus
   - Diagnoses order failures
   - Checks health before trading

3. **Optimus Integration** (`agents/optimus.py`)
   - Direct plugin into Optimus
   - Initializes on Optimus startup
   - Blocks trading when unhealthy
   - Diagnoses failures automatically

## How It Works

### 1. Continuous Monitoring

The engine runs a monitoring loop every 60 seconds that checks:

- ✅ **API Connection** - Verifies Tradier API is accessible
- ✅ **Account State** - Checks cash, buying power, equity
- ✅ **Options Approval** - Verifies options trading permissions
- ✅ **Buying Power** - Ensures sufficient funds available
- ✅ **Endpoint** - Validates live vs sandbox configuration
- ✅ **Account Restrictions** - Checks account status (active/restricted)

### 2. Issue Detection

When an issue is detected:

1. **Issue is recorded** with:
   - Issue ID (unique identifier)
   - Severity (INFO, WARNING, ERROR, CRITICAL)
   - Category (connection, permissions, funds, etc.)
   - Description
   - Timestamp
   - Auto-fixable flag

2. **Callback is triggered** (`on_issue_detected`)
   - Optimus logs the issue
   - Can trigger alerts
   - Can update dashboards

3. **Auto-fix is attempted** (if enabled and fixable)
   - Safe, reversible fixes applied
   - Result logged (FIXED, PARTIAL, FAILED)
   - Callback triggered (`on_auto_fix_applied`)

### 3. Order Failure Diagnosis

When an order fails:

1. **Order is analyzed** using `TradierOrderHandler`
2. **Exact error messages** are captured from Tradier
3. **Diagnostic issue** is created with:
   - Exact error message
   - Category (order, symbol, permissions, etc.)
   - Auto-fixable status
   - Metadata (order details, fixes applied, warnings)

4. **Issue is logged** and tracked
5. **Optimus is notified** via callback

### 4. Health Score

The engine calculates a health score (0.0 to 1.0):

- **1.0** = Perfect health (no issues)
- **0.7-0.99** = Healthy (minor warnings)
- **0.3-0.69** = Degraded (some errors)
- **0.0-0.29** = Unhealthy (critical issues)

**Trading is blocked** if:
- Health score < 0.5
- Any CRITICAL issues present

## Usage

### Automatic (Already Integrated)

The engine is **automatically initialized** when Optimus starts (if `TRADIER_API_KEY` is set). No code changes needed!

### Manual Access

```python
from agents.optimus import OptimusAgent

# Initialize Optimus (self-healing engine auto-initializes)
optimus = OptimusAgent(sandbox=False)

# Check health status
if optimus.self_healing_engine:
    health = optimus.self_healing_engine.get_health_status()
    print(f"Health: {health['status']} (score: {health['health_score']})")
    
    # Check if can trade
    can_trade = optimus.self_healing_engine.can_trade()
    print(f"Can trade: {can_trade}")
```

### Custom Callbacks

```python
from execution.self_healing import OptimusSelfHealingIntegration, DiagnosticIssue

def on_issue_detected(issue: DiagnosticIssue):
    print(f"Issue detected: {issue.description}")
    # Send alert, update dashboard, etc.

def on_issue_resolved(issue: DiagnosticIssue):
    print(f"Issue resolved: {issue.issue_id}")
    # Update dashboard, log resolution, etc.

# Create integration with custom callbacks
integration = OptimusSelfHealingIntegration(
    optimus_agent=optimus,
    on_issue_detected=on_issue_detected,
    on_issue_resolved=on_issue_resolved
)
```

## Auto-Fixes

### Currently Supported

1. **Endpoint Mismatch**
   - Detects when `TRADIER_SANDBOX` env var doesn't match adapter
   - Logs warning (requires restart to fully fix)

2. **Symbol Formatting**
   - Validates OCC format for options
   - Auto-uppercases equity symbols
   - Sets `class=option` for option orders

3. **Required Fields**
   - Validates all required fields present
   - Auto-fills missing fields when possible
   - Ensures limit orders have price, stop orders have stop

### Future Auto-Fixes

- Account re-authentication
- Token refresh
- Symbol lookup and correction
- Order size adjustment based on buying power

## Monitoring

### Logs

All issues are logged to Optimus logs:
```
🔍 [Self-Healing] Issue detected: connection_failed - Cannot connect to Tradier API
✅ [Self-Healing] Issue resolved: connection_failed
🔧 [Self-Healing] Auto-fixed: endpoint_mismatch
```

### Statistics

Tracked statistics:
- Total diagnostic checks
- Issues detected
- Issues auto-fixed
- Issues resolved
- Failed orders
- Successful orders

### Health Status

Query health status:
```python
health = optimus.self_healing_engine.get_health_status()
# Returns:
# {
#   "status": "healthy" | "degraded" | "unhealthy",
#   "health_score": 0.0-1.0,
#   "active_issues": 0,
#   "issues": [...],
#   "current_state": {...},
#   "stats": {...}
# }
```

## Issue Categories

### Connection Issues
- `connection_failed` - Cannot connect to Tradier API
- `connection_error` - Connection check error

### Permission Issues
- `options_not_approved` - Options trading not approved
- `account_restricted` - Account is restricted/closed

### Fund Issues
- `no_buying_power` - No buying power available

### Configuration Issues
- `endpoint_mismatch` - Live/sandbox mismatch

### Order Issues
- `order_failure_*` - Order submission failures (diagnosed)

## Best Practices

1. **Monitor Health Score**
   - Check health before executing trades
   - Alert if score drops below 0.7

2. **Review Issue History**
   - Check `issue_history` for patterns
   - Identify recurring issues

3. **Use Callbacks**
   - Implement custom callbacks for alerts
   - Update dashboards in real-time

4. **Enable Auto-Fix**
   - Keep `enable_auto_fix=True` for automatic remediation
   - Review fixes in logs

## Status

✅ **Fully Integrated** - Plugs directly into Optimus
✅ **Production Ready** - Comprehensive error handling
✅ **Real-Time** - Continuous monitoring (60s intervals)
✅ **Self-Healing** - Automatic fixes when possible
✅ **Well Logged** - Clear messages for Optimus
✅ **Extensible** - Easy to add new checks and fixes

## Files

- `execution/self_healing/tradier_self_healing_engine.py` - Core engine
- `execution/self_healing/optimus_integration.py` - Integration layer
- `execution/self_healing/__init__.py` - Package exports
- `agents/optimus.py` - Optimus integration (modified)

---

**NAE now has a self-aware trading agent that automatically detects and fixes issues!** 🔥


# NAE Monitoring & Dashboard Specification

## Overview

Comprehensive monitoring system for NAE with real-time dashboards, alerts, and performance tracking.

## Metrics Collection

### Trading Metrics (Daily/Weekly)

**PnL Metrics**:
- Daily PnL
- Weekly PnL
- Monthly PnL
- YTD PnL
- Realized PnL
- Unrealized PnL

**Risk Metrics**:
- Realized volatility (30/90/365 day)
- Max drawdown (30/90/365 day)
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- VaR (95%, 99%)
- CVaR (95%, 99%)

**Performance Metrics**:
- Hit rate (win rate)
- Average return per trade
- Profit factor
- Total trades
- Winning trades
- Losing trades

**Latency Metrics**:
- Decision latency (p50, p95, p99)
- Execution latency
- Data feed latency
- Model inference latency

**Model Metrics**:
- Model drift score
- Model confidence
- Prediction accuracy
- Feature importance

### System Metrics

**Agent Health**:
- Agent status (online/offline)
- Agent response time
- Agent error rate
- Agent throughput

**Data Quality**:
- Data freshness
- Missing data percentage
- Data feed delays
- Schema validation failures

**Risk Controls**:
- Circuit breaker status
- Kill switch status
- Position limits utilization
- Pre-trade check failures

## Dashboard Layout

### Main Dashboard

**Section 1: Portfolio Overview**
- Current NAV
- Daily PnL (with sparkline)
- Total return %
- Sharpe ratio
- Max drawdown

**Section 2: Risk Metrics**
- Realized volatility
- VaR/CVaR
- Position exposure
- Circuit breaker status

**Section 3: Performance**
- Hit rate
- Average return per trade
- Profit factor
- Trade count

**Section 4: Agent Status**
- Agent health indicators
- Last activity timestamps
- Error rates
- Throughput

**Section 5: Recent Alerts**
- Critical alerts
- Warning alerts
- Info alerts

### Agent-Specific Dashboards

**Optimus Dashboard**:
- Trading activity
- Position details
- Execution metrics
- Risk controls status

**Ralph Dashboard**:
- Strategy generation rate
- Strategy approval rate
- Model performance
- Learning metrics

**Donnie Dashboard**:
- Strategy validation rate
- Execution coordination
- Meta-labeling confidence

## Alert Rules

### Critical Alerts

1. **PnL Drawdown > 5% in 24h**
   - Metric: `daily_pnl`
   - Condition: `< -0.05`
   - Action: Email + Slack + Kill switch activation

2. **Circuit Breaker Triggered**
   - Metric: `circuit_breaker_status`
   - Condition: `== "triggered"`
   - Action: Immediate notification

3. **Data Feed Delay > 5s**
   - Metric: `data_feed_delay`
   - Condition: `> 5.0`
   - Action: Alert + Fallback activation

4. **Kill Switch Activated**
   - Metric: `kill_switch_status`
   - Condition: `== "active"`
   - Action: Critical alert

### Warning Alerts

1. **Strategy Exposure > 20%**
   - Metric: `strategy_exposure`
   - Condition: `> 0.20`
   - Action: Warning notification

2. **Model Confidence < 50%**
   - Metric: `model_confidence`
   - Condition: `< 0.50`
   - Action: Warning notification

3. **Consecutive Losses > 3**
   - Metric: `consecutive_losses`
   - Condition: `> 3`
   - Action: Warning notification

4. **High Volatility Detected**
   - Metric: `realized_volatility`
   - Condition: `> 0.30`
   - Action: Warning notification

## Implementation

### Prometheus Integration

**Metrics Endpoint**: `http://localhost:8000/metrics`

**Key Metrics**:
```python
# Trading metrics
nae_pnl{agent="Optimus", period="daily"}
nae_realized_volatility{agent="Optimus", period="30d"}
nae_max_drawdown{agent="Optimus", period="30d"}
nae_sharpe_ratio{agent="Optimus", period="30d"}
nae_hit_rate{agent="Optimus"}
nae_avg_return_per_trade{agent="Optimus"}

# Performance metrics
nae_decision_latency_seconds{agent="Optimus", model_id="xgboost_v1"}
nae_model_drift_score{agent="Ralph", model_id="ebm_v1"}
nae_model_confidence{agent="Optimus", model_id="xgboost_v1"}

# Risk metrics
nae_position_size{agent="Optimus", symbol="AAPL"}
nae_daily_loss{agent="Optimus"}
nae_consecutive_losses{agent="Optimus"}

# System metrics
nae_data_feed_delay_seconds{source="polygon"}
```

### Grafana Dashboard

**Dashboard JSON**: `config/grafana_dashboard.json`

**Panels**:
1. Portfolio value over time
2. PnL distribution
3. Sharpe ratio trend
4. Drawdown chart
5. Hit rate gauge
6. Agent status table
7. Alert timeline
8. Risk metrics overview

### Alert Manager

**Configuration**: `config/alertmanager.yml`

**Receivers**:
- Email (critical alerts)
- Slack (all alerts)
- PagerDuty (critical only)

## Usage

### View Metrics

```python
from tools.metrics_collector import get_metrics_collector

metrics = get_metrics_collector()

# Get dashboard data
dashboard = metrics.get_dashboard_data()
print(f"Optimus PnL: ${dashboard['agents']['Optimus']['pnl']:.2f}")
print(f"Sharpe Ratio: {dashboard['agents']['Optimus']['sharpe_ratio']:.2f}")

# Get metrics
pnl_metrics = metrics.get_metrics("pnl", agent="Optimus", start_time=datetime.now() - timedelta(days=7))
```

### Add Custom Alert

```python
from tools.metrics_collector import AlertRule

metrics = get_metrics_collector()
metrics.add_alert_rule(AlertRule(
    name="custom_alert",
    metric="custom_metric",
    condition=">",
    threshold=100.0,
    severity="warning"
))
```

---

**Last Updated**: 2024  
**Status**: Specification Complete  
**Implementation**: In Progress


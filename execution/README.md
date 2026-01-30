# NAE Execution Architecture

## Broker-Abstraction + Signal-Middleware Architecture

This directory contains the execution layer that bridges NAE's decision-making with broker execution.

## Architecture Overview

```
NAE (Decision Layer)
    ↓ (signals)
Signal Middleware (FastAPI)
    ↓ (validated signals)
Redis Queue
    ↓ (execution jobs)
Execution Engine (LEAN/QuantTrader/NautilusTrader)
    ↓ (orders)
Schwab Brokerage (via custom adapter)
    ↓ (fills)
Reconciliation & Monitoring
```

## Execution Engines

NAE supports multiple open-source execution engines:

1. **LEAN Self-Hosted** (Primary) - `execution_engine/lean_self_hosted.py`
   - Mature, supports many asset classes
   - Custom broker adapters
   - Full control over execution

2. **QuantTrader + PyBroker** - `execution_engine/quanttrader_adapter.py`
   - Good for simpler strategies
   - Research-to-live pipeline
   - Easy integration

3. **NautilusTrader** - `execution_engine/nautilus_adapter.py`
   - High performance and concurrency
   - Best for multiple strategies
   - High data throughput

## Components

1. **Signal Middleware** (`signal_middleware/`) - Receives and validates NAE signals
2. **Pre-Trade Validator** (`pre_trade_validator/`) - Circuit breakers and risk checks
3. **Queue System** (`queue/`) - Redis-based message queue
4. **Execution Engine** (`execution_engine/`) - Multiple engine adapters
5. **Broker Adapters** (`broker_adapters/`) - Schwab and other broker integrations
6. **Monitoring** (`monitoring/`) - Prometheus, Grafana, alerts
7. **Reconciliation** (`reconciliation/`) - Position and PnL reconciliation
8. **Failover** (`failover/`) - Secondary broker routing

## Quick Start

See `docs/DEPLOYMENT.md` for deployment instructions.

## Execution Engine Selection

Set `EXECUTION_ENGINE` environment variable:
- `lean_self_hosted` (default)
- `quanttrader_pybroker`
- `nautilus_trader`

## Security

All secrets managed via Vault. See `docs/SECURITY.md` for security guidelines.


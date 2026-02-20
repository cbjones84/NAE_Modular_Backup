# Execution Engines Guide

## Overview

NAE uses **LEAN Self-Hosted as the primary execution engine** with automatic failover to backup engines:

1. **LEAN Self-Hosted** (Primary) - Mature, flexible, full control
2. **QuantTrader + PyBroker** (Backup 1) - Simpler, good for research-to-live
3. **NautilusTrader** (Backup 2) - High performance, multiple strategies

## Architecture

The execution manager automatically:
- Starts with LEAN as primary
- Monitors for failures
- Fails over to QuantTrader/PyBroker if LEAN fails
- Fails over to NautilusTrader if QuantTrader fails
- Automatically switches back to LEAN when it recovers

## LEAN Self-Hosted

### Advantages
- ✅ Very mature and stable
- ✅ Supports many asset classes (equities, options, futures, crypto)
- ✅ Custom broker adapters
- ✅ Full control over execution
- ✅ Active community and documentation
- ✅ No cloud dependencies

### Setup

1. **Install LEAN**:
```bash
git clone https://github.com/QuantConnect/Lean.git
cd Lean
dotnet build
```

2. **Configure**:
```bash
export LEAN_PATH=/path/to/Lean
export ALGORITHM_PATH=./algorithms/nae_signal_consumer
```

3. **Create Algorithm**:
```python
# algorithms/nae_signal_consumer/main.py
from AlgorithmImports import *

# NOTE: This example is for QuantConnect Cloud (deprecated).
# For self-hosted LEAN, use execution_engine/lean_self_hosted.py instead.
# See lean_self_hosted.py for the current implementation.

class NAESignalConsumer(QCAlgorithm):
    def initialize(self):
        # Initialize Redis connection
        # Initialize Schwab broker adapter
        # Schedule signal consumption
        pass
```

4. **Deploy**:
```bash
docker-compose up -d lean-engine
```

### Broker Adapter

Use custom Schwab adapter (`broker_adapters/schwab_adapter.py`) with LEAN's brokerage plugin interface.

## QuantTrader + PyBroker

### Advantages
- ✅ Simple Python API
- ✅ Good for research-to-live pipeline
- ✅ Built-in backtesting
- ✅ Easy to integrate

### Setup

1. **Install**:
```bash
pip install pybroker
```

2. **Configure**:
```bash
export EXECUTION_ENGINE=quanttrader_pybroker
export PRIMARY_BROKER=alpaca  # PyBroker has Alpaca support
```

3. **Deploy**:
```bash
docker-compose up -d quanttrader-engine
```

### Limitations
- Limited broker support (Alpaca built-in, Schwab needs custom adapter)
- Less mature than LEAN
- Fewer asset classes

## NautilusTrader

### Advantages
- ✅ High performance and concurrency
- ✅ Best for multiple strategies
- ✅ High data throughput
- ✅ Modern architecture

### Setup

1. **Install**:
```bash
pip install nautilus_trader
```

2. **Configure**:
```bash
export EXECUTION_ENGINE=nautilus_trader
export REDIS_HOST=localhost
```

3. **Deploy**:
```bash
docker-compose up -d nautilus-engine
```

### Limitations
- More complex setup
- Steeper learning curve
- Less documentation than LEAN

## Comparison

| Feature | LEAN | QuantTrader/PyBroker | NautilusTrader |
|---------|------|---------------------|----------------|
| Maturity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Performance | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ease of Use | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Broker Support | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Asset Classes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Documentation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## Primary Engine: LEAN Self-Hosted

**LEAN is the primary execution engine** by default:
- Most mature and stable
- Best broker adapter support
- Full control
- Active community
- Automatic failover to backups if needed

## Failover Behavior

### Automatic Failover
- If LEAN fails 5+ times, automatically switches to QuantTrader/PyBroker
- If QuantTrader fails, switches to NautilusTrader
- After 5 minutes of recovery, automatically switches back to LEAN

### Manual Override
You can force a specific engine by setting environment variable:
```bash
# Force QuantTrader (not recommended - use automatic failover)
export EXECUTION_ENGINE=quanttrader_pybroker

# Force NautilusTrader (not recommended - use automatic failover)
export EXECUTION_ENGINE=nautilus_trader
```

## Configuration

### Failover Thresholds
```bash
# Number of failures before failover (default: 5)
export EXECUTION_FAILOVER_THRESHOLD=5

# Timeout before switching back to primary (default: 300 seconds)
export EXECUTION_FAILOVER_TIMEOUT=300
```

## Monitoring

Check execution engine status:
```bash
curl http://localhost:8001/admin/execution-status
```

Response shows:
- Current active engine
- Primary engine
- Backup engines available
- Failure counts
- Recovery status


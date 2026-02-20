# NAE Execution Architecture

## Overview

The NAE execution layer provides a broker-abstraction and signal-middleware architecture that bridges NAE's decision-making with broker execution.

## Architecture Diagram

```
┌─────────────────┐
│   NAE Agents    │
│   (Optimus)     │
└────────┬────────┘
         │ Signals (HTTP/Webhook)
         ▼
┌─────────────────────────────────┐
│   Signal Middleware (FastAPI)   │
│   - HMAC Validation             │
│   - Schema Validation           │
│   - Audit Logging               │
└────────┬────────────────────────┘
         │ Validated Signals
         ▼
┌─────────────────────────────────┐
│   Pre-Trade Validator           │
│   - Circuit Breakers            │
│   - Exposure Limits             │
│   - Risk Checks                 │
└────────┬────────────────────────┘
         │ Accepted Signals
         ▼
┌─────────────────────────────────┐
│   Redis Queue                   │
│   (execution.signals)           │
└────────┬────────────────────────┘
         │ Queued Signals
         ▼
┌─────────────────────────────────┐
│   Execution Engine              │
│   - LEAN Algorithm (QC/LEAN)    │
│   - Strategy Router             │
│   - Order Management            │
└────────┬────────────────────────┘
         │ Orders
         ▼
┌─────────────────────────────────┐
│   Broker Adapter                │
│   - Schwab (Primary)            │
│   - IBKR/Tradier (Failover)     │
└────────┬────────────────────────┘
         │ Fills
         ▼
┌─────────────────────────────────┐
│   Execution Ledger              │
│   - Position Tracking           │
│   - PnL Calculation             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Reconciliation Engine         │
│   - Position Reconciliation     │
│   - PnL Reconciliation          │
└─────────────────────────────────┘
```

## Components

### 1. Signal Middleware

**Purpose**: Receives and validates signals from NAE

**Key Features**:
- HMAC signature verification
- JSON schema validation
- Audit logging to PostgreSQL
- Pre-trade validation integration

**API Endpoints**:
- `POST /v1/signals` - Receive signal
- `GET /health` - Health check

### 2. Pre-Trade Validator

**Purpose**: Validates signals before execution

**Key Features**:
- Circuit breakers (system, execution, strategy)
- Exposure limits
- Position size limits
- Correlation group checks

**Validation Status**:
- `ACCEPTED` - Signal approved for execution
- `REJECTED` - Signal rejected (with reason)
- `PENDING` - Requires manual review

### 3. Queue System (Redis)

**Purpose**: Reliable message queue for signals

**Queues**:
- `execution.signals` - Validated signals awaiting execution
- `execution.events` - Execution events (fills, errors)
- `monitoring.events` - Monitoring events (failover, alerts)

### 4. Execution Engine

**Purpose**: Executes orders via broker

**Options**:
- **QuantConnect Cloud**: Single algorithm consuming queue
- **Self-Hosted LEAN**: Full control, multi-strategy support

**Features**:
- Strategy router for multiple NAE strategies
- Risk allocation per strategy
- Signal aggregation
- Order management

### 5. Broker Adapter

**Purpose**: Abstracts broker-specific APIs

**Primary**: Schwab (via QuantConnect adapter or direct API)
**Failover**: IBKR or Tradier

**Features**:
- OAuth token management
- Order submission
- Position queries
- Fill reporting

### 6. Monitoring

**Purpose**: Real-time metrics and alerts

**Metrics**:
- Orders submitted/filled/rejected
- Execution latency
- Strategy drawdown
- OAuth token expiry
- Broker connectivity

**Dashboards**: Grafana with Prometheus

### 7. Reconciliation

**Purpose**: Reconciles NAE ledger with broker

**Frequency**: Every 15 minutes

**Checks**:
- Position reconciliation
- PnL reconciliation
- Discrepancy detection

### 8. Failover Manager

**Purpose**: Automatic failover to secondary broker

**Triggers**:
- Primary broker failures > threshold
- Connectivity issues > 5 minutes

**Features**:
- Automatic routing switch
- Manual failover/failback
- Status monitoring

## Data Flow

### Signal Flow

1. NAE (Optimus) generates trade signal
2. Signal sent to middleware via HTTP POST
3. Middleware validates (HMAC, schema)
4. Signal saved to audit DB
5. Pre-trade validator checks
6. If accepted, queued to Redis
7. Execution engine consumes from queue
8. Order submitted via broker adapter
9. Fill reported back to NAE

### Execution Flow

1. Execution engine polls Redis queue
2. Signal routed through strategy router
3. Order created and submitted
4. Order tracked in execution ledger
5. Fill events sent back to NAE
6. Position and PnL updated

### Reconciliation Flow

1. Reconciliation job runs every 15 minutes
2. Fetches positions from broker API
3. Compares with NAE ledger
4. Detects discrepancies
5. Alerts if threshold exceeded
6. Stores reconciliation results

## Security

### Authentication
- HMAC signatures for signal validation
- JWT tokens for admin endpoints
- mTLS for internal communication

### Secrets Management
- Vault for OAuth tokens
- Encrypted storage for credentials
- Rotation policies

### Network Security
- TLS everywhere
- IP allowlisting
- Network isolation

## Scalability

### Horizontal Scaling
- Multiple middleware instances
- Multiple execution engines
- Redis cluster for queue

### Vertical Scaling
- Increase execution engine resources
- Optimize database queries
- Cache frequently accessed data

## Failover Strategy

### Primary Broker Failure
1. Failover manager detects failures
2. Automatically switches to secondary
3. Routes new orders to secondary
4. Monitors primary recovery
5. Switches back when recovered

### Execution Engine Failure
1. Health checks detect failure
2. Alert sent to monitoring
3. Restart execution engine
4. Resume queue consumption

## Monitoring & Alerting

### Key Metrics
- Order submission rate
- Fill rate
- Execution latency
- Error rate
- Queue depth

### Alerts
- High execution failures
- PnL drawdown
- OAuth expiry
- Queue backup
- Broker connectivity

## Testing Strategy

### Unit Tests
- Signal validation
- Pre-trade checks
- Circuit breakers

### Integration Tests
- End-to-end signal flow
- Broker adapter
- Reconciliation

### Paper Trading
- 2-week paper trading period
- Validate PnL drift
- Test failover

### Canary Deployment
- Small capital deployment
- 72-hour monitoring
- Gradual rollout


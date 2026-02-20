# NAE Execution Deployment Guide

## Overview

This guide covers deployment of the NAE execution architecture including signal middleware, execution engine, monitoring, and failover.

## Prerequisites

- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Python 3.10+
- QuantConnect account (for QC Cloud) OR LEAN setup (for self-hosted)

## Phase 0: Preparation

### 1. Accounts Setup

- **Schwab**: Obtain brokerage account and API access
- **QuantConnect**: Create account and obtain API keys
- **Secondary Broker**: Setup IBKR or Tradier account for failover

### 2. Infrastructure Setup

```bash
# Clone repository
git clone https://github.com/cbjones84/NAE.git
cd NAE/execution

# Create environment file
cp .env.example .env
# Edit .env with your credentials
```

### 3. Database Setup

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Run migrations
psql -h localhost -U nae -d nae_execution -f database/schema.sql
```

## Phase 1: Signal Middleware Deployment

### 1. Build and Start

```bash
# Build signal middleware
docker-compose build signal-middleware

# Start services
docker-compose up -d signal-middleware redis postgres
```

### 2. Verify

```bash
# Health check
curl http://localhost:8001/health

# Test signal submission
curl -X POST http://localhost:8001/v1/signals \
  -H "Content-Type: application/json" \
  -H "X-Signature: <hmac_signature>" \
  -d @test_signal.json
```

## Phase 2: Execution Engine Deployment

### Option A: QuantConnect Cloud

1. Upload LEAN algorithm to QuantConnect
2. Configure Schwab brokerage connection
3. Set algorithm to consume from Redis queue
4. Deploy algorithm

### Option B: Self-Hosted LEAN

```bash
# Build LEAN container
docker build -f Dockerfile.lean -t nae-lean .

# Run LEAN
docker run -d \
  --name nae-lean \
  --network nae-execution_nae-execution \
  -e REDIS_HOST=redis \
  -e SCHWAB_API_KEY=<key> \
  nae-lean
```

## Phase 3: Monitoring Deployment

```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access Grafana
# URL: http://localhost:3000
# User: admin
# Password: (from .env)
```

## Phase 4: Reconciliation Setup

```bash
# Start reconciliation service
docker-compose up -d reconciliation

# Schedule reconciliation job (cron)
# Runs every 15 minutes
```

## Phase 5: Failover Configuration

1. Configure secondary broker credentials
2. Test failover mechanism
3. Set up alerts for failover events

## Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
# Test end-to-end flow
python tests/integration_test.py
```

### Paper Trading

1. Configure paper trading mode
2. Run for 2 weeks
3. Validate PnL drift < tolerance

### Canary Deployment

1. Deploy with $1000 capital
2. Monitor for 72 hours
3. Validate reconciliation

## Production Checklist

- [ ] All secrets in Vault
- [ ] OAuth token refresh automated
- [ ] Monitoring alerts configured
- [ ] Failover tested
- [ ] Reconciliation validated
- [ ] Circuit breakers operational
- [ ] Documentation complete
- [ ] Runbooks prepared

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues and solutions.


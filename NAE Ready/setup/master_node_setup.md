# Master-Node Setup: Mac (Master) + HP OmniBook X (Node)

## Overview

Master-Node architecture for NAE:
- **Mac (Master)**: Command & control, strategy updates, monitoring
- **HP OmniBook X (Node)**: Production execution, live trading

## Architecture

```
┌─────────────────────┐
│   Mac (Master)      │
│                     │
│  - Strategy Gen     │
│  - Monitoring       │
│  - Control Dashboard│
│  - Model Updates    │
└──────────┬──────────┘
           │
           │ API / Git
           │
┌──────────▼──────────┐
│  HP OmniBook (Node) │
│                     │
│  - Live Trading     │
│  - Execution        │
│  - Production       │
└─────────────────────┘
```

## Communication Protocol

### Master → Node
- Strategy updates (via Git)
- Model bundles (via API)
- Control commands (via API)
- Monitoring requests (via API)

### Node → Master
- Execution status (via API)
- Trade confirmations (via API)
- Health metrics (via API)
- Error reports (via API)

## Setup Steps

1. **Configure Master (Mac)**
   ```bash
   ./setup/configure_master.sh
   ```

2. **Configure Node (HP)**
   ```bash
   ./setup/configure_node.sh
   ```

3. **Test Communication**
   ```bash
   python3 setup/test_master_node_communication.py
   ```

## API Endpoints

### Master API
- `POST /api/strategy/update` - Push strategy update
- `POST /api/model/bundle` - Push model bundle
- `GET /api/node/status` - Get node status
- `POST /api/control/command` - Send control command

### Node API
- `POST /api/execution/status` - Report execution status
- `POST /api/trade/confirmation` - Confirm trade
- `GET /api/health` - Health check
- `POST /api/error/report` - Report error

## Security

- API key authentication
- Device ID verification
- Encrypted communication
- Rate limiting


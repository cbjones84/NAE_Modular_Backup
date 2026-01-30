# Primary-Backup Execution Architecture

## Overview

NAE execution uses a **primary-backup architecture** with automatic failover:

- **Primary**: LEAN Self-Hosted
- **Backup 1**: QuantTrader/PyBroker  
- **Backup 2**: NautilusTrader

## Architecture Diagram

```
Signal Queue (Redis)
    ↓
Execution Manager
    ├─→ LEAN Self-Hosted (Primary) ✅
    │   └─→ Fails → Switch to Backup 1
    │
    ├─→ QuantTrader/PyBroker (Backup 1) 🔄
    │   └─→ Fails → Switch to Backup 2
    │
    └─→ NautilusTrader (Backup 2) 🔄
        └─→ Last resort
```

## Failover Logic

### Failure Detection
- Execution engine failures are tracked per engine
- Failure threshold: 5 failures (configurable)
- Failures reset on successful execution

### Failover Sequence
1. **LEAN fails** → Switch to QuantTrader/PyBroker
2. **QuantTrader fails** → Switch to NautilusTrader
3. **All engines fail** → Alert and pause execution

### Recovery Logic
- After 5 minutes (configurable), check if primary recovered
- If primary available, automatically switch back
- Prevents ping-ponging between engines

## Benefits

### High Availability
- Automatic failover ensures continuous execution
- No manual intervention needed
- Graceful degradation

### Reliability
- Primary engine (LEAN) is most mature
- Backups provide redundancy
- Multiple fallback options

### Performance
- Primary engine optimized for production
- Backups available if needed
- No performance impact when primary healthy

## Configuration

### Environment Variables

```bash
# Failover threshold (default: 5 failures)
EXECUTION_FAILOVER_THRESHOLD=5

# Recovery timeout (default: 300 seconds)
EXECUTION_FAILOVER_TIMEOUT=300

# Enable/disable execution engine failover (default: true)
EXECUTION_ENGINE_FAILOVER=true
```

### Monitoring

Check execution engine status:
```bash
# Get current status
curl http://localhost:8001/admin/execution-status

# Response:
{
  "active_engine": "lean_self_hosted",
  "primary_engine": "lean_self_hosted",
  "backup_engines": ["quanttrader_pybroker", "nautilus_trader"],
  "failure_counts": {
    "lean_self_hosted": 0,
    "quanttrader_pybroker": 0,
    "nautilus_trader": 0
  },
  "engines_available": {
    "lean_self_hosted": true,
    "quanttrader_pybroker": true,
    "nautilus_trader": true
  }
}
```

## Failover Scenarios

### Scenario 1: LEAN Temporary Failure
1. LEAN fails 5 times
2. Automatic switch to QuantTrader/PyBroker
3. Execution continues
4. After 5 minutes, LEAN recovers
5. Automatic switch back to LEAN

### Scenario 2: LEAN Extended Failure
1. LEAN fails repeatedly
2. Switch to QuantTrader/PyBroker
3. Execution continues on backup
4. Monitor LEAN recovery
5. Switch back when stable

### Scenario 3: All Engines Fail
1. LEAN fails → Switch to QuantTrader
2. QuantTrader fails → Switch to NautilusTrader
3. NautilusTrader fails → Alert and pause
4. Manual intervention required

## Best Practices

1. **Monitor Primary Engine**: Keep LEAN healthy
2. **Test Failover**: Regularly test backup engines
3. **Alert on Failover**: Get notified when failover occurs
4. **Review Failures**: Investigate why primary failed
5. **Keep Backups Updated**: Ensure backup engines are ready

## Troubleshooting

### Failover Not Working
- Check engine availability
- Verify failover threshold
- Review execution logs

### Frequent Failovers
- Investigate primary engine issues
- Check broker connectivity
- Review system resources

### Recovery Not Happening
- Check recovery timeout setting
- Verify primary engine health
- Review failure counts


# Continuous Autonomous Operation Guide

## Overview

NAE now runs continuously, autonomously, learning from errors and improving holistically in both dev/sandbox and prod/live modes.

## Features

### ✅ Continuous Operation
- Runs 24/7 without interruption
- Automatic process management
- Health monitoring every 30 seconds
- Auto-restart on failures

### ✅ Self-Healing
- Automatic error detection
- Auto-fix common issues
- Pattern recognition
- Proactive problem resolution

### ✅ Continuous Learning
- Learns from errors
- Analyzes performance patterns
- Adapts behavior
- Applies optimizations

### ✅ Holistic Enhancement
- System-wide improvements
- Resource optimization
- Performance tuning
- Efficiency gains

## Operation Modes

### Dual Mode (Default)
Runs both sandbox and live simultaneously:
```bash
./scripts/start_continuous_nae.sh dual
```

### Sandbox Only
Testing and validation:
```bash
./scripts/start_continuous_nae.sh sandbox
```

### Live Only
Production trading:
```bash
./scripts/start_continuous_nae.sh live
```

## Components

### 1. Continuous Operation Manager
**Location**: `execution/autonomous/continuous_operation.py`

Manages:
- Process lifecycle
- Health monitoring
- Error detection
- Auto-restart
- Learning loops
- Enhancement application

### 2. Self-Healing System
**Location**: `execution/autonomous/self_healing.py`

Features:
- Error pattern detection
- Automatic fixes
- Import error resolution
- Connection retry logic
- Timeout handling

### 3. Health Monitoring
- CPU and memory usage
- Process status
- Error rates
- System resources
- Uptime tracking

## Usage

### Start Continuous Operation

```bash
cd "NAE Ready"
./scripts/start_continuous_nae.sh [mode]
```

### Monitor Status

```bash
# View logs
tail -f logs/continuous_operation.log

# Check process status
ps aux | grep continuous_operation

# View health history
python3 -c "
from execution.autonomous.continuous_operation import ContinuousOperationManager
manager = ContinuousOperationManager()
print(manager.get_status())
"
```

### Stop Operation

```bash
# Find PID
cat logs/continuous_operation.pid

# Stop
kill $(cat logs/continuous_operation.pid)
```

## macOS Service (LaunchDaemon)

Install as system service:

```bash
# Copy plist
sudo cp com.nae.continuous.plist /Library/LaunchDaemons/

# Load service
sudo launchctl load /Library/LaunchDaemons/com.nae.continuous.plist

# Start service
sudo launchctl start com.nae.continuous

# Check status
sudo launchctl list | grep nae.continuous
```

## Health Checks

### Automatic Health Monitoring

- **Interval**: Every 30 seconds
- **Checks**:
  - Process status
  - CPU/memory usage
  - Error rates
  - System resources
  - Log errors

### Health Status Levels

- **HEALTHY**: All systems operational
- **DEGRADED**: Minor issues detected
- **UNHEALTHY**: Significant problems
- **CRITICAL**: Immediate action needed

### Auto-Recovery

- Critical processes auto-restart
- Unhealthy processes monitored
- Error patterns trigger fixes
- Performance issues optimized

## Learning System

### Error Learning
- Analyzes error patterns
- Groups by type and process
- Identifies common issues
- Suggests fixes

### Performance Learning
- Tracks resource usage
- Monitors error rates
- Identifies trends
- Optimizes intervals

### Enhancement Application
- Adjusts monitoring frequency
- Pre-emptive restarts
- Resource optimization
- Performance tuning

## Self-Healing

### Auto-Fixable Issues

1. **Import Errors**
   - Missing module imports
   - Auto-add imports

2. **Connection Errors**
   - Retry with backoff
   - Connection recovery

3. **Timeout Errors**
   - Increase timeouts
   - Retry logic

4. **Key Errors**
   - Suggest .get() usage
   - Default value handling

### Healing Process

1. Detect issue from logs
2. Identify issue type
3. Apply fix pattern
4. Verify fix
5. Log resolution

## Monitoring

### Log Files

- `logs/continuous_operation.log` - Main operation log
- `logs/continuous_service.log` - Service log (if using LaunchDaemon)
- `logs/*.log` - Individual process logs

### Metrics

Tracked metrics:
- Process uptime
- Error counts
- CPU usage
- Memory usage
- Error rates
- Enhancements applied

### Status API

Get current status:
```python
from execution.autonomous.continuous_operation import ContinuousOperationManager

manager = ContinuousOperationManager()
status = manager.get_status()
print(status)
```

## Troubleshooting

### Process Not Starting

1. Check logs: `tail -f logs/continuous_operation.log`
2. Verify Python: `python3 --version`
3. Check permissions: `ls -la scripts/start_continuous_nae.sh`
4. Verify dependencies: `pip3 list`

### High Error Rate

1. Review error log: `cat logs/change_log.json`
2. Check process logs: `tail -f logs/*.log`
3. Review health status: Check `get_status()` output
4. Apply manual fixes if needed

### Resource Issues

1. Check system resources: `top` or `htop`
2. Review health history
3. Adjust intervals if needed
4. Scale resources if necessary

## Best Practices

1. **Monitor Regularly**: Check logs daily
2. **Review Health**: Weekly health review
3. **Update Dependencies**: Keep packages updated
4. **Backup Logs**: Archive logs periodically
5. **Test Changes**: Test in sandbox first

## Configuration

### Adjustable Parameters

- `health_check_interval`: Health check frequency (default: 30s)
- `learning_interval`: Learning loop frequency (default: 300s)
- `error_threshold`: Error rate threshold (default: 10/hour)
- `restart_threshold`: Restarts before escalation (default: 3)

### Customization

Edit `execution/autonomous/continuous_operation.py` to customize:
- Process configurations
- Health check logic
- Learning algorithms
- Enhancement rules

---

**Status**: ✅ Fully Operational
**Mode**: Dual (Sandbox + Live)
**Automation Level**: 100%


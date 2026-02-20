# NAE Autonomous Operation Setup

## Overview

NAE is now configured to run **autonomously and continuously by any means necessary**. The system includes multiple layers of redundancy, auto-restart mechanisms, health monitoring, and process management to ensure uninterrupted operation.

## System Architecture

### 1. Autonomous Master Controller (`nae_autonomous_master.py`)
- **Purpose**: Master orchestrator that ensures all components run continuously
- **Features**:
  - Process monitoring and auto-restart
  - Health monitoring (CPU, memory, disk)
  - Automatic recovery from failures
  - Unlimited restart attempts
  - Graceful shutdown handling

### 2. Tradier Funds Activation (`execution/integration/tradier_funds_activation.py`)
- **Purpose**: Monitors Tradier account and activates trading when funds available
- **Features**:
  - Continuous balance monitoring (every 5 minutes)
  - Automatic trading activation
  - Goal reinforcement
  - Day trading compliance

### 3. Continuous Trading Engine (`execution/integration/continuous_trading_engine.py`)
- **Purpose**: Executes trading operations continuously
- **Features**:
  - Continuous trading loop
  - Compliance checking
  - Automatic recovery

### 4. Balance Monitor (`execution/monitoring/tradier_balance_monitor.py`)
- **Purpose**: Monitors Tradier account balance
- **Features**:
  - Real-time balance tracking
  - Fund detection
  - Activation signals

## Installation Methods

### Method 1: macOS LaunchAgent (Recommended)

```bash
# Install as macOS service (runs on boot, auto-restarts)
./scripts/install_autonomous_service.sh
```

This installs NAE as a macOS LaunchAgent that:
- Starts automatically on system boot
- Auto-restarts if it crashes
- Runs continuously in background
- Survives user logout

### Method 2: Manual Start

```bash
# Start autonomous system manually
./scripts/start_nae_autonomous.sh
```

### Method 3: Direct Python Execution

```bash
# Run master controller directly
python3 nae_autonomous_master.py
```

## Service Management

### Check Service Status

```bash
# Check if service is running
launchctl list | grep com.nae.autonomous

# View logs
tail -f logs/nae_autonomous.out
tail -f logs/nae_autonomous.err
```

### Stop Service

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.nae.autonomous.plist
```

### Start Service

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nae.autonomous.plist
```

### Restart Service

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.nae.autonomous.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nae.autonomous.plist
```

## Monitoring

### Log Files

- `logs/nae_autonomous_master.log` - Master controller logs
- `logs/nae_autonomous.out` - Service stdout
- `logs/nae_autonomous.err` - Service stderr
- `logs/tradier_funds_activation.log` - Funds activation logs
- `logs/continuous_trading_engine.log` - Trading engine logs
- `logs/tradier_balance_monitor.log` - Balance monitor logs

### Health Checks

The system performs automatic health checks:
- **CPU Usage**: Monitored every 5 minutes
- **Memory Usage**: Monitored every 5 minutes
- **Disk Space**: Monitored every 5 minutes
- **Process Status**: Checked every 30 seconds

### Process Monitoring

All critical processes are monitored:
- Auto-restart on failure
- Unlimited restart attempts
- Staggered restarts to avoid conflicts
- Graceful shutdown handling

## Redundancy Features

### 1. Auto-Restart
- All processes auto-restart on failure
- Unlimited restart attempts
- Configurable restart delays

### 2. Health Monitoring
- System resource monitoring
- Automatic alerts on degradation
- Self-healing capabilities

### 3. Process Isolation
- Each component runs independently
- Failure in one doesn't affect others
- Individual restart capabilities

### 4. Signal Handling
- Graceful shutdown on SIGTERM/SIGINT
- Proper cleanup on exit
- State preservation

## Continuous Operation Guarantees

### ✅ Automatic Startup
- Starts on system boot (LaunchAgent)
- Starts on user login
- Auto-restarts on crash

### ✅ Process Monitoring
- Checks every 30 seconds
- Restarts failed processes immediately
- Unlimited restart attempts

### ✅ Health Monitoring
- System health checked every 5 minutes
- Automatic recovery from issues
- Resource usage tracking

### ✅ Trading Continuity
- Trading operations run continuously
- Automatic activation when funds available
- Compliance checks on every operation

## Troubleshooting

### Service Not Starting

1. Check Python path in plist:
   ```bash
   which python3
   ```

2. Update plist with correct path:
   ```bash
   # Edit com.nae.autonomous.plist
   # Update ProgramArguments[0] with correct python3 path
   ```

3. Check logs:
   ```bash
   tail -f logs/nae_autonomous.err
   ```

### Process Not Restarting

1. Check master controller logs:
   ```bash
   tail -f logs/nae_autonomous_master.log
   ```

2. Verify process configs in `nae_autonomous_master.py`

3. Check script paths are correct

### High Resource Usage

1. Check health monitor logs
2. Review system resource usage
3. Adjust check intervals if needed

## Configuration

### Process Configs

Edit `nae_autonomous_master.py` to modify:
- Process scripts
- Restart delays
- Max restarts (set to 10000 for unlimited)
- Required processes

### Monitoring Intervals

- Process check: 30 seconds
- Health check: 5 minutes
- Balance check: 5 minutes
- Trading cycle: 5 minutes

## Status Verification

### Check All Systems

```bash
# Check master controller
ps aux | grep nae_autonomous_master

# Check funds activation
ps aux | grep tradier_funds_activation

# Check trading engine
ps aux | grep continuous_trading_engine

# Check balance monitor
ps aux | grep tradier_balance_monitor
```

### Verify Service

```bash
# Check LaunchAgent status
launchctl list | grep com.nae.autonomous

# Check service is loaded
launchctl list com.nae.autonomous
```

## Important Notes

1. **Unlimited Restarts**: System configured for unlimited restart attempts
2. **Auto-Start**: Service starts automatically on boot
3. **Survives Logout**: Service continues running after user logout
4. **Health Monitoring**: Automatic health checks and recovery
5. **Process Isolation**: Each component runs independently

## Next Steps

1. **Install Service**:
   ```bash
   ./scripts/install_autonomous_service.sh
   ```

2. **Verify Installation**:
   ```bash
   launchctl list | grep com.nae.autonomous
   ```

3. **Monitor Logs**:
   ```bash
   tail -f logs/nae_autonomous.out
   ```

4. **System is Now Autonomous**: NAE will run continuously and automatically!

---

**Status**: ✅ Fully Autonomous
**Auto-Restart**: ✅ Enabled (Unlimited)
**Health Monitoring**: ✅ Active
**Service Installation**: ✅ Ready


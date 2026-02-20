# NAE Automation System

## Overview

The NAE Automation System automatically starts, orchestrates, and maintains all NAE agents without manual intervention.

## Quick Start

### Option 1: Direct Python Execution
```bash
cd NAE
python3 nae_automation.py
```

### Option 2: Using Startup Script
```bash
cd NAE
./start_nae.sh
```

### Option 3: Systemd Service (Linux)
1. Edit `nae.service` and update paths:
   - Set `WorkingDirectory` to your NAE directory
   - Set `ExecStart` to full path of `nae_automation.py`
   - Set `User` to your username

2. Install and start:
```bash
sudo cp nae.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nae.service
sudo systemctl start nae.service
```

3. Check status:
```bash
sudo systemctl status nae.service
```

## What It Does

### Automatic Agent Management
- **Starts all agents** automatically on system startup
- **Orchestrates** agent interactions via Splinter
- **Schedules** agent execution cycles based on their roles
- **Monitors** agent health continuously

### Self-Improvement
- **Feedback loops** automatically trigger after agent actions
- **Performance feedback** adjusts Optimus trading parameters
- **Risk feedback** triggers kill switches when needed
- **Research feedback** integrates new algorithms into Casey

### Self-Healing
- **Detects** failed or unhealthy agents
- **Automatically restarts** failed agents (up to 5 times)
- **Disables** agents that exceed restart limits
- **Logs** all failures for debugging

### Continuous Operation
- **Runs 24/7** without manual intervention
- **Handles errors** gracefully
- **Recovers** from crashes automatically
- **Maintains** system state across restarts

## Architecture

```
NAE Automation System
├── Master Scheduler (schedules all agent cycles)
├── Splinter (orchestrates agent interactions)
├── Casey (monitors and builds agents)
├── Feedback Loops (self-improvement)
│   ├── Performance Feedback (Optimus)
│   ├── Risk Feedback (Optimus)
│   └── Research Feedback (Casey)
└── Monitoring & Healing (restarts failed agents)
```

## Agent Scheduling

| Agent | Frequency | Purpose |
|-------|-----------|---------|
| Ralph | Every 30 min | Generate trading strategies |
| Donnie | Every 30 min | Execute strategies in sandbox |
| Optimus | Every 10 sec | Execute live trades |
| Casey | Every 2 hours | Build/refine agents |
| Bebop | Every 15 min | Monitor system |
| Splinter | Every 60 min | Orchestrate agents |
| Phisher | Every 60 min | Security scanning |
| Rocksteady | Every 6 hours | Security sweeps |
| Shredder | Every 60 min | Allocate profits |

## Logs

All automation logs are written to:
- `logs/nae_automation.log` - Main automation system logs
- `logs/master_scheduler.log` - Scheduler logs
- `logs/casey.log` - Casey agent logs
- `logs/splinter.log` - Splinter orchestration logs
- Individual agent logs in `logs/` directory

## Troubleshooting

### System Won't Start
1. Check Python version: `python3 --version` (requires 3.7+)
2. Check dependencies: `pip install -r requirements.txt`
3. Check logs: `tail -f logs/nae_automation.log`

### Agents Not Running
1. Check scheduler status: Look for "Initialized Agents" in logs
2. Check agent errors: Look for "ERROR" entries in logs
3. Verify agent configuration in `config/` directory

### Feedback Loops Not Triggering
1. Check that agents are actually executing (see logs)
2. Verify feedback loops are registered (check startup logs)
3. Check `logs/` for feedback loop errors

## Manual Control

### Stop the System
Press `Ctrl+C` or:
```bash
sudo systemctl stop nae.service  # If using systemd
```

### Restart the System
```bash
sudo systemctl restart nae.service  # If using systemd
```

### Check Status
```bash
sudo systemctl status nae.service  # If using systemd
# Or check logs directly:
tail -f logs/nae_automation.log
```

## Configuration

Edit `nae_master_scheduler.py` to adjust:
- Agent execution frequencies
- Enable/disable specific agents
- Restart limits
- Monitoring intervals

## Next Steps

Once automation is running:
1. Monitor logs to ensure agents are executing
2. Check agent performance via Splinter monitoring
3. Review feedback loop recommendations
4. Adjust scheduling frequencies as needed

## Support

For issues or questions:
1. Check logs first: `logs/nae_automation.log`
2. Review agent-specific logs in `logs/` directory
3. Check system status: `sudo systemctl status nae.service`


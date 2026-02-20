# NAE Agent Automation Guide

## Overview

The NAE Master Automation Scheduler automates all agents and their roles within the Neural Agency Engine system. This comprehensive automation ensures continuous operation of all agents without manual intervention.

## Agents Automated

### Core Trading Agents
1. **Ralph** - Strategy Generation
   - Generates trading strategies from AI and web sources
   - Runs every 60 minutes
   - Cycle: `run_cycle()` → Generates approved strategies

2. **Donnie** - Strategy Validation & Execution
   - Receives strategies from Ralph
   - Validates strategies (trust score >= 70)
   - Executes strategies in sandbox mode
   - Runs every 30 minutes
   - Cycle: `run_cycle()` → Processes strategies → Sends to Optimus

3. **Optimus** - Trade Execution
   - Executes trades in sandbox/paper/live mode
   - Tracks P&L and positions
   - Runs every 10 seconds (checks inbox)
   - Cycle: `run_cycle()` → Executes trades from inbox

### Monitoring Agents
4. **Casey** - Agent Builder/Refiner
   - Builds and refines agents dynamically
   - Monitors agent performance
   - Runs every 2 hours
   - Cycle: `run()` → Checks/builds agents

5. **Bebop** - System Monitor
   - Monitors all agents and system health
   - Tracks agent status
   - Runs every 15 minutes
   - Cycle: `run()` → Monitors all agents

6. **Splinter** - Orchestrator
   - Orchestrates agent communication
   - Manages agent coordination
   - Runs every 60 minutes
   - Cycle: `run()` → Orchestrates agent activities

### Security Agents
7. **Rocksteady** - Security Enforcer
   - Performs security sweeps
   - Verifies agent integrity
   - Runs every 6 hours
   - Cycle: `run()` → Security sweep

8. **Phisher** - Security Scanner
   - Scans code for security issues
   - Analyzes runtime logs
   - Runs every hour (updated for compliance)
   - Cycle: `run()` → Security scan

### Other Agents
9. **Genny** - Generational Wealth Tracker
   - Tracks generational wealth goals
   - Runs every 3 hours
   - Cycle: `run_cycle()` → Tracks wealth metrics

## Usage

### Starting the Master Scheduler

```bash
python3 nae_master_scheduler.py
```

This will:
- Initialize all agents
- Set up schedules for each agent
- Start continuous automation
- Run agents on their configured intervals

### Configuration

Edit `AutomationConfig` class in `nae_master_scheduler.py` to customize:

```python
class AutomationConfig:
    RALPH_INTERVAL_MINUTES = 60  # Change frequency
    DONNIE_INTERVAL_MINUTES = 30
    OPTIMUS_INTERVAL_SECONDS = 10
    
    # Enable/disable agents
    ENABLE_RALPH = True
    ENABLE_DONNIE = True
    # ... etc
```

### Running Individual Agent Cycles

Each agent can also be run individually:

```python
from agents.ralph import RalphAgent
ralph = RalphAgent()
result = ralph.run_cycle()  # Generate strategies

from agents.donnie import DonnieAgent
donnie = DonnieAgent()
donnie.receive_strategies(strategies)
donnie.run_cycle(sandbox=True, optimus_agent=optimus)
```

## Agent Flow

```
Ralph (every hour)
  ↓ Generates strategies
Donnie (every 30 min)
  ↓ Validates & executes
Optimus (every 10 sec)
  ↓ Executes trades
  ↓ Tracks P&L

Casey (every 2 hours)
  ↓ Builds/refines agents
  
Bebop (every 15 min)
  ↓ Monitors system
  
Splinter (every hour)
  ↓ Orchestrates
  
Rocksteady (every 6 hours)
  ↓ Security sweep
  
Phisher (every hour)
  ↓ Security scan
```

## Testing Automation

Run the comprehensive test suite:

```bash
python3 test_agent_automation.py
```

This tests:
1. ✅ Agent initialization
2. ✅ Scheduler initialization
3. ✅ Agent cycle execution
4. ✅ Scheduler execution
5. ✅ Agent communication
6. ✅ Schedule configuration

## Monitoring

### Status Check

```python
from nae_master_scheduler import NAEMasterScheduler

scheduler = NAEMasterScheduler()
status = scheduler.get_status()

print(f"Running: {status['running']}")
print(f"Agents: {len(status['agents'])}")
for name, agent_status in status['agents'].items():
    print(f"{name}: {agent_status['run_count']} runs, {agent_status['success_rate']:.1f}% success")
```

### Logs

All agent activities are logged:
- `logs/master_scheduler.log` - Scheduler activities
- `logs/ralph.log` - Strategy generation
- `logs/donnie.log` - Strategy execution
- `logs/optimus.log` - Trade execution
- `logs/casey.log` - Agent building
- `logs/bebop.log` - Monitoring
- `logs/splinter.log` - Orchestration
- `logs/rocksteady.log` - Security sweeps
- `logs/phisher.log` - Security scans

## Enabling/Disabling Agents

```python
scheduler = NAEMasterScheduler()

# Disable an agent
scheduler.disable_agent('Ralph')

# Enable an agent
scheduler.enable_agent('Ralph')
```

## Troubleshooting

### Schedule Module Not Available

If you see `[WARNING] schedule module not available`, the scheduler will use a fallback time-based scheduling system. This works without the `schedule` library but is less precise.

To install the schedule library:
```bash
pip install schedule
```

### Agent Not Running

1. Check if agent is enabled: `status['agents']['AgentName']['enabled']`
2. Check agent logs: `logs/agentname.log`
3. Check scheduler logs: `logs/master_scheduler.log`
4. Verify agent initialization in test output

### Agent Communication Issues

Ensure agents are initialized in the correct order:
1. Ralph → Generates strategies
2. Donnie → Receives from Ralph
3. Optimus → Receives from Donnie

The scheduler automatically handles this flow.

## Best Practices

1. **Start with Sandbox Mode**: Always run Optimus in sandbox mode initially
2. **Monitor Logs**: Regularly check logs for errors or warnings
3. **Test Before Production**: Run test suite before deploying to production
4. **Gradual Enablement**: Enable agents one at a time to verify functionality
5. **Regular Backups**: Backup strategy database and execution history

## Example: Full Automation Workflow

```python
from nae_master_scheduler import NAEMasterScheduler

# Initialize scheduler
scheduler = NAEMasterScheduler()

# Start automation
scheduler.start()

# All agents now run automatically:
# - Ralph generates strategies every hour
# - Donnie processes them every 30 minutes
# - Optimus executes trades every 10 seconds
# - Other agents run on their schedules
```

## Agent Responsibilities Summary

| Agent | Role | Frequency | Key Method |
|-------|------|-----------|------------|
| Ralph | Strategy Generation | 60 min | `run_cycle()` |
| Donnie | Strategy Execution | 30 min | `run_cycle()` |
| Optimus | Trade Execution | 10 sec | `run_cycle()` |
| Casey | Agent Builder | 2 hours | `run()` |
| Bebop | System Monitor | 15 min | `run()` |
| Splinter | Orchestrator | 60 min | `run()` |
| Rocksteady | Security | 6 hours | `run()` |
| Phisher | Security Scan | 1 hour | `run()` |
| Genny | Wealth Tracker | 3 hours | `run_cycle()` |

## Next Steps

1. ✅ All agents are automated
2. ✅ Test suite verifies automation
3. ✅ Scheduler coordinates all agents
4. ✅ Logging tracks all activities
5. ✅ Status monitoring available

The NAE system is now fully automated and running continuously!


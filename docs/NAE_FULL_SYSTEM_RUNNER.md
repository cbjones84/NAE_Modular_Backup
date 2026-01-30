# NAE Full System Runner Guide

## Overview

The `run_nae_full_system.py` script runs the complete NAE system with all features enabled:

- **Ralph**: Continuous GitHub learning, strategy discovery, improvement tracking
- **Optimus**: Intelligent Tradier monitoring and trading decisions
- **Accelerator**: Aggressive micro-scalp strategy execution (30-second cycles)
- **Continuous Learning**: Automatic improvement loops
- **Full Orchestration**: All agents communicating and enhancing together

## Features

### 1. Ralph's GitHub Learning
- Automatically searches GitHub for trading tools, algorithms, and strategies
- Sends discoveries to Donnie for implementation
- Generates high-confidence strategies and sends directly to Optimus
- Tracks improvement metrics over time

### 2. Optimus Intelligent Tradier Monitoring
- Continuously monitors Tradier account health (every 60 seconds)
- Evaluates market opportunities
- Processes strategies from Ralph/Donnie
- Executes trades intelligently based on market conditions

### 3. Accelerator Module (Aggressive Mode)
- Runs micro-scalp strategy every 30 seconds (aggressive)
- Supports both sandbox and live modes simultaneously
- Tracks P&L and trade statistics
- Automatically adapts based on performance

### 4. Continuous Learning & Enhancement
- Learning loop runs every 10 minutes
- Enhancement loop runs every 15 minutes
- Automatic performance tracking and improvement
- Self-healing and error recovery

## Usage

### Basic Usage (Live Mode, Aggressive)

```bash
cd "NAE Ready"
python run_nae_full_system.py
```

This runs with:
- **Live mode**: Enabled (production trading)
- **Sandbox mode**: Disabled
- **Aggressive accelerator**: Enabled (30-second cycles)

### Sandbox Only Mode

```bash
python run_nae_full_system.py --sandbox --no-live
```

### Dual Mode (Sandbox + Live)

```bash
python run_nae_full_system.py --sandbox
```

This runs both sandbox (testing) and live (production) simultaneously.

### Disable Aggressive Mode

```bash
python run_nae_full_system.py --no-aggressive
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--sandbox` | Enable sandbox/testing mode |
| `--no-live` | Disable live/production mode |
| `--aggressive` | Enable aggressive accelerator (default: True) |
| `--no-aggressive` | Disable aggressive accelerator (use 60s cycles) |

## Configuration

### Cycle Intervals (in code)

You can modify intervals in `run_nae_full_system.py`:

```python
self.ralph_cycle_interval = 300      # 5 minutes (Ralph learning)
self.optimus_check_interval = 60     # 1 minute (Tradier checks)
self.learning_interval = 600         # 10 minutes (learning loop)
self.enhancement_interval = 900      # 15 minutes (enhancement loop)
```

### Accelerator Aggressiveness

Aggressive mode uses 30-second cycles (default):
- Faster execution
- More frequent opportunities
- Higher risk/reward

Normal mode uses 60-second cycles:
- More conservative
- Lower risk
- Standard execution

## What Happens When Running

1. **Initialization** (one-time):
   - Initializes all agents (Ralph, Optimus, Donnie, Casey, Splinter)
   - Sets up GitHub client with your token
   - Initializes Tradier self-healing engine
   - Starts Accelerator Controller
   - Sets up Main Orchestrator

2. **Ralph Learning Cycle** (every 5 minutes):
   - Searches GitHub for new tools/algorithms
   - Generates strategies from multiple sources
   - Evaluates and filters strategies
   - Sends high-confidence strategies to Optimus
   - Sends GitHub discoveries to Donnie

3. **Optimus Intelligence Loop** (every 60 seconds):
   - Checks Tradier account health
   - Evaluates market opportunities
   - Processes pending strategies
   - Monitors accelerator status

4. **Accelerator Cycles** (every 30 seconds in aggressive mode):
   - Runs micro-scalp strategy
   - Executes trades based on market conditions
   - Tracks P&L and statistics

5. **Learning Loop** (every 10 minutes):
   - Collects metrics from all agents
   - Analyzes performance
   - Applies learning insights

6. **Enhancement Loop** (every 15 minutes):
   - Checks for improvements
   - Applies enhancements to agents
   - Optimizes system performance

## Monitoring

### Logs

All logs are written to:
- `logs/nae_full_system.log` - Main system log
- `logs/ralph.log` - Ralph's learning activities
- `logs/optimus.log` - Optimus trading activities
- `logs/accelerator_controller.log` - Accelerator status

### Console Output

The script provides real-time console output:
- ✅ Successful operations
- ⚠️  Warnings
- ❌ Errors
- 📊 Metrics and statistics
- 🔄 Cycle completions

## Stopping the System

Press `Ctrl+C` to gracefully shutdown:
- Stops all loops
- Closes connections
- Saves state
- Exits cleanly

## Requirements

- GitHub token configured in `config/api_keys.json`
- Tradier credentials (if using Tradier)
- All NAE dependencies installed
- Python 3.8+

## Troubleshooting

### GitHub Not Working
- Verify token is in `config/api_keys.json`
- Check token permissions (repo read access)
- Check rate limits in logs

### Tradier Not Connecting
- Verify `TRADIER_API_KEY`, `TRADIER_CLIENT_ID`, etc. in environment
- Check Tradier account status
- Review self-healing engine logs

### Accelerator Not Running
- Check if agents initialized successfully
- Verify trading account balance
- Check accelerator logs for errors

## Performance Tips

1. **Start Conservative**: Begin with `--no-aggressive` to test
2. **Monitor Logs**: Watch logs for first few cycles
3. **Check Metrics**: Review learning metrics after 1 hour
4. **Adjust Intervals**: Modify cycle intervals based on performance
5. **Use Dual Mode**: Test in sandbox while running live

## Next Steps

After running successfully:
1. Monitor performance metrics
2. Review GitHub discoveries from Ralph
3. Check strategy quality and execution
4. Adjust aggressiveness based on results
5. Review learning improvements over time

---

**Ready to run?** Start with: `python run_nae_full_system.py`


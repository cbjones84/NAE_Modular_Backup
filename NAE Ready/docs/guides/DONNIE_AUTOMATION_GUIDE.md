# Automated Donnie Cycle - Usage Guide

## Overview

The automated Donnie cycle processes high-quality strategies from Ralph and passes them to Optimus for execution. This automates the complete strategy pipeline: **Ralph → Donnie → Optimus**.

---

## Quick Start

### Run Donnie Cycle Once

```bash
python3 run_donnie_cycle.py
```

This will:
1. Load the latest high-quality strategies from Ralph (trust_score >= 70)
2. Pass them to Donnie for validation
3. Execute them in sandbox mode
4. Send execution instructions to Optimus

### Run Automated Scheduler

```bash
# Run continuously every 30 minutes
python3 automated_donnie_scheduler.py

# Run with custom interval (e.g., every 15 minutes)
python3 automated_donnie_scheduler.py --interval 15

# Auto-generate strategies before each cycle
python3 automated_donnie_scheduler.py --auto-generate

# Run once and exit
python3 automated_donnie_scheduler.py --once
```

---

## Command Line Options

### `automated_donnie_scheduler.py`

- `--interval MINUTES`: Set cycle interval in minutes (default: 30)
- `--auto-generate`: Auto-generate strategies from Ralph before each cycle
- `--once`: Run one cycle and exit (instead of continuous)

### Examples

```bash
# Run every hour with auto-generation
python3 automated_donnie_scheduler.py --interval 60 --auto-generate

# Run once every 15 minutes
python3 automated_donnie_scheduler.py --interval 15

# Quick test run
python3 automated_donnie_scheduler.py --once
```

---

## How It Works

### Strategy Flow

1. **Ralph** generates and approves strategies
   - Filters by trust_score >= 50 (Ralph's threshold)
   - Saves to `logs/ralph_approved_strategies_*.json`

2. **Donnie** validates and executes strategies
   - Validates trust_score >= 70 AND backtest_score >= 50
   - Executes in sandbox mode
   - Sends execution instructions to Optimus

3. **Optimus** receives and executes strategies
   - Receives execution instructions via inbox
   - Executes trades in sandbox/paper/live mode
   - Maintains execution history

### Validation Criteria

**Donnie's Validation:**
- `trust_score >= 70.0` (required)
- `backtest_score >= 50.0` (required)

**Strategies that pass Donnie's validation are automatically sent to Optimus.**

---

## Monitoring

### Check Strategy Flow

```bash
python3 check_strategy_flow.py
```

This shows:
- Ralph's approved strategies
- Donnie's execution history
- Optimus's received messages

### Check Logs

```bash
# Donnie's log
tail -f logs/donnie.log

# Optimus's log  
tail -f logs/optimus.log

# Ralph's log
tail -f logs/ralph.log
```

---

## Current Status

**Latest Run Results:**
- ✅ 7 strategies passed from Ralph to Donnie
- ✅ 7 strategies executed by Donnie
- ✅ 7 strategies passed to Optimus

**Strategies in Optimus's inbox:**
1. Grok Insight 1
2. Grok Insight 2
3. DeepSeek Insight 1
4. DeepSeek Insight 2
5. Claude Insight 1
6. Claude Insight 2
7. optionsforum.com Strategy 2

---

## Automation Setup

### Background Service (Linux/macOS)

Create a systemd service or launchd plist to run the scheduler automatically:

```bash
# Example: Run every 30 minutes
python3 automated_donnie_scheduler.py --interval 30 --auto-generate &
```

### Cron Job Example

```bash
# Run every hour
0 * * * * cd /path/to/NAE && python3 automated_donnie_scheduler.py --once --auto-generate
```

---

## Troubleshooting

### No Strategies Found

If Donnie reports "No high-quality strategies found":
1. Run Ralph's enhanced cycle: `python3 run_ralph_max_quality.py`
2. Check strategy files: `ls -lt logs/ralph_approved_strategies_*.json`
3. Verify trust scores are >= 70

### Strategies Not Passing to Optimus

Check Donnie's validation:
- Trust score must be >= 70
- Backtest score must be >= 50
- Check `logs/donnie.log` for rejection reasons

### Optimus Not Receiving Strategies

Ensure Optimus is initialized when Donnie runs:
- Donnie's `run_cycle()` must receive `optimus_agent` parameter
- Optimus must have `receive_message()` method

---

## Files Created

- `run_donnie_cycle.py` - Single Donnie cycle execution
- `automated_donnie_scheduler.py` - Continuous automated scheduler
- `check_strategy_flow.py` - Strategy flow analysis tool

---

## Summary

✅ **Automation Complete!**

The Donnie cycle is now automated and running. Strategies flow automatically:
- Ralph generates high-quality strategies (trust >= 70)
- Donnie validates and executes them
- Optimus receives execution instructions

Use `automated_donnie_scheduler.py` to keep the pipeline running continuously!


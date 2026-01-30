# Running NAE - Quick Reference Guide

## ✅ **Status: NAE is Currently Running!**

The NAE master scheduler is already running in the background (PID: 30791).

---

## 🚀 **How to Run NAE**

### **Option 1: Master Scheduler (Full Automation) - RECOMMENDED**

Runs all agents with automated scheduling:

```bash
cd NAE
python3 scripts/start_nae_automation.py
```

**Or run in background:**
```bash
python3 scripts/start_nae_automation.py --background
```

**Or directly:**
```bash
python3 nae_master_scheduler.py
```

**What it does:**
- ✅ Starts all agents (Ralph, Donnie, Optimus, Casey, etc.)
- ✅ Schedules agent runs at appropriate intervals
- ✅ Monitors agent health and performance
- ✅ Manages agent communication

---

### **Option 2: Run All Agents (Simple)**

Starts all agents in separate threads:

```bash
python3 scripts/run_all.py
```

**What it does:**
- ✅ Starts all agents simultaneously
- ✅ Uses threading for parallel execution
- ✅ Casey monitors agent resource usage

---

### **Option 3: Individual Agent Testing**

Test specific agents:

```bash
# Ralph - Strategy Generation
python3 agents/ralph.py
python3 scripts/run_ralph_cycle.py

# Donnie - Strategy Execution
python3 scripts/run_donnie_cycle.py

# Optimus - Live Trading
python3 agents/optimus.py

# Casey - Agent Builder
python3 agents/casey.py
```

---

### **Option 4: Demo Mode**

Interactive demo:

```bash
python3 nae_demo.py
```

---

### **Option 5: AutoGen Integration**

Run with AutoGen framework:

```bash
# Full AutoGen integration (requires API keys)
python3 nae_autogen_integrated.py

# Simple demo (no API keys needed)
python3 nae_casey_autogen_demo.py
```

---

## 📊 **Check NAE Status**

```bash
# Check if scheduler is running
ps aux | grep nae_master_scheduler

# View scheduler logs
tail -f logs/master_scheduler.log

# View agent logs
tail -f logs/ralph.log
tail -f logs/casey.log
tail -f logs/optimus.log

# Check Redis (if using kill switch)
python3 redis_kill_switch.py --status
```

---

## 🛑 **Stop NAE**

```bash
# Find and kill the scheduler process
ps aux | grep nae_master_scheduler
kill <PID>

# Or kill all Python NAE processes
pkill -f "nae_master_scheduler"
```

---

## ⚙️ **Agent Intervals (Master Scheduler)**

- **Ralph**: Every 30 minutes (strategy generation)
- **Donnie**: Every 30 minutes (strategy execution)
- **Optimus**: Every 10 seconds (live trading check)
- **Casey**: Every 2 hours (agent building/refinement)
- **Bebop**: Every 15 minutes (system monitoring)
- **Splinter**: Every hour (orchestration)
- **Rocksteady**: Every 6 hours (security sweep)
- **Phisher**: Every hour (security scan)
- **Genny**: Every 3 hours (wealth tracking)

---

## 🔧 **Troubleshooting**

### **If NAE won't start:**
1. Check dependencies: `pip3 install -r requirements.txt`
2. Verify setup: `python3 verify_setup.py`
3. Check logs: `ls -la logs/`

### **If agents aren't communicating:**
1. Check Redis is running: `redis-cli ping`
2. Verify API keys: `python3 scripts/check_api_keys.py`
3. Check agent logs for errors

### **If you see import errors:**
1. Make sure you're in NAE directory: `cd NAE`
2. Check Python path: `python3 -c "import sys; print(sys.path)"`

---

## 📝 **Current Status**

✅ **NAE Master Scheduler**: Running (PID: 30791)  
✅ **Agents**: All initialized successfully  
✅ **Latest Run**: 17 strategies approved by Ralph  
✅ **System**: Operational  

---

## 🎯 **Quick Start (Right Now)**

If NAE is already running, you can:

1. **Check status**: `tail -f logs/master_scheduler.log`
2. **View agent activity**: `tail -f logs/ralph.log`
3. **Test individual agents**: `python3 agents/ralph.py`
4. **Interact with Casey**: `python3 nae_casey_autogen_demo.py`

---

**NAE is ready to go! 🚀**




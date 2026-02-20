# NAE Production Setup - Complete ✅

## Status: Production Mode Active

**Date:** Sat Nov 29 08:32:38 EST 2025

---

## ✅ Setup Complete

NAE is now configured to run continuously and autonomously in production mode with the following features:

### Features Implemented

1. **Continuous Operation**
   - NAE runs continuously in background
   - Auto-restarts on failure
   - Monitors all agents and processes

2. **Sleep Prevention**
   - Mac prevented from sleeping during NAE operation
   - Uses `caffeinate` to keep system awake

3. **Auto-Restart**
   - Launchd service ensures NAE starts on login
   - Auto-restarts if process crashes
   - Monitors and restarts individual agents

4. **Update Mechanism**
   - Script to pull updates from dev (HP)
   - Graceful restart after updates
   - Maintains continuous operation

5. **Background Operation**
   - All processes run in background
   - Logs to dedicated log files
   - No terminal required

---

## 📋 Files Created

### Core Scripts
- `nae_autonomous_master.py` - Main controller for continuous operation
- `start_nae_production.sh` - Start NAE in production mode
- `stop_nae_production.sh` - Stop NAE production mode
- `update_from_dev.sh` - Update production from dev branch

### Service Files
- `com.nae.production.plist` - Launchd service configuration
- `install_production_service.sh` - Install launchd service

---

## 🚀 Usage

### Start NAE Production
```bash
./start_nae_production.sh
```

### Stop NAE Production
```bash
./stop_nae_production.sh
```

### Update from Dev (HP)
```bash
./update_from_dev.sh
```

### Check Status
```bash
# Check if running
ps aux | grep nae_autonomous_master

# View logs
tail -f logs/nae_autonomous_master.log

# Check launchd service
launchctl list | grep nae
```

### Install Launchd Service (Auto-start on login)
```bash
./install_production_service.sh
```

---

## 📊 Process Management

### Monitored Processes
- **Master API Server** - `master_api/server.py`
- **All Agents** - Automatically discovered from `agents/` directory
- **Health Monitoring** - System resources, disk, memory, CPU

### Auto-Restart
- All processes auto-restart on failure
- Maximum restart attempts: 10,000 (essentially unlimited)
- Restart delay: 5-10 seconds

---

## 🔄 Update Workflow

### From Development (HP)

1. **Develop on HP** (dev branch)
   - Make changes
   - Test in dev environment
   - Commit to dev branch

2. **Update Production** (Mac)
   ```bash
   ./update_from_dev.sh
   ```
   - Fetches changes from dev
   - Merges into prod branch
   - Gracefully restarts NAE
   - Maintains continuous operation

---

## 📝 Logs

### Log Files
- `logs/nae_autonomous_master.log` - Main controller log
- `logs/nae_startup.log` - Startup log
- `logs/agent_*.log` - Individual agent logs
- `logs/caffeinate.log` - Sleep prevention log
- `logs/launchd.out` - Launchd stdout
- `logs/launchd.err` - Launchd stderr

### View Logs
```bash
# Main log
tail -f logs/nae_autonomous_master.log

# All logs
tail -f logs/*.log

# Specific agent
tail -f logs/agent_*.log
```

---

## ⚙️ Configuration

### Environment
- **Production Mode:** `PRODUCTION=true`
- **Branch:** `prod`
- **Sandbox Mode:** `false`
- **Live Trading:** `enabled`

### Launchd Service
- **Auto-start:** On login
- **Keep-alive:** Restarts on crash
- **Check interval:** Every 5 minutes
- **Working directory:** NAE root

---

## 🔧 Troubleshooting

### NAE Not Starting
```bash
# Check logs
tail -20 logs/nae_startup.log

# Check Python
python3 --version

# Check environment
cat .env | grep PRODUCTION
```

### Service Not Running
```bash
# Check launchd
launchctl list | grep nae

# Reload service
launchctl unload ~/Library/LaunchAgents/com.nae.production.plist
launchctl load ~/Library/LaunchAgents/com.nae.production.plist
```

### Mac Still Sleeping
```bash
# Check caffeinate
ps aux | grep caffeinate

# Restart caffeinate
pkill caffeinate
./start_nae_production.sh
```

---

## 🎯 Current Status

✅ **NAE Running:** Continuous autonomous operation active
✅ **Sleep Prevention:** Caffeinate active
✅ **Auto-Restart:** Enabled via launchd
✅ **Update Mechanism:** Ready for dev updates
✅ **Background Mode:** All processes in background

---

## 📌 Next Steps

1. **Monitor Operation**
   - Check logs regularly
   - Monitor system resources
   - Verify agents are running

2. **Develop on HP**
   - Make changes in dev environment
   - Test thoroughly
   - Push to dev branch

3. **Update Production**
   - Run `./update_from_dev.sh` when ready
   - Review changes before merging
   - Monitor after update

---

**NAE Production Mode is now fully operational! 🚀**


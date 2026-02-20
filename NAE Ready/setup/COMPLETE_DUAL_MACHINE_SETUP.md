# Complete Dual Machine Setup Guide

## Overview

This guide completes the setup for Mac (dev) + HP OmniBook X (prod) with full Cursor integration for remote execution.

## Current Status

✅ **Mac (Development)**
- Environment configured (dev branch, PRODUCTION=false)
- Safety systems initialized
- Master API configured
- Duplicate NAE instance stopped
- Remote execution bridge created

⏳ **HP OmniBook X (Production)**
- Needs: Clone, configure, connect

## Step-by-Step Setup

### Part 1: Mac Setup (Already Complete)

✅ Environment configured
✅ Safety systems initialized  
✅ Master API configured
✅ Remote execution bridge created
✅ Dependencies installed

### Part 2: HP OmniBook X Setup

#### 1. Clone Repository on HP

```bash
cd ~
git clone https://github.com/cbjones84/NAE.git
cd NAE
```

#### 2. Configure Environment

```bash
./setup/configure_environments.sh
```

This will:
- Detect HP OmniBook X (Linux)
- Set PRODUCTION=true
- Create .env.prod
- Generate device ID

#### 3. Initialize Safety Systems

```bash
./setup/initialize_safety_systems.sh
```

#### 4. Configure as Node

```bash
./setup/configure_node.sh
```

Enter the Master API key from Mac:
```
72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7
```

#### 5. Create Production Branch

```bash
git checkout -b prod
git push -u origin prod
```

#### 6. Enable SSH (if not already)

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

#### 7. Get HP IP Address

```bash
hostname -I
# or
ip addr show
```

Note the IP address for Mac connection.

### Part 3: Connect Mac to HP

#### 1. Configure Remote Connection on Mac

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
./setup/configure_remote_connection.sh
```

Enter:
- HP hostname/IP: (from step 7 above)
- HP username: (your HP username)
- HP port: 22 (default)
- NAE path: ~/NAE (or actual path)

#### 2. Copy SSH Key to HP

If SSH key was generated:

```bash
ssh-copy-id -p 22 <username>@<hostname>
```

Or manually:
```bash
cat ~/.ssh/id_rsa.pub | ssh <username>@<hostname> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

#### 3. Test Connection

```bash
python3 setup/cursor_remote_integration.py --verify
```

Should output: ✅ Connection verified

### Part 4: Verify Full Integration

#### 1. Get Production Status

```bash
python3 setup/cursor_remote_integration.py --status
```

Should show:
- Branch: prod
- Production Mode: true
- NAE Running: 0 (or 1 if running)

#### 2. Sync Changes

```bash
python3 setup/cursor_remote_integration.py --sync
```

This will:
- Push Mac changes to GitHub
- Pull changes on HP

#### 3. Start NAE on Production

```bash
python3 setup/cursor_remote_integration.py --start
```

#### 4. Verify NAE Running

```bash
python3 setup/cursor_remote_integration.py --status
```

Should show: NAE Running: 1

## Daily Workflow

### Development (Mac)

1. **Work on dev branch**
   ```bash
   git checkout dev
   git pull origin dev
   # Make changes, test
   ```

2. **Commit and push**
   ```bash
   git add .
   git commit -m "Description"
   git push origin dev
   ```

3. **When ready for production**
   ```bash
   git checkout main
   git merge dev
   git push origin main
   ```

4. **Deploy to production**
   ```bash
   python3 setup/cursor_remote_integration.py --sync
   python3 setup/cursor_remote_integration.py --start
   ```

### Production (HP OmniBook X)

- Automatically pulls from main
- Runs on prod branch
- Production safety checks active
- Live trading enabled (when PRODUCTION=true)

## Cursor Integration

Cursor can now:

1. **Execute commands on HP**
   ```bash
   python3 setup/cursor_remote_integration.py --command "git status"
   ```

2. **Monitor production**
   ```bash
   python3 setup/cursor_remote_integration.py --status
   ```

3. **Control production services**
   ```bash
   python3 setup/cursor_remote_integration.py --start
   python3 setup/cursor_remote_integration.py --stop
   ```

4. **Sync changes**
   ```bash
   python3 setup/cursor_remote_integration.py --sync
   ```

## Safety Features

✅ **Anti-Double-Trade**
- Device ID tracking
- Trade nonce system
- Timestamp validation

✅ **Production Guards**
- Branch must be `prod`
- Production lock file required
- Device ID verification

✅ **Environment Separation**
- Mac: dev, PRODUCTION=false
- HP: prod, PRODUCTION=true

## Troubleshooting

### Connection Issues

1. **SSH connection failed**
   - Check HP is powered on
   - Verify SSH is enabled: `sudo systemctl status ssh`
   - Check firewall allows port 22
   - Verify IP address is correct

2. **Permission denied**
   - Check SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
   - Verify authorized_keys: `chmod 600 ~/.ssh/authorized_keys`

3. **Command execution failed**
   - Verify NAE path on HP
   - Check user permissions
   - Verify Python3 available

### Production Issues

1. **NAE won't start**
   - Check production safety: `python3 safety/production_guard.py`
   - Verify branch is `prod`
   - Check production lock file exists

2. **Sync failed**
   - Verify git is configured on both machines
   - Check network connectivity
   - Verify GitHub access

## Status Check

Run this to check everything:

```bash
echo "=== Mac Status ==="
git branch --show-current
grep PRODUCTION .env
ps aux | grep nae_autonomous_master | grep -v grep | wc -l

echo ""
echo "=== HP Status ==="
python3 setup/cursor_remote_integration.py --status
```

## Next Steps

1. ✅ Complete HP OmniBook X setup (steps above)
2. ✅ Configure remote connection
3. ✅ Test connection
4. ✅ Verify full integration
5. ✅ Begin development workflow


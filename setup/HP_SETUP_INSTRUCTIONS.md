# HP OmniBook X Setup Instructions

## Quick Setup (Easiest Method)

### Method 1: Copy and Run Standalone Script

1. **On Mac**, the script `hp_local_setup.sh` has been created
2. **Copy to HP** (via USB, network share, or scp):
   ```bash
   scp hp_local_setup.sh <username>@<hp_ip>:~/
   ```
3. **On HP**, run:
   ```bash
   bash hp_local_setup.sh
   ```

### Method 2: Direct Setup on HP

1. **On HP OmniBook X**, open terminal
2. **Clone repository**:
   ```bash
   git clone https://github.com/cbjones84/NAE.git
   cd NAE
   ```
3. **Run automated setup**:
   ```bash
   bash setup/automated_hp_setup.sh
   ```

### Method 3: Remote Setup from Mac (If SSH Already Configured)

1. **On Mac**, configure remote connection:
   ```bash
   ./setup/configure_remote_connection.sh
   ```
2. **Run remote setup**:
   ```bash
   python3 setup/remote_hp_setup.py
   ```

## What the Setup Does

1. ✅ Clones NAE repository (if needed)
2. ✅ Configures environment (PRODUCTION=true)
3. ✅ Initializes safety systems
4. ✅ Configures as Node with Master API key
5. ✅ Creates production branch
6. ✅ Enables SSH
7. ✅ Generates device ID
8. ✅ Creates production lock file

## After HP Setup

### On Mac

1. **Configure remote connection**:
   ```bash
   ./setup/configure_remote_connection.sh
   ```
   Enter HP IP address when prompted

2. **Copy SSH key**:
   ```bash
   ssh-copy-id <username>@<hp_ip>
   ```

3. **Test connection**:
   ```bash
   ./setup/quick_connect.sh
   ```

4. **Verify production status**:
   ```bash
   python3 setup/cursor_remote_integration.py --status
   ```

5. **Start NAE on production**:
   ```bash
   python3 setup/cursor_remote_integration.py --start
   ```

## Verification

### On HP

```bash
cd NAE
python3 safety/production_guard.py
```

Should show:
- Production mode: true
- All checks passed
- Live trading allowed: true (if all checks pass)

### From Mac

```bash
python3 setup/cursor_remote_integration.py --status
```

Should show:
- Branch: prod
- Production Mode: true
- NAE Running: 0 or 1

## Troubleshooting

### HP Setup Issues

1. **Git clone fails**
   - Check internet connection
   - Verify GitHub access
   - Try: `git config --global http.sslVerify false` (if SSL issues)

2. **Setup scripts not executable**
   - Run: `chmod +x setup/*.sh`

3. **SSH not starting**
   - Check: `sudo systemctl status ssh`
   - Install: `sudo apt install openssh-server` (Ubuntu/Debian)

### Connection Issues

1. **Cannot connect from Mac**
   - Verify HP is on same network
   - Check firewall: `sudo ufw allow 22`
   - Test: `ssh <username>@<hp_ip>`

2. **Permission denied**
   - Copy SSH key: `ssh-copy-id <username>@<hp_ip>`
   - Check key permissions: `chmod 600 ~/.ssh/id_rsa`

## Master API Key

**Save this key** - you'll need it for HP setup:
```
72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7
```

## Status

Once setup is complete:
- ✅ HP configured as production Node
- ✅ Mac can connect and control HP
- ✅ Cursor can execute commands on HP
- ✅ Code syncs automatically via Git
- ✅ Production safety checks active


# Dual Machine Connection Setup

## Overview

This guide sets up the connection between Mac (dev) and HP OmniBook X (prod) so that Cursor can execute commands on production from the development environment.

## Prerequisites

1. **HP OmniBook X Setup Complete**
   - NAE cloned and configured
   - Production environment set up
   - SSH enabled

2. **Mac Setup Complete**
   - Development environment configured
   - SSH key generated

## Step 1: Configure Remote Connection

On Mac, run:

```bash
cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
./setup/configure_remote_connection.sh
```

This will:
- Prompt for HP hostname/IP, username, port
- Check for SSH key
- Generate SSH key if needed
- Create remote configuration file

## Step 2: Copy SSH Key to HP

If SSH key was generated, copy it to HP:

```bash
ssh-copy-id -p <port> <username>@<hostname>
```

Or manually:
```bash
cat ~/.ssh/id_rsa.pub | ssh <username>@<hostname> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

## Step 3: Verify Connection

Test the connection:

```bash
python3 setup/cursor_remote_integration.py --verify
```

## Step 4: Test Remote Execution

Execute a test command:

```bash
python3 setup/cursor_remote_integration.py --command "echo 'Test successful'"
```

## Usage

### Get Production Status

```bash
python3 setup/cursor_remote_integration.py --status
```

### Sync Changes to Production

```bash
python3 setup/cursor_remote_integration.py --sync
```

This will:
1. Push changes from Mac to GitHub
2. Pull changes on HP OmniBook X

### Start NAE on Production

```bash
python3 setup/cursor_remote_integration.py --start
```

### Stop NAE on Production

```bash
python3 setup/cursor_remote_integration.py --stop
```

### Execute Custom Command

```bash
python3 setup/cursor_remote_integration.py --command "git status"
```

## Cursor Integration

Cursor can now:
- Execute commands on HP OmniBook X
- Sync code changes automatically
- Monitor production status
- Start/stop production services

## Troubleshooting

### Connection Failed

1. Check HP is powered on and connected
2. Verify SSH is enabled: `sudo systemctl status ssh` (Linux)
3. Check firewall allows SSH (port 22)
4. Verify SSH key is copied correctly

### Permission Denied

1. Check SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
2. Verify authorized_keys on HP: `chmod 600 ~/.ssh/authorized_keys`
3. Check SSH config on HP

### Command Execution Failed

1. Verify NAE path on HP is correct
2. Check user has permissions in NAE directory
3. Verify Python3 is available on HP

## Security Notes

- SSH keys are preferred over passwords
- Keep remote_config.json secure (it's git-ignored)
- Use SSH key passphrase for additional security
- Consider using SSH config file for easier management


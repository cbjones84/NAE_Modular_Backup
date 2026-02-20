# HP OmniBook X SSH Setup (IP: 192.168.132.68)

## Current Status

✅ **IP Address:** 192.168.132.68  
⚠️ **SSH:** Not accessible yet (connection timeout)

## Enable SSH on HP OmniBook X

### Step 1: Install SSH Server (if needed)

On HP OmniBook X, open terminal and run:

```bash
sudo apt update
sudo apt install openssh-server -y
```

### Step 2: Enable and Start SSH

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

### Step 3: Verify SSH is Running

```bash
sudo systemctl status ssh
```

Should show: `Active: active (running)`

### Step 4: Allow SSH Through Firewall

```bash
sudo ufw allow 22
```

Or if ufw is not active:
```bash
sudo ufw enable
sudo ufw allow 22
```

### Step 5: Verify IP Address

```bash
hostname -I
```

Should show: `192.168.132.68` (or include it)

## After SSH is Enabled

### On Mac

1. **Copy SSH key to HP:**
   ```bash
   cd "/Users/melissabishop/Downloads/Neural Agency Engine/NAE"
   ssh-copy-id <username>@192.168.132.68
   ```
   (Replace `<username>` with HP username)

2. **Test connection:**
   ```bash
   ssh <username>@192.168.132.68 'echo Connected'
   ```

3. **Run remote setup:**
   ```bash
   python3 setup/remote_hp_setup.py
   ```

## Alternative: Use Standalone Script

If remote setup doesn't work, use the standalone script:

1. **Copy to HP:**
   ```bash
   scp hp_local_setup.sh <username>@192.168.132.68:~/
   ```

2. **On HP, run:**
   ```bash
   bash hp_local_setup.sh
   ```

## Quick Test

Once SSH is enabled, test from Mac:

```bash
ssh <username>@192.168.132.68 "echo 'SSH working!'"
```

If this works, you can proceed with remote setup.


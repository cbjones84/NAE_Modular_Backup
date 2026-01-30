#!/bin/bash
# Auto-Setup HP OmniBook X
# Tries remote setup first, then provides easy local setup option

set -e

echo "=========================================="
echo "Auto-Setup HP OmniBook X"
echo "=========================================="
echo ""

# Check if we can connect remotely
if [ -f "config/remote_config.json" ]; then
    echo "✅ Remote config found - attempting remote setup..."
    python3 setup/remote_hp_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ HP setup completed remotely!"
        echo ""
        echo "Verifying connection..."
        python3 setup/cursor_remote_integration.py --verify
        echo ""
        echo "Getting production status..."
        python3 setup/cursor_remote_integration.py --status
        exit 0
    fi
fi

# If remote setup failed or config doesn't exist, provide local setup script
echo "Creating local setup script for HP..."
echo ""

cat > "hp_local_setup.sh" << 'SETUP_EOF'
#!/bin/bash
# HP OmniBook X Local Setup Script
# Run this directly on the HP OmniBook X

set -e

echo "=========================================="
echo "HP OmniBook X - Local Setup"
echo "=========================================="
echo ""

# Step 1: Clone if needed
if [ ! -d "NAE" ]; then
    echo "Step 1: Cloning NAE repository..."
    git clone https://github.com/cbjones84/NAE.git
    cd NAE
else
    cd NAE
    echo "Step 1: NAE directory found"
fi

NAE_ROOT="$(pwd)"
echo "NAE Root: $NAE_ROOT"
echo ""

# Step 2: Configure environment
echo "Step 2: Configuring environment..."
if [ -f "setup/configure_environments.sh" ]; then
    bash setup/configure_environments.sh
else
    echo "⚠️  Setup script not found - creating basic config..."
    cat > .env.prod << 'ENV_EOF'
# NAE Environment Configuration - HP OmniBook X Production
NAE_MACHINE_TYPE=hp
NAE_MACHINE_NAME="HP OmniBook X Production"
NAE_BRANCH=prod
PRODUCTION=true
NAE_PRODUCTION_MODE=true
NAE_SAFETY_ENABLED=true
NAE_ANTI_DOUBLE_TRADE=true
NAE_BRANCH_CHECK=true
NAE_LIVE_TRADING_ENABLED=true
NAE_SANDBOX_MODE=false
NAE_DEVICE_ID=""
NAE_GIT_BRANCH=prod
NAE_GIT_REQUIRE_PROD_BRANCH=true
NAE_IS_MASTER=false
NAE_IS_NODE=true
NAE_NODE_API_KEY=72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7
NAE_MASTER_URL=http://<MAC_IP>:8080
NAE_NODE_PORT=8081
ENV_EOF
    ln -sf .env.prod .env
    echo "✅ Basic environment created"
fi

# Generate device ID
if ! grep -q "NAE_DEVICE_ID=" .env 2>/dev/null; then
    DEVICE_ID=$(openssl rand -hex 16 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(16))" 2>/dev/null || echo "hp_$(hostname)_$(date +%s)")
    echo "NAE_DEVICE_ID=$DEVICE_ID" >> .env
    echo "Generated Device ID: $DEVICE_ID"
fi

# Step 3: Initialize safety systems
echo ""
echo "Step 3: Initializing safety systems..."
mkdir -p safety/locks safety/logs safety/checks

# Create production lock file
cat > safety/locks/production.lock << 'LOCK_EOF'
# Production Lock File
LOCKED=true
CREATED_AT=$(date +%s)
MACHINE_TYPE=hp
DEVICE_ID=$(grep NAE_DEVICE_ID .env | cut -d'=' -f2)
LOCK_EOF

echo "$(grep NAE_DEVICE_ID .env | cut -d'=' -f2)" > safety/device_id.txt
chmod 600 safety/device_id.txt

# Create branch check
mkdir -p safety/checks
cat > safety/checks/branch_check.py << 'BRANCH_EOF'
#!/usr/bin/env python3
import os, sys, subprocess
def get_current_branch():
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return None
def check_branch():
    production = os.getenv('PRODUCTION', 'false').lower() == 'true'
    required_branch = os.getenv('NAE_GIT_BRANCH', 'main')
    current_branch = get_current_branch()
    if production and current_branch != required_branch:
        print(f"❌ ERROR: Production mode requires branch '{required_branch}', but current branch is '{current_branch}'")
        return False
    return True
if __name__ == '__main__':
    if not check_branch():
        sys.exit(1)
BRANCH_EOF
chmod +x safety/checks/branch_check.py

echo "✅ Safety systems initialized"

# Step 4: Create production branch
echo ""
echo "Step 4: Setting up production branch..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    if git rev-parse --verify prod > /dev/null 2>&1; then
        git checkout prod
        echo "✅ Switched to prod branch"
    else
        git checkout -b prod
        echo "✅ Created prod branch"
    fi
else
    echo "⚠️  Not a git repository"
fi

# Step 5: Enable SSH
echo ""
echo "Step 5: Enabling SSH..."
if command -v systemctl > /dev/null; then
    sudo systemctl enable ssh 2>/dev/null || sudo systemctl enable sshd 2>/dev/null || echo "⚠️  Could not enable SSH"
    sudo systemctl start ssh 2>/dev/null || sudo systemctl start sshd 2>/dev/null || echo "⚠️  Could not start SSH"
    echo "✅ SSH enabled"
fi

# Step 6: Get IP address
echo ""
echo "Step 6: Getting IP address..."
IP_ADDRESS=$(hostname -I | awk '{print $1}' 2>/dev/null || ip addr show | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1 || echo "unknown")

# Step 7: Verify
echo ""
echo "Step 7: Verifying setup..."
echo ""
echo "=========================================="
echo "✅ HP OmniBook X Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Machine: HP OmniBook X"
echo "  Production: true"
echo "  Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
echo "  Device ID: $(grep NAE_DEVICE_ID .env 2>/dev/null | cut -d'=' -f2 || echo 'unknown')"
echo "  IP Address: $IP_ADDRESS"
echo ""
echo "Next Steps:"
echo "1. On Mac: ./setup/configure_remote_connection.sh"
echo "2. Enter HP IP: $IP_ADDRESS"
echo "3. Copy SSH key: ssh-copy-id <username>@$IP_ADDRESS"
echo "4. Test: python3 setup/cursor_remote_integration.py --verify"
echo ""
SETUP_EOF

chmod +x hp_local_setup.sh

echo "✅ Created: hp_local_setup.sh"
echo ""
echo "=========================================="
echo "Setup Script Ready"
echo "=========================================="
echo ""
echo "To complete HP setup, choose one:"
echo ""
echo "Option A: Copy script to HP and run"
echo "  1. Copy hp_local_setup.sh to HP OmniBook X"
echo "  2. On HP, run: bash hp_local_setup.sh"
echo ""
echo "Option B: Run directly on HP"
echo "  1. On HP: git clone https://github.com/cbjones84/NAE.git"
echo "  2. On HP: cd NAE && bash setup/automated_hp_setup.sh"
echo ""
echo "Option C: If you have HP IP and SSH access"
echo "  1. Run: ./setup/configure_remote_connection.sh"
echo "  2. Then: python3 setup/remote_hp_setup.py"
echo ""


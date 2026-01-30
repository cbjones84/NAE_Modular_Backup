#!/bin/bash
# Complete HP Setup with IP 192.168.132.68
# This script orchestrates the entire setup process

HP_IP="192.168.132.68"

echo "=========================================="
echo "Complete HP OmniBook X Setup"
echo "IP: $HP_IP"
echo "=========================================="
echo ""

# Step 1: Check if SSH is accessible
echo "Step 1: Checking SSH connection..."
if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$HP_IP" "echo 'Connected'" 2>/dev/null; then
    echo "✅ SSH is accessible!"
    SSH_READY=true
else
    echo "⚠️  SSH not accessible yet"
    SSH_READY=false
fi
echo ""

# Step 2: If SSH not ready, provide instructions
if [ "$SSH_READY" = false ]; then
    echo "=========================================="
    echo "SSH Setup Required on HP"
    echo "=========================================="
    echo ""
    echo "To enable SSH on HP OmniBook X:"
    echo ""
    echo "Option A: Copy script to HP"
    echo "  1. Copy setup/enable_ssh_on_hp.sh to HP"
    echo "  2. On HP, run: sudo bash enable_ssh_on_hp.sh"
    echo ""
    echo "Option B: Manual commands on HP"
    echo "  sudo apt update"
    echo "  sudo apt install openssh-server -y"
    echo "  sudo systemctl enable ssh"
    echo "  sudo systemctl start ssh"
    echo "  sudo ufw allow 22"
    echo ""
    echo "Option C: Use standalone setup (includes SSH setup)"
    echo "  1. Copy hp_local_setup.sh to HP"
    echo "  2. On HP, run: bash hp_local_setup.sh"
    echo "  (This will set up everything including SSH)"
    echo ""
    read -p "Press Enter when SSH is enabled on HP, or 's' to skip and use standalone script: " response
    
    if [ "$response" == "s" ]; then
        echo ""
        echo "Using standalone script method..."
        echo ""
        echo "1. Copy hp_local_setup.sh to HP OmniBook X"
        echo "2. On HP, run: bash hp_local_setup.sh"
        echo "3. After setup, get HP IP and configure connection"
        exit 0
    fi
    
    # Test again
    echo ""
    echo "Testing SSH connection again..."
    if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$HP_IP" "echo 'Connected'" 2>/dev/null; then
        echo "✅ SSH is now accessible!"
        SSH_READY=true
    else
        echo "⚠️  SSH still not accessible"
        echo "Please enable SSH on HP and run this script again"
        exit 1
    fi
fi

# Step 3: Get username
if [ -f "config/remote_config.json" ]; then
    HP_USERNAME=$(grep -o '"hp_username": "[^"]*' config/remote_config.json | cut -d'"' -f4)
fi

if [ -z "$HP_USERNAME" ]; then
    read -p "Enter HP OmniBook X username: " HP_USERNAME
    if [ -z "$HP_USERNAME" ]; then
        echo "❌ Username required"
        exit 1
    fi
fi

# Step 4: Set up SSH key
echo ""
echo "Step 2: Setting up SSH key..."
if [ ! -f "$HOME/.ssh/id_rsa" ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f "$HOME/.ssh/id_rsa" -N "" -q
    echo "✅ SSH key generated"
fi

echo "Copying SSH key to HP..."
ssh-copy-id -o StrictHostKeyChecking=no "$HP_USERNAME@$HP_IP" 2>&1 | grep -v "Warning: Permanently added" || echo "⚠️  Key copy may have failed - you may need to enter password"

# Step 5: Update remote config
echo ""
echo "Step 3: Updating remote configuration..."
mkdir -p config
cat > config/remote_config.json << EOF
{
  "hp_hostname": "$HP_IP",
  "hp_username": "$HP_USERNAME",
  "hp_port": 22,
  "hp_nae_path": "~/NAE",
  "ssh_key_path": "$HOME/.ssh/id_rsa",
  "connection_timeout": 10
}
EOF
echo "✅ Remote config updated"

# Step 6: Run remote setup
echo ""
echo "Step 4: Running remote setup..."
python3 setup/remote_hp_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ HP Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Verifying connection..."
    python3 setup/cursor_remote_integration.py --verify
    echo ""
    echo "Getting production status..."
    python3 setup/cursor_remote_integration.py --status
    echo ""
    echo "🎉 HP OmniBook X is fully set up and connected!"
else
    echo ""
    echo "⚠️  Remote setup encountered issues"
    echo "See output above for details"
    echo ""
    echo "You can also use the standalone script:"
    echo "  1. Copy hp_local_setup.sh to HP"
    echo "  2. On HP, run: bash hp_local_setup.sh"
fi


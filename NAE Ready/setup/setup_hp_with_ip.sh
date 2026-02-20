#!/bin/bash
# Setup HP OmniBook X with known IP address
# IP: 192.168.132.68

set -e

HP_IP="192.168.132.68"
CONFIG_FILE="config/remote_config.json"

echo "=========================================="
echo "HP OmniBook X Setup (IP: $HP_IP)"
echo "=========================================="
echo ""

# Create config directory
mkdir -p config

# Get username
read -p "Enter HP OmniBook X username: " HP_USERNAME
if [ -z "$HP_USERNAME" ]; then
    echo "❌ Username required"
    exit 1
fi

# Check for SSH key
SSH_KEY="$HOME/.ssh/id_rsa"
if [ ! -f "$SSH_KEY" ]; then
    echo "SSH key not found. Generating..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY" -N "" -q
    echo "✅ SSH key generated"
fi

# Create remote config
cat > "$CONFIG_FILE" << EOF
{
  "hp_hostname": "$HP_IP",
  "hp_username": "$HP_USERNAME",
  "hp_port": 22,
  "hp_nae_path": "~/NAE",
  "ssh_key_path": "$SSH_KEY",
  "connection_timeout": 10
}
EOF

echo "✅ Remote configuration created"
echo ""

# Test connection
echo "Testing connection to $HP_IP..."
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$HP_USERNAME@$HP_IP" "echo 'Connection successful'" 2>/dev/null; then
    echo "✅ Connection successful!"
    echo ""
    echo "Attempting remote setup..."
    python3 setup/remote_hp_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ HP setup completed remotely!"
        exit 0
    fi
else
    echo "⚠️  Connection failed"
    echo ""
    echo "SSH connection to HP is not available yet."
    echo ""
    echo "=========================================="
    echo "Enable SSH on HP OmniBook X"
    echo "=========================================="
    echo ""
    echo "On HP OmniBook X, run these commands:"
    echo ""
    echo "1. Install SSH (if not installed):"
    echo "   sudo apt update"
    echo "   sudo apt install openssh-server"
    echo ""
    echo "2. Enable and start SSH:"
    echo "   sudo systemctl enable ssh"
    echo "   sudo systemctl start ssh"
    echo ""
    echo "3. Check SSH status:"
    echo "   sudo systemctl status ssh"
    echo ""
    echo "4. Allow SSH through firewall:"
    echo "   sudo ufw allow 22"
    echo ""
    echo "5. Get IP address (verify):"
    echo "   hostname -I"
    echo "   (Should show: $HP_IP)"
    echo ""
    echo "=========================================="
    echo "After SSH is enabled:"
    echo "=========================================="
    echo ""
    echo "1. Copy SSH key to HP:"
    echo "   ssh-copy-id $HP_USERNAME@$HP_IP"
    echo ""
    echo "2. Test connection:"
    echo "   ssh $HP_USERNAME@$HP_IP 'echo Connected'"
    echo ""
    echo "3. Run remote setup:"
    echo "   python3 setup/remote_hp_setup.py"
    echo ""
    echo "OR use standalone script:"
    echo "   Copy hp_local_setup.sh to HP and run it"
    echo ""
fi


#!/bin/bash
# Configure Remote Connection to HP OmniBook X

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$NAE_ROOT/config"

echo "=========================================="
echo "Remote Connection Configuration"
echo "HP OmniBook X (Production) Setup"
echo "=========================================="
echo ""

# Create config directory
mkdir -p "$CONFIG_DIR"

# Get HP connection details
read -p "Enter HP OmniBook X hostname/IP: " HP_HOSTNAME
read -p "Enter HP OmniBook X username: " HP_USERNAME
read -p "Enter HP OmniBook X SSH port (default: 22): " HP_PORT
HP_PORT=${HP_PORT:-22}

read -p "Enter NAE path on HP (default: ~/NAE): " HP_NAE_PATH
HP_NAE_PATH=${HP_NAE_PATH:-"~/NAE"}

# Check for SSH key
SSH_KEY_PATH="$HOME/.ssh/id_rsa"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo ""
    echo "⚠️  SSH key not found at $SSH_KEY_PATH"
    read -p "Generate SSH key? [y/N]: " GENERATE_KEY
    
    if [ "$GENERATE_KEY" == "y" ]; then
        ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N ""
        echo "✅ SSH key generated"
        echo ""
        echo "Next step: Copy public key to HP OmniBook X:"
        echo "  ssh-copy-id -p $HP_PORT $HP_USERNAME@$HP_HOSTNAME"
        echo ""
        read -p "Press Enter after copying SSH key..."
    else
        read -p "Enter SSH key path (or leave empty for password auth): " SSH_KEY_PATH
        SSH_KEY_PATH=${SSH_KEY_PATH:-""}
    fi
fi

# Create remote config
REMOTE_CONFIG="$CONFIG_DIR/remote_config.json"
cat > "$REMOTE_CONFIG" << EOF
{
  "hp_hostname": "$HP_HOSTNAME",
  "hp_username": "$HP_USERNAME",
  "hp_port": $HP_PORT,
  "hp_nae_path": "$HP_NAE_PATH",
  "ssh_key_path": "$SSH_KEY_PATH",
  "connection_timeout": 10
}
EOF

echo "✅ Remote configuration saved to $REMOTE_CONFIG"
echo ""

# Test connection
echo "Testing connection..."
python3 "$SCRIPT_DIR/remote_execution_bridge.py" "echo 'Connection test successful'" 2>&1 | grep -q "success" && echo "✅ Connection successful!" || echo "⚠️  Connection test failed - check configuration"

echo ""
echo "=========================================="
echo "✅ Remote connection configured!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  python3 setup/remote_execution_bridge.py 'command'"
echo "  python3 setup/remote_execution_bridge.py --status"
echo "  python3 setup/remote_execution_bridge.py --start"
echo "  python3 setup/remote_execution_bridge.py --stop"
echo ""


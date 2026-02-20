#!/bin/bash
# Complete HP OmniBook X Setup from Mac
# This script attempts to set up HP remotely or provides instructions

set -e

echo "=========================================="
echo "Complete HP OmniBook X Setup from Mac"
echo "=========================================="
echo ""

# Check if remote config exists
if [ -f "config/remote_config.json" ]; then
    echo "✅ Remote configuration found"
    echo "Attempting remote setup..."
    echo ""
    
    # Try remote setup
    python3 setup/remote_hp_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Remote setup completed successfully!"
        exit 0
    else
        echo ""
        echo "⚠️  Remote setup encountered issues"
        echo "Falling back to manual setup instructions..."
        echo ""
    fi
else
    echo "⚠️  Remote configuration not found"
    echo "You have two options:"
    echo ""
fi

# Option 1: Generate setup script for HP
echo "Option 1: Generate setup script for HP"
echo "--------------------------------------"
echo ""
echo "This will create a script you can copy to HP and run:"
echo ""

# Create standalone setup script
STANDALONE_SCRIPT="hp_setup_standalone.sh"
cat > "$STANDALONE_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash
# Standalone HP OmniBook X Setup Script
# Copy this to HP and run: bash hp_setup_standalone.sh

set -e

echo "=========================================="
echo "HP OmniBook X - Standalone Setup"
echo "=========================================="
echo ""

# Clone if needed
if [ ! -d "NAE" ]; then
    echo "Cloning NAE repository..."
    git clone https://github.com/cbjones84/NAE.git
    cd NAE
else
    cd NAE
fi

# Run setup scripts
echo "Configuring environment..."
bash setup/configure_environments.sh

echo "Initializing safety systems..."
bash setup/initialize_safety_systems.sh

# Configure as node
MASTER_API_KEY="72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7"
echo "Configuring as Node..."
if [ -f ".env.prod" ]; then
    cat >> .env.prod << EOF

# Node Settings
NAE_IS_MASTER=false
NAE_IS_NODE=true
NAE_NODE_API_KEY=$MASTER_API_KEY
NAE_MASTER_URL=http://<MAC_IP>:8080
NAE_NODE_PORT=8081
EOF
    echo "✅ Node configured (update MASTER_URL with Mac IP)"
fi

# Create prod branch
echo "Setting up production branch..."
git checkout -b prod 2>/dev/null || git checkout prod
git push -u origin prod 2>/dev/null || echo "⚠️  Push failed - may need to set remote"

# Enable SSH
echo "Enabling SSH..."
sudo systemctl enable ssh 2>/dev/null || sudo systemctl enable sshd 2>/dev/null || true
sudo systemctl start ssh 2>/dev/null || sudo systemctl start sshd 2>/dev/null || true

# Get IP
IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "unknown")
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "HP IP Address: $IP"
echo "Save this for Mac connection"
echo ""
SCRIPT_EOF

chmod +x "$STANDALONE_SCRIPT"
echo "✅ Created: $STANDALONE_SCRIPT"
echo ""
echo "To use:"
echo "1. Copy $STANDALONE_SCRIPT to HP OmniBook X"
echo "2. On HP, run: bash $STANDALONE_SCRIPT"
echo ""

# Option 2: Manual instructions
echo "Option 2: Manual Setup Instructions"
echo "------------------------------------"
echo ""
echo "On HP OmniBook X, run these commands:"
echo ""
echo "1. Clone repository:"
echo "   git clone https://github.com/cbjones84/NAE.git"
echo "   cd NAE"
echo ""
echo "2. Configure environment:"
echo "   ./setup/configure_environments.sh"
echo ""
echo "3. Initialize safety:"
echo "   ./setup/initialize_safety_systems.sh"
echo ""
echo "4. Configure as node:"
echo "   ./setup/configure_node.sh"
echo "   (Enter Master API Key when prompted)"
echo ""
echo "5. Create prod branch:"
echo "   git checkout -b prod"
echo "   git push -u origin prod"
echo ""
echo "6. Enable SSH:"
echo "   sudo systemctl enable ssh"
echo "   sudo systemctl start ssh"
echo ""
echo "7. Get IP address:"
echo "   hostname -I"
echo ""
echo "Then on Mac:"
echo "1. ./setup/configure_remote_connection.sh"
echo "2. Enter HP IP address"
echo "3. Copy SSH key: ssh-copy-id <username>@<hp_ip>"
echo "4. Test: ./setup/quick_connect.sh"
echo ""


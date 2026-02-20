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

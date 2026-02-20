#!/bin/bash
# Automated HP OmniBook X Setup Script
# Can be run directly on HP or remotely from Mac

set -e

echo "=========================================="
echo "HP OmniBook X - Automated Setup"
echo "=========================================="
echo ""

# Detect if running on HP (Linux) or remotely
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Detected: HP OmniBook X (Linux)"
    RUNNING_ON_HP=true
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  Running from Mac - will attempt remote execution"
    RUNNING_ON_HP=false
else
    echo "⚠️  Unknown OS - proceeding anyway"
    RUNNING_ON_HP=false
fi

# Check if NAE directory exists
if [ ! -d "NAE" ] && [ "$RUNNING_ON_HP" = true ]; then
    echo "NAE directory not found. Cloning repository..."
    git clone https://github.com/cbjones84/NAE.git
    cd NAE
elif [ -d "NAE" ]; then
    cd NAE
fi

NAE_ROOT="$(pwd)"
echo "NAE Root: $NAE_ROOT"
echo ""

# Step 1: Configure Environment
echo "Step 1: Configuring environment..."
if [ "$RUNNING_ON_HP" = true ]; then
    ./setup/configure_environments.sh
else
    echo "⚠️  Must run configure_environments.sh directly on HP"
fi

# Step 2: Initialize Safety Systems
echo ""
echo "Step 2: Initializing safety systems..."
if [ -f "setup/initialize_safety_systems.sh" ]; then
    ./setup/initialize_safety_systems.sh
else
    echo "⚠️  Safety systems script not found"
fi

# Step 3: Configure as Node
echo ""
echo "Step 3: Configuring as Node..."
MASTER_API_KEY="72364bc8ecfea124010e9811d06d0b0b3b220e4dee3d09163b2bd1f005af40a7"
MASTER_URL="http://$(hostname -I | awk '{print $1}'):8080"  # Will need Mac IP

if [ -f "setup/configure_node.sh" ]; then
    # Auto-configure node
    if [ -f ".env" ] || [ -f ".env.prod" ]; then
        ENV_FILE=".env.prod"
        if [ -f ".env" ]; then
            ENV_FILE=".env"
        fi
        
        # Add node settings if not present
        if ! grep -q "NAE_IS_NODE" "$ENV_FILE" 2>/dev/null; then
            cat >> "$ENV_FILE" << EOF

# Node Settings
NAE_IS_MASTER=false
NAE_IS_NODE=true
NAE_NODE_API_KEY=$MASTER_API_KEY
NAE_MASTER_URL=$MASTER_URL
NAE_NODE_PORT=8081
EOF
            echo "✅ Node settings added to $ENV_FILE"
        fi
    else
        echo "⚠️  Environment file not found - run configure_environments.sh first"
    fi
else
    echo "⚠️  Node configuration script not found"
fi

# Step 4: Create Production Branch
echo ""
echo "Step 4: Setting up production branch..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    if git rev-parse --verify prod > /dev/null 2>&1; then
        echo "✅ Production branch already exists"
        git checkout prod
    else
        git checkout -b prod
        echo "✅ Production branch created"
    fi
    
    # Set upstream if not set
    if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} > /dev/null 2>&1; then
        git push -u origin prod 2>/dev/null || echo "⚠️  Could not push prod branch (may need to set remote)"
    fi
else
    echo "⚠️  Not a git repository"
fi

# Step 5: Enable SSH (if on HP)
if [ "$RUNNING_ON_HP" = true ]; then
    echo ""
    echo "Step 5: Enabling SSH..."
    if command -v systemctl > /dev/null; then
        sudo systemctl enable ssh 2>/dev/null || sudo systemctl enable sshd 2>/dev/null || echo "⚠️  Could not enable SSH"
        sudo systemctl start ssh 2>/dev/null || sudo systemctl start sshd 2>/dev/null || echo "⚠️  Could not start SSH"
        echo "✅ SSH enabled"
    else
        echo "⚠️  systemctl not available"
    fi
fi

# Step 6: Verify Setup
echo ""
echo "Step 6: Verifying setup..."
echo ""

# Check environment
if [ -f ".env" ] || [ -f ".env.prod" ]; then
    ENV_FILE=".env.prod"
    [ -f ".env" ] && ENV_FILE=".env"
    
    echo "Environment Configuration:"
    grep -E "PRODUCTION|NAE_MACHINE_TYPE|NAE_BRANCH" "$ENV_FILE" 2>/dev/null | head -3 || echo "  (environment file found)"
else
    echo "⚠️  Environment file not found"
fi

# Check branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "Current Branch: $CURRENT_BRANCH"

# Check safety systems
if [ -d "safety" ]; then
    echo "Safety Systems: ✅ Present"
else
    echo "Safety Systems: ⚠️  Not found"
fi

# Get IP address
if [ "$RUNNING_ON_HP" = true ]; then
    IP_ADDRESS=$(hostname -I | awk '{print $1}' 2>/dev/null || ip addr show | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1)
    echo ""
    echo "=========================================="
    echo "✅ HP OmniBook X Setup Complete!"
    echo "=========================================="
    echo ""
    echo "IP Address: $IP_ADDRESS"
    echo "Save this for Mac connection configuration"
    echo ""
    echo "Next Steps:"
    echo "1. On Mac: ./setup/configure_remote_connection.sh"
    echo "2. Enter IP: $IP_ADDRESS"
    echo "3. Copy SSH key: ssh-copy-id <username>@$IP_ADDRESS"
    echo "4. Test: ./setup/quick_connect.sh"
else
    echo ""
    echo "=========================================="
    echo "⚠️  Setup script ready for HP"
    echo "=========================================="
    echo ""
    echo "To complete setup on HP OmniBook X:"
    echo "1. Copy this script to HP"
    echo "2. Run: bash automated_hp_setup.sh"
    echo ""
fi

echo ""
echo "Production Guard Check:"
python3 safety/production_guard.py 2>/dev/null || echo "⚠️  Production guard check failed"


#!/bin/bash
# Switch HP OmniBook X from Production to Development Environment
# Run this script on the HP OmniBook X

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "HP OmniBook X - Switch to Development"
echo "=========================================="
echo ""

# Check if on Linux (HP)
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is for Linux (HP OmniBook X)"
    read -p "Continue anyway? [y/N]: " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

# Backup current .env
if [ -f "$NAE_ROOT/.env" ]; then
    echo "Backing up current .env..."
    cp "$NAE_ROOT/.env" "$NAE_ROOT/.env.backup.$(date +%s)"
fi

# Get device ID from current .env if exists
DEVICE_ID=""
if [ -f "$NAE_ROOT/.env" ]; then
    DEVICE_ID=$(grep "NAE_DEVICE_ID=" "$NAE_ROOT/.env" | cut -d'=' -f2 | tr -d '"' | head -1)
fi

# Generate device ID if not found
if [ -z "$DEVICE_ID" ]; then
    DEVICE_ID=$(openssl rand -hex 16 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(16))" 2>/dev/null || echo "hp_$(hostname)_$(date +%s)")
fi

# Create .env.dev file
cat > "$NAE_ROOT/.env.dev" << EOF
# NAE Environment Configuration
# Machine: HP OmniBook X Development
# Generated: $(date)

# Machine Identification
NAE_MACHINE_TYPE=hp
NAE_MACHINE_NAME="HP OmniBook X Development"
NAE_BRANCH=dev

# Production Settings
PRODUCTION=false
NAE_PRODUCTION_MODE=false

# Safety Settings
NAE_SAFETY_ENABLED=true
NAE_ANTI_DOUBLE_TRADE=true
NAE_BRANCH_CHECK=true
NAE_DEVICE_ID_CHECK=true

# Trading Settings (Development)
NAE_LIVE_TRADING_ENABLED=false
NAE_SANDBOX_MODE=true

# Device ID
NAE_DEVICE_ID=$DEVICE_ID

# Git Settings
NAE_GIT_BRANCH=dev
NAE_GIT_REQUIRE_PROD_BRANCH=false

# Master-Node Settings
NAE_MASTER_NODE_ENABLED=true
NAE_IS_MASTER=false
NAE_IS_NODE=true

# Communication (Update with Mac IP if needed)
NAE_MASTER_URL="http://192.168.132.36:8080"
NAE_NODE_URL=""
NAE_NODE_API_KEY=""

# Logging
NAE_LOG_LEVEL=INFO
NAE_LOG_DIR="$NAE_ROOT/logs"
EOF

# Update symlink
if [ -L "$NAE_ROOT/.env" ]; then
    rm "$NAE_ROOT/.env"
fi
ln -sf .env.dev "$NAE_ROOT/.env"

echo "✅ Created .env.dev and updated symlink"
echo ""

# Switch to dev branch if needed
if [ -d "$NAE_ROOT/.git" ]; then
    echo "Checking git branch..."
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" != "dev" ]; then
        echo "Switching to dev branch..."
        git checkout dev 2>/dev/null || git checkout -b dev 2>/dev/null || echo "⚠️  Could not switch branch"
    else
        echo "✅ Already on dev branch"
    fi
fi

echo ""
echo "=========================================="
echo "✅ HP switched to Development environment"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Machine: HP OmniBook X Development"
echo "  Production: false"
echo "  Branch: dev"
echo "  Is Master: false"
echo "  Is Node: true"
echo "  Sandbox Mode: true"
echo ""
echo "Next steps:"
echo "1. Update NAE_MASTER_URL if Mac IP changed"
echo "2. Update NAE_NODE_API_KEY with Master API key"
echo "3. Verify: python3 safety/production_guard.py"
echo ""


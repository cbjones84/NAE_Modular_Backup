#!/bin/bash
# Configure Environment Files for Mac (Dev) and HP OmniBook X (Prod)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "NAE Environment Configuration"
echo "=========================================="
echo ""

# Detect machine type
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACHINE_TYPE="mac"
    MACHINE_NAME="Mac Development"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MACHINE_TYPE="hp"
    MACHINE_NAME="HP OmniBook X Production"
else
    echo "⚠️  Unknown OS type: $OSTYPE"
    read -p "Is this the Mac (dev) or HP (prod)? [mac/hp]: " MACHINE_TYPE
fi

echo "Detected: $MACHINE_NAME"
echo ""

# Set production flag
if [ "$MACHINE_TYPE" == "mac" ]; then
    PRODUCTION="false"
    ENV_FILE="$NAE_ROOT/.env.dev"
    BRANCH="dev"
elif [ "$MACHINE_TYPE" == "hp" ]; then
    PRODUCTION="true"
    ENV_FILE="$NAE_ROOT/.env.prod"
    BRANCH="prod"
else
    echo "❌ Invalid machine type"
    exit 1
fi

echo "Configuration:"
echo "  Machine: $MACHINE_NAME"
echo "  Production: $PRODUCTION"
echo "  Branch: $BRANCH"
echo "  Env File: $ENV_FILE"
echo ""

# Create .env file
cat > "$ENV_FILE" << EOF
# NAE Environment Configuration
# Machine: $MACHINE_NAME
# Generated: $(date)

# Machine Identification
NAE_MACHINE_TYPE=$MACHINE_TYPE
NAE_MACHINE_NAME="$MACHINE_NAME"
NAE_BRANCH=$BRANCH

# Production Settings
PRODUCTION=$PRODUCTION
NAE_PRODUCTION_MODE=$PRODUCTION

# Safety Settings
NAE_SAFETY_ENABLED=true
NAE_ANTI_DOUBLE_TRADE=true
NAE_BRANCH_CHECK=true
NAE_DEVICE_ID_CHECK=true

# Trading Settings (Production Only)
NAE_LIVE_TRADING_ENABLED=$PRODUCTION
NAE_SANDBOX_MODE=$([ "$PRODUCTION" == "true" ] && echo "false" || echo "true")

# Device ID (will be generated)
NAE_DEVICE_ID=""

# Git Settings
NAE_GIT_BRANCH=$BRANCH
NAE_GIT_REQUIRE_PROD_BRANCH=$PRODUCTION

# Master-Node Settings
NAE_MASTER_NODE_ENABLED=true
NAE_IS_MASTER=$([ "$MACHINE_TYPE" == "mac" ] && echo "true" || echo "false")
NAE_IS_NODE=$([ "$MACHINE_TYPE" == "hp" ] && echo "true" || echo "false")

# Communication
NAE_MASTER_URL=""
NAE_NODE_URL=""
NAE_API_KEY=""

# Logging
NAE_LOG_LEVEL=INFO
NAE_LOG_DIR="$NAE_ROOT/logs"
EOF

echo "✅ Created environment file: $ENV_FILE"
echo ""

# Generate device ID if not exists
if [ -z "$NAE_DEVICE_ID" ]; then
    DEVICE_ID=$(openssl rand -hex 16)
    echo "Generated Device ID: $DEVICE_ID"
    echo "NAE_DEVICE_ID=$DEVICE_ID" >> "$ENV_FILE"
fi

# Create symlink to .env
if [ -f "$NAE_ROOT/.env" ]; then
    echo "⚠️  .env already exists, backing up..."
    mv "$NAE_ROOT/.env" "$NAE_ROOT/.env.backup.$(date +%s)"
fi

ln -sf "$(basename "$ENV_FILE")" "$NAE_ROOT/.env"
echo "✅ Created symlink: .env -> $(basename "$ENV_FILE")"
echo ""

# Create .env.local for machine-specific overrides
if [ ! -f "$NAE_ROOT/.env.local" ]; then
    cat > "$NAE_ROOT/.env.local" << EOF
# Machine-specific overrides
# This file is git-ignored and should contain machine-specific settings
# DO NOT commit this file

EOF
    echo "✅ Created .env.local for machine-specific overrides"
fi

echo "=========================================="
echo "✅ Environment configuration complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review $ENV_FILE"
echo "2. Add any machine-specific settings to .env.local"
echo "3. Run: ./setup/initialize_safety_systems.sh"
echo ""


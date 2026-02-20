#!/bin/bash
# Update Production from Development (HP)
# Pulls changes from dev branch and restarts NAE if needed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🔄 Updating Production from Development"
echo "=========================================="
echo ""

# Check if we're on prod branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "prod" ]; then
    echo "⚠️  Warning: Not on prod branch (current: $CURRENT_BRANCH)"
    read -p "Continue anyway? [y/N]: " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

# Check if NAE is running
NAE_RUNNING=false
if [ -f "logs/nae.pid" ]; then
    NAE_PID=$(cat logs/nae.pid)
    if ps -p $NAE_PID > /dev/null 2>&1; then
        NAE_RUNNING=true
    fi
fi

# Fetch latest changes
echo "📥 Fetching latest changes..."
git fetch origin

# Check for updates from dev
echo "🔍 Checking for updates from dev branch..."
DEV_COMMITS=$(git rev-list --count origin/prod..origin/dev 2>/dev/null || echo "0")

if [ "$DEV_COMMITS" = "0" ]; then
    echo "✅ No updates available from dev"
    exit 0
fi

echo "📦 Found $DEV_COMMITS new commit(s) from dev"
echo ""

# Show what will be merged
echo "Changes to be merged:"
git log --oneline origin/prod..origin/dev | head -10
echo ""

read -p "Merge changes from dev to prod? [y/N]: " confirm
if [ "$confirm" != "y" ]; then
    echo "Update cancelled"
    exit 0
fi

# Stop NAE if running
if [ "$NAE_RUNNING" = true ]; then
    echo "🛑 Stopping NAE for update..."
    ./stop_nae_production.sh
    sleep 3
fi

# Merge dev into prod
echo "🔄 Merging dev into prod..."
git merge origin/dev --no-edit

if [ $? -eq 0 ]; then
    echo "✅ Merge successful"
    
    # Restart NAE if it was running
    if [ "$NAE_RUNNING" = true ]; then
        echo "🚀 Restarting NAE..."
        ./start_nae_production.sh
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ Update Complete"
    echo "=========================================="
else
    echo "❌ Merge failed - resolve conflicts manually"
    if [ "$NAE_RUNNING" = true ]; then
        echo "🔄 Restarting NAE..."
        ./start_nae_production.sh
    fi
    exit 1
fi


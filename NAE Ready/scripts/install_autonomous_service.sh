#!/bin/bash
# Install NAE Autonomous Service for macOS (launchd)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_FILE="$NAE_DIR/com.nae.autonomous.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
INSTALLED_PLIST="$LAUNCHD_DIR/com.nae.autonomous.plist"
LAUNCH_USER="gui/$(id -u)"

echo "=========================================="
echo "NAE AUTONOMOUS SERVICE INSTALLER"
echo "=========================================="
echo ""

# Check if plist exists
if [ ! -f "$PLIST_FILE" ]; then
    echo "❌ Error: Plist file not found: $PLIST_FILE"
    exit 1
fi

# Update plist with correct paths
echo "📝 Updating plist with correct paths..."

# Get Python path
PYTHON_PATH=$(which python3)
if [ -z "$PYTHON_PATH" ]; then
    PYTHON_PATH="/usr/local/bin/python3"
fi

# Update plist file
sed -i.bak "s|/usr/local/bin/python3|$PYTHON_PATH|g" "$PLIST_FILE"
sed -i.bak "s|/Users/melissabishop/Downloads/Neural Agency Engine/NAE|$NAE_DIR|g" "$PLIST_FILE"

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCHD_DIR"

# Stop existing service if running (bootstrap/bootout for macOS Ventura+)
if launchctl list | grep -q "com.nae.autonomous"; then
    echo "🛑 Stopping existing service..."
    launchctl bootout "$LAUNCH_USER" "$INSTALLED_PLIST" 2>/dev/null || launchctl unload "$INSTALLED_PLIST" 2>/dev/null || true
fi

# Copy plist to LaunchAgents
echo "📋 Installing plist..."
cp "$PLIST_FILE" "$INSTALLED_PLIST"

# Load service (bootstrap for macOS Ventura+)
echo "🚀 Loading service..."
launchctl bootstrap "$LAUNCH_USER" "$INSTALLED_PLIST" 2>/dev/null || launchctl load "$INSTALLED_PLIST"

# Check status
sleep 2
if launchctl list | grep -q "com.nae.autonomous"; then
    echo ""
    echo "✅ NAE Autonomous Service installed and running!"
    echo ""
    echo "Service Status:"
    launchctl list | grep "com.nae.autonomous"
    echo ""
    echo "To check logs:"
    echo "  tail -f $NAE_DIR/logs/nae_autonomous.out"
    echo "  tail -f $NAE_DIR/logs/nae_autonomous.err"
    echo ""
    echo "To stop service:"
    echo "  launchctl bootout $LAUNCH_USER $INSTALLED_PLIST"
    echo ""
    echo "To start service:"
    echo "  launchctl bootstrap $LAUNCH_USER $INSTALLED_PLIST"
else
    echo "⚠️ Service may not have started. Check logs:"
    echo "  tail -f $NAE_DIR/logs/nae_autonomous.err"
fi


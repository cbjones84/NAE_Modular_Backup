#!/bin/bash
# Install NAE Production Service (launchd)
# This ensures NAE runs continuously, even after reboot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_FILE="$SCRIPT_DIR/com.nae.production.plist"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCH_AGENT_FILE="$LAUNCH_AGENT_DIR/com.nae.production.plist"

echo "=========================================="
echo "🔧 Installing NAE Production Service"
echo "=========================================="
echo ""

# Check if plist exists
if [ ! -f "$PLIST_FILE" ]; then
    echo "❌ Plist file not found: $PLIST_FILE"
    exit 1
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENT_DIR"

# Update paths in plist
echo "📝 Updating paths in plist..."
sed "s|/Users/melissabishop/Downloads/Neural Agency Engine/NAE|$SCRIPT_DIR|g" "$PLIST_FILE" > "$LAUNCH_AGENT_FILE"

# Unload existing service if it exists
if [ -f "$LAUNCH_AGENT_FILE" ]; then
    echo "🛑 Unloading existing service..."
    launchctl unload "$LAUNCH_AGENT_FILE" 2>/dev/null || true
fi

# Load the service
echo "🚀 Loading NAE Production service..."
launchctl load -w "$LAUNCH_AGENT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ NAE Production service installed successfully!"
    echo ""
    echo "The service will:"
    echo "  - Start automatically on login"
    echo "  - Restart if NAE crashes"
    echo "  - Run continuously in background"
    echo ""
    echo "To check status:"
    echo "  launchctl list | grep nae"
    echo ""
    echo "To uninstall:"
    echo "  launchctl unload $LAUNCH_AGENT_FILE"
    echo "  rm $LAUNCH_AGENT_FILE"
else
    echo "❌ Failed to install service"
    exit 1
fi


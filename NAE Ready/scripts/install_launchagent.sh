#!/bin/bash
#
# Install NAE Launch Agent for macOS
# ==================================
# This script installs the NAE continuous operation launch agent
# with proper path configuration for your system.
#
# Usage: ./scripts/install_launchagent.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PATH="$(which python3 2>/dev/null || echo '/usr/local/bin/python3')"
PLIST_TEMPLATE="$NAE_PATH/com.nae.continuous.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.nae.continuous.plist"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NAE Launch Agent Installer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Detected Configuration:"
echo "  NAE Path:    $NAE_PATH"
echo "  Python Path: $PYTHON_PATH"
echo ""

# Verify Python exists
if [ ! -x "$PYTHON_PATH" ]; then
    echo -e "${RED}Error: Python not found at $PYTHON_PATH${NC}"
    echo "Please install Python 3 or update the path."
    exit 1
fi

# Verify template exists
if [ ! -f "$PLIST_TEMPLATE" ]; then
    echo -e "${RED}Error: Template not found at $PLIST_TEMPLATE${NC}"
    exit 1
fi

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Unload existing agent if present (bootstrap/bootout for macOS Ventura+)
LAUNCH_USER="gui/$(id -u)"
if launchctl list | grep -q "com.nae.continuous"; then
    echo -e "${YELLOW}Unloading existing launch agent...${NC}"
    launchctl bootout "$LAUNCH_USER" "$PLIST_DEST" 2>/dev/null || launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

# Create configured plist
echo "Creating configured plist..."
sed -e "s|__NAE_PATH__|$NAE_PATH|g" \
    -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
    "$PLIST_TEMPLATE" > "$PLIST_DEST"

echo -e "${GREEN}✓ Plist installed to: $PLIST_DEST${NC}"

# Create logs directory
mkdir -p "$NAE_PATH/logs"
echo -e "${GREEN}✓ Logs directory created${NC}"

# Load the agent (bootstrap for macOS Ventura+)
echo "Loading launch agent..."
launchctl bootstrap "$LAUNCH_USER" "$PLIST_DEST" 2>/dev/null || launchctl load "$PLIST_DEST"

# Verify it's running
sleep 2
if launchctl list | grep -q "com.nae.continuous"; then
    echo -e "${GREEN}✓ Launch agent loaded and running${NC}"
else
    echo -e "${YELLOW}⚠ Launch agent loaded but may not be running yet${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Commands:"
echo "  Check status:  launchctl list | grep nae"
echo "  View logs:     tail -f $NAE_PATH/logs/continuous_service.log"
echo "  Stop:          launchctl bootout $LAUNCH_USER $PLIST_DEST"
echo "  Start:         launchctl bootstrap $LAUNCH_USER $PLIST_DEST"
echo ""


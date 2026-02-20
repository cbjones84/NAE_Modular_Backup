#!/bin/bash
#
# Install NAE Autonomous Launch Agent for macOS
# ==============================================
# Installs com.nae.autonomous with correct paths for your system.
# Uses launchctl bootstrap (macOS Ventura+) to fix "Load failed: 5" errors.
#
# Usage: ./scripts/install_autonomous_launchagent.sh [--relocate | --no-relocate]
#
# By default, auto-detects if NAE is in Documents/Downloads (macOS blocks
# LaunchAgents from those folders) and relocates to ~/NAE automatically.
#
# --relocate:    Force relocate to ~/NAE
# --no-relocate: Skip relocate even if in Documents/Downloads (will likely fail)
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse flags
RELOCATE=""
[[ "$1" == "--relocate" ]] && RELOCATE=true
[[ "$1" == "--no-relocate" ]] && RELOCATE=false

# Detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_READY_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
NAE_PROJECT_ROOT="$(dirname "$NAE_READY_PATH")"
PLIST_TEMPLATE="$NAE_READY_PATH/com.nae.autonomous.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.nae.autonomous.plist"

# Auto-detect Documents/Downloads (macOS blocks LaunchAgents from these)
REAL_DOCS="$(cd "$HOME/Documents" 2>/dev/null && pwd)" || REAL_DOCS=""
REAL_DL="$(cd "$HOME/Downloads" 2>/dev/null && pwd)" || REAL_DL=""
IN_PROTECTED=""
case "$NAE_PROJECT_ROOT" in
    "$REAL_DOCS"|"$REAL_DOCS"/*) IN_PROTECTED="Documents" ;;
    "$REAL_DL"|"$REAL_DL"/*)     IN_PROTECTED="Downloads" ;;
esac

# Default: relocate when in protected folder (so it works without manual flags)
if [[ -z "$RELOCATE" ]]; then
    if [[ -n "$IN_PROTECTED" ]]; then
        RELOCATE=true
        echo -e "${YELLOW}NAE is in ~/$IN_PROTECTED (LaunchAgents cannot access). Auto-relocating to ~/NAE.${NC}"
    else
        RELOCATE=false
    fi
fi

# Relocate to ~/NAE if requested (avoids Documents/Downloads permission issues)
if $RELOCATE; then
    RELOCATE_DEST="$HOME/NAE"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}NAE Relocate + Launch Agent Install${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Syncing NAE to $RELOCATE_DEST (LaunchAgents can access this location)..."
    if [ -d "$RELOCATE_DEST" ] && command -v rsync &>/dev/null; then
        rsync -a --delete "$NAE_PROJECT_ROOT/" "$RELOCATE_DEST/"
    else
        rm -rf "$RELOCATE_DEST" 2>/dev/null || true
        cp -a "$NAE_PROJECT_ROOT" "$RELOCATE_DEST"
    fi
    NAE_READY_PATH="$RELOCATE_DEST/NAE Ready"
    NAE_PROJECT_ROOT="$RELOCATE_DEST"
    echo -e "${GREEN}✓ Relocated to $RELOCATE_DEST${NC}"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NAE Autonomous Launch Agent Installer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Detected:"
echo "  NAE Ready Path: $NAE_READY_PATH"
echo "  Plist will be:  $PLIST_DEST"
echo ""

# Verify template exists
if [ ! -f "$PLIST_TEMPLATE" ]; then
    echo -e "${RED}Error: Template not found at $PLIST_TEMPLATE${NC}"
    exit 1
fi

# Verify wrapper exists
if [ ! -f "$NAE_READY_PATH/nae_autonomous_wrapper.sh" ]; then
    echo -e "${RED}Error: nae_autonomous_wrapper.sh not found${NC}"
    exit 1
fi

# Logs path (project root / logs)
NAE_PROJECT_ROOT="$(dirname "$NAE_READY_PATH")"
NAE_LOGS_PATH="$NAE_PROJECT_ROOT/logs"
mkdir -p "$NAE_LOGS_PATH"
echo -e "${GREEN}✓ Logs directory: $NAE_LOGS_PATH${NC}"

# Unload existing agent if present (use both legacy and modern commands)
echo -e "${YELLOW}Unloading existing launch agent (if any)...${NC}"
launchctl bootout "gui/$(id -u)/com.nae.autonomous" 2>/dev/null || true
launchctl unload "$PLIST_DEST" 2>/dev/null || true

# Create configured plist
sed -e "s|__NAE_READY_PATH__|$NAE_READY_PATH|g" \
    -e "s|__NAE_LOGS_PATH__|$NAE_LOGS_PATH|g" \
    "$PLIST_TEMPLATE" > "$PLIST_DEST"

# Validate plist
if ! plutil -lint "$PLIST_DEST" >/dev/null 2>&1; then
    echo -e "${RED}Error: Invalid plist generated${NC}"
    plutil -lint "$PLIST_DEST"
    exit 1
fi
echo -e "${GREEN}✓ Plist installed and validated${NC}"

# Load the agent (use bootstrap for macOS Ventura/Sonoma/Sequoia)
echo "Loading launch agent..."
LAUNCH_USER="gui/$(id -u)"
if launchctl bootstrap "$LAUNCH_USER" "$PLIST_DEST" 2>&1; then
    echo -e "${GREEN}✓ Launch agent loaded successfully${NC}"
elif launchctl load "$PLIST_DEST" 2>&1; then
    echo -e "${GREEN}✓ Launch agent loaded (legacy)${NC}"
else
    echo -e "${RED}Load failed. Trying bootout first then bootstrap...${NC}"
    launchctl bootout "$LAUNCH_USER" "$PLIST_DEST" 2>/dev/null || true
    sleep 1
    if launchctl bootstrap "$LAUNCH_USER" "$PLIST_DEST" 2>&1; then
        echo -e "${GREEN}✓ Launch agent loaded on retry${NC}"
    else
        echo -e "${RED}Load failed. If project is in ~/Documents, macOS may block access.${NC}"
        echo "  Option: Run with --relocate to copy NAE to ~/NAE for LaunchAgent compatibility"
        echo "  Or grant Full Disk Access: System Preferences → Privacy → Full Disk Access → Add Terminal"
        exit 1
    fi
fi

# Verify
sleep 2
if launchctl list 2>/dev/null | grep -q "com.nae.autonomous"; then
    STATUS=$(launchctl list | grep com.nae.autonomous)
    echo ""
    echo -e "${GREEN}Status: $STATUS${NC}"
    if echo "$STATUS" | grep -qE "32256|126|Exit"; then
        echo -e "${YELLOW}⚠ Agent loaded but may have exited. Check:${NC}"
        echo "   tail -f $NAE_LOGS_PATH/nae_autonomous.err"
        echo ""
        echo "   If you see 'Operation not permitted', the project may be in a protected folder (Documents/Downloads)."
        echo "   Run with --relocate to copy to ~/NAE: ./scripts/install_autonomous_launchagent.sh --relocate"
    fi
else
    echo -e "${YELLOW}⚠ Agent not in list - may have exited immediately. Check logs.${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Commands:"
echo "  Check status:  launchctl list | grep nae"
echo "  View logs:     tail -f $NAE_LOGS_PATH/nae_autonomous.out"
echo "  Stop:          launchctl bootout $LAUNCH_USER $PLIST_DEST"
echo "  Start:         launchctl bootstrap $LAUNCH_USER $PLIST_DEST"
echo ""

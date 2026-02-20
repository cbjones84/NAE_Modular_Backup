#!/bin/bash
# Start Mac Auto-Updater Service
# Run this on Mac to enable remote updates from HP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Starting Mac Auto-Updater Service                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Export environment
export NAE_ROOT="$NAE_ROOT"
export NAE_UPDATE_PORT=8081
export NAE_UPDATE_API_KEY="a07e9de261c6eb815fbcd9cb6263f0862534af1cd3cc3540c87ed70ce0e4438d"

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip3 install flask
fi

# Start the updater service
cd "$NAE_ROOT"
python3 scripts/mac_auto_updater.py



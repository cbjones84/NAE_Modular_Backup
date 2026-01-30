#!/bin/bash
# Update HP OmniBook X from Cursor changes
# This script can be run manually or integrated into Cursor workflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔄 Syncing Cursor changes to HP OmniBook X..."

cd "$NAE_ROOT"

# Stage all changes
git add -A

# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ No changes to sync"
    exit 0
fi

# Commit changes
git commit -m "Auto-sync from Cursor: $(date '+%Y-%m-%d %H:%M:%S')"

# Push to prod branch
git push origin prod

# Pull on HP OmniBook X (if remote execution is configured)
if [ -f "setup/cursor_remote_integration.py" ]; then
    python setup/cursor_remote_integration.py --sync --branch prod
fi

echo "✅ Changes synced to HP OmniBook X"


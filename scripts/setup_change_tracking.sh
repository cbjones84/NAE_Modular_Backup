#!/bin/bash
#
# Setup Automated Change Tracking for NAE
#
# Installs git hooks and configures automatic changelog updates
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"
SCRIPTS_DIR="$REPO_ROOT/NAE Ready/scripts"

echo "Setting up automated change tracking for NAE..."
echo ""

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "Error: Not a git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Install pre-commit hook
echo "Installing pre-commit hook..."
cat > "$GIT_HOOKS_DIR/pre-commit" <<'HOOK_EOF'
#!/bin/bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CHANGE_TRACKER="$REPO_ROOT/NAE Ready/scripts/track_changes.sh"

CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -n "$CHANGED_FILES" ] && [ -f "$CHANGE_TRACKER" ] && [ -x "$CHANGE_TRACKER" ]; then
    "$CHANGE_TRACKER" --pre-commit $CHANGED_FILES
fi

exit 0
HOOK_EOF

chmod +x "$GIT_HOOKS_DIR/pre-commit"
echo "✓ Pre-commit hook installed"

# Install post-commit hook
echo "Installing post-commit hook..."
cat > "$GIT_HOOKS_DIR/post-commit" <<'HOOK_EOF'
#!/bin/bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CHANGE_TRACKER="$REPO_ROOT/NAE Ready/scripts/track_changes.sh"

COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
COMMIT_DATE=$(git log -1 --pretty=%ci)
AUTHOR=$(git log -1 --pretty=%an)

if [ -f "$CHANGE_TRACKER" ] && [ -x "$CHANGE_TRACKER" ]; then
    "$CHANGE_TRACKER" --post-commit "$COMMIT_HASH" "$COMMIT_MSG" "$COMMIT_DATE" "$AUTHOR"
fi

exit 0
HOOK_EOF

chmod +x "$GIT_HOOKS_DIR/post-commit"
echo "✓ Post-commit hook installed"

# Create logs directory
mkdir -p "$REPO_ROOT/NAE Ready/logs"
echo "✓ Logs directory created"

# Make scripts executable
chmod +x "$SCRIPTS_DIR/track_changes.sh"
chmod +x "$SCRIPTS_DIR/auto_changelog.py"
echo "✓ Scripts made executable"

echo ""
echo "✅ Automated change tracking setup complete!"
echo ""
echo "Features enabled:"
echo "  - Pre-commit: Auto-updates CHANGELOG.md before commits"
echo "  - Post-commit: Records commits in change_log.json"
echo "  - Change tracking: Monitors all file changes"
echo ""
echo "To manually sync changelog from git:"
echo "  python3 $SCRIPTS_DIR/auto_changelog.py --sync --days 7"


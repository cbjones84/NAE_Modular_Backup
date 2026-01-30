#!/bin/bash
#
# Automated Change Tracker for NAE
#
# Tracks all changes and automatically updates CHANGELOG.md
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHANGELOG="$REPO_ROOT/NAE Ready/CHANGELOG.md"
CHANGE_LOG="$REPO_ROOT/NAE Ready/logs/change_log.json"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Ensure logs directory exists
mkdir -p "$(dirname "$CHANGE_LOG")"

# Function to categorize changes
categorize_change() {
    local file="$1"
    local ext="${file##*.}"
    
    case "$ext" in
        py)
            echo "code"
            ;;
        md|txt|rst)
            echo "documentation"
            ;;
        sh|bash)
            echo "script"
            ;;
        json|yaml|yml|toml)
            echo "config"
            ;;
        *)
            echo "other"
            ;;
    esac
}

# Function to detect change type
detect_change_type() {
    local file="$1"
    local status="$2"
    
    case "$status" in
        A|AM|AD)
            echo "Added"
            ;;
        M|MM|MD)
            echo "Changed"
            ;;
        D|DM|DD)
            echo "Removed"
            ;;
        R|RM|RD)
            echo "Renamed"
            ;;
        *)
            echo "Modified"
            ;;
    esac
}

# Function to get file description
get_file_description() {
    local file="$1"
    local basename=$(basename "$file")
    
    # Extract meaningful description from path
    if [[ "$file" == *"agent"* ]]; then
        echo "Agent: $(basename "$file" .py)"
    elif [[ "$file" == *"tool"* ]]; then
        echo "Tool: $(basename "$file" .py)"
    elif [[ "$file" == *"adapter"* ]]; then
        echo "Adapter: $(basename "$file" .py)"
    elif [[ "$file" == *"script"* ]]; then
        echo "Script: $(basename "$file")"
    elif [[ "$file" == *"doc"* ]] || [[ "$file" == *.md ]]; then
        echo "Documentation: $(basename "$file" .md)"
    else
        echo "$basename"
    fi
}

# Pre-commit processing
if [ "$1" == "--pre-commit" ]; then
    shift
    CHANGED_FILES="$@"
    
    # Analyze changes
    ADDED_FILES=()
    CHANGED_FILES_LIST=()
    REMOVED_FILES=()
    
    for file in $CHANGED_FILES; do
        status=$(git diff --cached --name-status -- "$file" | cut -c1)
        change_type=$(detect_change_type "$file" "$status")
        
        case "$change_type" in
            Added)
                ADDED_FILES+=("$file")
                ;;
            Removed)
                REMOVED_FILES+=("$file")
                ;;
            *)
                CHANGED_FILES_LIST+=("$file")
                ;;
        esac
    done
    
    # Update changelog if there are significant changes
    if [ ${#ADDED_FILES[@]} -gt 0 ] || [ ${#CHANGED_FILES_LIST[@]} -gt 0 ] || [ ${#REMOVED_FILES[@]} -gt 0 ]; then
        # Auto-update changelog using Python script
        if [ -f "$CHANGELOG" ]; then
            # Write Python script to temp file to avoid escaping issues
            TEMP_PY=$(mktemp)
            cat > "$TEMP_PY" <<PYEOF
import re
import sys
from datetime import datetime

changelog_path = "$CHANGELOG"
added_files = $(python3 -c "import json; print(json.dumps(list('${ADDED_FILES[@]}'.split())))" 2>/dev/null || echo "[]")
changed_files = $(python3 -c "import json; print(json.dumps(list('${CHANGED_FILES_LIST[@]}'.split())))" 2>/dev/null || echo "[]")
removed_files = $(python3 -c "import json; print(json.dumps(list('${REMOVED_FILES[@]}'.split())))" 2>/dev/null || echo "[]")

def get_file_category(filepath):
    if 'agent' in filepath.lower():
        return 'Agent'
    elif 'tool' in filepath.lower():
        return 'Tool'
    elif 'adapter' in filepath.lower():
        return 'Adapter'
    elif 'script' in filepath.lower():
        return 'Script'
    elif 'doc' in filepath.lower() or filepath.endswith('.md'):
        return 'Documentation'
    return 'Other'

def get_short_path(filepath):
    # Extract meaningful part of path
    parts = filepath.split('/')
    if 'NAE Ready' in parts:
        idx = parts.index('NAE Ready')
        return '/'.join(parts[idx+1:])
    return filepath

try:
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    # Find [Unreleased] section
    unreleased_pattern = r'(## \[Unreleased\]\n)'
    
    if re.search(unreleased_pattern, content):
        # Add entries to Unreleased section
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        additions = []
        changes = []
        removals = []
        
        for f in added_files:
            category = get_file_category(f)
            short_path = get_short_path(f)
            additions.append(f"- {category}: `{short_path}`")
        
        for f in changed_files:
            category = get_file_category(f)
            short_path = get_short_path(f)
            changes.append(f"- {category}: `{short_path}`")
        
        for f in removed_files:
            category = get_file_category(f)
            short_path = get_short_path(f)
            removals.append(f"- {category}: `{short_path}`")
        
        # Build entry
        entry = f"\n### [{timestamp}] - Automated Update\n\n"
        
        if additions:
            entry += "#### Added\n"
            entry += "\n".join(additions) + "\n\n"
        
        if changes:
            entry += "#### Changed\n"
            entry += "\n".join(changes) + "\n\n"
        
        if removals:
            entry += "#### Removed\n"
            entry += "\n".join(removals) + "\n\n"
        
        # Insert after [Unreleased] header
        content = re.sub(
            unreleased_pattern,
            r'\1' + entry,
            content,
            count=1
        )
        
        with open(changelog_path, 'w') as f:
            f.write(content)
        
        print("CHANGELOG.md updated automatically")
    else:
        print("No [Unreleased] section found in CHANGELOG.md")
        
except Exception as e:
    print(f"Error updating changelog: {e}")
    # Don't fail the commit if changelog update fails
    pass
PYEOF
            
            # Run Python script
            python3 "$TEMP_PY" 2>/dev/null || true
            rm -f "$TEMP_PY"
        fi
    fi
    
    exit 0
fi

# Post-commit processing
if [ "$1" == "--post-commit" ]; then
    COMMIT_HASH="$2"
    COMMIT_MSG="$3"
    COMMIT_DATE="$4"
    AUTHOR="$5"
    
    # Record in change log JSON
    python3 <<EOF
import json
import os
from datetime import datetime

change_log_path = "$CHANGE_LOG"

# Load existing log or create new
if os.path.exists(change_log_path):
    with open(change_log_path, 'r') as f:
        try:
            change_log = json.load(f)
        except:
            change_log = {"commits": []}
else:
    change_log = {"commits": []}

# Add new entry
entry = {
    "hash": "$COMMIT_HASH",
    "message": """$COMMIT_MSG""",
    "date": "$COMMIT_DATE",
    "author": "$AUTHOR",
    "timestamp": datetime.now().isoformat()
}

change_log["commits"].append(entry)

# Keep only last 1000 entries
if len(change_log["commits"]) > 1000:
    change_log["commits"] = change_log["commits"][-1000:]

# Save
with open(change_log_path, 'w') as f:
    json.dump(change_log, f, indent=2)

print(f"Change logged: $COMMIT_HASH")
EOF
    
    exit 0
fi

# Default: track current changes
echo "Change tracker for NAE"
echo "Usage:"
echo "  --pre-commit <files>  : Process before commit"
echo "  --post-commit <hash> <msg> <date> <author> : Record after commit"


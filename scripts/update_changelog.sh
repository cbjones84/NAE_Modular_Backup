#!/bin/bash
#
# Update CHANGELOG.md with new changes
#
# Usage: ./scripts/update_changelog.sh [version] [description]
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CHANGELOG_FILE="NAE Ready/CHANGELOG.md"
TIMESTAMP=$(date +"%Y-%m-%d")

if [ ! -f "$CHANGELOG_FILE" ]; then
    echo "Error: CHANGELOG.md not found"
    exit 1
fi

# Get git changes since last commit
GIT_CHANGES=$(git diff --name-only HEAD)
GIT_ADDED=$(git diff --cached --name-only)

# Create temporary file
TEMP_FILE=$(mktemp)

# Process changelog
awk -v timestamp="$TIMESTAMP" -v version="$1" -v description="$2" '
BEGIN {
    in_unreleased = 0
    added_section = 0
}

/^## \[Unreleased\]/ {
    in_unreleased = 1
    print
    print ""
    if (version) {
        print "### [" timestamp "] - Version " version
    } else {
        print "### [" timestamp "] - " (description ? description : "Update")
    }
    print ""
    added_section = 1
    next
}

/^### (Added|Changed|Fixed|Removed|Deprecated|Security)/ {
    if (in_unreleased && !added_section) {
        print ""
        if (version) {
            print "### [" timestamp "] - Version " version
        } else {
            print "### [" timestamp "] - " (description ? description : "Update")
        }
        print ""
        added_section = 1
    }
    print
    next
}

/^## \[/ && !/^## \[Unreleased\]/ {
    if (in_unreleased && !added_section) {
        print ""
        if (version) {
            print "### [" timestamp "] - Version " version
        } else {
            print "### [" timestamp "] - " (description ? description : "Update")
        }
        print ""
        added_section = 1
    }
    in_unreleased = 0
    print
    next
}

{ print }
' "$CHANGELOG_FILE" > "$TEMP_FILE"

mv "$TEMP_FILE" "$CHANGELOG_FILE"

echo -e "${GREEN}✓ CHANGELOG.md updated${NC}"
echo -e "${YELLOW}Review and add specific changes under appropriate sections${NC}"


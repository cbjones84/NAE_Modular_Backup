#!/bin/bash
#
# Deploy Accelerator Strategy - GitHub Push + Sandbox Testing + Live Production
#
# This script:
# 1. Commits all changes to GitHub
# 2. Starts sandbox testing
# 3. Starts live production trading
# 4. Monitors both environments
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Find git repo root (go up from NAE Ready to parent)
if [ -d "$SCRIPT_DIR/../../.git" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -d "$SCRIPT_DIR/../.git" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
LOG_DIR="$REPO_DIR/NAE Ready/logs"
GIT_BRANCH="${GIT_BRANCH:-prod}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ACCELERATOR STRATEGY DEPLOYMENT${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Verify we're in the right directory
echo -e "${YELLOW}Step 1: Verifying repository...${NC}"
cd "$REPO_DIR"
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not a git repository in $REPO_DIR${NC}"
    echo -e "${YELLOW}Trying parent directory...${NC}"
    REPO_DIR="$(cd "$REPO_DIR/.." && pwd)"
    cd "$REPO_DIR"
    if [ ! -d ".git" ]; then
        echo -e "${RED}Error: Still not a git repository${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Repository verified at: $REPO_DIR${NC}"
echo ""

# Step 2: Check for uncommitted changes
echo -e "${YELLOW}Step 2: Checking for changes...${NC}"
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${GREEN}Changes detected, preparing commit...${NC}"
    
    # Add all changes
    git add -A
    
    # Update CHANGELOG.md
    echo -e "${YELLOW}Updating CHANGELOG.md...${NC}"
    CHANGELOG_FILE="$REPO_DIR/NAE Ready/CHANGELOG.md"
    if [ -f "$CHANGELOG_FILE" ]; then
        # Add entry to Unreleased section
        TIMESTAMP=$(date +"%Y-%m-%d")
        TEMP_FILE=$(mktemp)
        
        # Insert new entry after [Unreleased] header
        awk -v timestamp="$TIMESTAMP" '
        /^## \[Unreleased\]/ {
            print
            print ""
            print "### [" timestamp "] - Accelerator Strategy Update"
            print ""
            print "#### Added"
            print "- Deployment automation improvements"
            print "- CHANGELOG.md maintenance"
            print ""
            print "#### Changed"
            print "- Updated deployment script to maintain changelog"
            print ""
            next
        }
        { print }
        ' "$CHANGELOG_FILE" > "$TEMP_FILE"
        mv "$TEMP_FILE" "$CHANGELOG_FILE"
        echo -e "${GREEN}✓ CHANGELOG.md updated${NC}"
    fi
    
    # Commit with descriptive message
    COMMIT_MSG="Deploy Accelerator Strategy: Dual-mode operation (sandbox + live)
    
- Updated target account size to \$8000-\$10000
- Integrated accelerator into NAE master controller
- Added dual-mode operation (sandbox testing + live production)
- Added settlement cash tracking
- Added Ralph signal integration
- Added comprehensive risk management features
- Updated CHANGELOG.md"
    
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ Changes committed${NC}"
fi
echo ""

# Step 3: Push to GitHub
echo -e "${YELLOW}Step 3: Pushing to GitHub...${NC}"
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "$GIT_BRANCH" ]; then
    echo -e "${YELLOW}Switching to branch: $GIT_BRANCH${NC}"
    git checkout "$GIT_BRANCH" || git checkout -b "$GIT_BRANCH"
fi

git push origin "$GIT_BRANCH"
echo -e "${GREEN}✓ Pushed to GitHub (branch: $GIT_BRANCH)${NC}"
echo ""

# Step 4: Create logs directory
echo -e "${YELLOW}Step 4: Setting up logs directory...${NC}"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓ Logs directory ready${NC}"
echo ""

# Step 5: Start sandbox testing
echo -e "${YELLOW}Step 5: Starting sandbox testing...${NC}"
cd "$REPO_DIR/NAE Ready" 2>/dev/null || cd "$REPO_DIR"
NAE_READY_DIR="$REPO_DIR/NAE Ready"
if [ ! -d "$NAE_READY_DIR" ]; then
    NAE_READY_DIR="$REPO_DIR"
fi
cd "$NAE_READY_DIR"
python3 execution/integration/accelerator_controller.py \
    --sandbox \
    --no-live \
    --interval 60 \
    > "$LOG_DIR/accelerator_sandbox.log" 2>&1 &
SANDBOX_PID=$!
echo "Sandbox PID: $SANDBOX_PID"
echo $SANDBOX_PID > "$LOG_DIR/accelerator_sandbox.pid"
echo -e "${GREEN}✓ Sandbox testing started (PID: $SANDBOX_PID)${NC}"
echo ""

# Step 6: Start live production
echo -e "${YELLOW}Step 6: Starting live production...${NC}"
cd "$NAE_READY_DIR"
python3 execution/integration/accelerator_controller.py \
    --live \
    --no-sandbox \
    --interval 60 \
    > "$LOG_DIR/accelerator_live.log" 2>&1 &
LIVE_PID=$!
echo "Live PID: $LIVE_PID"
echo $LIVE_PID > "$LOG_DIR/accelerator_live.pid"
echo -e "${GREEN}✓ Live production started (PID: $LIVE_PID)${NC}"
echo ""

# Step 7: Start NAE master controller (includes accelerator)
echo -e "${YELLOW}Step 7: Starting NAE master controller...${NC}"
cd "$NAE_READY_DIR"
python3 nae_autonomous_master.py > "$LOG_DIR/nae_master.log" 2>&1 &
MASTER_PID=$!
echo "Master PID: $MASTER_PID"
echo $MASTER_PID > "$LOG_DIR/nae_master.pid"
echo -e "${GREEN}✓ NAE master controller started (PID: $MASTER_PID)${NC}"
echo ""

# Step 8: Wait a moment and verify processes
echo -e "${YELLOW}Step 8: Verifying processes...${NC}"
sleep 3

if ps -p $SANDBOX_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Sandbox process running${NC}"
else
    echo -e "${RED}✗ Sandbox process not running - check logs${NC}"
fi

if ps -p $LIVE_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Live process running${NC}"
else
    echo -e "${RED}✗ Live process not running - check logs${NC}"
fi

if ps -p $MASTER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Master controller running${NC}"
else
    echo -e "${RED}✗ Master controller not running - check logs${NC}"
fi
echo ""

# Step 9: Display status
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Process Status:"
echo "  Sandbox Testing: PID $SANDBOX_PID"
echo "  Live Production: PID $LIVE_PID"
echo "  Master Controller: PID $MASTER_PID"
echo ""
echo "Log Files:"
echo "  Sandbox: $LOG_DIR/accelerator_sandbox.log"
echo "  Live: $LOG_DIR/accelerator_live.log"
echo "  Master: $LOG_DIR/nae_master.log"
echo ""
echo "Monitor logs with:"
echo "  tail -f $LOG_DIR/accelerator_sandbox.log"
echo "  tail -f $LOG_DIR/accelerator_live.log"
echo "  tail -f $LOG_DIR/nae_master.log"
echo ""
echo "Stop processes with:"
echo "  kill $SANDBOX_PID $LIVE_PID $MASTER_PID"
echo ""
echo -e "${YELLOW}Accelerator Strategy is now running in dual-mode!${NC}"
echo -e "${YELLOW}Target: Grow account to \$8000-\$10000${NC}"
echo ""


#!/bin/bash
#
# Initialize Continuous Autonomous Operation for NAE
#
# Sets up and starts NAE in continuous autonomous mode
# Ensures all systems are running, learning, and improving
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-dual}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NAE CONTINUOUS AUTONOMOUS OPERATION${NC}"
echo -e "${BLUE}Initialization & Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Verify environment
echo -e "${YELLOW}Step 1: Verifying environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python: $(python3 --version)${NC}"

# Check required packages
REQUIRED_PACKAGES=("psutil" "numpy" "pandas")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓ Package: $pkg${NC}"
    else
        echo -e "${YELLOW}⚠ Package: $pkg (not installed, may cause issues)${NC}"
    fi
done
echo ""

# Step 2: Create necessary directories
echo -e "${YELLOW}Step 2: Creating directories...${NC}"
mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/execution/autonomous"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 3: Verify continuous operation script exists
echo -e "${YELLOW}Step 3: Verifying scripts...${NC}"
if [ ! -f "$REPO_DIR/execution/autonomous/continuous_operation.py" ]; then
    echo -e "${RED}❌ continuous_operation.py not found${NC}"
    exit 1
fi
chmod +x "$REPO_DIR/execution/autonomous/continuous_operation.py"
echo -e "${GREEN}✓ Scripts verified${NC}"
echo ""

# Step 4: Check for existing processes
echo -e "${YELLOW}Step 4: Checking for existing processes...${NC}"
EXISTING=$(ps aux | grep -i "continuous_operation" | grep -v grep | wc -l)
if [ "$EXISTING" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found existing continuous operation processes${NC}"
    echo -e "${YELLOW}  Stopping existing processes...${NC}"
    pkill -f "continuous_operation.py" || true
    sleep 2
fi
echo -e "${GREEN}✓ Process check complete${NC}"
echo ""

# Step 5: Start continuous operation
echo -e "${YELLOW}Step 5: Starting continuous operation...${NC}"
cd "$REPO_DIR"

python3 execution/autonomous/continuous_operation.py \
    --mode "$MODE" \
    --health-interval 30 \
    --learning-interval 300 \
    > logs/continuous_operation.log 2>&1 &

CONTINUOUS_PID=$!
echo $CONTINUOUS_PID > logs/continuous_operation.pid

# Wait a moment to verify it started
sleep 3

if ps -p $CONTINUOUS_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Continuous operation started (PID: $CONTINUOUS_PID)${NC}"
else
    echo -e "${RED}❌ Failed to start continuous operation${NC}"
    echo -e "${YELLOW}Check logs: tail -f logs/continuous_operation.log${NC}"
    exit 1
fi
echo ""

# Step 6: Display status
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CONTINUOUS OPERATION ACTIVE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Mode: $MODE"
echo "PID: $CONTINUOUS_PID"
echo ""
echo "Features Active:"
echo "  ✓ Continuous health monitoring (every 30s)"
echo "  ✓ Automatic error detection and recovery"
echo "  ✓ Self-healing (auto-fix common issues)"
echo "  ✓ Continuous learning (every 5min)"
echo "  ✓ Holistic enhancement"
echo "  ✓ Auto-restart on failures"
echo "  ✓ Resource optimization"
echo "  ✓ Performance tuning"
echo ""
echo "Monitor:"
echo "  tail -f logs/continuous_operation.log"
echo ""
echo "Status:"
echo "  python3 -c \"from execution.autonomous.continuous_operation import ContinuousOperationManager; m = ContinuousOperationManager(); print(m.get_status())\""
echo ""
echo "Stop:"
echo "  kill $CONTINUOUS_PID"
echo ""


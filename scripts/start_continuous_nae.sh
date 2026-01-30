#!/bin/bash
#
# Start NAE in Continuous Autonomous Operation Mode
#
# Ensures NAE runs continuously, learns from errors, and improves holistically
# in both dev/sandbox and prod/live modes.
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
MODE="${1:-dual}"  # sandbox, live, or dual

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NAE CONTINUOUS AUTONOMOUS OPERATION${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Mode: ${MODE}${NC}"
echo ""

# Create logs directory
mkdir -p "$LOG_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Start continuous operation manager
echo -e "${YELLOW}Starting continuous operation manager...${NC}"
cd "$REPO_DIR"

python3 execution/autonomous/continuous_operation.py \
    --mode "$MODE" \
    --health-interval 30 \
    --learning-interval 300 \
    > "$LOG_DIR/continuous_operation.log" 2>&1 &

CONTINUOUS_PID=$!
echo $CONTINUOUS_PID > "$LOG_DIR/continuous_operation.pid"

echo -e "${GREEN}✅ Continuous operation started (PID: $CONTINUOUS_PID)${NC}"
echo ""
echo "Features enabled:"
echo "  ✓ Continuous health monitoring"
echo "  ✓ Automatic error detection and recovery"
echo "  ✓ Self-healing (auto-restart failed processes)"
echo "  ✓ Continuous learning from errors"
echo "  ✓ Performance optimization"
echo "  ✓ Holistic enhancement"
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/continuous_operation.log"
echo ""
echo "Stop:"
echo "  kill $CONTINUOUS_PID"
echo ""


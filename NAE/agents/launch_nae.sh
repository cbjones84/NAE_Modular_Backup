#!/bin/bash
# Launch NAE Trading System and All Agents

cd "$(dirname "$0")/../.." || exit 1

# Set environment variables
export TRADIER_API_KEY=27Ymk28vtbgqY1LFYxhzaEmIuwJb
export TRADIER_ACCOUNT_ID=6YB66744
export TRADIER_SANDBOX=false

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "NAE Launch Script"
echo "=========================================="
echo ""

# Function to check if process is running
is_running() {
    local process_name=$1
    pgrep -f "$process_name" > /dev/null 2>&1
}

# Function to launch process in background
launch_agent() {
    local agent_name=$1
    local script_path=$2
    local log_file="logs/${agent_name}.log"
    
    if is_running "$agent_name"; then
        echo -e "${YELLOW}⚠️  $agent_name is already running${NC}"
        return 1
    else
        echo -e "${GREEN}🚀 Launching $agent_name...${NC}"
        mkdir -p logs
        nohup python3 "$script_path" >> "$log_file" 2>&1 &
        echo "   PID: $!"
        echo "   Log: $log_file"
        sleep 2
        return 0
    fi
}

# Create logs directory
mkdir -p logs

# Launch NAE (Ralph Continuous Trading)
echo "Checking NAE Trading System..."
if is_running "ralph_github_continuous"; then
    echo -e "${YELLOW}⚠️  NAE (ralph_github_continuous) is already running${NC}"
else
    echo -e "${GREEN}🚀 Launching NAE Trading System...${NC}"
    launch_agent "ralph_github_continuous" "NAE/agents/ralph_github_continuous.py"
fi

echo ""
echo "Checking other agents..."

# Launch Optimus
launch_agent "optimus.py" "NAE Ready/agents/optimus.py"

# Launch Donnie
launch_agent "donnie.py" "NAE Ready/agents/donnie.py"

# Launch Splinter
launch_agent "splinter.py" "NAE Ready/agents/splinter.py"

# Launch Genny
launch_agent "genny.py" "NAE Ready/agents/genny.py"

# Launch Casey
launch_agent "casey.py" "NAE Ready/agents/casey.py"

# Launch Ralph (research)
launch_agent "ralph.py" "NAE Ready/agents/ralph.py"

echo ""
echo "=========================================="
echo "Launch Complete"
echo "=========================================="
echo ""
echo "To view logs:"
echo "  tail -f logs/ralph_github_continuous.log"
echo "  tail -f logs/optimus.log"
echo ""
echo "To stop all agents:"
echo "  pkill -f 'python.*agents'"
echo ""


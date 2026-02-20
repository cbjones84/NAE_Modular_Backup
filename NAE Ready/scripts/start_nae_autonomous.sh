#!/bin/bash
# Start NAE Autonomous System

cd "$(dirname "$0")/.."

echo "=========================================="
echo "STARTING NAE AUTONOMOUS SYSTEM"
echo "=========================================="
echo ""

# Activate Python virtual environment if it exists
if [ -d "venv_python311" ]; then
    source venv_python311/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the autonomous master controller
echo "🚀 Starting NAE Autonomous Master Controller..."
python3 nae_autonomous_master.py


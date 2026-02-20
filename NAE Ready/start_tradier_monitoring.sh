#!/bin/bash
# Start Tradier Balance Monitoring and Funds Activation System

cd "$(dirname "$0")"

echo "=========================================="
echo "TRADIER FUNDS ACTIVATION SYSTEM"
echo "=========================================="
echo ""
echo "Starting Tradier balance monitoring..."
echo "System will automatically activate trading when funds are detected"
echo ""

# Activate Python virtual environment if it exists
if [ -d "venv_python311" ]; then
    source venv_python311/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the funds activation system
python3 execution/integration/tradier_funds_activation.py


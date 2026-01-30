#!/bin/bash
# NAE Live Trading Startup Script
# Verifies connection and starts live trading

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "NAE Live Trading Startup"
echo "=========================================="

# Step 1: Verify Alpaca connection
echo ""
echo "Step 1: Verifying Alpaca connection..."
python3 verify_alpaca_keys.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Alpaca connection failed!"
    echo ""
    echo "Please verify:"
    echo "  1. API Access is enabled in Alpaca dashboard"
    echo "  2. Trading permissions are enabled"
    echo "  3. Account is fully activated"
    echo ""
    echo "Once verified, run this script again."
    exit 1
fi

echo ""
echo "✅ Alpaca connection verified!"
echo ""

# Step 2: Confirm live trading
read -p "⚠️  Ready to start LIVE trading? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Startup cancelled."
    exit 0
fi

# Step 3: Start NAE automation
echo ""
echo "Starting NAE automation system..."
echo ""

python3 nae_automation.py


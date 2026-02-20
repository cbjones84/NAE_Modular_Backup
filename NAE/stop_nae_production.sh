#!/bin/bash
# Stop NAE Production Mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🛑 Stopping NAE Production Mode"
echo "=========================================="
echo ""

# Stop NAE Autonomous Master
if [ -f "logs/nae.pid" ]; then
    NAE_PID=$(cat logs/nae.pid)
    if ps -p $NAE_PID > /dev/null 2>&1; then
        echo "Stopping NAE (PID: $NAE_PID)..."
        kill $NAE_PID 2>/dev/null || true
        sleep 2
        if ps -p $NAE_PID > /dev/null 2>&1; then
            echo "Force killing NAE..."
            kill -9 $NAE_PID 2>/dev/null || true
        fi
        echo "✅ NAE stopped"
    else
        echo "NAE process not found"
    fi
    rm -f logs/nae.pid
else
    echo "Stopping all NAE processes..."
    pkill -f "nae_autonomous_master.py" 2>/dev/null || true
    sleep 2
    pkill -9 -f "nae_autonomous_master.py" 2>/dev/null || true
fi

# Stop caffeinate
if [ -f "logs/caffeinate.pid" ]; then
    CAFFEINATE_PID=$(cat logs/caffeinate.pid)
    if ps -p $CAFFEINATE_PID > /dev/null 2>&1; then
        echo "Stopping caffeinate (PID: $CAFFEINATE_PID)..."
        kill $CAFFEINATE_PID 2>/dev/null || true
        echo "✅ Caffeinate stopped"
    fi
    rm -f logs/caffeinate.pid
fi

# Stop any remaining agent processes
echo "Stopping agent processes..."
pkill -f "master_api/server.py" 2>/dev/null || true
pkill -f "agents/.*\.py" 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ NAE Production Mode Stopped"
echo "=========================================="
echo ""


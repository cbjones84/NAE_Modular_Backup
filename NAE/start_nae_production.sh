#!/bin/bash
# Start NAE in Production Mode - Continuous Autonomous Operation
# Prevents Mac sleep, runs in background, auto-restarts on failure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load Tradier credentials and other env vars for production
if [ -f ".env.prod" ]; then
    set -a
    source .env.prod
    set +a
    echo "✅ Loaded .env.prod (TRADIER_API_KEY, TRADIER_ACCOUNT_ID, etc.)"
elif [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "✅ Loaded .env"
else
    echo "⚠️  No .env.prod or .env found - Tradier-dependent agents may fail"
fi

echo "=========================================="
echo "🚀 Starting NAE Production Mode"
echo "Continuous Autonomous Operation"
echo "=========================================="
echo ""

# Check if already running
if pgrep -f "nae_autonomous_master.py" > /dev/null; then
    echo "✅ NAE is already running!"
    ps aux | grep "nae_autonomous_master.py" | grep -v grep
    echo ""
    echo "To restart, first stop: ./stop_nae_production.sh"
    exit 0
fi

# Create logs directory
mkdir -p logs

# Prevent Mac from sleeping (caffeinate)
echo "🔋 Preventing Mac sleep during NAE operation..."
CAFFEINATE_PID=$(pgrep -f "caffeinate.*nae" || echo "")
if [ -z "$CAFFEINATE_PID" ]; then
    # Start caffeinate to prevent sleep
    nohup caffeinate -d -i -m -s -u -w $$ > logs/caffeinate.log 2>&1 &
    CAFFEINATE_PID=$!
    echo "✅ Caffeinate started (PID: $CAFFEINATE_PID)"
    echo "$CAFFEINATE_PID" > logs/caffeinate.pid
else
    echo "✅ Caffeinate already running (PID: $CAFFEINATE_PID)"
fi

# Start NAE Autonomous Master
echo "🚀 Starting NAE Autonomous Master..."
nohup python3 nae_autonomous_master.py > logs/nae_startup.log 2>&1 &
NAE_PID=$!

sleep 3

# Check if started successfully
if ps -p $NAE_PID > /dev/null; then
    echo "✅ NAE started successfully!"
    echo "🆔 Process ID: $NAE_PID"
    echo "$NAE_PID" > logs/nae.pid
    echo ""
    echo "📝 Logs:"
    echo "   Main: logs/nae_autonomous_master.log"
    echo "   Startup: logs/nae_startup.log"
    echo "   Agents: logs/agent_*.log"
    echo ""
    echo "📋 To check status:"
    echo "   tail -f logs/nae_autonomous_master.log"
    echo ""
    echo "🛑 To stop:"
    echo "   ./stop_nae_production.sh"
    echo "   or: kill $NAE_PID"
else
    echo "❌ Failed to start NAE"
    echo "Check logs/nae_startup.log for errors"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ NAE is now running in Production Mode"
echo "Continuous & Autonomous Operation Active"
echo "=========================================="
echo ""


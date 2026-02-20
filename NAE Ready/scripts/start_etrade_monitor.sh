#!/bin/bash
# Start E*TRADE Status Monitor with OpenSSL Python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/python_openssl.sh"

cd "$NAE_DIR"

# Check if already running
if [ -f "logs/etrade_monitor.pid" ]; then
    PID=$(cat logs/etrade_monitor.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Monitor is already running (PID: $PID)"
        echo "   Stop it first: bash scripts/stop_etrade_monitor.sh"
        exit 1
    fi
fi

# Start monitor with OpenSSL Python
echo "🚀 Starting E*TRADE monitor with OpenSSL Python..."
echo "   Checking every 60 seconds"
echo "   Log: logs/etrade_monitor.log"
echo

cd "$NAE_DIR"
nohup "$PYTHON_SCRIPT" scripts/monitor_etrade_status.py --interval 60 >> logs/etrade_monitor.log 2>&1 &
MONITOR_PID=$!
sleep 1
echo $MONITOR_PID > logs/etrade_monitor.pid

sleep 2

# Verify it's running
if ps -p $MONITOR_PID > /dev/null 2>&1; then
    echo "✅ Monitor started successfully"
    echo "   PID: $MONITOR_PID"
    echo "   Using: $("$PYTHON_SCRIPT" -c 'import ssl; print(ssl.OPENSSL_VERSION)' 2>/dev/null || echo 'Python (checking SSL...)')"
    echo
    echo "📋 Commands:"
    echo "   Check status: bash scripts/check_etrade_monitor.sh"
    echo "   View log: tail -f logs/etrade_monitor.log"
    echo "   Stop: bash scripts/stop_etrade_monitor.sh"
else
    echo "❌ Failed to start monitor"
    echo "   Check logs/etrade_monitor.log for errors"
    rm -f logs/etrade_monitor.pid
    exit 1
fi


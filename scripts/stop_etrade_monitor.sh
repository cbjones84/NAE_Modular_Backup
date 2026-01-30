#!/bin/bash
# Stop E*TRADE Status Monitor (Updated to use OpenSSL Python)

MONITOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$MONITOR_DIR/logs/etrade_monitor.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping E*TRADE monitor (PID: $PID)..."
        kill $PID
        sleep 1
        
        if ps -p $PID > /dev/null 2>&1; then
            echo "Force stopping..."
            kill -9 $PID
        fi
        
        rm -f "$PID_FILE"
        echo "✅ Monitor stopped"
    else
        echo "⚠️  Monitor process not found (PID: $PID)"
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  PID file not found. Trying to find process..."
    
    # Try to find and kill by process name
    PIDS=$(ps aux | grep "monitor_etrade_status.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo "❌ No monitor process found"
    else
        for PID in $PIDS; do
            echo "Stopping process $PID..."
            kill $PID
        done
        echo "✅ Monitor stopped"
    fi
fi


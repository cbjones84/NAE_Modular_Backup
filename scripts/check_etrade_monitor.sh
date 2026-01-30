#!/bin/bash
# Check E*TRADE Status Monitor Status

MONITOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$MONITOR_DIR/logs/etrade_monitor.pid"
LOG_FILE="$MONITOR_DIR/logs/etrade_monitor.log"

echo "="*80
echo "E*TRADE MONITOR STATUS"
echo "="*80
echo

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Monitor is RUNNING"
        echo "   PID: $PID"
        echo "   Started: $(ps -p $PID -o lstart= | awk '{print $1, $2, $3, $4}')"
        echo "   CPU: $(ps -p $PID -o %cpu= | tr -d ' ')%"
        echo "   Memory: $(ps -p $PID -o %mem= | tr -d ' ')%"
        echo
        echo "📝 Recent log output:"
        echo "   $(tail -1 "$LOG_FILE" 2>/dev/null || echo 'No log entries yet')"
    else
        echo "❌ Monitor is NOT running (PID file exists but process is dead)"
        echo "   PID: $PID"
    fi
else
    echo "❌ Monitor is NOT running (no PID file)"
fi

echo
echo "📋 Log file: $LOG_FILE"
if [ -f "$LOG_FILE" ]; then
    echo "   Size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "   Last updated: $(stat -f "%Sm" "$LOG_FILE" 2>/dev/null || stat -c "%y" "$LOG_FILE" 2>/dev/null | cut -d' ' -f1-2)"
fi

echo
echo "💡 Commands:"
echo "   Check status: bash scripts/check_etrade_monitor.sh"
echo "   View log: tail -f logs/etrade_monitor.log"
echo "   Stop monitor: bash scripts/stop_etrade_monitor.sh"



#!/bin/bash
# Start NAE autonomously in the background
# This script starts NAE and sets it up to run continuously

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🚀 Starting NAE Autonomous Mode"
echo "=========================================="
echo ""

# Check if already running
if pgrep -f "splinter" > /dev/null || pgrep -f "nae_main_orchestrator.py" > /dev/null || pgrep -f "nae_continuous_automation.py" > /dev/null; then
    echo "✅ NAE is already running!"
    ps aux | grep -E "splinter|nae_main_orchestrator|nae_continuous_automation" | grep -v grep
    exit 0
fi

# Create logs directory
mkdir -p logs

# Start NAE in background (using Splinter as comprehensive orchestrator)
echo "🚀 Starting NAE automation system via Splinter..."
nohup python3 start_splinter_autonomous.py > logs/automation.out 2>&1 &
NAE_PID=$!

sleep 3

# Check if started successfully
if ps -p $NAE_PID > /dev/null; then
    echo "✅ NAE started successfully!"
    echo "🆔 Process ID: $NAE_PID"
    echo "📝 Logs: logs/automation.out"
    echo ""
    echo "📋 To check status:"
    echo "   tail -f logs/automation.out"
    echo ""
    echo "🛑 To stop:"
    echo "   kill $NAE_PID"
    echo "   or: pkill -f splinter"
else
    echo "❌ Failed to start NAE"
    echo "Check logs/automation.out for errors"
    exit 1
fi

# Set up LaunchAgent for macOS (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "🔧 Setting up LaunchAgent for automatic startup..."
    
    LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$LAUNCH_AGENT_DIR/com.nae.automation.plist"
    
    # Create LaunchAgents directory if it doesn't exist
    mkdir -p "$LAUNCH_AGENT_DIR"
    
    # Copy plist file if it exists
    if [ -f "com.nae.automation.plist" ]; then
        cp com.nae.automation.plist "$PLIST_FILE"
        
        # Update paths in plist
        sed -i '' "s|/Users/melissabishop/Downloads/Neural Agency Engine/NAE|$SCRIPT_DIR|g" "$PLIST_FILE"
        
        # Load LaunchAgent (bootstrap/bootout for macOS Ventura+)
        LAUNCH_USER="gui/$(id -u)"
        launchctl bootout "$LAUNCH_USER" "$PLIST_FILE" 2>/dev/null || true
        launchctl bootstrap "$LAUNCH_USER" "$PLIST_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✅ LaunchAgent installed! NAE will start automatically on login."
        else
            echo "⚠️  Could not install LaunchAgent, but NAE is running now."
        fi
    else
        echo "⚠️  LaunchAgent plist not found, but NAE is running now."
    fi
fi

echo ""
echo "=========================================="
echo "✅ NAE is now running autonomously!"
echo "=========================================="


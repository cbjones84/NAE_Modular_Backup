#!/bin/bash
# Start AutoGen Studio for NAE

echo "=========================================="
echo "NAE AutoGen Studio Launcher"
echo "=========================================="
echo ""

# Check if autogenstudio is installed
if ! command -v autogenstudio &> /dev/null; then
    echo "❌ AutoGen Studio not found. Installing..."
    pip install autogenstudio
fi

# Check if bridge server should be started
if [ "$1" == "--with-bridge" ]; then
    echo "🌉 Starting bridge server..."
    python3 autogen_studio_bridge.py &
    BRIDGE_PID=$!
    echo "Bridge server PID: $BRIDGE_PID"
    sleep 2
fi

# Start AutoGen Studio
echo "🚀 Starting AutoGen Studio..."
echo "📱 AutoGen Studio will be available at: http://localhost:8081/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

autogenstudio ui --port 8081

# Cleanup bridge if started
if [ ! -z "$BRIDGE_PID" ]; then
    echo ""
    echo "🛑 Stopping bridge server..."
    kill $BRIDGE_PID 2>/dev/null
fi


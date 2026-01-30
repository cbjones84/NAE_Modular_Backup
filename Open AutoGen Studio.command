#!/bin/bash
# AutoGen Studio Launcher for macOS
# Double-click this file to start AutoGen Studio and open it in your browser

cd "$(dirname "$0")"

echo "=========================================="
echo "🚀 Starting AutoGen Studio for NAE"
echo "=========================================="
echo ""

# Check if autogenstudio is installed
if ! command -v autogenstudio &> /dev/null; then
    echo "❌ AutoGen Studio not found. Installing..."
    pip3 install autogenstudio
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install AutoGen Studio"
        echo "Please install manually: pip install autogenstudio"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Check if port 8081 is already in use
if lsof -Pi :8081 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ AutoGen Studio appears to be running on port 8081"
    echo "🌐 Opening browser..."
    sleep 1
    open "http://localhost:8081"
    echo ""
    echo "✅ Browser opened! AutoGen Studio should be available."
    echo ""
    read -p "Press Enter to exit..."
    exit 0
fi

echo "🚀 Starting AutoGen Studio server..."
echo "📱 AutoGen Studio will be available at: http://localhost:8081/"
echo ""
echo "⏳ Waiting for server to start..."
echo ""

# Start AutoGen Studio in background
autogenstudio ui --port 8081 > /tmp/autogen_studio.log 2>&1 &
AUTOGEN_PID=$!

# Wait a bit for server to start
sleep 5

# Check if server started successfully
if ps -p $AUTOGEN_PID > /dev/null; then
    echo "✅ AutoGen Studio server started (PID: $AUTOGEN_PID)"
    echo "🌐 Opening browser..."
    
    # Open browser
    sleep 2
    open "http://localhost:8081"
    
    echo ""
    echo "=========================================="
    echo "✅ AutoGen Studio is running!"
    echo "=========================================="
    echo ""
    echo "📱 Browser opened at: http://localhost:8081"
    echo "🆔 Server PID: $AUTOGEN_PID"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Create Casey agent using autogen_studio_config.json"
    echo "   2. Create a workflow with Casey and User agents"
    echo "   3. Start chatting with Casey!"
    echo ""
    echo "🛑 To stop the server, press Ctrl+C or close this window"
    echo ""
    
    # Wait for user to stop
    wait $AUTOGEN_PID
else
    echo "❌ Failed to start AutoGen Studio server"
    echo "Check logs: /tmp/autogen_studio.log"
    read -p "Press Enter to exit..."
    exit 1
fi


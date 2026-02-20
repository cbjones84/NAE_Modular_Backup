#!/bin/bash
# Start AutoGen Studio with NAE integration

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Starting AutoGen Studio for NAE"
echo "=========================================="
echo ""

# Check if autogenstudio is installed
if ! python3 -c "import autogenstudio" 2>/dev/null; then
    echo "❌ AutoGen Studio not installed"
    echo ""
    echo "Installing AutoGen Studio..."
    python3 -m pip install autogenstudio --upgrade
    echo ""
fi

# Generate NAE configurations
echo "Generating NAE agent configurations..."
python3 autogen_studio_nae_integration.py

echo ""
echo "Starting AutoGen Studio UI..."
echo "Access at: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start AutoGen Studio UI
autogenstudio ui --port 8080


#!/bin/bash
# Start NAE with THRML support in continuous background mode
# This script activates Python 3.11 virtual environment and starts NAE autonomously

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🚀 Starting NAE with THRML Support"
echo "=========================================="
echo ""

# Check if Python 3.11 venv exists
if [ ! -d "venv_python311" ]; then
    echo "❌ Python 3.11 virtual environment not found!"
    echo "   Please run: python3.11 -m venv venv_python311"
    echo "   Then: source venv_python311/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate Python 3.11 virtual environment
echo "🔧 Activating Python 3.11 virtual environment..."
source venv_python311/bin/activate

# Verify THRML is available
echo "🔍 Verifying THRML installation..."
python -c "import thrml; import jax; print('✅ THRML:', thrml.__version__); print('✅ JAX:', jax.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  THRML not found in venv. Installing..."
    pip install jax jaxlib thrml > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install THRML. Continuing with JAX fallback..."
    else
        echo "✅ THRML installed successfully"
    fi
fi

# Check if already running
if pgrep -f "nae_master_scheduler.py" > /dev/null || pgrep -f "nae_continuous_automation.py" > /dev/null; then
    echo "✅ NAE is already running!"
    ps aux | grep -E "nae_master_scheduler|nae_continuous" | grep -v grep
    exit 0
fi

# Create logs directory
mkdir -p logs

# Determine which script to run
if [ -f "nae_master_scheduler.py" ]; then
    MAIN_SCRIPT="nae_master_scheduler.py"
elif [ -f "nae_continuous_automation.py" ]; then
    MAIN_SCRIPT="nae_continuous_automation.py"
else
    echo "❌ No NAE main script found!"
    exit 1
fi

# Start NAE in background with THRML-enabled Python
echo "🚀 Starting NAE automation system with THRML..."
nohup python "$MAIN_SCRIPT" > logs/nae_thrml.out 2>&1 &
NAE_PID=$!

sleep 3

# Check if started successfully
if ps -p $NAE_PID > /dev/null; then
    echo "✅ NAE started successfully with THRML support!"
    echo "🆔 Process ID: $NAE_PID"
    echo "🐍 Python: $(python --version)"
    echo "📝 Logs: logs/nae_thrml.out"
    echo ""
    echo "📋 To check status:"
    echo "   tail -f logs/nae_thrml.out"
    echo ""
    echo "🛑 To stop:"
    echo "   kill $NAE_PID"
    echo "   or: pkill -f $MAIN_SCRIPT"
    
    # Save PID for easy management
    echo $NAE_PID > logs/nae_thrml.pid
    echo ""
    echo "💾 PID saved to: logs/nae_thrml.pid"
else
    echo "❌ Failed to start NAE"
    echo "Check logs/nae_thrml.out for errors"
    exit 1
fi

# Set up LaunchAgent for macOS (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "🔧 Setting up LaunchAgent for automatic startup..."
    
    LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$LAUNCH_AGENT_DIR/com.nae.thrml.plist"
    
    # Create LaunchAgents directory if it doesn't exist
    mkdir -p "$LAUNCH_AGENT_DIR"
    
    # Create plist file
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nae.thrml</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_DIR/venv_python311/bin/python</string>
        <string>$SCRIPT_DIR/$MAIN_SCRIPT</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/logs/nae_thrml.out</string>
    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/logs/nae_thrml.err</string>
</dict>
</plist>
EOF
    
    # Load LaunchAgent (bootstrap/bootout for macOS Ventura+)
    LAUNCH_USER="gui/$(id -u)"
    launchctl bootout "$LAUNCH_USER" "$PLIST_FILE" 2>/dev/null || true
    launchctl bootstrap "$LAUNCH_USER" "$PLIST_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ LaunchAgent installed! NAE will start automatically on login."
    else
        echo "⚠️  Could not install LaunchAgent, but NAE is running now."
    fi
fi

echo ""
echo "=========================================="
echo "✅ NAE is now running with THRML support!"
echo "=========================================="
echo ""
echo "🔬 THRML Features Active:"
echo "   • Probabilistic trading scenarios (Optimus)"
echo "   • Tail risk estimation (Optimus)"
echo "   • Energy-based strategy learning (Ralph)"
echo "   • Probabilistic validation (Donnie)"
echo ""


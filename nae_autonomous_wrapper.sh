#!/bin/bash
# NAE Autonomous Wrapper Script
# Handles macOS security restrictions

cd "/Users/melissabishop/NAE_Ready"

# Activate virtual environment if it exists
if [ -d "venv_python311" ]; then
    source venv_python311/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the autonomous master
exec /usr/bin/python3 nae_autonomous_master.py


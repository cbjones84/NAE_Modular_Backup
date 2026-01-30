#!/bin/bash
# NAE Startup Script
# Automatically starts the NAE automation system

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Starting Neural Agency Engine (NAE)"
echo "=========================================="

# Check Python version
python3 --version || {
    echo "ERROR: Python 3 is required"
    exit 1
}

# Create logs directory
mkdir -p logs

# Start NAE automation system
python3 nae_automation.py


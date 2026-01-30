#!/bin/bash
# Complete HP OmniBook X Setup - Master Script
# Tries all methods to complete HP setup

set -e

echo "=========================================="
echo "Complete HP OmniBook X Setup"
echo "=========================================="
echo ""

# Method 1: Try remote setup if config exists
if [ -f "config/remote_config.json" ]; then
    echo "Method 1: Attempting remote setup..."
    python3 setup/remote_hp_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ HP setup completed remotely!"
        echo ""
        echo "Verifying connection..."
        python3 setup/cursor_remote_integration.py --verify && echo "✅ Connection verified"
        echo ""
        echo "Getting production status..."
        python3 setup/cursor_remote_integration.py --status
        echo ""
        echo "🎉 HP OmniBook X is fully set up and connected!"
        exit 0
    else
        echo "⚠️  Remote setup failed - trying other methods..."
        echo ""
    fi
else
    echo "Method 1: Remote config not found"
    echo ""
fi

# Method 2: Check if we can configure remote connection
echo "Method 2: Checking if we can configure remote connection..."
echo ""
read -p "Do you have the HP OmniBook X IP address? [y/N]: " HAS_IP

if [ "$HAS_IP" == "y" ] || [ "$HAS_IP" == "Y" ]; then
    echo ""
    echo "Configuring remote connection..."
    ./setup/configure_remote_connection.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Attempting remote setup..."
        python3 setup/remote_hp_setup.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ HP setup completed remotely!"
            python3 setup/cursor_remote_integration.py --verify
            exit 0
        fi
    fi
fi

# Method 3: Provide standalone script
echo ""
echo "Method 3: Creating standalone setup script..."
echo ""

if [ -f "hp_local_setup.sh" ]; then
    echo "✅ Standalone script already exists: hp_local_setup.sh"
else
    ./setup/auto_setup_hp.sh
fi

echo ""
echo "=========================================="
echo "Setup Instructions"
echo "=========================================="
echo ""
echo "Since remote setup isn't possible right now, use the standalone script:"
echo ""
echo "1. Copy hp_local_setup.sh to HP OmniBook X"
echo "   (via USB, network share, or email)"
echo ""
echo "2. On HP OmniBook X, run:"
echo "   bash hp_local_setup.sh"
echo ""
echo "3. After HP setup, on Mac:"
echo "   ./setup/configure_remote_connection.sh"
echo "   (Enter HP IP address)"
echo ""
echo "4. Test connection:"
echo "   ./setup/quick_connect.sh"
echo ""
echo "=========================================="
echo ""

# Show script location
SCRIPT_PATH="$(pwd)/hp_local_setup.sh"
echo "Standalone script location:"
echo "$SCRIPT_PATH"
echo ""
echo "File size: $(du -h "$SCRIPT_PATH" | cut -f1)"
echo ""


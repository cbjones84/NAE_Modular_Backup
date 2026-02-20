#!/bin/bash
# Quick connection test and setup verification

echo "=== NAE Dual Machine Connection Test ==="
echo ""

# Check if remote config exists
if [ ! -f "config/remote_config.json" ]; then
    echo "❌ Remote config not found"
    echo "Run: ./setup/configure_remote_connection.sh"
    exit 1
fi

# Test connection
echo "Testing connection to HP OmniBook X..."
python3 setup/cursor_remote_integration.py --verify

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Connection successful!"
    echo ""
    echo "Getting production status..."
    python3 setup/cursor_remote_integration.py --status
else
    echo ""
    echo "❌ Connection failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Verify HP OmniBook X is powered on"
    echo "2. Check SSH is enabled on HP"
    echo "3. Verify SSH key is copied"
    echo "4. Check config/remote_config.json"
fi


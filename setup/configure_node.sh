#!/bin/bash
# Configure HP OmniBook X as Node

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "NAE Node Configuration (HP OmniBook X)"
echo "=========================================="
echo ""

# Check if on Linux (HP)
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is for Linux (HP OmniBook X)"
    read -p "Continue anyway? [y/N]: " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

# Get Master API key
read -p "Enter Master API Key: " MASTER_API_KEY
if [ -z "$MASTER_API_KEY" ]; then
    echo "❌ API key required"
    exit 1
fi

read -p "Enter Master URL (default: http://localhost:8080): " MASTER_URL
MASTER_URL=${MASTER_URL:-"http://localhost:8080"}

# Update .env
if [ -f "$NAE_ROOT/.env" ]; then
    # Add node settings
    if ! grep -q "NAE_IS_NODE" "$NAE_ROOT/.env"; then
        cat >> "$NAE_ROOT/.env" << EOF

# Node Settings
NAE_IS_MASTER=false
NAE_IS_NODE=true
NAE_NODE_API_KEY=$MASTER_API_KEY
NAE_MASTER_URL=$MASTER_URL
NAE_NODE_PORT=8081
EOF
        echo "✅ Updated .env with node settings"
    fi
fi

# Create node API directory
mkdir -p "$NAE_ROOT/node_api"
mkdir -p "$NAE_ROOT/node_api/endpoints"

echo "✅ Created node API directories"
echo ""

# Create node API client
NODE_API_FILE="$NAE_ROOT/node_api/client.py"
cat > "$NODE_API_FILE" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Node API Client"""

import os
import requests
import json

class NodeAPIClient:
    """Client for communicating with Master"""
    
    def __init__(self):
        self.master_url = os.getenv('NAE_MASTER_URL', 'http://localhost:8080')
        self.api_key = os.getenv('NAE_NODE_API_KEY', '')
        self.headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def report_status(self, status: dict):
        """Report node status to master"""
        try:
            response = requests.post(
                f"{self.master_url}/api/node/status",
                json=status,
                headers=self.headers,
                timeout=5
            )
            return response.json()
        except Exception as e:
            print(f"Error reporting status: {e}")
            return None
    
    def report_trade(self, trade: dict):
        """Report trade to master"""
        try:
            response = requests.post(
                f"{self.master_url}/api/trade/confirmation",
                json=trade,
                headers=self.headers,
                timeout=5
            )
            return response.json()
        except Exception as e:
            print(f"Error reporting trade: {e}")
            return None

if __name__ == '__main__':
    client = NodeAPIClient()
    status = client.report_status({"status": "online"})
    print(f"Status report: {status}")
PYTHON_EOF

chmod +x "$NODE_API_FILE"
echo "✅ Created node API client"

echo ""
echo "=========================================="
echo "✅ Node configuration complete!"
echo "=========================================="
echo ""
echo "Node configured to connect to: $MASTER_URL"
echo ""
echo "Next steps:"
echo "1. Test connection: python3 node_api/client.py"
echo "2. Start production services"
echo ""


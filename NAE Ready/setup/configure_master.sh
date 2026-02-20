#!/bin/bash
# Configure Mac as Master

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "NAE Master Configuration (Mac)"
echo "=========================================="
echo ""

# Check if on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for Mac only"
    exit 1
fi

# Generate API key
API_KEY=$(openssl rand -hex 32)
echo "Generated Master API Key: $API_KEY"
echo ""

# Update .env
if [ -f "$NAE_ROOT/.env" ]; then
    # Add master settings
    if ! grep -q "NAE_IS_MASTER" "$NAE_ROOT/.env"; then
        cat >> "$NAE_ROOT/.env" << EOF

# Master Settings
NAE_IS_MASTER=true
NAE_IS_NODE=false
NAE_MASTER_API_KEY=$API_KEY
NAE_MASTER_PORT=8080
NAE_MASTER_URL="http://localhost:8080"
EOF
        echo "✅ Updated .env with master settings"
    fi
fi

# Create master API directory
mkdir -p "$NAE_ROOT/master_api"
mkdir -p "$NAE_ROOT/master_api/endpoints"
mkdir -p "$NAE_ROOT/master_api/models"
mkdir -p "$NAE_ROOT/master_api/strategies"

echo "✅ Created master API directories"
echo ""

# Create master API server
MASTER_API_FILE="$NAE_ROOT/master_api/server.py"
cat > "$MASTER_API_FILE" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Master API Server"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load config
API_KEY = os.getenv('NAE_MASTER_API_KEY', '')
PORT = int(os.getenv('NAE_MASTER_PORT', '8080'))

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "role": "master"})

@app.route('/api/strategy/update', methods=['POST'])
def update_strategy():
    # Verify API key
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    # Process strategy update
    return jsonify({"status": "success", "message": "Strategy updated"})

@app.route('/api/node/status', methods=['GET'])
def node_status():
    # Get node status
    return jsonify({"status": "ok", "nodes": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
PYTHON_EOF

chmod +x "$MASTER_API_FILE"
echo "✅ Created master API server"

echo ""
echo "=========================================="
echo "✅ Master configuration complete!"
echo "=========================================="
echo ""
echo "Master API Key: $API_KEY"
echo "Save this key - you'll need it for the Node configuration"
echo ""
echo "Next steps:"
echo "1. Configure Node (HP): ./setup/configure_node.sh"
echo "2. Start Master API: python3 master_api/server.py"
echo ""


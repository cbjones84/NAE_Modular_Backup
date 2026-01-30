#!/bin/bash
# Setup Tradier Environment Variables
# Run this script to set up your Tradier credentials for NAE

# Tradier Configuration
export TRADIER_SANDBOX="false"  # Use LIVE trading
export TRADIER_API_KEY="27Ymk28vtbgqY1LFYxhzaEmIuwJb"
export TRADIER_ACCOUNT_ID="6YB66744"

# Verify environment variables are set
echo "✅ Tradier Environment Variables Set:"
echo "   TRADIER_SANDBOX: $TRADIER_SANDBOX"
echo "   TRADIER_API_KEY: ${TRADIER_API_KEY:0:10}..."
echo "   TRADIER_ACCOUNT_ID: $TRADIER_ACCOUNT_ID"
echo ""
echo "To make these permanent, add to your ~/.bashrc or ~/.zshrc:"
echo "   source $(pwd)/setup_tradier_env.sh"
echo ""
echo "Or run this script before starting NAE:"
echo "   source setup_tradier_env.sh"


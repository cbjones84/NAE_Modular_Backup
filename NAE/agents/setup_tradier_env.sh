#!/bin/bash
# NAE Tradier Environment Setup Script
# Sets TRADIER_API_KEY and TRADIER_ACCOUNT_ID environment variables

echo "🔧 Setting up Tradier environment variables..."

# Set Tradier API Key
export TRADIER_API_KEY=27Ymk28vtbgqY1LFYxhzaEmIuwJb
echo "✅ TRADIER_API_KEY set"

# Set Tradier Account ID
export TRADIER_ACCOUNT_ID=6YB66744
echo "✅ TRADIER_ACCOUNT_ID set"

# Set sandbox mode (PRODUCTION account - set to false for live trading)
export TRADIER_SANDBOX=false
echo "✅ TRADIER_SANDBOX set to: $TRADIER_SANDBOX (PRODUCTION)"

echo ""
echo "📝 To make these permanent, add to your shell profile (~/.bashrc or ~/.zshrc):"
echo "   export TRADIER_API_KEY=27Ymk28vtbgqY1LFYxhzaEmIuwJb"
echo "   export TRADIER_ACCOUNT_ID=6YB66744"
echo "   export TRADIER_SANDBOX=false"
echo ""
echo "⚠️  WARNING: This is a PRODUCTION account. Trades will be real!"
echo ""
echo "✅ Environment variables set for current session"


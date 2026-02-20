#!/bin/bash
# NAE Environment Variables Setup Script
# Usage: source setup_env.sh or ./setup_env.sh

# Check if .env file exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found. Creating template..."
    cat > .env << 'EOF'
# NAE Environment Configuration
NAE_ENVIRONMENT=sandbox

# LLM API Keys (Replace with your actual keys)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: Vault Password
NAE_VAULT_PASSWORD=your-vault-password
EOF
    echo "📝 Created .env template. Please edit it with your API keys."
    echo "   Then run: source setup_env.sh"
fi

# Verify variables are set
echo ""
echo "🔍 Environment Variables Status:"
echo "  NAE_ENVIRONMENT: ${NAE_ENVIRONMENT:-sandbox (default)}"
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-key-here" ]; then
    echo "  OPENAI_API_KEY: ❌ Not set"
else
    echo "  OPENAI_API_KEY: ✅ Set (${#OPENAI_API_KEY} chars)"
fi

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-key-here" ]; then
    echo "  ANTHROPIC_API_KEY: ❌ Not set"
else
    echo "  ANTHROPIC_API_KEY: ✅ Set (${#ANTHROPIC_API_KEY} chars)"
fi

echo ""
echo "💡 To set variables manually:"
echo "  export OPENAI_API_KEY=\"your-key\""
echo "  export ANTHROPIC_API_KEY=\"your-key\""
echo "  export NAE_ENVIRONMENT=\"sandbox\""



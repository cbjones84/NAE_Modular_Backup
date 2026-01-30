#!/bin/bash
# NAE Environment Variables Export Script
# Source this file: source export_env.sh

# Check if .env exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found"
fi

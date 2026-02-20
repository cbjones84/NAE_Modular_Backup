#!/bin/bash
# Start Ralph GitHub Research in continuous mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🚀 Starting Ralph GitHub Research"
echo "Continuous Improvement Discovery"
echo "=========================================="
echo ""

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN not set"
    echo "   For better rate limits, set: export GITHUB_TOKEN=your_token"
    echo "   Continuing with limited API access..."
    echo ""
fi

# Start continuous research
echo "🔄 Starting continuous research loop..."
echo "   Research interval: 24 hours"
echo "   Logs: logs/github_research/"
echo ""

nohup python3 agents/ralph_github_research.py --full > logs/ralph_github_research.log 2>&1 &
RESEARCH_PID=$!

echo "✅ Ralph GitHub Research started (PID: $RESEARCH_PID)"
echo "$RESEARCH_PID" > logs/ralph_github_research.pid

echo ""
echo "📋 To check status:"
echo "   tail -f logs/ralph_github_research.log"
echo ""
echo "🛑 To stop:"
echo "   kill $RESEARCH_PID"
echo ""


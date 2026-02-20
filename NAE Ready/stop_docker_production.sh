#!/bin/bash
# =============================================================================
# NAE Docker Production Stop
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🛑 Stopping NAE Docker Production"
echo "=========================================="

docker-compose down

echo ""
echo "✅ All NAE containers stopped"
echo ""
echo "   Data persists in ./data/, ./logs/, ./config/"
echo "   To remove all data: docker-compose down -v"
echo ""

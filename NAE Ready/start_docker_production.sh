#!/bin/bash
# =============================================================================
# NAE Docker Production Startup
# =============================================================================
# Starts all NAE agents in Docker containers with:
# - Process isolation (each agent in its own container)
# - Automatic restarts (Docker handles it, not Python)
# - Resource limits (CPU/memory per container)
# - Redis for inter-agent communication
# - Persistent volumes for logs, data, config
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "NAE Docker Production Mode"
echo "=========================================="

# Check Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker Desktop first."
    echo "   Download: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

echo "Checking Docker daemon..."
if ! docker info &> /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "OK - Docker is running"

# Check for .env.docker
if [ ! -f ".env.docker" ]; then
    echo "ERROR: .env.docker not found."
    echo "   Please create .env.docker with your Tradier credentials."
    exit 1
fi
echo "OK - Environment file found"

# Create data directories
mkdir -p logs data/optimus config

# Build and start services (combined for efficiency)
echo ""
echo "Building and starting containers..."
echo "(First build will pull base images - this may take several minutes)"
echo ""

# Use docker compose (v2) with build
docker compose up --build -d 2>&1

echo ""
echo "=========================================="
echo "NAE Docker Production Mode Active"
echo "=========================================="
echo ""
echo "Container Status:"
docker compose ps
echo ""
echo "Commands:"
echo "   Logs:     docker compose logs -f"
echo "   Optimus:  docker compose logs -f optimus"
echo "   Ralph:    docker compose logs -f ralph"
echo "   Donnie:   docker compose logs -f donnie"
echo "   Status:   docker compose ps"
echo "   Stop:     docker compose down"
echo "   Restart:  docker compose restart <service>"
echo ""

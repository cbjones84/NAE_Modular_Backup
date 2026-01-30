#!/bin/bash
# NAE Complete Setup Script
# Configures API keys, Redis, owner credentials, and prepares for production deployment

set -e

echo "🚀 Starting NAE Complete Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from correct directory
if [ ! -f "config/api_keys.json" ]; then
    print_error "Please run this script from the NAE root directory"
    exit 1
fi

print_status "NAE Complete Setup Script"
print_status "========================="

# Step 1: Install system dependencies
print_status "Step 1: Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip redis-server docker.io docker-compose
elif command -v brew &> /dev/null; then
    brew install python3 redis docker docker-compose
else
    print_warning "Please install Python3, Redis, and Docker manually"
fi

# Step 2: Install Python dependencies
print_status "Step 2: Installing Python dependencies..."
pip3 install -r requirements.txt

# Step 3: Start Redis service
print_status "Step 3: Starting Redis service..."
if command -v systemctl &> /dev/null; then
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
elif command -v brew &> /dev/null; then
    brew services start redis
else
    print_warning "Please start Redis manually"
fi

# Step 4: Test Redis connection
print_status "Step 4: Testing Redis connection..."
python3 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"

# Step 5: Configure API keys
print_status "Step 5: API Keys Configuration"
print_warning "IMPORTANT: You need to configure your API keys manually"
echo ""
echo "Please edit the following files with your actual API keys:"
echo "  📄 config/api_keys.json - Trading platform API keys"
echo "  📄 config/goal_manager.json - Owner credentials"
echo ""
echo "Required API keys:"
echo "  🔑 Polygon.io - Market data API"
echo "  🔑 QuantConnect - Backtesting API"
echo "  🔑 Interactive Brokers - Live trading API"
echo "  🔑 Alpaca - Paper trading API"
echo ""

# Step 6: Test Redis kill switch
print_status "Step 6: Testing Redis kill switch..."
python3 redis_kill_switch.py --status
python3 redis_kill_switch.py --health

# Step 7: Create environment file
print_status "Step 7: Creating environment configuration..."
cat > .env << EOF
# NAE Environment Configuration
NAE_ENVIRONMENT=sandbox
TRADING_MODE=sandbox
REDIS_URL=redis://localhost:6379/0

# API Keys (set these manually)
POLYGON_API_KEY=YOUR_POLYGON_API_KEY_HERE
QUANTCONNECT_USER_ID=YOUR_QUANTCONNECT_USER_ID_HERE
QUANTCONNECT_API_KEY=YOUR_QUANTCONNECT_API_KEY_HERE
IBKR_API_KEY=YOUR_IBKR_API_KEY_HERE
IBKR_API_SECRET=YOUR_IBKR_API_SECRET_HERE
ALPACA_API_KEY=YOUR_ALPACA_API_KEY_HERE
ALPACA_API_SECRET=YOUR_ALPACA_API_SECRET_HERE

# Security
OWNER_API_KEY=YOUR_OWNER_API_KEY_HERE
ENCRYPTION_KEY=YOUR_ENCRYPTION_KEY_HERE
EOF

# Step 8: Test Docker setup
print_status "Step 8: Testing Docker setup..."
if command -v docker &> /dev/null; then
    docker --version
    docker-compose --version
    print_success "Docker is available"
else
    print_warning "Docker not found - required for production deployment"
fi

# Step 9: Create production deployment package
print_status "Step 9: Creating production deployment package..."
python3 production_deploy.py --package

# Step 10: Display next steps
print_success "Setup completed successfully!"
echo ""
print_status "Next Steps:"
echo "============"
echo ""
echo "1. 🔑 Configure API Keys:"
echo "   Edit config/api_keys.json with your actual API keys"
echo "   Edit config/goal_manager.json with owner credentials"
echo ""
echo "2. 🧪 Test Sandbox Environment:"
echo "   python3 agents/optimus.py  # Test Optimus agent"
echo "   python3 agents/ralph.py    # Test Ralph agent"
echo ""
echo "3. 🔄 Test Kill Switch:"
echo "   python3 redis_kill_switch.py --status"
echo "   python3 redis_kill_switch.py --disable --reason 'Testing'"
echo "   python3 redis_kill_switch.py --enable --reason 'Testing complete'"
echo ""
echo "4. 🐳 Start Services with Docker:"
echo "   docker-compose up -d"
echo "   docker-compose logs -f"
echo ""
echo "5. 🚀 Deploy to Production VPS:"
echo "   python3 production_deploy.py --deploy nae_deployment_*.tar.gz"
echo ""
echo "6. 📊 Monitor System:"
echo "   python3 redis_kill_switch.py --health"
echo "   docker-compose ps"
echo ""
print_warning "Remember to:"
echo "  - Keep API keys secure and never commit them to version control"
echo "  - Test thoroughly in sandbox before moving to live trading"
echo "  - Monitor system health regularly"
echo "  - Follow the sandbox phase progression"
echo ""
print_success "NAE setup complete! 🎉"

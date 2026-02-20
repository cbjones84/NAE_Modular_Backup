# Neural Agency Engine (NAE) - Production Setup Guide

## 🚀 Quick Start

The NAE system is now configured for production deployment with comprehensive API key management, Redis kill switch functionality, owner authentication, and sandbox phase progression.

### Prerequisites

- Python 3.11+
- Redis Server
- Docker & Docker Compose
- API Keys for trading platforms

### One-Command Setup

```bash
./setup.sh
```

This script will:
- Install system dependencies
- Install Python packages
- Start Redis service
- Test Redis kill switch
- Create environment configuration
- Generate production deployment package

## 📋 Configuration Files

### API Keys Configuration (`config/api_keys.json`)
```json
{
  "polygon": {
    "api_key": "YOUR_POLYGON_API_KEY_HERE",
    "description": "Polygon.io market data API"
  },
  "quantconnect": {
    "user_id": "YOUR_QUANTCONNECT_USER_ID_HERE",
    "api_key": "YOUR_QUANTCONNECT_API_KEY_HERE",
    "description": "QuantConnect backtesting API"
  },
  "interactive_brokers": {
    "api_key": "YOUR_IBKR_API_KEY_HERE",
    "api_secret": "YOUR_IBKR_API_SECRET_HERE",
    "description": "Interactive Brokers live trading API"
  },
  "alpaca": {
    "api_key": "YOUR_ALPACA_API_KEY_HERE",
    "api_secret": "YOUR_ALPACA_API_SECRET_HERE",
    "description": "Alpaca paper trading API"
  }
}
```

### System Settings (`config/settings.json`)
- Redis configuration
- Kill switch settings
- Trading mode configuration
- Safety limits
- Production deployment settings

### Owner Credentials (`config/goal_manager.json`)
- Owner authentication
- Security settings
- Goal management
- Permission controls

### Sandbox Phases (`config/sandbox_phases.json`)
- Phase progression rules
- Risk limits per phase
- Success criteria
- Monitoring requirements

## 🔧 Redis Kill Switch Management

### Command Line Interface

```bash
# Check kill switch status
python3 redis_kill_switch.py --status

# Enable trading
python3 redis_kill_switch.py --enable --reason "Manual activation"

# Disable trading
python3 redis_kill_switch.py --disable --reason "Risk management"

# View history
python3 redis_kill_switch.py --history 10

# Health check
python3 redis_kill_switch.py --health
```

### Programmatic Usage

```python
from redis_kill_switch import RedisKillSwitchManager

manager = RedisKillSwitchManager()
manager.activate_kill_switch("Risk limit exceeded")
manager.deactivate_kill_switch("Risk cleared")
```

## 🐳 Docker Deployment

### Local Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps

# Stop services
docker-compose down
```

### Services Included

- **Redis**: Kill switch and state management
- **Optimus**: Trading execution agent
- **Ralph**: Strategy development agent
- **Shredder**: Risk management agent

## 🚀 Production Deployment

### 1. Create Deployment Package

```bash
python3 production_deploy.py --package
```

### 2. Deploy to VPS

```bash
python3 production_deploy.py --deploy nae_deployment_*.tar.gz
```

### 3. Validate Configuration

```bash
python3 production_deploy.py --validate
```

## 📊 Sandbox Phase Progression

### Phase 1: Initial Testing (7 days)
- Simulated trading only
- Basic functionality testing
- Limited symbols: AAPL, MSFT, GOOGL
- Max order size: $1,000
- Max positions: 3

### Phase 2: Paper Trading (14 days)
- Real market data
- Paper trading execution
- Extended symbol list
- Max order size: $5,000
- Max positions: 5

### Phase 3: Live Trading Preparation (7 days)
- Advanced paper trading
- Full symbol access
- Manual approval required
- Max order size: $10,000
- Max positions: 10

### Phase 4: Live Trading (365 days)
- Live trading execution
- Full capabilities
- Automated execution
- Max order size: $50,000
- Max positions: 20

## 🔒 Security Features

### Authentication
- Owner credential management
- API key encryption
- Session timeout controls
- Login attempt limiting

### Audit Logging
- Immutable audit trails
- All actions logged
- Hash-based integrity
- Compliance reporting

### Kill Switch
- Redis-based state management
- Automatic activation triggers
- Manual override capabilities
- Historical logging

## 📈 Monitoring & Health Checks

### System Health

```bash
# Redis health
python3 redis_kill_switch.py --health

# Docker services
docker-compose ps

# Agent logs
docker-compose logs optimus
docker-compose logs ralph
```

### Metrics Collection
- CPU usage monitoring
- Memory usage tracking
- Error rate monitoring
- Trading performance metrics

## 🛠️ Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   sudo systemctl start redis-server
   python3 redis_kill_switch.py --health
   ```

2. **Docker Services Not Starting**
   ```bash
   docker-compose down
   docker-compose up -d
   docker-compose logs
   ```

3. **API Key Errors**
   - Verify keys in `config/api_keys.json`
   - Check environment variables
   - Test API connectivity

4. **Permission Denied**
   ```bash
   chmod +x setup.sh
   chmod +x redis_kill_switch.py
   chmod +x production_deploy.py
   ```

### Log Locations

- Agent logs: `logs/`
- Redis logs: System logs
- Docker logs: `docker-compose logs`

## 📞 Support

### Configuration Help
- Check configuration files in `config/`
- Validate settings with deployment script
- Review error logs for details

### API Integration
- Polygon.io: Market data API
- QuantConnect: Backtesting API
- Interactive Brokers: Live trading
- Alpaca: Paper trading

### Production Deployment
- Use production deployment script
- Follow VPS setup instructions
- Monitor system health regularly

## 🎯 Next Steps

1. **Configure API Keys**: Edit configuration files with actual keys
2. **Test Sandbox**: Run agents in sandbox mode
3. **Validate Kill Switch**: Test Redis kill switch functionality
4. **Deploy to VPS**: Use production deployment script
5. **Monitor Progress**: Follow sandbox phase progression
6. **Go Live**: Transition to live trading after validation

---

**⚠️ Important Security Notes:**
- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys
- Monitor system access logs
- Keep kill switch accessible at all times

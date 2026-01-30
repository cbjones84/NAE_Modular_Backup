# NAE Quick Commands Reference
# Neural Agency Engine - Essential Commands

## 🚀 Initial Setup Commands

# Complete setup (installs dependencies, configures Redis, creates deployment package)
./setup.sh

# Verify system configuration and test all components
python3 verify_setup.py

## 🔴 Kill Switch Management

# Check kill switch status
python3 redis_kill_switch.py --status

# Enable trading
python3 redis_kill_switch.py --enable --reason "Manual activation"

# Disable trading (activate kill switch)
python3 redis_kill_switch.py --disable --reason "Risk management"

# View kill switch history
python3 redis_kill_switch.py --history 10

# Health check
python3 redis_kill_switch.py --health

## 🐳 Docker Services

# Start all services
docker-compose up -d

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f optimus
docker-compose logs -f ralph
docker-compose logs -f redis

# Check service status
docker-compose ps

# Stop all services
docker-compose down

# Restart services
docker-compose restart

## 🚀 Production Deployment

# Create deployment package
python3 production_deploy.py --package

# Deploy to production VPS
python3 production_deploy.py --deploy nae_deployment_*.tar.gz

# Validate configuration
python3 production_deploy.py --validate

# Create sandbox phases config
python3 production_deploy.py --phases

## 🧪 Testing Commands

# Test Optimus agent
python3 agents/optimus.py

# Test Ralph agent
python3 agents/ralph.py

# Test Redis connection
python3 -c "import redis; r=redis.Redis(); print('Redis OK' if r.ping() else 'Redis FAIL')"

# Test Docker
docker --version
docker-compose --version

## 📊 Monitoring Commands

# Check Redis info
python3 -c "import redis; r=redis.Redis(); print(r.info())"

# Check system resources
docker stats

# View recent logs
tail -f logs/optimus.log
tail -f logs/ralph.log

# Check configuration files
python3 -c "import json; print(json.load(open('config/settings.json')))"

## 🔧 Troubleshooting Commands

# Restart Redis
sudo systemctl restart redis-server

# Check Redis status
sudo systemctl status redis-server

# Rebuild Docker containers
docker-compose down
docker-compose up --build -d

# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep python

## 📁 File Management

# View configuration files
cat config/api_keys.json
cat config/settings.json
cat config/goal_manager.json
cat config/sandbox_phases.json

# Edit configuration files
nano config/api_keys.json
nano config/settings.json

# View environment file
cat .env

# Check file permissions
ls -la *.py *.sh

## 🔒 Security Commands

# Generate password hash (for owner credentials)
python3 -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"

# Check API key configuration
python3 -c "import json; config=json.load(open('config/api_keys.json')); print('Configured' if not any('YOUR_' in str(v) for v in config.values()) else 'Needs config')"

# Verify SSL certificates (production)
openssl x509 -in /etc/ssl/certs/nae.crt -text -noout

## 📈 Performance Monitoring

# Monitor Redis performance
redis-cli --latency

# Monitor Docker resource usage
docker stats --no-stream

# Check system load
uptime

# Monitor network connections
netstat -tulpn | grep :6379  # Redis
netstat -tulpn | grep :8006  # Optimus
netstat -tulpn | grep :8007  # Ralph

## 🎯 Phase Progression Commands

# Check current phase
python3 -c "import json; phases=json.load(open('config/sandbox_phases.json')); print(f'Current phase: {phases[\"current_phase\"]}')"

# View phase requirements
python3 -c "import json; phases=json.load(open('config/sandbox_phases.json')); print(json.dumps(phases['phase_1'], indent=2))"

## 🔄 Backup Commands

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/

# Backup Redis data
redis-cli BGSAVE

## 📞 Emergency Commands

# Emergency stop all trading
python3 redis_kill_switch.py --disable --reason "EMERGENCY STOP"

# Kill all Docker containers
docker kill $(docker ps -q)

# Stop all Python processes
pkill -f python

# Restart entire system
sudo reboot

---

## 📝 Notes

- Always test in sandbox before live trading
- Monitor kill switch status regularly
- Keep API keys secure and never commit to version control
- Follow phase progression rules
- Regular backups recommended
- Monitor system health continuously

## 🆘 Emergency Contacts

- Kill Switch: `python3 redis_kill_switch.py --disable`
- System Restart: `docker-compose restart`
- Full Reset: `docker-compose down && docker-compose up -d`

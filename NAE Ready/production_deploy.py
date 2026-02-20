#!/usr/bin/env python3
"""
NAE Production Deployment Script
Handles deployment to VPS with proper configuration and security
"""

import os
import json
import subprocess
import shutil
import time
from datetime import datetime
from typing import Dict, Any, List
import argparse

class NAEProductionDeployer:
    """Handles production deployment of NAE system"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self._load_config(config_path)
        self.deployment_config = self.config.get('production', {})
        self.vps_config = self.deployment_config.get('vps', {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            return {}
    
    def validate_configuration(self) -> bool:
        """Validate production configuration"""
        required_fields = [
            'production.vps.host',
            'production.vps.user',
            'production.vps.deployment_path'
        ]
        
        missing_fields = []
        for field in required_fields:
            keys = field.split('.')
            current = self.config
            for key in keys:
                if key not in current:
                    missing_fields.append(field)
                    break
                current = current[key]
        
        if missing_fields:
            print(f"Missing required configuration fields: {missing_fields}")
            return False
        
        return True
    
    def create_deployment_package(self) -> str:
        """Create deployment package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"nae_deployment_{timestamp}.tar.gz"
        
        # Files to include in deployment
        include_files = [
            "agents/",
            "config/",
            "tools/",
            "requirements.txt",
            "docker-compose.yml",
            "Dockerfile.optimus",
            "Dockerfile.ralph",
            "redis_kill_switch.py",
            "goal_manager.py",
            "human_safety_gates.py"
        ]
        
        # Create temporary directory
        temp_dir = f"temp_deployment_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Copy files to temp directory
            for item in include_files:
                if os.path.exists(item):
                    if os.path.isdir(item):
                        shutil.copytree(item, os.path.join(temp_dir, item))
                    else:
                        shutil.copy2(item, temp_dir)
            
            # Create production configuration
            self._create_production_config(temp_dir)
            
            # Create deployment script
            self._create_deployment_script(temp_dir)
            
            # Create tar.gz package
            subprocess.run([
                "tar", "-czf", package_name, "-C", temp_dir, "."
            ], check=True)
            
            print(f"Deployment package created: {package_name}")
            return package_name
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _create_production_config(self, temp_dir: str):
        """Create production-specific configuration - LIVE MODE with Tradier"""
        prod_config = {
            "environment": "production",
            "redis": {
                "host": "redis",
                "port": 6379,
                "db": 0,
                "password": None,
                "decode_responses": True
            },
            "trading": {
                "default_mode": "live",
                "broker": "tradier",
                "aggressiveness": "VERY_AGGRESSIVE",
                "risk_adjustment_factor": 1.5,
                "live": {
                    "enabled": True,
                    "broker": "tradier",
                    "requires_manual_approval": False
                },
                "paper": {
                    "enabled": False
                },
                "sandbox": {
                    "enabled": False
                }
            },
            "safety_limits": {
                "max_order_size_usd": 50000.0,
                "max_order_size_pct_nav": 0.50,
                "daily_loss_limit_pct": 0.50,
                "consecutive_loss_limit": 10,
                "max_open_positions": 30
            },
            "logging": {
                "level": "INFO",
                "file_rotation": True,
                "max_file_size_mb": 100,
                "backup_count": 5
            },
            "security": {
                "encrypt_sensitive_data": True,
                "require_authentication": True,
                "session_timeout": 3600
            }
        }
        
        with open(os.path.join(temp_dir, "config", "production_settings.json"), 'w') as f:
            json.dump(prod_config, f, indent=2)
    
    def _create_deployment_script(self, temp_dir: str):
        """Create deployment script for VPS"""
        script_content = f"""#!/bin/bash
# NAE Production Deployment Script
# Generated on {datetime.now().isoformat()}

set -e

echo "Starting NAE Production Deployment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y docker.io docker-compose python3 python3-pip redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Create NAE user if not exists
if ! id "nae_user" &>/dev/null; then
    sudo useradd -m -s /bin/bash nae_user
    sudo usermod -aG docker nae_user
fi

# Create deployment directory
sudo mkdir -p {self.vps_config.get('deployment_path', '/opt/nae')}
sudo chown nae_user:nae_user {self.vps_config.get('deployment_path', '/opt/nae')}

# Copy deployment files
cp -r * {self.vps_config.get('deployment_path', '/opt/nae')}/

# Set proper permissions
sudo chown -R nae_user:nae_user {self.vps_config.get('deployment_path', '/opt/nae')}
sudo chmod +x {self.vps_config.get('deployment_path', '/opt/nae')}/redis_kill_switch.py

# Install Python dependencies
cd {self.vps_config.get('deployment_path', '/opt/nae')}
pip3 install -r requirements.txt

# Start services with Docker Compose
docker-compose up -d

# Wait for services to start
sleep 30

# Verify services are running
docker-compose ps

# Test Redis connection
python3 redis_kill_switch.py --health

echo "NAE Production Deployment Completed Successfully!"
echo "Services are running. Check logs with: docker-compose logs"
"""
        
        script_path = os.path.join(temp_dir, "deploy.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
    
    def deploy_to_vps(self, package_path: str) -> bool:
        """Deploy package to VPS"""
        if not self.validate_configuration():
            return False
        
        host = self.vps_config['host']
        user = self.vps_config['user']
        port = self.vps_config.get('port', 22)
        
        try:
            # Copy package to VPS
            print(f"Copying deployment package to {host}...")
            subprocess.run([
                "scp", "-P", str(port), package_path, f"{user}@{host}:~/"
            ], check=True)
            
            # Extract and deploy on VPS
            package_name = os.path.basename(package_path)
            print(f"Deploying on {host}...")
            subprocess.run([
                "ssh", "-p", str(port), f"{user}@{host}", 
                f"tar -xzf {package_name} && chmod +x deploy.sh && ./deploy.sh"
            ], check=True)
            
            print("Deployment completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Deployment failed: {e}")
            return False
    
    def create_production_phases(self) -> Dict[str, Any]:
        """Create production phase configuration - LIVE TRADING ONLY via Tradier"""
        phases = {
            "phase_1": {
                "name": "LIVE Trading - VERY_AGGRESSIVE",
                "description": "LIVE trading with Tradier - Growth milestones: Year 7 $6.2M, Year 8 $15.7M",
                "growth_milestones": {
                    1: 9411, 2: 44110, 3: 152834, 4: 388657,
                    5: 982500, 6: 2477897, 7: 6243561, 8: 15726144
                },
                "duration_days": 365,
                "trading_enabled": True,
                "trading_mode": "live",
                "broker": "tradier",
                "aggressiveness": "VERY_AGGRESSIVE",
                "risk_adjustment_factor": 1.5,
                "max_order_size_usd": 50000,
                "max_order_size_pct_nav": 0.50,
                "allowed_symbols": "all",
                "risk_limits": {
                    "daily_loss_limit_pct": 0.50,
                    "max_open_positions": 30,
                    "consecutive_loss_limit": 10
                },
                "monitoring": {
                    "log_level": "INFO",
                    "audit_all_actions": True,
                    "require_manual_approval": False
                }
            }
        }
        
        return phases

def main():
    """CLI interface for production deployment"""
    parser = argparse.ArgumentParser(description="NAE Production Deployment")
    parser.add_argument("--config", default="config/settings.json", help="Config file path")
    parser.add_argument("--package", action="store_true", help="Create deployment package")
    parser.add_argument("--deploy", help="Deploy package to VPS")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--phases", action="store_true", help="Create sandbox phases config")
    
    args = parser.parse_args()
    
    deployer = NAEProductionDeployer(args.config)
    
    if args.validate:
        if deployer.validate_configuration():
            print("Configuration is valid")
        else:
            print("Configuration validation failed")
    
    elif args.package:
        package_path = deployer.create_deployment_package()
        print(f"Deployment package created: {package_path}")
    
    elif args.deploy:
        if deployer.deploy_to_vps(args.deploy):
            print("Deployment successful")
        else:
            print("Deployment failed")
    
    elif args.phases:
        phases = deployer.create_sandbox_phases()
        with open("config/sandbox_phases.json", 'w') as f:
            json.dump(phases, f, indent=2)
        print("Sandbox phases configuration created")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

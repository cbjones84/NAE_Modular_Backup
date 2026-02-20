#!/usr/bin/env python3
"""
Remote Execution Bridge

Enables Mac (dev) to execute commands on HP OmniBook X (prod) via SSH.
Allows Cursor to control production environment from development machine.
"""

import os
import sys
import json
import subprocess
import paramiko
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemoteExecutionBridge:
    """Bridge for remote execution on HP OmniBook X"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize remote execution bridge"""
        self.config_file = config_file or os.path.join(
            Path(__file__).parent.parent, "config", "remote_config.json"
        )
        self.config = self._load_config()
        self.ssh_client: Optional[paramiko.SSHClient] = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load remote configuration"""
        default_config = {
            "hp_hostname": "",
            "hp_username": "",
            "hp_port": 22,
            "hp_nae_path": "/path/to/NAE",
            "ssh_key_path": os.path.expanduser("~/.ssh/id_rsa"),
            "connection_timeout": 10
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def _connect(self) -> bool:
        """Establish SSH connection to HP"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Try key-based auth first
            key_path = self.config.get("ssh_key_path")
            if key_path and os.path.exists(key_path):
                key = paramiko.RSAKey.from_private_key_file(key_path)
                self.ssh_client.connect(
                    hostname=self.config["hp_hostname"],
                    username=self.config["hp_username"],
                    pkey=key,
                    port=self.config.get("hp_port", 22),
                    timeout=self.config.get("connection_timeout", 10)
                )
            else:
                # Fallback to password (not recommended for production)
                password = self.config.get("password")
                if not password:
                    logger.error("No SSH key or password configured")
                    return False
                
                self.ssh_client.connect(
                    hostname=self.config["hp_hostname"],
                    username=self.config["hp_username"],
                    password=password,
                    port=self.config.get("hp_port", 22),
                    timeout=self.config.get("connection_timeout", 10)
                )
            
            logger.info(f"✅ Connected to HP OmniBook X at {self.config['hp_hostname']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def execute_remote(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute command on HP OmniBook X"""
        if not self.ssh_client:
            if not self._connect():
                return {"success": False, "error": "Connection failed"}
        
        try:
            work_dir = cwd or self.config.get("hp_nae_path", "~/NAE")
            full_command = f"cd {work_dir} && {command}"
            
            stdin, stdout, stderr = self.ssh_client.exec_command(full_command)
            
            exit_status = stdout.channel.recv_exit_status()
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            return {
                "success": exit_status == 0,
                "exit_code": exit_status,
                "output": output,
                "error": error,
                "command": command
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def push_changes(self, local_path: str, remote_path: str) -> bool:
        """Push changes from Mac to HP using git"""
        try:
            # Use git to push changes
            result = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True,
                text=True,
                cwd=local_path
            )
            
            if result.returncode == 0:
                # Pull on remote
                remote_result = self.execute_remote("git pull origin main")
                return remote_result.get("success", False)
            else:
                logger.error(f"Git push failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Push error: {e}")
            return False
    
    def get_remote_status(self) -> Dict[str, Any]:
        """Get status of HP OmniBook X"""
        status_commands = {
            "branch": "git branch --show-current",
            "production_mode": "grep PRODUCTION .env 2>/dev/null | head -1",
            "nae_running": "ps aux | grep nae_autonomous_master | grep -v grep | wc -l",
            "uptime": "uptime"
        }
        
        status = {}
        for key, cmd in status_commands.items():
            result = self.execute_remote(cmd)
            status[key] = result.get("output", "").strip() if result.get("success") else "unknown"
        
        return status
    
    def start_nae_production(self) -> Dict[str, Any]:
        """Start NAE on HP OmniBook X in production mode"""
        # Verify production mode
        check_result = self.execute_remote("python3 safety/production_guard.py")
        if not check_result.get("success"):
            return {
                "success": False,
                "error": "Production safety check failed",
                "details": check_result.get("error", "")
            }
        
        # Start NAE
        start_command = "nohup python3 nae_autonomous_master.py > logs/nae_production.log 2>&1 &"
        result = self.execute_remote(start_command)
        
        return result
    
    def stop_nae_production(self) -> Dict[str, Any]:
        """Stop NAE on HP OmniBook X"""
        result = self.execute_remote("pkill -f nae_autonomous_master.py")
        return result
    
    def close(self):
        """Close SSH connection"""
        if self.ssh_client:
            self.ssh_client.close()
            logger.info("Connection closed")


def main():
    """CLI interface for remote execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remote Execution Bridge for HP OmniBook X")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("--status", action="store_true", help="Get remote status")
    parser.add_argument("--start", action="store_true", help="Start NAE production")
    parser.add_argument("--stop", action="store_true", help="Stop NAE production")
    
    args = parser.parse_args()
    
    bridge = RemoteExecutionBridge()
    
    try:
        if args.status:
            status = bridge.get_remote_status()
            print(json.dumps(status, indent=2))
        elif args.start:
            result = bridge.start_nae_production()
            print(json.dumps(result, indent=2))
        elif args.stop:
            result = bridge.stop_nae_production()
            print(json.dumps(result, indent=2))
        else:
            result = bridge.execute_remote(args.command)
            print(json.dumps(result, indent=2))
    finally:
        bridge.close()


if __name__ == "__main__":
    main()


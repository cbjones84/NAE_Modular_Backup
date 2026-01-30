#!/usr/bin/env python3
"""
Redis Configuration Utility for NAE Kill Switch
Manages Redis connection, kill switch state, and monitoring
"""

import redis
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class RedisKillSwitchManager:
    """Manages Redis-based kill switch for NAE trading system"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self._load_config(config_path)
        self.redis_client = None
        self.kill_switch_key = self.config.get('kill_switch', {}).get('key', 'TRADING_ENABLED')
        self.default_state = self.config.get('kill_switch', {}).get('default_state', True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis connection
        self._initialize_redis()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing config file: {e}")
            return {}
    
    def _initialize_redis(self):
        """Initialize Redis connection with configuration"""
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                decode_responses=redis_config.get('decode_responses', True),
                socket_timeout=redis_config.get('socket_timeout', 5),
                socket_connect_timeout=redis_config.get('socket_connect_timeout', 5),
                retry_on_timeout=redis_config.get('retry_on_timeout', True)
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connection established successfully")
            
            # Initialize kill switch state if not exists
            if not self.redis_client.exists(self.kill_switch_key):
                self.set_kill_switch_state(self.default_state, "Initial setup")
                
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        except Exception as e:
            self.logger.error(f"Redis initialization error: {e}")
            self.redis_client = None
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get_kill_switch_state(self) -> bool:
        """Get current kill switch state"""
        if not self.is_redis_available():
            self.logger.warning("Redis not available, returning default state")
            return self.default_state
        
        try:
            state = self.redis_client.get(self.kill_switch_key)
            return state == "true" if state else self.default_state
        except Exception as e:
            self.logger.error(f"Error getting kill switch state: {e}")
            return False  # Fail-safe: assume trading disabled
    
    def set_kill_switch_state(self, enabled: bool, reason: str = "Manual operation") -> bool:
        """Set kill switch state"""
        if not self.is_redis_available():
            self.logger.error("Cannot set kill switch state: Redis not available")
            return False
        
        try:
            state_value = "true" if enabled else "false"
            self.redis_client.set(self.kill_switch_key, state_value)
            
            # Log the state change
            self._log_kill_switch_change(enabled, reason)
            
            self.logger.info(f"Kill switch {'ENABLED' if enabled else 'DISABLED'}: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting kill switch state: {e}")
            return False
    
    def activate_kill_switch(self, reason: str = "Manual activation") -> bool:
        """Activate kill switch (disable trading)"""
        return self.set_kill_switch_state(False, reason)
    
    def deactivate_kill_switch(self, reason: str = "Manual deactivation") -> bool:
        """Deactivate kill switch (enable trading)"""
        return self.set_kill_switch_state(True, reason)
    
    def _log_kill_switch_change(self, enabled: bool, reason: str):
        """Log kill switch state changes"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "KILL_SWITCH_CHANGE",
                "enabled": enabled,
                "reason": reason,
                "state": "ENABLED" if enabled else "DISABLED"
            }
            
            # Store in Redis with timestamp key
            log_key = f"kill_switch_log:{int(time.time())}"
            self.redis_client.setex(log_key, 86400, json.dumps(log_entry))  # 24 hour TTL
            
            # Also append to a list for easy retrieval
            self.redis_client.lpush("kill_switch_history", json.dumps(log_entry))
            self.redis_client.ltrim("kill_switch_history", 0, 99)  # Keep last 100 entries
            
        except Exception as e:
            self.logger.error(f"Error logging kill switch change: {e}")
    
    def get_kill_switch_history(self, limit: int = 10) -> list:
        """Get recent kill switch history"""
        if not self.is_redis_available():
            return []
        
        try:
            history = self.redis_client.lrange("kill_switch_history", 0, limit - 1)
            return [json.loads(entry) for entry in history]
        except Exception as e:
            self.logger.error(f"Error getting kill switch history: {e}")
            return []
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        if not self.is_redis_available():
            return {"status": "disconnected"}
        
        try:
            info = self.redis_client.info()
            return {
                "status": "connected",
                "version": info.get("redis_version"),
                "uptime": info.get("uptime_in_seconds"),
                "memory_used": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed")
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis and kill switch"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "redis_available": self.is_redis_available(),
            "kill_switch_state": self.get_kill_switch_state(),
            "config_loaded": bool(self.config)
        }
        
        if self.is_redis_available():
            health_status["redis_info"] = self.get_redis_info()
        
        return health_status

def main():
    """CLI interface for Redis kill switch management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NAE Redis Kill Switch Manager")
    parser.add_argument("--config", default="config/settings.json", help="Config file path")
    parser.add_argument("--status", action="store_true", help="Show kill switch status")
    parser.add_argument("--enable", action="store_true", help="Enable trading")
    parser.add_argument("--disable", action="store_true", help="Disable trading")
    parser.add_argument("--reason", default="CLI operation", help="Reason for state change")
    parser.add_argument("--history", type=int, help="Show kill switch history (number of entries)")
    parser.add_argument("--health", action="store_true", help="Show health check")
    
    args = parser.parse_args()
    
    manager = RedisKillSwitchManager(args.config)
    
    if args.status:
        state = manager.get_kill_switch_state()
        print(f"Kill Switch Status: {'ENABLED' if state else 'DISABLED'}")
        print(f"Redis Available: {manager.is_redis_available()}")
    
    elif args.enable:
        success = manager.deactivate_kill_switch(args.reason)
        print(f"Kill switch {'enabled' if success else 'failed to enable'}")
    
    elif args.disable:
        success = manager.activate_kill_switch(args.reason)
        print(f"Kill switch {'disabled' if success else 'failed to disable'}")
    
    elif args.history:
        history = manager.get_kill_switch_history(args.history)
        print(f"Kill Switch History (last {len(history)} entries):")
        for entry in history:
            print(f"  {entry['timestamp']}: {entry['state']} - {entry['reason']}")
    
    elif args.health:
        health = manager.health_check()
        print(json.dumps(health, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

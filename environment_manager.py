# NAE/environment_manager.py
"""
Environment Profile Manager with Auto-Switching
Manages different environment profiles (sandbox, paper, live, test)
"""

import os
import json
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    SANDBOX = "sandbox"
    PAPER = "paper"
    LIVE = "live"
    TEST = "test"

@dataclass
class EnvironmentProfile:
    """Environment profile configuration"""
    name: str
    trading_mode: str
    api_base_urls: Dict[str, str]
    rate_limits: Dict[str, int]
    safety_limits: Dict[str, float]
    enable_live_trading: bool
    require_approval: bool
    logging_level: str

class EnvironmentManager:
    """Manages environment profiles and auto-switching"""
    
    def __init__(self, config_file: str = "config/environment_profiles.json"):
        self.config_file = config_file
        self.current_environment: Optional[Environment] = None
        self.profiles: Dict[str, EnvironmentProfile] = {}
        self._load_profiles()
        self._detect_environment()
    
    def _load_profiles(self):
        """Load environment profiles from config"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                for env_name, profile_data in config.items():
                    self.profiles[env_name] = EnvironmentProfile(**profile_data)
            else:
                # Create default profiles
                self._create_default_profiles()
                self._save_profiles()
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            self._create_default_profiles()
    
    def _create_default_profiles(self):
        """Create default environment profiles"""
        self.profiles = {
            "sandbox": EnvironmentProfile(
                name="sandbox",
                trading_mode="sandbox",
                api_base_urls={
                    "alpaca": "https://paper-api.alpaca.markets",
                    "ibkr": "https://api.ibkr.com/v1/paper",
                    "polygon": "https://api.polygon.io"
                },
                rate_limits={
                    "alpaca": 200,
                    "ibkr": 200,
                    "polygon": 5
                },
                safety_limits={
                    "max_order_size_usd": 1000.0,
                    "daily_loss_limit_pct": 0.05,
                    "max_open_positions": 5
                },
                enable_live_trading=False,
                require_approval=False,
                logging_level="DEBUG"
            ),
            "paper": EnvironmentProfile(
                name="paper",
                trading_mode="paper",
                api_base_urls={
                    "alpaca": "https://paper-api.alpaca.markets",
                    "ibkr": "https://api.ibkr.com/v1/paper",
                    "polygon": "https://api.polygon.io"
                },
                rate_limits={
                    "alpaca": 200,
                    "ibkr": 200,
                    "polygon": 5
                },
                safety_limits={
                    "max_order_size_usd": 10000.0,
                    "daily_loss_limit_pct": 0.02,
                    "max_open_positions": 10
                },
                enable_live_trading=False,
                require_approval=False,
                logging_level="INFO"
            ),
            "live": EnvironmentProfile(
                name="live",
                trading_mode="live",
                api_base_urls={
                    "alpaca": "https://api.alpaca.markets",
                    "ibkr": "https://api.ibkr.com/v1",
                    "polygon": "https://api.polygon.io"
                },
                rate_limits={
                    "alpaca": 200,
                    "ibkr": 200,
                    "polygon": 5
                },
                safety_limits={
                    "max_order_size_usd": 50000.0,
                    "daily_loss_limit_pct": 0.01,
                    "max_open_positions": 20
                },
                enable_live_trading=True,
                require_approval=True,
                logging_level="WARNING"
            ),
            "test": EnvironmentProfile(
                name="test",
                trading_mode="sandbox",
                api_base_urls={
                    "alpaca": "https://paper-api.alpaca.markets",
                    "ibkr": "https://api.ibkr.com/v1/paper",
                    "polygon": "https://api.polygon.io"
                },
                rate_limits={
                    "alpaca": 200,
                    "ibkr": 200,
                    "polygon": 5
                },
                safety_limits={
                    "max_order_size_usd": 100.0,
                    "daily_loss_limit_pct": 0.10,
                    "max_open_positions": 2
                },
                enable_live_trading=False,
                require_approval=False,
                logging_level="DEBUG"
            )
        }
    
    def _save_profiles(self):
        """Save profiles to config file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {name: asdict(profile) for name, profile in self.profiles.items()}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
    
    def _detect_environment(self):
        """Auto-detect current environment"""
        # Check environment variable first
        env_var = os.getenv("NAE_ENVIRONMENT", "").lower()
        if env_var in self.profiles:
            self.current_environment = Environment(env_var)
            return
        
        # Check config file
        try:
            if os.path.exists("config/settings.json"):
                with open("config/settings.json", 'r') as f:
                    settings = json.load(f)
                    env_name = settings.get("environment", "sandbox")
                    if env_name in self.profiles:
                        self.current_environment = Environment(env_name)
                        return
        except:
            pass
        
        # Default to sandbox
        self.current_environment = Environment.SANDBOX
    
    def set_environment(self, env: Environment) -> bool:
        """Set current environment"""
        if env.value in self.profiles:
            self.current_environment = env
            os.environ["NAE_ENVIRONMENT"] = env.value
            logger.info(f"Environment switched to: {env.value}")
            return True
        return False
    
    def get_current_profile(self) -> Optional[EnvironmentProfile]:
        """Get current environment profile"""
        if self.current_environment:
            return self.profiles.get(self.current_environment.value)
        return None
    
    def get_profile(self, env_name: str) -> Optional[EnvironmentProfile]:
        """Get specific environment profile"""
        return self.profiles.get(env_name)
    
    def auto_switch_for_trading_mode(self, trading_mode: str) -> bool:
        """Auto-switch environment based on trading mode"""
        if trading_mode == "sandbox":
            return self.set_environment(Environment.SANDBOX)
        elif trading_mode == "paper":
            return self.set_environment(Environment.PAPER)
        elif trading_mode == "live":
            return self.set_environment(Environment.LIVE)
        return False
    
    def validate_profile(self, env_name: str) -> tuple[bool, list[str]]:
        """Validate environment profile configuration"""
        errors = []
        if env_name not in self.profiles:
            errors.append(f"Environment '{env_name}' not found")
            return False, errors
        
        profile = self.profiles[env_name]
        
        # Validate required fields
        if not profile.api_base_urls:
            errors.append("api_base_urls missing")
        if not profile.safety_limits:
            errors.append("safety_limits missing")
        
        if profile.enable_live_trading and not profile.require_approval:
            errors.append("Live trading requires approval flag")
        
        return (len(errors) == 0, errors)


# Global environment manager instance
_env_manager = None

def get_env_manager() -> EnvironmentManager:
    """Get global environment manager instance"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager

def get_current_environment() -> Environment:
    """Get current environment"""
    return get_env_manager().current_environment or Environment.SANDBOX


if __name__ == "__main__":
    # Test environment manager
    manager = EnvironmentManager()
    
    print(f"Current environment: {manager.current_environment}")
    print(f"Available profiles: {list(manager.profiles.keys())}")
    
    # Test switching
    manager.set_environment(Environment.PAPER)
    profile = manager.get_current_profile()
    print(f"Paper profile: {profile.name}, Trading mode: {profile.trading_mode}")


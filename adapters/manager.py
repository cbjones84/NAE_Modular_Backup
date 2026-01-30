# NAE/adapters/manager.py
"""
Adapter Manager - Tradier ONLY Broker Adapter

NAE uses Tradier as the EXCLUSIVE broker for all trading operations.
LIVE MODE ONLY - No paper trading or sandbox mode.
"""

from typing import Dict, Optional, List
import os
import json
import sys

# Add execution path for Tradier adapter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class AdapterManager:
    """Manages Tradier broker adapter - LIVE MODE ONLY"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize adapter manager for Tradier LIVE trading
        
        Args:
            config: Optional config dict. If None, uses Tradier defaults.
        """
        self._instances: Dict[str, object] = {}
        
        if config is None:
            config = self._load_config()
        
        self.config = config
        
    def _load_config(self) -> Dict:
        """Load adapter configuration - Tradier ONLY"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "broker_adapters.json",
        )
        config_path = os.path.abspath(config_path)
        
        # Try to load from config file
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load adapter config: {e}")
        
        # Default config - Tradier LIVE ONLY
        return {
            "default": "tradier",
            "adapters": {
                "tradier": {
                    "module": "execution.broker_adapters.tradier_adapter",
                    "class": "TradierBrokerAdapter",
                    "config": {
                        "sandbox": False,
                        "mode": "LIVE"
                    }
                }
            }
        }
    
    def get(self, name: Optional[str] = None) -> object:
        """
        Get Tradier adapter instance
        
        Args:
            name: Adapter name (ignored - always returns Tradier)
        
        Returns:
            TradierBrokerAdapter instance
        """
        # ALWAYS use Tradier - ignore any other name
        name = "tradier"
        
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]
        
        # Create Tradier adapter
        try:
            from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter
            
            # LIVE MODE - sandbox=False
            instance = TradierBrokerAdapter(sandbox=False)
            self._instances[name] = instance
            
            print(f"✅ TradierBrokerAdapter initialized in LIVE MODE")
            return instance
            
        except ImportError as e:
            raise ImportError(f"Could not import Tradier adapter: {e}")
        except Exception as e:
            raise Exception(f"Error creating Tradier adapter: {e}")
    
    def list_available(self) -> List[str]:
        """List available adapter names - Tradier only"""
        return ["tradier"]
    
    def get_default(self) -> str:
        """Get default adapter name - always Tradier"""
        return "tradier"
    
    def get_tradier(self) -> object:
        """Convenience method to get Tradier adapter"""
        return self.get("tradier")


# Global adapter manager instance
_manager: Optional[AdapterManager] = None

def get_adapter_manager() -> AdapterManager:
    """Get global adapter manager instance"""
    global _manager
    if _manager is None:
        _manager = AdapterManager()
    return _manager

def get_tradier_adapter():
    """Convenience function to get Tradier adapter directly"""
    return get_adapter_manager().get_tradier()

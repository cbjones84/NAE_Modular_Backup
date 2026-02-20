# NAE/env_loader.py
"""
Environment Variables Auto-Loader
Automatically loads API keys from multiple sources with fallback chain
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple

class EnvLoader:
    """Auto-load environment variables from multiple sources"""
    
    def __init__(self):
        self.env_file = Path(".env")
        self.api_keys_file = Path("config/api_keys.json")
        self.vault_file = Path("config/.vault.encrypted")
        self._load_all()
    
    def _load_all(self):
        """Load environment variables from all sources"""
        # 1. Check current environment
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        # 2. If not in env, try .env file
        if not openai_key or not anthropic_key:
            self._load_from_env_file()
            openai_key = os.getenv('OPENAI_API_KEY') or openai_key
            anthropic_key = os.getenv('ANTHROPIC_API_KEY') or anthropic_key
        
        # 3. If still not found, try vault
        if not openai_key or not anthropic_key:
            vault_openai, vault_anthropic = self._load_from_vault()
            if not openai_key and vault_openai:
                os.environ['OPENAI_API_KEY'] = vault_openai
                openai_key = vault_openai
            if not anthropic_key and vault_anthropic:
                os.environ['ANTHROPIC_API_KEY'] = vault_anthropic
                anthropic_key = vault_anthropic
        
        # 4. If still not found, try api_keys.json (for backwards compatibility)
        if not openai_key or not anthropic_key:
            json_openai, json_anthropic = self._load_from_json()
            if not openai_key and json_openai:
                os.environ['OPENAI_API_KEY'] = json_openai
                openai_key = json_openai
            if not anthropic_key and json_anthropic:
                os.environ['ANTHROPIC_API_KEY'] = json_anthropic
                anthropic_key = json_anthropic
    
    def _load_from_env_file(self):
        """Load from .env file"""
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            if value and value not in ['your-openai-key-here', 'your-anthropic-key-here']:
                                os.environ[key] = value
            except Exception:
                pass
    
    def _load_from_vault(self) -> Tuple[Optional[str], Optional[str]]:
        """Try to load from secure vault"""
        try:
            from secure_vault import get_vault
            vault = get_vault()
            openai_key = vault.get_secret("openai", "api_key")
            anthropic_key = vault.get_secret("anthropic", "api_key")
            return openai_key, anthropic_key
        except Exception:
            return None, None
    
    def _load_from_json(self) -> Tuple[Optional[str], Optional[str]]:
        """Try to load from api_keys.json"""
        try:
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r') as f:
                    data = json.load(f)
                    openai_key = data.get('openai', {}).get('api_key')
                    anthropic_key = data.get('anthropic', {}).get('api_key')
                    # Filter out placeholders
                    if openai_key and 'YOUR_' not in openai_key:
                        return openai_key, anthropic_key if anthropic_key and 'YOUR_' not in anthropic_key else None
                    return None, None
        except Exception:
            pass
        return None, None
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return os.getenv('OPENAI_API_KEY')
    
    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        return os.getenv('ANTHROPIC_API_KEY')
    
    def status(self) -> dict:
        """Get status of API keys"""
        return {
            'OPENAI_API_KEY': '✅ Set' if self.get_openai_key() else '❌ Not set',
            'ANTHROPIC_API_KEY': '✅ Set' if self.get_anthropic_key() else '❌ Not set',
            'NAE_ENVIRONMENT': os.getenv('NAE_ENVIRONMENT', 'sandbox (default)')
        }


# Global loader instance
_loader = None

def get_env_loader() -> EnvLoader:
    """Get global env loader instance"""
    global _loader
    if _loader is None:
        _loader = EnvLoader()
    return _loader


# Auto-load on import
if __name__ != "__main__":
    get_env_loader()

if __name__ == "__main__":
    loader = EnvLoader()
    status = loader.status()
    print("Environment Variables Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")



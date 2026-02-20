# NAE/secure_vault.py
"""
Secure Key Vault System with Encryption
Provides secure storage and retrieval of API keys and sensitive data
"""

import os
import json
import base64
import hashlib
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import getpass

class SecureVault:
    """Secure vault for storing encrypted API keys and sensitive data"""
    
    def __init__(self, vault_file: str = "config/.vault.encrypted", master_key_file: str = "config/.master.key"):
        self.vault_file = vault_file
        self.master_key_file = master_key_file
        self.cipher = None
        self._initialize_vault()
    
    def _initialize_vault(self):
        """Initialize or load vault encryption"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.vault_file), exist_ok=True)
            
            # Try to load existing master key
            if os.path.exists(self.master_key_file):
                with open(self.master_key_file, 'rb') as f:
                    master_key = f.read()
            else:
                # Generate new master key from password
                # Try environment variable first, then interactive input, then default
                password = os.getenv('NAE_VAULT_PASSWORD', None)
                
                if password is None:
                    try:
                        password = getpass.getpass("Enter vault master password (or press Enter for default): ")
                    except (EOFError, OSError):
                        # Non-interactive environment, use default
                        password = None
                
                if not password:
                    password = "NAE_DEFAULT_MASTER_KEY_CHANGE_IN_PRODUCTION"  # CHANGE IN PRODUCTION
                
                # Derive key from password
                salt = b'nae_vault_salt_2025'  # Should be random in production
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                master_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                
                # Save master key (in production, use proper key management)
                with open(self.master_key_file, 'wb') as f:
                    f.write(master_key)
            
            self.cipher = Fernet(master_key)
            
            # Initialize vault if it doesn't exist
            if not os.path.exists(self.vault_file):
                self._save_vault({})
                
        except Exception as e:
            print(f"Error initializing vault: {e}")
            raise
    
    def _load_vault(self) -> Dict[str, Any]:
        """Load encrypted vault data"""
        try:
            if os.path.exists(self.vault_file):
                with open(self.vault_file, 'rb') as f:
                    encrypted_data = f.read()
                
                if encrypted_data:
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    return json.loads(decrypted_data.decode())
            return {}
        except Exception as e:
            print(f"Error loading vault: {e}")
            return {}
    
    def _save_vault(self, data: Dict[str, Any]):
        """Save encrypted vault data"""
        try:
            json_data = json.dumps(data).encode()
            encrypted_data = self.cipher.encrypt(json_data)
            with open(self.vault_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            print(f"Error saving vault: {e}")
            raise
    
    def set_secret(self, path: str, key: str, value: str) -> bool:
        """Store a secret in the vault"""
        try:
            vault_data = self._load_vault()
            
            if path not in vault_data:
                vault_data[path] = {}
            
            vault_data[path][key] = value
            self._save_vault(vault_data)
            return True
        except Exception as e:
            print(f"Error setting secret: {e}")
            return False
    
    def get_secret(self, path: str, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret from the vault"""
        try:
            vault_data = self._load_vault()
            return vault_data.get(path, {}).get(key, default)
        except Exception as e:
            print(f"Error getting secret: {e}")
            return default
    
    def delete_secret(self, path: str, key: str) -> bool:
        """Delete a secret from the vault"""
        try:
            vault_data = self._load_vault()
            if path in vault_data and key in vault_data[path]:
                del vault_data[path][key]
                self._save_vault(vault_data)
                return True
            return False
        except Exception as e:
            print(f"Error deleting secret: {e}")
            return False
    
    def list_secrets(self, path: Optional[str] = None) -> Dict[str, Any]:
        """List all secrets or secrets in a path"""
        try:
            vault_data = self._load_vault()
            if path:
                return vault_data.get(path, {})
            return vault_data
        except Exception as e:
            print(f"Error listing secrets: {e}")
            return {}
    
    def migrate_from_json(self, json_file: str) -> int:
        """Migrate secrets from plain JSON to encrypted vault"""
        try:
            if not os.path.exists(json_file):
                print(f"JSON file not found: {json_file}")
                return 0
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            migrated = 0
            for path, secrets in data.items():
                if isinstance(secrets, dict):
                    for key, value in secrets.items():
                        if isinstance(value, str) and not value.startswith('YOUR_'):
                            self.set_secret(path, key, value)
                            migrated += 1
            
            print(f"Migrated {migrated} secrets to vault")
            return migrated
        except Exception as e:
            print(f"Error migrating secrets: {e}")
            return 0


# Global vault instance
_vault_instance = None

def get_vault() -> SecureVault:
    """Get global vault instance"""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = SecureVault()
    return _vault_instance


if __name__ == "__main__":
    # Test vault functionality
    vault = SecureVault()
    
    # Test set/get
    vault.set_secret("test", "api_key", "test_value_123")
    value = vault.get_secret("test", "api_key")
    print(f"Retrieved value: {value}")
    
    # Migrate from existing JSON
    json_file = "config/api_keys.json"
    if os.path.exists(json_file):
        vault.migrate_from_json(json_file)


#!/usr/bin/env python3
"""
Anthropic API Key Retriever
Retrieves API key from Anthropic API using admin key
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def retrieve_anthropic_key(admin_key: str = None) -> str:
    """Retrieve Anthropic API key using admin key"""
    
    # Get admin key from environment or parameter
    if not admin_key:
        admin_key = os.getenv('ANTHROPIC_ADMIN_KEY')
    
    if not admin_key:
        print("❌ ANTHROPIC_ADMIN_KEY not found")
        print("   Set it with: export ANTHROPIC_ADMIN_KEY='your-admin-key'")
        return None
    
    # API key ID from the curl command
    api_key_id = "apikey_01Rj2N8SVvo6BePZj99NhmiT"
    
    # Build curl command
    url = f"https://api.anthropic.com/v1/organizations/api_keys/{api_key_id}"
    
    curl_command = [
        'curl',
        url,
        '--header', 'anthropic-version: 2023-06-01',
        '--header', 'content-type: application/json',
        '--header', f'x-api-key: {admin_key}'
    ]
    
    try:
        # Execute curl command
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ Error retrieving API key: {result.stderr}")
            return None
        
        # Parse response
        try:
            response_data = json.loads(result.stdout)
            
            # Check various possible fields for the API key
            api_key = (
                response_data.get('api_key') or 
                response_data.get('key') or 
                response_data.get('value') or
                response_data.get('secret') or
                response_data.get('secret_key')
            )
            
            # If full key not found, check for partial hint
            if not api_key:
                partial_hint = response_data.get('partial_key_hint')
                if partial_hint:
                    print(f"⚠️  API returned partial key hint: {partial_hint}")
                    print("   Note: Anthropic only shows full key when first created.")
                    print("   The full key is not returned by the API for security.")
                    print("   You need to use the key value you saved when it was created.")
                    return None
            
            if api_key:
                return api_key
            else:
                print(f"⚠️  Response format: {json.dumps(response_data, indent=2)}")
                print("   Note: Anthropic API only shows full key when first created.")
                print("   If you have the full key, add it manually to .env file.")
                return None
                
        except json.JSONDecodeError:
            # Maybe it's just the key itself?
            api_key = result.stdout.strip()
            if api_key and api_key.startswith('sk-ant-'):
                return api_key
            else:
                print(f"⚠️  Unexpected response: {result.stdout}")
                return None
                
    except subprocess.TimeoutExpired:
        print("❌ Request timed out")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def save_to_env_file(api_key: str):
    """Save API key to .env file"""
    env_file = Path(".env")
    
    # Read existing content
    lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Update or add ANTHROPIC_API_KEY
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith('ANTHROPIC_API_KEY='):
            new_lines.append(f'ANTHROPIC_API_KEY={api_key}\n')
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        new_lines.append(f'ANTHROPIC_API_KEY={api_key}\n')
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    return True

def save_to_vault(api_key: str):
    """Save API key to secure vault"""
    try:
        from secure_vault import get_vault
        vault = get_vault()
        vault.set_secret("anthropic", "api_key", api_key)
        return True
    except Exception as e:
        print(f"⚠️  Could not save to vault: {e}")
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("ANTHROPIC API KEY RETRIEVER")
    print("=" * 70)
    print()
    
    # Check for admin key
    admin_key = os.getenv('ANTHROPIC_ADMIN_KEY')
    if not admin_key:
        print("⚠️  ANTHROPIC_ADMIN_KEY not set in environment")
        print()
        print("Set it with:")
        print("  export ANTHROPIC_ADMIN_KEY='your-admin-key'")
        print()
        print("Then run this script again")
        return
    
    print("1. Retrieving API key from Anthropic API...")
    api_key = retrieve_anthropic_key(admin_key)
    
    if not api_key:
        print("❌ Failed to retrieve API key")
        return
    
    print(f"✅ API key retrieved: {api_key[:20]}...{api_key[-10:]}")
    print()
    
    # Save to .env file
    print("2. Saving to .env file...")
    if save_to_env_file(api_key):
        print("✅ Saved to .env file")
    else:
        print("❌ Failed to save to .env file")
    print()
    
    # Save to vault
    print("3. Saving to secure vault...")
    if save_to_vault(api_key):
        print("✅ Saved to secure vault")
    else:
        print("⚠️  Could not save to vault (non-critical)")
    print()
    
    # Set environment variable
    print("4. Setting environment variable...")
    os.environ['ANTHROPIC_API_KEY'] = api_key
    print("✅ Set in current session")
    print()
    
    # Verify
    print("5. Verifying...")
    from env_loader import get_env_loader
    loader = get_env_loader()
    status = loader.status()
    print(f"   {status['ANTHROPIC_API_KEY']}")
    print()
    
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    print("✅ Anthropic API key has been:")
    print("   - Retrieved from Anthropic API")
    print("   - Saved to .env file")
    print("   - Saved to secure vault")
    print("   - Set in current environment")
    print()
    print("💡 To use in future sessions:")
    print("   source export_env.sh")
    print("=" * 70)

if __name__ == "__main__":
    main()


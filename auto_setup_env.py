#!/usr/bin/env python3
"""
NAE Environment Variables Auto-Setup
Automatically checks and configures environment variables
"""

import os
import sys
import json
from pathlib import Path

def check_vault_for_keys():
    """Check if API keys exist in vault"""
    try:
        from secure_vault import get_vault
        vault = get_vault()
        
        # Check for OpenAI key
        openai_key = vault.get_secret("openai", "api_key")
        anthropic_key = vault.get_secret("anthropic", "api_key")
        
        return openai_key, anthropic_key
    except Exception as e:
        return None, None

def check_env_file():
    """Check if .env file exists and has keys"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            has_openai = 'OPENAI_API_KEY=' in content and 'your-openai-key-here' not in content
            has_anthropic = 'ANTHROPIC_API_KEY=' in content and 'your-anthropic-key-here' not in content
            return has_openai, has_anthropic, env_file
    return False, False, None

def check_current_env():
    """Check current environment variables"""
    openai = os.getenv('OPENAI_API_KEY')
    anthropic = os.getenv('ANTHROPIC_API_KEY')
    return openai is not None and openai != '', anthropic is not None and anthropic != ''

def update_env_file(openai_key=None, anthropic_key=None):
    """Update .env file with keys"""
    env_file = Path(".env")
    
    # Read existing content
    lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Update or add keys
    updated_openai = False
    updated_anthropic = False
    
    new_lines = []
    for line in lines:
        if line.startswith('OPENAI_API_KEY='):
            if openai_key:
                new_lines.append(f'OPENAI_API_KEY={openai_key}\n')
                updated_openai = True
            else:
                new_lines.append(line)
        elif line.startswith('ANTHROPIC_API_KEY='):
            if anthropic_key:
                new_lines.append(f'ANTHROPIC_API_KEY={anthropic_key}\n')
                updated_anthropic = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add if not present
    if openai_key and not updated_openai:
        new_lines.append(f'OPENAI_API_KEY={openai_key}\n')
    if anthropic_key and not updated_anthropic:
        new_lines.append(f'ANTHROPIC_API_KEY={anthropic_key}\n')
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    return updated_openai or (openai_key is not None), updated_anthropic or (anthropic_key is not None)

def create_shell_script():
    """Create shell script to export variables"""
    script_content = """#!/bin/bash
# NAE Environment Variables Export Script
# Source this file: source export_env.sh

# Check if .env exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found"
fi
"""
    script_file = Path("export_env.sh")
    with open(script_file, 'w') as f:
        f.write(script_content)
    script_file.chmod(0o755)
    return script_file

def main():
    """Main setup function"""
    print("=" * 70)
    print("NAE ENVIRONMENT VARIABLES AUTO-SETUP")
    print("=" * 70)
    print()
    
    # Check current environment
    print("1. Checking current environment variables...")
    env_openai, env_anthropic = check_current_env()
    print(f"   OPENAI_API_KEY: {'✅ Set' if env_openai else '❌ Not set'}")
    print(f"   ANTHROPIC_API_KEY: {'✅ Set' if env_anthropic else '❌ Not set'}")
    print()
    
    if env_openai and env_anthropic:
        print("✅ Both API keys are already set in environment!")
        return
    
    # Check vault
    print("2. Checking secure vault for API keys...")
    vault_openai, vault_anthropic = check_vault_for_keys()
    if vault_openai or vault_anthropic:
        print(f"   Found in vault: OpenAI={'✅' if vault_openai else '❌'}, Anthropic={'✅' if vault_anthropic else '❌'}")
    else:
        print("   No keys found in vault")
    print()
    
    # Check .env file
    print("3. Checking .env file...")
    env_file_openai, env_file_anthropic, env_file = check_env_file()
    if env_file:
        print(f"   .env file exists: OpenAI={'✅' if env_file_openai else '❌'}, Anthropic={'✅' if env_file_anthropic else '❌'}")
    else:
        print("   No .env file found")
    print()
    
    # Update .env file with vault keys if available
    updated = False
    if vault_openai or vault_anthropic:
        print("4. Updating .env file with keys from vault...")
        updated_openai, updated_anthropic = update_env_file(
            vault_openai if vault_openai and not env_openai else None,
            vault_anthropic if vault_anthropic and not env_anthropic else None
        )
        if updated_openai or updated_anthropic:
            print(f"   ✅ Updated .env file")
            updated = True
        print()
    
    # Create export script
    print("5. Creating export script...")
    export_script = create_shell_script()
    print(f"   ✅ Created: {export_script}")
    print()
    
    # Final status
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    
    if env_openai and env_anthropic:
        print("✅ Both API keys are set in environment!")
        print("   No further action needed.")
    elif vault_openai and vault_anthropic:
        print("✅ API keys found in vault and added to .env")
        print()
        print("To use them, run:")
        print(f"   source {export_script}")
        print("   # or")
        print("   export $(cat .env | grep -v '^#' | xargs)")
    else:
        print("⚠️  API keys not found in vault or environment")
        print()
        print("To set them manually:")
        print("   1. Edit .env file:")
        print("      OPENAI_API_KEY=your-openai-key")
        print("      ANTHROPIC_API_KEY=your-anthropic-key")
        print()
        print("   2. Or set environment variables:")
        print("      export OPENAI_API_KEY='your-key'")
        print("      export ANTHROPIC_API_KEY='your-key'")
        print()
        print("   3. Then load .env:")
        print(f"      source {export_script}")
        print()
        print("To get API keys:")
        print("   OpenAI: https://platform.openai.com/api-keys")
        print("   Anthropic: https://console.anthropic.com/")
    
    print("=" * 70)

if __name__ == "__main__":
    main()



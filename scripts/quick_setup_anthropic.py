#!/usr/bin/env python3
"""
Quick Anthropic API Key Setup
Add your full API key directly
"""

import os
import sys
from pathlib import Path

def setup_anthropic_key():
    """Quick setup for Anthropic API key"""
    print("=" * 70)
    print("ANTHROPIC API KEY QUICK SETUP")
    print("=" * 70)
    print()
    print("From your API response:")
    print("  Partial Hint: sk-ant-api03-R2D...igAA")
    print("  Status: Active ✅")
    print()
    print("⚠️  The full key is NOT in the API response.")
    print("   You need to use the key you saved when creating it.")
    print()
    
    # Ask for the key
    api_key = input("Enter your full Anthropic API key (starts with sk-ant-api03-R2D...): ").strip()
    
    if not api_key:
        print("❌ No key provided")
        return False
    
    if not api_key.startswith('sk-ant-'):
        print("⚠️  Warning: Key doesn't start with 'sk-ant-'")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    # Save to .env file
    print()
    print("Saving to .env file...")
    env_file = Path(".env")
    
    lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
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
    
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Saved to .env file")
    
    # Save to vault
    print("Saving to secure vault...")
    try:
        from secure_vault import get_vault
        vault = get_vault()
        vault.set_secret("anthropic", "api_key", api_key)
        print("✅ Saved to secure vault")
    except Exception as e:
        print(f"⚠️  Could not save to vault: {e}")
    
    # Set environment variable
    os.environ['ANTHROPIC_API_KEY'] = api_key
    print("✅ Set in current environment")
    
    # Verify
    print()
    print("Verifying...")
    try:
        from env_loader import get_env_loader
        loader = get_env_loader()
        status = loader.status()
        print(f"  {status['ANTHROPIC_API_KEY']}")
    except Exception as e:
        print(f"  Verification error: {e}")
    
    print()
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    print("✅ Anthropic API key configured!")
    print("   The system will automatically find it from .env file")
    return True

if __name__ == "__main__":
    setup_anthropic_key()



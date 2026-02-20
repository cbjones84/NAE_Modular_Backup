#!/usr/bin/env python3
"""
NAE Secure Line - Telegram Setup Script
========================================
Interactive setup for Telegram bot integration.

This script will:
1. Guide you through creating a Telegram bot
2. Help you get your chat ID
3. Test the connection
4. Save configuration
"""

import os
import sys
import requests
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

def print_header():
    print("\n" + "="*60)
    print("🔐 NAE SECURE LINE - TELEGRAM SETUP")
    print("="*60)

def check_existing_config():
    """Check if Telegram is already configured"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if bot_token and chat_id:
        print("\n✅ Existing configuration found!")
        print(f"   Bot Token: {bot_token[:15]}...{bot_token[-5:]}")
        print(f"   Chat ID: {chat_id}")
        return True
    return False

def print_setup_instructions():
    """Print setup instructions"""
    print("""
📱 SETUP INSTRUCTIONS
=====================

STEP 1: Create a Telegram Bot
-----------------------------
1. Open Telegram on your phone
2. Search for @BotFather
3. Send: /newbot
4. Choose a name: "NAE Secure Line"
5. Choose a username: "NAE_SecureLine_bot" (must end in 'bot')
6. Copy the API token BotFather gives you

STEP 2: Get Your Chat ID  
-------------------------
1. Message your new bot (just say "hi")
2. Search for @userinfobot in Telegram
3. Start a chat with @userinfobot
4. It will show your Chat ID (a number like 123456789)

STEP 3: Set Environment Variables
---------------------------------
Open PowerShell and run:

$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

Or add to your system environment variables permanently.
""")

def test_connection(bot_token: str, chat_id: str) -> bool:
    """Test Telegram connection"""
    print("\n🔄 Testing connection...")
    
    try:
        # Test sending a message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "🔔 NAE Secure Line - Connection Test Successful!\n\nCasey is now connected to your phone.",
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        result = response.json()
        
        if result.get("ok"):
            print("✅ Connection successful! Check your Telegram for a test message.")
            return True
        else:
            print(f"❌ Connection failed: {result.get('description')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def interactive_setup():
    """Interactive setup flow"""
    print_setup_instructions()
    
    print("\n" + "-"*60)
    print("ENTER YOUR CREDENTIALS")
    print("-"*60)
    
    bot_token = input("\nPaste your Bot Token: ").strip()
    if not bot_token:
        print("❌ Bot token is required")
        return
    
    chat_id = input("Paste your Chat ID: ").strip()
    if not chat_id:
        print("❌ Chat ID is required")
        return
    
    # Test connection
    if test_connection(bot_token, chat_id):
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        
        print(f"""
To make this permanent, add to your environment:

PowerShell (temporary - current session):
$env:TELEGRAM_BOT_TOKEN = "{bot_token}"
$env:TELEGRAM_CHAT_ID = "{chat_id}"

PowerShell (permanent - for all sessions):
[System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "{bot_token}", "User")
[System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "{chat_id}", "User")

Or create a .env file in NAE Ready/:
TELEGRAM_BOT_TOKEN={bot_token}
TELEGRAM_CHAT_ID={chat_id}

To start Casey's Telegram interface:
python scripts/start_casey_telegram.py
""")
    else:
        print("\n❌ Setup failed. Please check your credentials and try again.")

def main():
    print_header()
    
    if check_existing_config():
        choice = input("\nWould you like to test the existing configuration? (y/n): ").strip().lower()
        if choice == 'y':
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            test_connection(bot_token, chat_id)
        
        choice = input("\nWould you like to set up a new configuration? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nKeeping existing configuration. Goodbye!")
            return
    
    interactive_setup()

if __name__ == "__main__":
    main()


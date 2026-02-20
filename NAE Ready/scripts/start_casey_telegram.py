#!/usr/bin/env python3
"""
Start Casey's Telegram Interface
================================
Run this script to start Casey's NAE Secure Line.

Usage:
    python scripts/start_casey_telegram.py

Prerequisites:
    - TELEGRAM_BOT_TOKEN environment variable set
    - TELEGRAM_CHAT_ID environment variable set
    
Run setup_telegram.py first if you haven't configured Telegram yet.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

def main():
    print("\n" + "🤖"*20)
    print("\n  NAE SECURE LINE - CASEY")
    print("  Two-Way Telegram Communication")
    print("\n" + "🤖"*20 + "\n")
    
    # Check environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("""
❌ TELEGRAM NOT CONFIGURED

Please set up Telegram first by running:
    python scripts/setup_telegram.py

Or set environment variables:
    $env:TELEGRAM_BOT_TOKEN = "your_token"
    $env:TELEGRAM_CHAT_ID = "your_chat_id"
""")
        sys.exit(1)
    
    print(f"✅ Bot Token: {bot_token[:10]}...{bot_token[-5:]}")
    print(f"✅ Chat ID: {chat_id}")
    print("\n" + "-"*40)
    
    # Import and start Casey
    try:
        from communication.casey_telegram_interface import CaseyTelegramInterface
        
        casey = CaseyTelegramInterface()
        
        print("""
🚀 Casey is starting up...

Available Commands (send via Telegram):
/help     - Show all commands
/status   - System status
/pnl      - P&L summary
/balance  - Account balance
/positions - Open positions
/trade    - Execute trade
/stop     - Stop trading
/emergency - Emergency stop

Press Ctrl+C to stop Casey.
""")
        
        # Start the listener (blocking)
        casey.start(blocking=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Casey is shutting down...")
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting Casey: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


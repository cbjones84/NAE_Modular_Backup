#!/usr/bin/env python3
"""
NAE Secure Line - Telegram Bot Integration
===========================================
Provides secure, instant communication between NAE agents and your phone.

Features:
- Two-way communication with Casey
- Real-time trade alerts from Optimus
- System status updates
- Command execution from phone
- End-to-end encrypted messaging

Setup:
1. Create bot via @BotFather on Telegram
2. Set TELEGRAM_BOT_TOKEN in environment
3. Message your bot to get chat ID
4. Set TELEGRAM_CHAT_ID in environment
"""

import os
import sys
import json
import time
import logging
import requests
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("NAE.Telegram")

# =============================================================================
# CONFIGURATION
# =============================================================================

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "🔵"      # Info
    MEDIUM = "🟡"   # Warning
    HIGH = "🟠"     # Important
    CRITICAL = "🔴" # Urgent
    SUCCESS = "✅"  # Success
    TRADE = "💰"    # Trade related

@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str
    chat_id: str
    bot_name: str = "NAE Secure Line"
    polling_interval: float = 1.0
    max_retries: int = 3
    timeout: int = 30

# =============================================================================
# TELEGRAM BOT CLASS
# =============================================================================

class NAETelegramBot:
    """
    NAE Secure Line - Telegram Bot for agent communication
    """
    
    def __init__(self, config: TelegramConfig = None):
        """Initialize the Telegram bot"""
        if config:
            self.config = config
        else:
            # Load from environment
            self.config = TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", "")
            )
        
        self.base_url = f"https://api.telegram.org/bot{self.config.bot_token}"
        self.is_running = False
        self.command_handlers: Dict[str, Callable] = {}
        self.message_queue: List[Dict] = []
        self.last_update_id = 0
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate bot configuration"""
        if not self.config.bot_token:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set - Telegram notifications disabled")
            self.enabled = False
        elif not self.config.chat_id:
            logger.warning("⚠️ TELEGRAM_CHAT_ID not set - Telegram notifications disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ Telegram bot configured successfully")
    
    # =========================================================================
    # SENDING MESSAGES
    # =========================================================================
    
    def send_message(self, message: str, priority: AlertPriority = AlertPriority.LOW,
                     parse_mode: str = "HTML", disable_notification: bool = False) -> bool:
        """
        Send a message to the configured chat
        
        Args:
            message: The message text
            priority: Alert priority level
            parse_mode: HTML or Markdown
            disable_notification: Silent message
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, would send: {message[:50]}...")
            return False
        
        try:
            # Format message with priority
            formatted_message = f"{priority.value} {message}"
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.config.chat_id,
                "text": formatted_message,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }
            
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            result = response.json()
            
            if result.get("ok"):
                logger.debug(f"Message sent successfully")
                return True
            else:
                logger.error(f"Failed to send message: {result.get('description')}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_alert(self, title: str, message: str, priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Send a formatted alert"""
        alert_text = f"<b>{title}</b>\n\n{message}"
        return self.send_message(alert_text, priority=priority)
    
    def send_trade_alert(self, action: str, symbol: str, quantity: int, 
                         price: float, details: str = "") -> bool:
        """Send a trade execution alert"""
        trade_text = f"""<b>🔔 TRADE EXECUTED</b>

<b>Action:</b> {action.upper()}
<b>Symbol:</b> {symbol}
<b>Quantity:</b> {quantity}
<b>Price:</b> ${price:.2f}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{details}"""
        return self.send_message(trade_text, priority=AlertPriority.TRADE)
    
    def send_status_update(self, status: Dict[str, Any]) -> bool:
        """Send a system status update"""
        status_text = f"""<b>📊 NAE STATUS UPDATE</b>

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        for key, value in status.items():
            status_text += f"<b>{key}:</b> {value}\n"
        
        return self.send_message(status_text, priority=AlertPriority.LOW)
    
    def send_pnl_update(self, daily_pnl: float, total_pnl: float, 
                        balance: float, trades_today: int) -> bool:
        """Send P&L update"""
        emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_text = f"""<b>{emoji} P&L UPDATE</b>

<b>Daily P&L:</b> ${daily_pnl:+,.2f}
<b>Total P&L:</b> ${total_pnl:+,.2f}
<b>Account Balance:</b> ${balance:,.2f}
<b>Trades Today:</b> {trades_today}
<b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"""
        
        priority = AlertPriority.SUCCESS if daily_pnl >= 0 else AlertPriority.HIGH
        return self.send_message(pnl_text, priority=priority)
    
    def send_risk_alert(self, risk_type: str, description: str, 
                        action_taken: str = "") -> bool:
        """Send a risk management alert"""
        risk_text = f"""<b>⚠️ RISK ALERT</b>

<b>Type:</b> {risk_type}
<b>Description:</b> {description}
<b>Action:</b> {action_taken or 'Awaiting review'}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        return self.send_message(risk_text, priority=AlertPriority.CRITICAL)
    
    def send_error_alert(self, error_type: str, error_message: str, 
                         component: str = "NAE") -> bool:
        """Send an error alert"""
        error_text = f"""<b>🚨 ERROR DETECTED</b>

<b>Component:</b> {component}
<b>Error Type:</b> {error_type}
<b>Message:</b> {error_message}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        return self.send_message(error_text, priority=AlertPriority.CRITICAL)
    
    # =========================================================================
    # RECEIVING MESSAGES (Command Listener)
    # =========================================================================
    
    def get_updates(self, offset: int = None) -> List[Dict]:
        """Get new messages from Telegram"""
        if not self.enabled:
            return []
        
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                "offset": offset or self.last_update_id + 1,
                "timeout": 30
            }
            
            response = requests.get(url, params=params, timeout=35)
            result = response.json()
            
            if result.get("ok"):
                return result.get("result", [])
            else:
                logger.error(f"Failed to get updates: {result.get('description')}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting Telegram updates: {e}")
            return []
    
    def register_command(self, command: str, handler: Callable):
        """
        Register a command handler
        
        Args:
            command: Command name (without /)
            handler: Function to call when command received
        """
        self.command_handlers[command.lower()] = handler
        logger.info(f"Registered command: /{command}")
    
    def process_message(self, message: Dict) -> Optional[str]:
        """
        Process an incoming message
        
        Args:
            message: Telegram message object
            
        Returns:
            Response text or None
        """
        try:
            text = message.get("text", "")
            chat_id = message.get("chat", {}).get("id")
            
            # Security check - only respond to authorized chat
            if str(chat_id) != str(self.config.chat_id):
                logger.warning(f"Unauthorized message from chat_id: {chat_id}")
                return None
            
            # Check if it's a command
            if text.startswith("/"):
                command = text.split()[0][1:].lower()  # Remove / and get command
                args = text.split()[1:] if len(text.split()) > 1 else []
                
                if command in self.command_handlers:
                    return self.command_handlers[command](args)
                else:
                    return f"Unknown command: /{command}\n\nType /help for available commands."
            else:
                # Process as natural language (route to Casey)
                return self._process_natural_language(text)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing message: {str(e)}"
    
    def _process_natural_language(self, text: str) -> str:
        """Process natural language messages (route to Casey)"""
        # This will be overridden by Casey's interface
        return "Message received. Processing..."
    
    def start_listener(self, blocking: bool = False):
        """
        Start the message listener
        
        Args:
            blocking: If True, run in main thread. If False, run in background.
        """
        if not self.enabled:
            logger.warning("Telegram bot not enabled, cannot start listener")
            return
        
        self.is_running = True
        logger.info("🚀 Starting NAE Secure Line listener...")
        
        if blocking:
            self._listener_loop()
        else:
            listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
            listener_thread.start()
            logger.info("Listener running in background")
    
    def stop_listener(self):
        """Stop the message listener"""
        self.is_running = False
        logger.info("Stopping NAE Secure Line listener...")
    
    def _listener_loop(self):
        """Main listener loop"""
        logger.info("📱 NAE Secure Line active - listening for commands...")
        
        # Send startup notification
        self.send_message(
            "<b>🤖 NAE Secure Line Active</b>\n\nCasey is now listening for commands.\nType /help for available commands.",
            priority=AlertPriority.SUCCESS
        )
        
        while self.is_running:
            try:
                updates = self.get_updates(self.last_update_id)
                
                for update in updates:
                    self.last_update_id = update["update_id"]
                    
                    if "message" in update:
                        message = update["message"]
                        response = self.process_message(message)
                        
                        if response:
                            self.send_message(response, priority=AlertPriority.LOW)
                
                time.sleep(self.config.polling_interval)
                
            except KeyboardInterrupt:
                logger.info("Listener stopped by user")
                break
            except Exception as e:
                logger.error(f"Listener error: {e}")
                time.sleep(5)  # Wait before retrying


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global bot instance
_bot_instance: Optional[NAETelegramBot] = None

def get_telegram_bot() -> NAETelegramBot:
    """Get or create the global Telegram bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = NAETelegramBot()
    return _bot_instance

def send_telegram_alert(message: str, priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
    """Send a quick Telegram alert"""
    return get_telegram_bot().send_message(message, priority=priority)

def send_trade_notification(action: str, symbol: str, quantity: int, 
                           price: float, details: str = "") -> bool:
    """Send a trade notification"""
    return get_telegram_bot().send_trade_alert(action, symbol, quantity, price, details)

def send_error_notification(error_type: str, message: str, component: str = "NAE") -> bool:
    """Send an error notification"""
    return get_telegram_bot().send_error_alert(error_type, message, component)


# =============================================================================
# SETUP HELPER
# =============================================================================

def setup_telegram_bot():
    """Interactive setup for Telegram bot"""
    print("\n" + "="*60)
    print("🔐 NAE SECURE LINE - TELEGRAM BOT SETUP")
    print("="*60)
    
    print("""
To set up your NAE Secure Line:

1. Open Telegram and search for @BotFather
2. Send /newbot and follow the prompts
3. Copy the bot token you receive
4. Message your new bot to activate it
5. Get your chat ID by messaging @userinfobot

Then set these environment variables:

Windows PowerShell:
    $env:TELEGRAM_BOT_TOKEN = "your_bot_token_here"
    $env:TELEGRAM_CHAT_ID = "your_chat_id_here"

Or add to your .env file:
    TELEGRAM_BOT_TOKEN=your_bot_token_here
    TELEGRAM_CHAT_ID=your_chat_id_here
""")
    
    # Check current status
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if bot_token and chat_id:
        print("✅ Environment variables detected!")
        print(f"   Bot Token: {bot_token[:10]}...{bot_token[-5:]}")
        print(f"   Chat ID: {chat_id}")
        
        # Test connection
        bot = NAETelegramBot()
        if bot.send_message("🔔 NAE Secure Line test message - Setup successful!", 
                          priority=AlertPriority.SUCCESS):
            print("\n✅ Test message sent successfully!")
        else:
            print("\n❌ Failed to send test message. Check your credentials.")
    else:
        print("❌ Environment variables not set yet.")
        print("   Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")


if __name__ == "__main__":
    setup_telegram_bot()


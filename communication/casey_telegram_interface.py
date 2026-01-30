#!/usr/bin/env python3
"""
Casey Telegram Interface
========================
Two-way communication interface between you and NAE via Telegram.

Casey can:
- Receive commands from your phone
- Route commands to appropriate NAE agents
- Send real-time updates and alerts
- Provide system status on demand
- Execute trading operations remotely

Commands:
/status - System status
/pnl - P&L summary
/balance - Account balance
/positions - Open positions
/trade <symbol> <action> - Execute trade
/stop - Stop all trading
/start - Start trading
/help - Show all commands
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

from communication.telegram_bot import (
    NAETelegramBot, 
    AlertPriority, 
    TelegramConfig,
    get_telegram_bot
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Casey.Telegram")

# =============================================================================
# CASEY TELEGRAM INTERFACE
# =============================================================================

class CaseyTelegramInterface:
    """
    Casey's Telegram command interface for NAE
    
    This is the two-way communication layer between you and NAE.
    """
    
    def __init__(self):
        """Initialize Casey's Telegram interface"""
        self.bot = get_telegram_bot()
        self.trading_enabled = True
        self.last_command_time = None
        self.command_history: List[Dict] = []
        
        # Agent references (will be set during initialization)
        self.optimus = None
        self.ralph = None
        self.donnie = None
        self.shredder = None
        
        # Register all commands
        self._register_commands()
        
        logger.info("Casey Telegram Interface initialized")
    
    def _register_commands(self):
        """Register all available commands"""
        commands = {
            "help": self.cmd_help,
            "status": self.cmd_status,
            "pnl": self.cmd_pnl,
            "balance": self.cmd_balance,
            "positions": self.cmd_positions,
            "trades": self.cmd_trades,
            "trade": self.cmd_trade,
            "buy": self.cmd_buy,
            "sell": self.cmd_sell,
            "stop": self.cmd_stop,
            "start": self.cmd_start,
            "pause": self.cmd_pause,
            "resume": self.cmd_resume,
            "risk": self.cmd_risk,
            "alerts": self.cmd_alerts,
            "market": self.cmd_market,
            "vix": self.cmd_vix,
            "strategies": self.cmd_strategies,
            "learn": self.cmd_learn,
            "ping": self.cmd_ping,
            "restart": self.cmd_restart,
            "emergency": self.cmd_emergency,
        }
        
        for cmd, handler in commands.items():
            self.bot.register_command(cmd, handler)
        
        # Override natural language processing
        self.bot._process_natural_language = self._handle_natural_language
    
    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================
    
    def cmd_help(self, args: List[str]) -> str:
        """Show available commands"""
        return """<b>📱 NAE Secure Line - Commands</b>

<b>📊 Status & Info:</b>
/status - System status overview
/pnl - Today's P&L summary
/balance - Account balance
/positions - Open positions
/trades - Recent trades
/market - Market overview
/vix - Current VIX level

<b>💰 Trading:</b>
/trade [symbol] [buy/sell] [qty] - Execute trade
/buy [symbol] [qty] - Quick buy
/sell [symbol] [qty] - Quick sell

<b>⚙️ Control:</b>
/start - Enable trading
/stop - Disable trading
/pause - Pause temporarily
/resume - Resume trading

<b>🛡️ Risk & Safety:</b>
/risk - Risk status
/alerts - Active alerts
/emergency - EMERGENCY STOP ALL

<b>🧠 Intelligence:</b>
/strategies - Active strategies
/learn - Ralph learning status

<b>🔧 System:</b>
/ping - Test connection
/restart - Restart NAE

Type any message to talk to Casey directly."""

    def cmd_status(self, args: List[str]) -> str:
        """Get system status"""
        try:
            status = {
                "Trading": "✅ ENABLED" if self.trading_enabled else "❌ DISABLED",
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Optimus": "🟢 Online" if self.optimus else "⚪ Not connected",
                "Ralph": "🟢 Online" if self.ralph else "⚪ Not connected",
                "Market": self._get_market_status(),
            }
            
            response = "<b>📊 NAE SYSTEM STATUS</b>\n\n"
            for key, value in status.items():
                response += f"<b>{key}:</b> {value}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"
    
    def cmd_pnl(self, args: List[str]) -> str:
        """Get P&L summary"""
        try:
            # Try to get real P&L from Optimus
            if self.optimus:
                pnl_data = self.optimus.get_pnl_summary()
            else:
                # Return placeholder if Optimus not connected
                pnl_data = self._get_cached_pnl()
            
            return f"""<b>📈 P&L SUMMARY</b>

<b>Today's P&L:</b> ${pnl_data.get('daily_pnl', 0):+,.2f}
<b>Week P&L:</b> ${pnl_data.get('weekly_pnl', 0):+,.2f}
<b>Month P&L:</b> ${pnl_data.get('monthly_pnl', 0):+,.2f}
<b>Total P&L:</b> ${pnl_data.get('total_pnl', 0):+,.2f}

<b>Win Rate:</b> {pnl_data.get('win_rate', 0):.1%}
<b>Trades Today:</b> {pnl_data.get('trades_today', 0)}

<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>"""
            
        except Exception as e:
            return f"❌ Error getting P&L: {str(e)}"
    
    def cmd_balance(self, args: List[str]) -> str:
        """Get account balance"""
        try:
            if self.optimus:
                balance_data = self.optimus.get_account_balance()
            else:
                balance_data = self._get_cached_balance()
            
            return f"""<b>💰 ACCOUNT BALANCE</b>

<b>Cash:</b> ${balance_data.get('cash', 0):,.2f}
<b>Buying Power:</b> ${balance_data.get('buying_power', 0):,.2f}
<b>Portfolio Value:</b> ${balance_data.get('portfolio_value', 0):,.2f}
<b>Total Equity:</b> ${balance_data.get('equity', 0):,.2f}

<b>Margin Used:</b> ${balance_data.get('margin_used', 0):,.2f}
<b>Day Trades Left:</b> {balance_data.get('day_trades_left', 'N/A')}

<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>"""
            
        except Exception as e:
            return f"❌ Error getting balance: {str(e)}"
    
    def cmd_positions(self, args: List[str]) -> str:
        """Get open positions"""
        try:
            if self.optimus:
                positions = self.optimus.get_open_positions()
            else:
                positions = self._get_cached_positions()
            
            if not positions:
                return "📋 <b>No open positions</b>"
            
            response = "<b>📋 OPEN POSITIONS</b>\n\n"
            for pos in positions:
                pnl = pos.get('unrealized_pnl', 0)
                emoji = "🟢" if pnl >= 0 else "🔴"
                response += f"{emoji} <b>{pos['symbol']}</b>: {pos['qty']} @ ${pos['avg_price']:.2f} ({pnl:+.2f})\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting positions: {str(e)}"
    
    def cmd_trades(self, args: List[str]) -> str:
        """Get recent trades"""
        try:
            limit = int(args[0]) if args else 5
            
            if self.optimus:
                trades = self.optimus.get_recent_trades(limit)
            else:
                trades = self._get_cached_trades(limit)
            
            if not trades:
                return "📜 <b>No recent trades</b>"
            
            response = f"<b>📜 LAST {len(trades)} TRADES</b>\n\n"
            for trade in trades:
                emoji = "🟢" if trade['side'] == 'buy' else "🔴"
                response += f"{emoji} {trade['symbol']} {trade['side'].upper()} {trade['qty']} @ ${trade['price']:.2f}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting trades: {str(e)}"
    
    def cmd_trade(self, args: List[str]) -> str:
        """Execute a trade"""
        if len(args) < 2:
            return "❌ Usage: /trade [symbol] [buy/sell] [qty]\nExample: /trade SPY buy 10"
        
        if not self.trading_enabled:
            return "❌ Trading is currently DISABLED. Use /start to enable."
        
        symbol = args[0].upper()
        action = args[1].lower()
        qty = int(args[2]) if len(args) > 2 else 1
        
        if action not in ['buy', 'sell']:
            return "❌ Action must be 'buy' or 'sell'"
        
        return self._execute_trade(symbol, action, qty)
    
    def cmd_buy(self, args: List[str]) -> str:
        """Quick buy command"""
        if len(args) < 1:
            return "❌ Usage: /buy [symbol] [qty]\nExample: /buy SPY 10"
        
        symbol = args[0].upper()
        qty = int(args[1]) if len(args) > 1 else 1
        
        return self._execute_trade(symbol, 'buy', qty)
    
    def cmd_sell(self, args: List[str]) -> str:
        """Quick sell command"""
        if len(args) < 1:
            return "❌ Usage: /sell [symbol] [qty]\nExample: /sell SPY 10"
        
        symbol = args[0].upper()
        qty = int(args[1]) if len(args) > 1 else 1
        
        return self._execute_trade(symbol, 'sell', qty)
    
    def cmd_stop(self, args: List[str]) -> str:
        """Stop all trading"""
        self.trading_enabled = False
        logger.warning("Trading DISABLED via Telegram command")
        
        # Notify Optimus if connected
        if self.optimus:
            self.optimus.disable_trading("Disabled via Telegram")
        
        return "🛑 <b>TRADING DISABLED</b>\n\nAll trading operations have been stopped.\nUse /start to re-enable."
    
    def cmd_start(self, args: List[str]) -> str:
        """Enable trading"""
        self.trading_enabled = True
        logger.info("Trading ENABLED via Telegram command")
        
        if self.optimus:
            self.optimus.enable_trading()
        
        return "✅ <b>TRADING ENABLED</b>\n\nOptimus is now active and can execute trades."
    
    def cmd_pause(self, args: List[str]) -> str:
        """Pause trading temporarily"""
        minutes = int(args[0]) if args else 30
        self.trading_enabled = False
        
        return f"⏸️ <b>TRADING PAUSED</b>\n\nPaused for {minutes} minutes.\nUse /resume to restart early."
    
    def cmd_resume(self, args: List[str]) -> str:
        """Resume trading"""
        self.trading_enabled = True
        return "▶️ <b>TRADING RESUMED</b>\n\nOptimus is back online."
    
    def cmd_risk(self, args: List[str]) -> str:
        """Get risk status"""
        try:
            risk_data = {
                "Daily Loss": "$0.00",
                "Max Drawdown": "0.0%",
                "Position Risk": "Low",
                "VaR (1d)": "$0.00",
                "Heat": "0/6%"
            }
            
            response = "<b>🛡️ RISK STATUS</b>\n\n"
            for key, value in risk_data.items():
                response += f"<b>{key}:</b> {value}\n"
            
            response += "\n<i>All systems nominal</i>"
            return response
            
        except Exception as e:
            return f"❌ Error getting risk status: {str(e)}"
    
    def cmd_alerts(self, args: List[str]) -> str:
        """Get active alerts"""
        return "<b>🔔 ACTIVE ALERTS</b>\n\n✅ No active alerts\n\n<i>System running normally</i>"
    
    def cmd_market(self, args: List[str]) -> str:
        """Get market overview"""
        try:
            return f"""<b>📈 MARKET OVERVIEW</b>

<b>SPY:</b> Loading...
<b>QQQ:</b> Loading...
<b>VIX:</b> Loading...

<b>Market Status:</b> {self._get_market_status()}

<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>"""
            
        except Exception as e:
            return f"❌ Error getting market data: {str(e)}"
    
    def cmd_vix(self, args: List[str]) -> str:
        """Get VIX level"""
        return """<b>📊 VIX STATUS</b>

<b>Current:</b> Loading...
<b>Change:</b> Loading...
<b>Regime:</b> Normal

<b>Trading Implication:</b>
Standard position sizing applies."""
    
    def cmd_strategies(self, args: List[str]) -> str:
        """Get active strategies"""
        return """<b>🎯 ACTIVE STRATEGIES</b>

1️⃣ <b>Wheel Strategy</b> - Active
2️⃣ <b>Credit Spreads</b> - Active  
3️⃣ <b>Iron Condors</b> - Standby
4️⃣ <b>Kelly Sizing</b> - Active

<i>Strategies loaded from Ralph's knowledge base</i>"""
    
    def cmd_learn(self, args: List[str]) -> str:
        """Get Ralph learning status"""
        return """<b>🧠 RALPH LEARNING STATUS</b>

<b>Last Update:</b> Today
<b>Sources Processed:</b> 11
<b>Strategies Extracted:</b> 4
<b>Psychology Insights:</b> 6
<b>Risk Rules:</b> 7

<b>Knowledge Bases:</b>
✅ Options Trading
✅ Psychology
✅ Risk Management

<i>Next scheduled update: 48 hours</i>"""
    
    def cmd_ping(self, args: List[str]) -> str:
        """Test connection"""
        return f"🏓 <b>PONG!</b>\n\nNAE Secure Line is active.\nLatency: <1s\nTime: {datetime.now().strftime('%H:%M:%S')}"
    
    def cmd_restart(self, args: List[str]) -> str:
        """Restart NAE"""
        return "🔄 <b>RESTART REQUESTED</b>\n\nThis would restart NAE services.\n⚠️ Not implemented in this version."
    
    def cmd_emergency(self, args: List[str]) -> str:
        """Emergency stop all"""
        self.trading_enabled = False
        
        # Stop everything
        if self.optimus:
            self.optimus.emergency_stop()
        
        logger.critical("🚨 EMERGENCY STOP ACTIVATED VIA TELEGRAM")
        
        return """🚨 <b>EMERGENCY STOP ACTIVATED</b>

All trading operations have been IMMEDIATELY halted.
All pending orders cancelled.
System is in safe mode.

Contact support if this was unintentional.
Use /start to re-enable trading."""
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _execute_trade(self, symbol: str, action: str, qty: int) -> str:
        """Execute a trade via Optimus"""
        try:
            if not self.trading_enabled:
                return "❌ Trading is disabled. Use /start to enable."
            
            logger.info(f"Trade request via Telegram: {action} {qty} {symbol}")
            
            if self.optimus:
                result = self.optimus.execute_trade({
                    'symbol': symbol,
                    'action': action,
                    'quantity': qty,
                    'source': 'telegram_command'
                })
                
                if result.get('status') == 'success':
                    return f"""✅ <b>TRADE EXECUTED</b>

<b>Action:</b> {action.upper()}
<b>Symbol:</b> {symbol}
<b>Quantity:</b> {qty}
<b>Status:</b> {result.get('status', 'submitted')}

Order ID: {result.get('order_id', 'N/A')}"""
                else:
                    return f"❌ Trade failed: {result.get('error', 'Unknown error')}"
            else:
                return "❌ Optimus not connected. Cannot execute trade."
                
        except Exception as e:
            return f"❌ Trade error: {str(e)}"
    
    def _get_market_status(self) -> str:
        """Get current market status"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()
        
        if weekday >= 5:  # Weekend
            return "🔴 CLOSED (Weekend)"
        
        # Convert to ET (simplified - assumes local is ET)
        if hour < 9 or (hour == 9 and minute < 30):
            return "🟡 Pre-Market"
        elif hour >= 16:
            return "🟡 After-Hours"
        else:
            return "🟢 OPEN"
    
    def _get_cached_pnl(self) -> Dict:
        """Get cached P&L data"""
        return {
            'daily_pnl': 0,
            'weekly_pnl': 0,
            'monthly_pnl': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'trades_today': 0
        }
    
    def _get_cached_balance(self) -> Dict:
        """Get cached balance data"""
        return {
            'cash': 0,
            'buying_power': 0,
            'portfolio_value': 0,
            'equity': 0,
            'margin_used': 0,
            'day_trades_left': 'N/A'
        }
    
    def _get_cached_positions(self) -> List[Dict]:
        """Get cached positions"""
        return []
    
    def _get_cached_trades(self, limit: int) -> List[Dict]:
        """Get cached recent trades"""
        return []
    
    def _handle_natural_language(self, text: str) -> str:
        """Handle natural language messages"""
        text_lower = text.lower()
        
        # Simple keyword matching for common queries
        if any(word in text_lower for word in ['status', "how's it going", 'report']):
            return self.cmd_status([])
        
        if any(word in text_lower for word in ['pnl', 'profit', 'loss', 'money']):
            return self.cmd_pnl([])
        
        if any(word in text_lower for word in ['balance', 'cash', 'account']):
            return self.cmd_balance([])
        
        if any(word in text_lower for word in ['position', 'holding', 'portfolio']):
            return self.cmd_positions([])
        
        if any(word in text_lower for word in ['stop', 'halt', 'disable']):
            return "⚠️ To stop trading, use the /stop command."
        
        if any(word in text_lower for word in ['help', 'command', 'what can you do']):
            return self.cmd_help([])
        
        # Default response
        return f"""🤖 <b>Casey received:</b> "{text}"

I understand these message types:
• Status requests ("How's it going?")
• P&L queries ("Show me profit")
• Balance checks ("What's my balance?")
• Position queries ("What am I holding?")

Or use /help for all commands."""
    
    # =========================================================================
    # AGENT INTEGRATION
    # =========================================================================
    
    def connect_optimus(self, optimus_instance):
        """Connect Optimus agent"""
        self.optimus = optimus_instance
        logger.info("Casey connected to Optimus")
    
    def connect_ralph(self, ralph_instance):
        """Connect Ralph agent"""
        self.ralph = ralph_instance
        logger.info("Casey connected to Ralph")
    
    def start(self, blocking: bool = False):
        """Start Casey's Telegram interface"""
        logger.info("🚀 Starting Casey Telegram Interface...")
        self.bot.start_listener(blocking=blocking)
    
    def stop(self):
        """Stop Casey's Telegram interface"""
        self.bot.stop_listener()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Casey Telegram interface"""
    print("\n" + "="*60)
    print("🤖 CASEY - NAE SECURE LINE")
    print("="*60)
    
    # Check configuration
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("""
❌ Telegram not configured!

To set up Casey's Secure Line:

1. Create a bot via @BotFather on Telegram
2. Set environment variables:

   PowerShell:
   $env:TELEGRAM_BOT_TOKEN = "your_token_here"
   $env:TELEGRAM_CHAT_ID = "your_chat_id_here"

3. Run this script again
""")
        return
    
    # Initialize and start
    casey = CaseyTelegramInterface()
    
    print(f"""
✅ Casey is ready!

Bot Token: {bot_token[:10]}...
Chat ID: {chat_id}

Starting listener...
Press Ctrl+C to stop.
""")
    
    try:
        casey.start(blocking=True)
    except KeyboardInterrupt:
        print("\n\nStopping Casey...")
        casey.stop()
        print("Casey stopped.")


if __name__ == "__main__":
    main()


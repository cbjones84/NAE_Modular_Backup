#!/usr/bin/env python3
"""
Check Optimus Sandbox Trading Profits
Analyzes Optimus's execution history and calculates profits in sandbox mode
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from agents.optimus import OptimusAgent
from env_loader import EnvLoader

# Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def check_optimus_profits():
    """Check Optimus's sandbox trading profits"""
    print_header("OPTIMUS SANDBOX TRADING PROFITS")
    
    # Initialize Optimus
    loader = EnvLoader()
    optimus = OptimusAgent(sandbox=True)
    
    # Get trading status
    status = optimus.get_trading_status()
    
    print_info("Trading Status:")
    print(f"  Mode: {status['trading_mode']}")
    print(f"  Trading Enabled: {status['trading_enabled']}")
    print(f"  Net Asset Value (NAV): ${status['nav']:,.2f}")
    print(f"  Daily P&L: ${status['daily_pnl']:,.2f}")
    print(f"  Open Positions: {status['open_positions']}")
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    print()
    
    # Check execution history
    execution_count = len(optimus.execution_history)
    print_info(f"Execution History: {execution_count} trades")
    
    if execution_count > 0:
        print_info("\nRecent Trades:")
        for i, trade in enumerate(optimus.execution_history[-10:], 1):  # Last 10
            details = trade.get('details', {})
            result = trade.get('result', {})
            timestamp = trade.get('timestamp', 'Unknown')
            
            strategy_name = details.get('strategy_name', 'Unknown')
            status_trade = result.get('status', 'Unknown')
            order_id = result.get('order_id', 'N/A')
            
            print(f"  {i}. {strategy_name}")
            print(f"     Order ID: {order_id}")
            print(f"     Status: {status_trade}")
            print(f"     Time: {timestamp[:19] if len(timestamp) > 19 else timestamp}")
            
            # Try to extract P&L if available
            if 'pnl' in result:
                pnl = result['pnl']
                pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
                print(f"     P&L: {pnl_str}")
            print()
    else:
        print_warning("No trades executed yet")
    
    # Check inbox for pending strategies
    if hasattr(optimus, 'inbox') and optimus.inbox:
        print_info(f"Pending Strategies in Inbox: {len(optimus.inbox)}")
        print_info("(These strategies have been received but not yet executed)")
    
    # Calculate summary
    print_header("PROFIT SUMMARY")
    
    daily_pnl = status['daily_pnl']
    nav = status['nav']
    
    if daily_pnl > 0:
        print_success(f"Daily Profit: ${daily_pnl:,.2f}")
        print_info(f"Return on NAV: {(daily_pnl/nav)*100:.2f}%")
    elif daily_pnl < 0:
        print_warning(f"Daily Loss: ${abs(daily_pnl):,.2f}")
        print_info(f"Loss on NAV: {(abs(daily_pnl)/nav)*100:.2f}%")
    else:
        print_info("Daily P&L: $0.00 (No trades executed or no P&L calculated)")
    
    print()
    print_info(f"Current NAV: ${nav:,.2f}")
    print_info(f"Total Trades: {execution_count}")
    
    # Check audit log for trade execution details
    audit_summary = optimus.get_audit_summary()
    if audit_summary.get('total_entries', 0) > 0:
        print_info(f"Audit Log Entries: {audit_summary['total_entries']}")
        action_counts = audit_summary.get('action_counts', {})
        if 'TRADE_EXECUTED' in action_counts:
            print_info(f"Trades Executed (from audit): {action_counts['TRADE_EXECUTED']}")
    
    print()
    print_info("Note: Sandbox mode uses simulated trades.")
    print_info("Actual P&L calculation depends on trade execution results.")
    
    return {
        'daily_pnl': daily_pnl,
        'nav': nav,
        'trades_executed': execution_count,
        'trading_mode': status['trading_mode']
    }

if __name__ == "__main__":
    result = check_optimus_profits()


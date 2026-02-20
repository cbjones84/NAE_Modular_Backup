#!/usr/bin/env python3
"""
Process Optimus Inbox and Check Sandbox Profits
Executes pending strategies and calculates profits
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from agents.optimus import OptimusAgent
from agents.donnie import DonnieAgent
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

def process_optimus_and_check_profits():
    """Process Optimus's inbox and check profits"""
    print_header("PROCESSING OPTIMUS INBOX & CHECKING PROFITS")
    
    loader = EnvLoader()
    
    # Initialize Optimus
    optimus = OptimusAgent(sandbox=True)
    
    # Load strategies and send to Optimus via Donnie
    logs_dir = Path('logs')
    strategy_files = sorted(logs_dir.glob('ralph_approved_strategies*.json'), reverse=True)
    
    if strategy_files:
        with open(strategy_files[0], 'r') as f:
            strategies = json.load(f)
        
        high_quality = [s for s in strategies if s.get('trust_score', 0) >= 70]
        
        if high_quality:
            # Send strategies through Donnie to Optimus
            donnie = DonnieAgent()
            donnie.receive_strategies(high_quality[:7])
            donnie.run_cycle(sandbox=True, optimus_agent=optimus)
    
    # Check current status before processing
    status_before = optimus.get_trading_status()
    print_info(f"Status Before Processing:")
    print(f"  Daily P&L: ${status_before['daily_pnl']:,.2f}")
    print(f"  Execution History: {status_before['execution_history_count']} trades")
    print(f"  Inbox Messages: {len(optimus.inbox) if hasattr(optimus, 'inbox') else 0}")
    print()
    
    # Process inbox messages
    if hasattr(optimus, 'inbox') and optimus.inbox:
        print_info(f"Processing {len(optimus.inbox)} messages from inbox...")
        
        executed_count = 0
        for msg in optimus.inbox:
            if isinstance(msg, dict):
                # Extract execution details
                execution_details = {
                    "strategy_name": msg.get('strategy_name', 'Unknown'),
                    "action": msg.get('action', 'execute_trade'),
                    "parameters": msg.get('parameters', {}),
                    "sandbox": msg.get('sandbox', True),
                    "symbol": "SPY",  # Default symbol for sandbox
                    "side": "buy",
                    "quantity": 10,
                    "price": 450.0  # Default price for sandbox simulation
                }
                
                # Execute trade
                result = optimus.execute_trade(execution_details)
                
                if result.get('status') == 'filled':
                    executed_count += 1
                    print_success(f"Executed: {execution_details['strategy_name']}")
                else:
                    print_warning(f"Failed: {execution_details['strategy_name']} - {result.get('status', 'Unknown')}")
        
        print_info(f"Processed {executed_count} trades")
        print()
    else:
        print_warning("No messages in Optimus inbox")
    
    # Check status after processing
    status_after = optimus.get_trading_status()
    
    print_header("OPTIMUS SANDBOX PROFITS")
    
    daily_pnl = status_after['daily_pnl']
    nav = status_after['nav']
    execution_count = status_after['execution_history_count']
    
    print_info("Trading Status:")
    print(f"  Mode: {status_after['trading_mode']}")
    print(f"  Net Asset Value (NAV): ${nav:,.2f}")
    print(f"  Daily P&L: ${daily_pnl:,.2f}")
    print(f"  Total Trades Executed: {execution_count}")
    print(f"  Open Positions: {status_after['open_positions']}")
    print(f"  Consecutive Losses: {status_after['consecutive_losses']}")
    print()
    
    # Show execution history
    if execution_count > 0:
        print_info("Recent Trades:")
        for i, trade in enumerate(optimus.execution_history[-10:], 1):
            details = trade.get('details', {})
            result = trade.get('result', {})
            strategy_name = details.get('strategy_name', 'Unknown')
            status_trade = result.get('status', 'Unknown')
            order_id = result.get('order_id', 'N/A')
            print(f"  {i}. {strategy_name} - {status_trade} (Order: {order_id})")
    
    print()
    print_header("PROFIT SUMMARY")
    
    if daily_pnl > 0:
        print_success(f"Daily Profit: ${daily_pnl:,.2f}")
        print_info(f"Return on NAV: {(daily_pnl/nav)*100:.2f}%")
    elif daily_pnl < 0:
        print_warning(f"Daily Loss: ${abs(daily_pnl):,.2f}")
        print_info(f"Loss on NAV: {(abs(daily_pnl)/nav)*100:.2f}%")
    else:
        print_info("Daily P&L: $0.00")
        print_warning("Note: Sandbox trades are simulated and may not calculate P&L automatically")
        print_info("P&L is typically calculated when trades are closed or marked-to-market")
    
    print()
    print_info(f"Current NAV: ${nav:,.2f}")
    print_info(f"Total Trades: {execution_count}")
    
    return {
        'daily_pnl': daily_pnl,
        'nav': nav,
        'trades_executed': execution_count,
        'trading_mode': status_after['trading_mode']
    }

if __name__ == "__main__":
    result = process_optimus_and_check_profits()


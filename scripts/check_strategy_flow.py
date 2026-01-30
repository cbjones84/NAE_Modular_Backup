#!/usr/bin/env python3
"""
Check Strategy Flow: Ralph → Donnie → Optimus
Analyzes how many strategies have been passed through the pipeline
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from agents.ralph import RalphAgent
from agents.donnie import DonnieAgent
from agents.optimus import OptimusAgent

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

def check_ralph_strategies():
    """Check Ralph's approved strategies"""
    print_header("STEP 1: Ralph's Approved Strategies")
    
    # Check approved strategy files
    logs_dir = Path('logs')
    strategy_files = sorted(logs_dir.glob('ralph_approved_strategies*.json'), reverse=True)
    
    if not strategy_files:
        print_warning("No approved strategy files found")
        return []
    
    # Get latest file
    latest_file = strategy_files[0]
    print_info(f"Latest strategy file: {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            strategies = json.load(f)
        
        if isinstance(strategies, list):
            print_success(f"Found {len(strategies)} approved strategies")
            
            # Show summary
            eligible = [s for s in strategies if s.get('trust_score', 0) >= 50.0]
            print_info(f"Eligible for Donnie (trust_score >= 50.0): {len(eligible)}")
            
            if strategies:
                print_info("\nTop strategies:")
                for i, s in enumerate(strategies[:5], 1):
                    name = s.get('name', 'Unknown')
                    trust = s.get('trust_score', 0)
                    backtest = s.get('backtest_score', 0)
                    print(f"  {i}. {name}")
                    print(f"     Trust Score: {trust:.1f} | Backtest Score: {backtest:.1f}")
            
            return strategies
        else:
            print_warning("Strategy file format unexpected")
            return []
    except Exception as e:
        print_warning(f"Error reading strategy file: {e}")
        return []

def check_donnie_execution():
    """Check Donnie's execution history"""
    print_header("STEP 2: Donnie's Execution History")
    
    try:
        donnie = DonnieAgent()
        
        # Check execution history
        if hasattr(donnie, 'execution_history') and donnie.execution_history:
            print_success(f"Donnie has executed {len(donnie.execution_history)} strategies")
            
            print_info("\nExecuted strategies:")
            for i, exec_detail in enumerate(donnie.execution_history[-10:], 1):  # Last 10
                name = exec_detail.get('strategy_name', 'Unknown')
                sandbox = exec_detail.get('sandbox', True)
                timestamp = exec_detail.get('timestamp', 'Unknown')
                print(f"  {i}. {name} (Sandbox: {sandbox})")
                print(f"     Timestamp: {timestamp}")
            
            return donnie.execution_history
        else:
            print_warning("Donnie has no execution history (no strategies executed yet)")
            return []
    except Exception as e:
        print_warning(f"Error checking Donnie: {e}")
        return []

def check_donnie_logs():
    """Check Donnie's log file"""
    print_info("\nChecking Donnie's log file...")
    
    log_file = Path('logs/donnie.log')
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Count strategy-related entries
        strategy_received = [l for l in lines if 'Received' in l and 'strategies' in l]
        strategy_executed = [l for l in lines if 'Executing strategy' in l]
        strategy_sent = [l for l in lines if 'Sent execution instruction to Optimus' in l]
        
        print_info(f"Log entries:")
        print(f"  Strategies received: {len(strategy_received)}")
        print(f"  Strategies executed: {len(strategy_executed)}")
        print(f"  Strategies sent to Optimus: {len(strategy_sent)}")
        
        if strategy_sent:
            print_success(f"Donnie has sent {len(strategy_sent)} strategies to Optimus")
            print_info("\nRecent messages to Optimus:")
            for msg in strategy_sent[-5:]:
                print(f"  {msg.strip()}")
        
        return len(strategy_sent)
    else:
        print_warning("Donnie log file not found")
        return 0

def check_optimus_received():
    """Check Optimus's received messages"""
    print_header("STEP 3: Optimus's Received Strategies")
    
    try:
        optimus = OptimusAgent(sandbox=True)
        
        # Check if Optimus has inbox/received messages
        if hasattr(optimus, 'inbox') and optimus.inbox:
            print_success(f"Optimus has received {len(optimus.inbox)} messages")
            
            print_info("\nReceived messages:")
            for i, msg in enumerate(optimus.inbox[-10:], 1):  # Last 10
                strategy_name = msg.get('strategy_name', 'Unknown')
                action = msg.get('action', 'Unknown')
                print(f"  {i}. Strategy: {strategy_name}")
                print(f"     Action: {action}")
            
            return len(optimus.inbox)
        else:
            print_warning("Optimus has no received messages (inbox is empty)")
            
            # Check execution history as alternative
            if hasattr(optimus, 'execution_history') and optimus.execution_history:
                print_info(f"However, Optimus has {len(optimus.execution_history)} execution history entries")
                return len(optimus.execution_history)
            else:
                return 0
    except Exception as e:
        print_warning(f"Error checking Optimus: {e}")
        return 0

def run_full_flow_check():
    """Run a complete flow check"""
    print_header("STRATEGY FLOW ANALYSIS: Ralph → Donnie → Optimus")
    
    # Step 1: Check Ralph
    ralph_strategies = check_ralph_strategies()
    
    # Step 2: Check Donnie
    donnie_executions = check_donnie_execution()
    donnie_sent_count = check_donnie_logs()
    
    # Step 3: Check Optimus
    optimus_received = check_optimus_received()
    
    # Summary
    print_header("SUMMARY")
    
    print_info("Strategy Flow Status:")
    print(f"  Ralph Approved: {len(ralph_strategies)} strategies")
    print(f"  Donnie Executed: {len(donnie_executions)} strategies")
    print(f"  Donnie Sent to Optimus: {donnie_sent_count} strategies")
    print(f"  Optimus Received: {optimus_received} strategies")
    
    # Analysis
    print_info("\nFlow Analysis:")
    if len(ralph_strategies) > 0:
        if len(donnie_executions) == 0:
            print_warning("⚠️  Ralph has strategies but Donnie hasn't executed any yet")
            print_info("   → Run Donnie.run_cycle() to process strategies")
        elif donnie_sent_count == 0:
            print_warning("⚠️  Donnie executed strategies but didn't send to Optimus")
            print_info("   → This happens when strategies don't meet Donnie's validation (trust_score >= 70, backtest_score >= 50)")
        elif optimus_received == 0:
            print_warning("⚠️  Strategies were sent but Optimus hasn't received them")
            print_info("   → Check if Optimus agent was initialized when Donnie sent messages")
        else:
            print_success(f"✅ Flow is working! {optimus_received} strategies made it to Optimus")
    else:
        print_warning("⚠️  No strategies approved by Ralph yet")
        print_info("   → Run Ralph.run_cycle() to generate strategies")
    
    return {
        'ralph_approved': len(ralph_strategies),
        'donnie_executed': len(donnie_executions),
        'donnie_sent': donnie_sent_count,
        'optimus_received': optimus_received
    }

if __name__ == "__main__":
    run_full_flow_check()


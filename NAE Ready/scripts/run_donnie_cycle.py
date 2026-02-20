#!/usr/bin/env python3
"""
Run Donnie Cycle - Process Ralph's Strategies and Pass to Optimus
Automates the Ralph → Donnie → Optimus flow
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

def load_latest_strategies():
    """Load the latest strategies from Ralph"""
    logs_dir = Path('logs')
    strategy_files = sorted(logs_dir.glob('ralph_approved_strategies*.json'), reverse=True)
    
    if not strategy_files:
        print_warning("No strategy files found")
        return []
    
    latest_file = strategy_files[0]
    print_info(f"Loading strategies from: {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            strategies = json.load(f)
        
        if isinstance(strategies, list):
            # Filter for high-quality strategies (trust_score >= 70)
            high_quality = [s for s in strategies if s.get('trust_score', 0) >= 70]
            print_success(f"Loaded {len(strategies)} total strategies")
            print_info(f"High-quality strategies (>= 70): {len(high_quality)}")
            return high_quality
        else:
            return []
    except Exception as e:
        print_warning(f"Error loading strategies: {e}")
        return []

def run_donnie_cycle():
    """Run Donnie's cycle to process strategies and pass to Optimus"""
    print_header("RUNNING DONNIE CYCLE")
    
    # Setup environment
    loader = EnvLoader()
    
    # Initialize agents
    print_info("Initializing agents...")
    optimus = OptimusAgent(sandbox=False)  # LIVE MODE ONLY
    donnie = DonnieAgent()
    
    # Load latest strategies from Ralph
    print_info("Loading latest strategies from Ralph...")
    strategies = load_latest_strategies()
    
    if not strategies:
        print_warning("No high-quality strategies found (trust_score >= 70)")
        print_info("Running Ralph cycle to generate new strategies...")
        
        # Use enhanced Ralph to generate high-quality strategies
        from run_ralph_max_quality import EnhancedRalphAgent
        
        enhanced_config = {
            "min_trust_score": 50.0,
            "min_backtest_score": 30.0,
            "min_consensus_sources": 1,
            "max_drawdown_pct": 0.6,
            "source_reputations": {
                "Grok": 92, "DeepSeek": 90, "Claude": 91,
                "toptrader.com": 85, "optionsforum.com": 75,
                "financeapi.local": 80, "reddit_r_options": 80,
                "seeking_alpha": 88, "tradingview": 85
            }
        }
        
        ralph = EnhancedRalphAgent(config=enhanced_config)
        result = ralph.run_cycle()
        
        # Filter for high-quality strategies
        strategies = [s for s in ralph.strategy_database if s.get('trust_score', 0) >= 70]
        
        if not strategies:
            print_warning("Ralph did not generate any strategies with trust_score >= 70")
            return
    
    print_success(f"Found {len(strategies)} high-quality strategies to process")
    
    # Show strategies
    print_info("\nStrategies to process:")
    for i, s in enumerate(strategies, 1):
        trust = s.get('trust_score', 0)
        backtest = s.get('backtest_score', 0)
        print(f"  {i}. {s.get('name', 'Unknown')} (Trust: {trust:.1f}, Backtest: {backtest:.1f})")
    
    # Pass strategies to Donnie
    print_info("\nPassing strategies to Donnie...")
    donnie.receive_strategies(strategies)
    
    # Run Donnie's cycle with Optimus connected
    print_info("Running Donnie's execution cycle...")
    donnie.run_cycle(sandbox=True, optimus_agent=optimus)
    
    # Summary
    print_header("DONNIE CYCLE COMPLETE")
    
    executed = len(donnie.execution_history)
    print_success(f"Strategies executed: {executed}")
    
    if executed > 0:
        print_info("\nExecuted strategies:")
        for i, exec_detail in enumerate(donnie.execution_history, 1):
            name = exec_detail.get('strategy_name', 'Unknown')
            sandbox = exec_detail.get('sandbox', True)
            print(f"  {i}. {name} (Sandbox: {sandbox})")
        
        # Check Optimus's inbox
        if hasattr(optimus, 'inbox') and optimus.inbox:
            print_success(f"\nOptimus received {len(optimus.inbox)} execution instructions")
        else:
            print_info("\nNote: Optimus inbox status checked")
    
    return {
        'strategies_received': len(strategies),
        'strategies_executed': executed,
        'optimus_received': len(optimus.inbox) if hasattr(optimus, 'inbox') else 0
    }

if __name__ == "__main__":
    result = run_donnie_cycle()
    
    if result:
        print_header("SUMMARY")
        print(f"Strategies Received by Donnie: {result['strategies_received']}")
        print(f"Strategies Executed by Donnie: {result['strategies_executed']}")
        print(f"Strategies Received by Optimus: {result['optimus_received']}")


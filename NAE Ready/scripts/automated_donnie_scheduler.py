#!/usr/bin/env python3
"""
Automated Donnie Cycle Scheduler
Continuously processes strategies from Ralph and passes them to Optimus
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import threading

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

class AutomatedDonnieScheduler:
    """Automated scheduler for Donnie cycle"""
    
    def __init__(self, interval_minutes=30, auto_generate=False):
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        self.auto_generate = auto_generate
        self.running = False
        self.cycle_count = 0
        self.stats = {
            'cycles_run': 0,
            'strategies_processed': 0,
            'strategies_passed_to_optimus': 0
        }
        
        # Initialize agents
        loader = EnvLoader()
        self.optimus = OptimusAgent(sandbox=False)  # LIVE MODE ONLY
        self.donnie = DonnieAgent()
        
        # Use enhanced Ralph if auto-generating
        if auto_generate:
            try:
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
                self.ralph = EnhancedRalphAgent(config=enhanced_config)
            except:
                self.ralph = RalphAgent()
        else:
            self.ralph = None
    
    def load_latest_strategies(self):
        """Load latest strategies from Ralph"""
        logs_dir = Path('logs')
        strategy_files = sorted(logs_dir.glob('ralph_approved_strategies*.json'), reverse=True)
        
        if not strategy_files:
            return []
        
        latest_file = strategy_files[0]
        try:
            with open(latest_file, 'r') as f:
                strategies = json.load(f)
            
            if isinstance(strategies, list):
                # Filter for high-quality strategies
                high_quality = [s for s in strategies if s.get('trust_score', 0) >= 70]
                return high_quality
        except:
            pass
        
        return []
    
    def run_cycle(self):
        """Run a single Donnie cycle"""
        self.cycle_count += 1
        print_header(f"DONNIE CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate strategies if auto-generate is enabled
        if self.auto_generate and self.ralph:
            print_info("Auto-generating strategies from Ralph...")
            try:
                result = self.ralph.run_cycle()
                strategies = [s for s in self.ralph.strategy_database if s.get('trust_score', 0) >= 70]
                print_success(f"Generated {len(strategies)} high-quality strategies")
            except Exception as e:
                print_warning(f"Error generating strategies: {e}")
                strategies = []
        else:
            # Load from file
            strategies = self.load_latest_strategies()
        
        if not strategies:
            print_warning("No high-quality strategies found (trust_score >= 70)")
            return
        
        print_info(f"Processing {len(strategies)} strategies...")
        
        # Pass to Donnie
        self.donnie.receive_strategies(strategies)
        
        # Run Donnie's cycle
        self.donnie.run_cycle(sandbox=False, optimus_agent=self.optimus)  # LIVE MODE
        
        # Update stats
        executed = len(self.donnie.execution_history)
        self.stats['cycles_run'] += 1
        self.stats['strategies_processed'] += len(strategies)
        self.stats['strategies_passed_to_optimus'] += executed
        
        # Summary
        print_success(f"Cycle complete: {executed} strategies executed")
        
        if executed > 0:
            print_info("Strategies passed to Optimus:")
            for exec_detail in self.donnie.execution_history[-executed:]:
                print(f"  - {exec_detail.get('strategy_name', 'Unknown')}")
        
        print_info(f"Next cycle in {self.interval_minutes} minutes...")
    
    def start(self):
        """Start the automated scheduler"""
        print_header("STARTING AUTOMATED DONNIE SCHEDULER")
        print_info(f"Cycle interval: {self.interval_minutes} minutes")
        print_info(f"Auto-generate strategies: {self.auto_generate}")
        print_info("Press Ctrl+C to stop")
        
        self.running = True
        
        try:
            while self.running:
                self.run_cycle()
                
                if self.running:
                    print_info(f"Waiting {self.interval_minutes} minutes until next cycle...")
                    time.sleep(self.interval_seconds)
        except KeyboardInterrupt:
            print_header("SCHEDULER STOPPED")
            self.print_stats()
        except Exception as e:
            print_warning(f"Scheduler error: {e}")
            self.print_stats()
    
    def print_stats(self):
        """Print scheduler statistics"""
        print_header("SCHEDULER STATISTICS")
        print(f"Total cycles run: {self.stats['cycles_run']}")
        print(f"Total strategies processed: {self.stats['strategies_processed']}")
        print(f"Total strategies passed to Optimus: {self.stats['strategies_passed_to_optimus']}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Donnie Cycle Scheduler')
    parser.add_argument('--interval', type=int, default=30,
                       help='Cycle interval in minutes (default: 30)')
    parser.add_argument('--auto-generate', action='store_true',
                       help='Auto-generate strategies from Ralph before each cycle')
    parser.add_argument('--once', action='store_true',
                       help='Run once instead of continuously')
    
    args = parser.parse_args()
    
    scheduler = AutomatedDonnieScheduler(
        interval_minutes=args.interval,
        auto_generate=args.auto_generate
    )
    
    if args.once:
        scheduler.run_cycle()
        scheduler.print_stats()
    else:
        scheduler.start()

if __name__ == "__main__":
    main()


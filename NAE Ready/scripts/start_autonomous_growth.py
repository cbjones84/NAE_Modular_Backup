#!/usr/bin/env python3
"""
NAE AUTONOMOUS GROWTH SYSTEM - STARTUP SCRIPT
==============================================
Starts the autonomous trading system with all agents aligned
on the $5M growth mission.

Usage:
    python scripts/start_autonomous_growth.py

This script:
1. Loads the mission configuration
2. Aligns all agents on growth targets
3. Starts autonomous trading operations
4. Monitors progress continuously
5. Adapts strategies proactively
"""

import os
import sys
import json
import time
import datetime
import threading
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
NAE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NAE_DIR))

# Environment setup
os.environ.setdefault("TRADIER_SANDBOX", "false")
os.environ.setdefault("TRADIER_API_KEY", "27Ymk28vtbgqY1LFYxhzaEmIuwJb")
os.environ.setdefault("TRADIER_ACCOUNT_ID", "6YB66744")

# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_prefix = {
        "INFO": "[INFO]",
        "SUCCESS": "[OK]",
        "WARNING": "[WARN]",
        "ERROR": "[ERR]",
        "TRADE": "[TRADE]",
        "MISSION": "[MISSION]"
    }.get(level, "[INFO]")
    print(f"[{timestamp}] {level_prefix} {message}", flush=True)

# =============================================================================
# MAIN AUTONOMOUS SYSTEM
# =============================================================================

class NAEAutonomousGrowthSystem:
    """
    Main autonomous growth system that coordinates all agents
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.mission_active = False
        self.current_balance = 100.0  # Will be updated from broker
        self.start_time = None
        self.agents_status = {}
        
    def _load_config(self) -> dict:
        """Load agent alignment configuration"""
        config_path = NAE_DIR / "core" / "agent_alignment_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def display_startup_banner(self):
        """Display startup banner"""
        print("\n" + "=" * 70)
        print("  NAE AUTONOMOUS GROWTH SYSTEM")
        print("  Mission: $100 -> $5,000,000 (8 Years)")
        print("=" * 70)
        print("""
    _   _    _    _____   __  __ ___ ____ ____ ___ ___  _   _ 
   | \ | |  / \  | ____| |  \/  |_ _/ ___/ ___|_ _/ _ \| \ | |
   |  \| | / _ \ |  _|   | |\/| || |\___ \___ \| | | | |  \| |
   | |\  |/ ___ \| |___  | |  | || | ___) ___) | | |_| | |\  |
   |_| \_/_/   \_|_____| |_|  |_|___|____|____/___\___/|_| \_|
                                                              
                    OPERATION $5M GROWTH                      
        """)
    
    def verify_broker_connection(self) -> bool:
        """Verify connection to trading broker"""
        log("Verifying broker connection...", "INFO")
        
        try:
            import requests
            api_key = os.environ.get("TRADIER_API_KEY")
            account_id = os.environ.get("TRADIER_ACCOUNT_ID")
            
            url = f"https://api.tradier.com/v1/accounts/{account_id}/balances"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                balances = data.get("balances", {})
                cash = float(balances.get("total_cash", 0))
                equity = float(balances.get("total_equity", cash))
                
                self.current_balance = equity if equity > 0 else cash
                log(f"Broker connected! Balance: ${self.current_balance:,.2f}", "SUCCESS")
                return True
            else:
                log(f"Broker connection failed: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            log(f"Broker connection error: {e}", "ERROR")
            return False
    
    def align_agents(self):
        """Align all agents on the mission"""
        log("Aligning agents on $5M growth mission...", "MISSION")
        
        agents = self.config.get("agents", {})
        for agent_name, agent_config in agents.items():
            self.agents_status[agent_name] = {
                "role": agent_config.get("role"),
                "priority": agent_config.get("priority"),
                "autonomous": agent_config.get("autonomous"),
                "status": "ALIGNED",
                "last_action": datetime.datetime.now().isoformat()
            }
            log(f"  [{agent_name.upper()}] Aligned - {agent_config.get('role')}", "SUCCESS")
        
        log(f"All {len(agents)} agents aligned on mission!", "SUCCESS")
    
    def get_phase_info(self) -> dict:
        """Get current phase information"""
        milestones = self.config.get("milestones", {})
        
        if self.current_balance < 1000:
            phase = "MICRO"
            target_return = 0.30
        elif self.current_balance < 10000:
            phase = "SMALL"
            target_return = 0.20
        elif self.current_balance < 100000:
            phase = "GROWTH"
            target_return = 0.12
        else:
            phase = "SCALE"
            target_return = 0.08
        
        return {
            "phase": phase,
            "balance": self.current_balance,
            "target_monthly_return": target_return,
            "target_yearly_return": (1 + target_return) ** 12 - 1
        }
    
    def display_mission_status(self):
        """Display current mission status"""
        phase_info = self.get_phase_info()
        
        # Calculate progress
        year = 1  # First year
        milestones = self.config.get("milestones", {})
        year_milestone = milestones.get(f"year_{year}", {}).get("target", 9411)
        progress = (self.current_balance / year_milestone) * 100
        
        print("\n" + "-" * 70)
        print("MISSION STATUS")
        print("-" * 70)
        print(f"  Current Balance:    ${self.current_balance:,.2f}")
        print(f"  Phase:              {phase_info['phase']}")
        print(f"  Target Monthly:     {phase_info['target_monthly_return']:.0%}")
        print(f"  Year 1 Milestone:   ${year_milestone:,.2f}")
        print(f"  Progress:           {progress:.1f}%")
        print(f"  Agents Active:      {len(self.agents_status)}")
        print("-" * 70)
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        log("Running trading cycle...", "INFO")
        
        try:
            # Import and run accelerator
            from strategies.optimus_accelerator import OptimusAccelerator, AcceleratorConfig
            
            config = AcceleratorConfig(
                starting_capital=self.current_balance,
                biweekly_deposit=100.0
            )
            
            accelerator = OptimusAccelerator(config)
            accelerator.current_balance = self.current_balance
            
            # Scan for opportunities
            opportunities = accelerator.scan_opportunities()
            
            if opportunities:
                log(f"Found {len(opportunities)} opportunities", "SUCCESS")
                
                # Generate trade plan
                trade_plan = accelerator.generate_trade_plan(opportunities)
                
                if trade_plan:
                    log(f"Generated {len(trade_plan)} trades", "INFO")
                    for trade in trade_plan:
                        log(f"  {trade['signal']} {trade['symbol']} @ ${trade['price']:.2f}", "TRADE")
                    
                    # In production, execute trades here
                    # results = accelerator.execute_trade_plan(trade_plan, dry_run=False)
            else:
                log("No opportunities meet criteria", "INFO")
                
        except Exception as e:
            log(f"Trading cycle error: {e}", "ERROR")
    
    def start(self, continuous: bool = True, interval_minutes: int = 5):
        """Start the autonomous growth system"""
        self.display_startup_banner()
        
        log("Starting NAE Autonomous Growth System...", "MISSION")
        
        # Verify broker connection
        if not self.verify_broker_connection():
            log("Cannot start without broker connection", "ERROR")
            return False
        
        # Align agents
        self.align_agents()
        
        # Display status
        self.display_mission_status()
        
        self.mission_active = True
        self.start_time = datetime.datetime.now()
        
        log("", "INFO")
        log("=" * 50, "INFO")
        log("AUTONOMOUS MODE ACTIVATED", "MISSION")
        log("=" * 50, "INFO")
        log(f"Trading cycle interval: {interval_minutes} minutes", "INFO")
        log("Press Ctrl+C to stop", "INFO")
        log("", "INFO")
        
        if continuous:
            cycle_count = 0
            while self.mission_active:
                try:
                    cycle_count += 1
                    log(f"--- Cycle {cycle_count} ---", "INFO")
                    
                    # Run trading cycle
                    self.run_trading_cycle()
                    
                    # Update balance from broker
                    self.verify_broker_connection()
                    
                    # Display status every 10 cycles
                    if cycle_count % 10 == 0:
                        self.display_mission_status()
                    
                    # Wait for next cycle
                    log(f"Next cycle in {interval_minutes} minutes...", "INFO")
                    time.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    log("Shutdown requested...", "WARNING")
                    self.mission_active = False
                    break
                except Exception as e:
                    log(f"Cycle error: {e}", "ERROR")
                    time.sleep(60)  # Wait 1 minute on error
        
        log("NAE Autonomous Growth System stopped", "MISSION")
        return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    system = NAEAutonomousGrowthSystem()
    
    # Start with 5-minute intervals
    system.start(continuous=True, interval_minutes=5)


if __name__ == "__main__":
    main()


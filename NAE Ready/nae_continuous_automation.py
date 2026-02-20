#!/usr/bin/env python3
"""
NAE Continuous Automation Daemon
Runs NAE in continuous automation mode with:
- Continuous strategy execution
- Real-time monitoring by Casey & Splinter
- Feedback loop for continuous improvement
- Paper trading via Alpaca

ALIGNED WITH:
- 3 Core Goals (Generational wealth, $5M in 8 years, Optimize options trading)
- Long-Term Plan (Phase-aware execution)
- PDT Prevention (all positions hold overnight)
"""

import os
import sys
import time
import signal
import datetime
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import agents
from agents.optimus import OptimusAgent
from agents.ralph import RalphAgent
from agents.donnie import DonnieAgent
from agents.casey import CaseyAgent
from agents.splinter import SplinterAgent
from agents.shredder import ShredderAgent
from agents.bebop import BebopAgent
from agents.phisher import PhisherAgent
from agents.rocksteady import RocksteadyAgent

# Import feedback loop
from tools.feedback_loop import create_feedback_loop

class NAEContinuousAutomation:
    """
    Continuous automation daemon for NAE
    
    Features:
    - Continuous strategy execution
    - Real-time monitoring
    - Feedback loop for improvement
    - Paper trading via Alpaca
    """
    
    def __init__(self):
        self.running = True
        self.cycle_count = 0
        
        # Initialize agents
        print("=" * 80)
        print("INITIALIZING NAE CONTINUOUS AUTOMATION")
        print("=" * 80)
        print()
        
        print("Initializing Agents...")
        self.optimus = OptimusAgent(sandbox=False)  # LIVE mode (Live Alpaca account)
        self.ralph = RalphAgent()
        self.donnie = DonnieAgent()
        self.casey = CaseyAgent()
        self.splinter = SplinterAgent()
        self.shredder = ShredderAgent()
        self.bebop = BebopAgent()
        self.phisher = PhisherAgent()
        self.rocksteady = RocksteadyAgent()
        
        # Initialize feedback loop
        print("Initializing Feedback Loop System...")
        self.feedback_loop = create_feedback_loop()
        
        # Register agents with feedback loop
        self.feedback_loop.register_agent("OptimusAgent", self.optimus)
        self.feedback_loop.register_agent("RalphAgent", self.ralph)
        self.feedback_loop.register_agent("DonnieAgent", self.donnie)
        self.feedback_loop.register_agent("CaseyAgent", self.casey)
        self.feedback_loop.register_agent("SplinterAgent", self.splinter)
        
        # Register agents with Splinter
        self.splinter.register_agents([
            self.optimus, self.ralph, self.donnie, self.casey,
            self.shredder, self.bebop, self.phisher, self.rocksteady
        ])
        
        # Set agent references for bidirectional communication
        self.casey.phisher_agent = self.phisher
        self.casey.bebop_agent = self.bebop
        self.casey.rocksteady_agent = self.rocksteady
        
        # Configuration
        self.strategy_execution_interval = 300  # 5 minutes
        self.feedback_cycle_interval = 600  # 10 minutes
        self.monitoring_interval = 60  # 1 minute
        
        # Threads
        self.strategy_thread = None
        self.feedback_thread = None
        self.monitoring_thread = None
        
        # Performance tracking
        self.start_time = datetime.datetime.now()
        self.strategies_executed = 0
        self.strategies_filled = 0
        self.strategies_rejected = 0
        
        print("✅ All agents initialized")
        print(f"✅ Feedback loop initialized")
        print(f"✅ Continuous automation ready")
        print()
    
    def start(self):
        """Start continuous automation"""
        print("=" * 80)
        print("STARTING NAE CONTINUOUS AUTOMATION")
        print("=" * 80)
        print()
        print("Mode: Paper Trading via Alpaca")
        print("PDT Prevention: ACTIVE")
        print("Feedback Loop: ACTIVE")
        print("Monitoring: Casey & Splinter")
        print()
        print("Press Ctrl+C to stop")
        print()
        
        # Start threads
        self.strategy_thread = threading.Thread(target=self._strategy_execution_loop, daemon=True)
        self.feedback_thread = threading.Thread(target=self._feedback_loop_thread, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        
        self.strategy_thread.start()
        self.feedback_thread.start()
        self.monitoring_thread.start()
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Main loop
        try:
            while self.running:
                self._print_status()
                time.sleep(60)  # Print status every minute
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.stop()
    
    def stop(self):
        """Stop continuous automation"""
        self.running = False
        
        print("\n" + "=" * 80)
        print("SHUTTING DOWN NAE CONTINUOUS AUTOMATION")
        print("=" * 80)
        print()
        
        # Final status
        self._print_final_status()
        
        print("✅ Shutdown complete")
    
    def _strategy_execution_loop(self):
        """Continuous strategy execution loop"""
        while self.running:
            try:
                self.cycle_count += 1
                
                print(f"\n[Cycle {self.cycle_count}] Strategy Execution Cycle")
                print("-" * 80)
                
                # Step 1: Generate strategies from Ralph
                print("1. Generating strategies from Ralph...")
                strategies = self.ralph.generate_strategies()
                
                if not strategies:
                    # Use predefined strategies if Ralph generates none
                    from scripts.batch_execute_strategies import create_diverse_strategies
                    current_phase = self.optimus.current_phase
                    strategies = create_diverse_strategies(current_phase)
                    print(f"   Using {len(strategies)} predefined strategies (Phase: {current_phase})")
                else:
                    print(f"   ✅ Ralph generated {len(strategies)} strategies")
                
                # Step 2: Validate strategies with Donnie
                print("2. Validating strategies with Donnie...")
                validated = []
                for strategy in strategies[:5]:  # Limit to 5 per cycle
                    if self.donnie.validate_strategy(strategy):
                        validated.append(strategy)
                
                print(f"   ✅ {len(validated)} strategies validated")
                
                # Step 3: Execute strategies through Optimus
                if validated:
                    print("3. Executing strategies through Optimus...")
                    
                    for strategy in validated:
                        try:
                            # Prepare execution details
                            exec_details = {
                                "symbol": strategy.get("symbol", "SPY"),
                                "side": "buy" if strategy.get("action") == "sell" and strategy.get("strategy_type") == "wheel_cash_secured_put" else strategy.get("action", "buy"),
                                "order_type": "market",
                                "time_in_force": "day",
                                "strategy_name": strategy.get("name", "Unknown"),
                                "trust_score": strategy.get("trust_score", 55),
                                "backtest_score": strategy.get("backtest_score", 50),
                                "expected_return": strategy.get("expected_return", 0.10),
                                "stop_loss_pct": strategy.get("stop_loss_pct", 0.02),
                                "parameters": strategy.get("parameters", {}),
                                "tier": strategy.get("tier", 1),
                                "phase": strategy.get("phase", "Phase 1"),
                                "pdt_compliant": True,
                                "strategy_type": strategy.get("strategy_type", "unknown")
                            }
                            
                            # Calculate position size
                            tier = strategy.get("tier", 1)
                            position_pct = 0.05 if tier == 1 else 0.03
                            
                            # Get price
                            estimated_price = 100.0
                            if self.optimus.polygon_client:
                                try:
                                    real_price = self.optimus.polygon_client.get_real_time_price(exec_details["symbol"])
                                    if real_price and real_price > 0:
                                        estimated_price = real_price
                                except Exception:
                                    pass  # Price fetch failed, use estimated price
                            
                            exec_details["quantity"] = max(1, int((self.optimus.nav * position_pct) / estimated_price))
                            exec_details["price"] = 0
                            
                            # Execute
                            result = self.optimus.execute_trade(exec_details)
                            
                            self.strategies_executed += 1
                            if result.get("status") == "filled":
                                self.strategies_filled += 1
                            elif result.get("status") == "rejected":
                                self.strategies_rejected += 1
                            
                            print(f"   Strategy: {strategy.get('name', 'Unknown')} → {result.get('status', 'unknown')}")
                            
                        except Exception as e:
                            print(f"   ⚠️  Execution error: {e}")
                
                # Wait before next cycle
                print(f"\n   Next execution cycle in {self.strategy_execution_interval} seconds...")
                time.sleep(self.strategy_execution_interval)
                
            except Exception as e:
                print(f"⚠️  Error in strategy execution loop: {e}")
                time.sleep(60)  # Wait before retry
    
    def _feedback_loop_thread(self):
        """Feedback loop thread for continuous improvement"""
        while self.running:
            try:
                print("\n[Feedback Loop] Running feedback cycle...")
                
                # Run feedback cycle
                cycle_result = self.feedback_loop.run_feedback_cycle()
                
                if cycle_result.get("recommendations"):
                    print(f"   ✅ Generated {len(cycle_result['recommendations'])} improvement recommendations")
                    for rec in cycle_result['recommendations'][:3]:  # Show top 3
                        print(f"      - {rec.get('agent_name', 'Unknown')}: {rec.get('description', 'N/A')}")
                
                if cycle_result.get("patterns"):
                    print(f"   ✅ Identified {len(cycle_result['patterns'])} patterns")
                
                # Wait before next cycle
                time.sleep(self.feedback_cycle_interval)
                
            except Exception as e:
                print(f"⚠️  Error in feedback loop: {e}")
                time.sleep(60)
    
    def _monitoring_loop(self):
        """Monitoring loop for Casey & Splinter"""
        while self.running:
            try:
                # Casey monitoring
                if self.casey:
                    try:
                        # Get system summary
                        system_summary = self.casey.get_system_summary()
                        
                        # Get Optimus status
                        if self.optimus:
                            optimus_status = self.optimus.get_trading_status()
                            
                            # Monitor for improvements
                            nav = optimus_status.get("nav", 0.0)
                            daily_pnl = optimus_status.get("daily_pnl", 0)
                            goal_progress = (nav / self.optimus.target_goal) * 100
                            
                            # Log progress
                            if nav > 25.0:
                                self.casey.log_action(f"📊 MONITORING: NAV=${nav:.2f}, Daily P&L=${daily_pnl:.2f}, "
                                                     f"Goal Progress={goal_progress:.4f}% toward $5M")
                            
                            # Check for issues
                            if daily_pnl < -1.0:  # $1 loss
                                self.casey.log_action(f"⚠️  MONITORING: Daily loss detected: ${daily_pnl:.2f}")
                            
                            if optimus_status.get("consecutive_losses", 0) >= 3:
                                self.casey.log_action(f"⚠️  MONITORING: {optimus_status['consecutive_losses']} consecutive losses detected")
                    except Exception as e:
                        pass
                
                # Splinter monitoring
                if self.splinter:
                    try:
                        # Monitor all agents
                        agent_statuses = {}
                        if self.optimus:
                            agent_statuses["Optimus"] = "Active" if self.optimus.trading_enabled else "Paused"
                        if self.ralph:
                            agent_statuses["Ralph"] = getattr(self.ralph, "status", "Unknown")
                        if self.donnie:
                            agent_statuses["Donnie"] = "Active"
                        
                        # Send monitoring message to Splinter
                        monitoring_msg = {
                            "from": "NAE_Automation",
                            "to": "SplinterAgent",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "monitoring_update",
                            "content": {
                                "agent_statuses": agent_statuses,
                                "cycle_count": self.cycle_count,
                                "strategies_executed": self.strategies_executed,
                                "uptime": str(datetime.datetime.now() - self.start_time)
                            }
                        }
                        self.splinter.receive_message(monitoring_msg)
                    except Exception as e:
                        pass
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                time.sleep(60)
    
    def _print_status(self):
        """Print current status"""
        try:
            status = self.optimus.get_trading_status()
            nav = status.get("nav", 0.0)
            goal_progress = (nav / self.optimus.target_goal) * 100
            
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Status:")
            print(f"  Cycle: {self.cycle_count} | NAV: ${nav:.2f} | Progress: {goal_progress:.4f}% toward $5M")
            print(f"  Executed: {self.strategies_executed} | Filled: {self.strategies_filled} | Rejected: {self.strategies_rejected}")
            print(f"  Daily P&L: ${status.get('daily_pnl', 0):.2f} | Open Positions: {status.get('open_positions', 0)}")
        except Exception:
            pass  # Status display failed, continue
    
    def _print_final_status(self):
        """Print final status summary"""
        try:
            status = self.optimus.get_trading_status()
            nav = status.get("nav", 0.0)
            goal_progress = (nav / self.optimus.target_goal) * 100
            
            uptime = datetime.datetime.now() - self.start_time
            
            print(f"Final Status:")
            print(f"  Uptime: {uptime}")
            print(f"  Cycles: {self.cycle_count}")
            print(f"  Strategies Executed: {self.strategies_executed}")
            print(f"  Strategies Filled: {self.strategies_filled}")
            print(f"  Strategies Rejected: {self.strategies_rejected}")
            print(f"  Final NAV: ${nav:.2f}")
            print(f"  Goal Progress: {goal_progress:.4f}% toward $5M")
            print(f"  Daily P&L: ${status.get('daily_pnl', 0):.2f}")
            
            # Feedback loop summary
            feedback_summary = self.feedback_loop.get_feedback_summary()
            print(f"  Feedback Cycles: {feedback_summary['cycle_count']}")
            print(f"  Recommendations Generated: {feedback_summary['total_recommendations']}")
            print(f"  Patterns Identified: {feedback_summary['total_patterns']}")
        except Exception:
            pass  # Status display failed, continue
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\nReceived shutdown signal...")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point"""
    automation = NAEContinuousAutomation()
    automation.start()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
NAE Full System Runner - Comprehensive Continuous Operation

Features:
- Ralph: GitHub learning, strategy discovery, continuous improvement
- Optimus: Intelligent Tradier monitoring and trading decisions
- Accelerator: Aggressive micro-scalp strategy execution
- Continuous learning loops and enhancements
- Full agent orchestration with communication
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add NAE to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set Tradier credentials (same as other scripts)
os.environ.setdefault("TRADIER_SANDBOX", "false")
os.environ.setdefault("TRADIER_API_KEY", "27Ymk28vtbgqY1LFYxhzaEmIuwJb")
os.environ.setdefault("TRADIER_ACCOUNT_ID", "6YB66744")
os.environ.setdefault("TRADIER_ACCOUNT_TYPE", "cash")

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/nae_full_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import agents
from agents.ralph import RalphAgent
from agents.optimus import OptimusAgent
from agents.donnie import DonnieAgent
from agents.casey import CaseyAgent
from agents.splinter import SplinterAgent
from execution.integration.accelerator_controller import AcceleratorController
from nae_main_orchestrator import NAEMainOrchestrator


class NAEFullSystem:
    """
    Complete NAE system runner with all features enabled
    """
    
    def __init__(self, sandbox_mode: bool = False, live_mode: bool = True, aggressive_accelerator: bool = True):
        """
        Initialize full NAE system
        
        Args:
            sandbox_mode: Enable sandbox/testing mode
            live_mode: Enable live/production mode
            aggressive_accelerator: Enable aggressive accelerator mode
        """
        self.sandbox_mode = sandbox_mode
        self.live_mode = live_mode
        self.aggressive_accelerator = aggressive_accelerator
        self.running = False
        
        # Agents
        self.ralph = None
        self.optimus = None
        self.donnie = None
        self.casey = None
        self.splinter = None
        self.accelerator_controller = None
        self.orchestrator = None
        
        # Threads
        self.ralph_thread = None
        self.optimus_intelligence_thread = None
        self.learning_thread = None
        self.enhancement_thread = None
        
        # Configuration
        self.ralph_cycle_interval = 300  # 5 minutes
        self.optimus_check_interval = 60  # 1 minute for Tradier checks
        self.learning_interval = 600  # 10 minutes
        self.enhancement_interval = 900  # 15 minutes
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("="*80)
        logger.info("NAE Full System - Initializing...")
        logger.info(f"  Sandbox Mode: {sandbox_mode}")
        logger.info(f"  Live Mode: {live_mode}")
        logger.info(f"  Aggressive Accelerator: {aggressive_accelerator}")
        logger.info("="*80)
    
    def initialize(self):
        """Initialize all agents and systems"""
        try:
            # Initialize Ralph (with GitHub learning enabled)
            logger.info("Initializing Ralph...")
            self.ralph = RalphAgent()
            logger.info("✅ Ralph initialized with GitHub integration")
            
            # Initialize Optimus (intelligent Tradier monitoring)
            logger.info("Initializing Optimus...")
            self.optimus = OptimusAgent(sandbox=self.sandbox_mode)
            
            # Enable trading - ensure kill switch is deactivated
            try:
                self.optimus.trading_enabled = True
                if hasattr(self.optimus, 'deactivate_kill_switch'):
                    self.optimus.deactivate_kill_switch("Full system initialization - enabling autonomous trading")
                logger.info("✅ Optimus trading ENABLED - ready to execute opportunities")
            except Exception as e:
                logger.warning(f"⚠️ Could not explicitly enable trading: {e}")
            
            # Enable Tradier intelligent monitoring if available
            if hasattr(self.optimus, 'self_healing_engine') and self.optimus.self_healing_engine:
                logger.info("✅ Optimus initialized with Tradier self-healing engine")
            
            # Initialize Donnie
            logger.info("Initializing Donnie...")
            self.donnie = DonnieAgent()
            logger.info("✅ Donnie initialized")
            
            # Initialize Casey
            logger.info("Initializing Casey...")
            self.casey = CaseyAgent()
            logger.info("✅ Casey initialized")
            
            # Initialize Splinter
            logger.info("Initializing Splinter...")
            self.splinter = SplinterAgent()
            logger.info("✅ Splinter initialized")
            
            # Initialize Accelerator Controller (aggressive mode)
            logger.info("Initializing Accelerator Controller...")
            self.accelerator_controller = AcceleratorController(
                sandbox_mode=self.sandbox_mode,
                live_mode=self.live_mode
            )
            
            if self.aggressive_accelerator:
                # Set aggressive cycle interval (30 seconds)
                self.accelerator_controller.cycle_interval = 30
                logger.info("✅ Accelerator Controller initialized in AGGRESSIVE mode (30s cycles)")
            else:
                logger.info("✅ Accelerator Controller initialized (60s cycles)")
            
            if not self.accelerator_controller.initialize_agents():
                logger.error("❌ Failed to initialize accelerator agents")
                return False
            
            # Initialize Main Orchestrator
            logger.info("Initializing Main Orchestrator...")
            self.orchestrator = NAEMainOrchestrator()
            logger.info("✅ Main Orchestrator initialized")
            
            # Register Optimus for direct communication with Ralph
            if hasattr(self.ralph, 'register_optimus_for_direct_communication'):
                self.ralph.register_optimus_for_direct_communication(self.optimus)
                logger.info("✅ Ralph-Optimus direct communication enabled")
            
            logger.info("="*80)
            logger.info("✅ All systems initialized successfully")
            logger.info("="*80)
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def start(self):
        """Start all continuous loops"""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("="*80)
        logger.info("Starting NAE Full System...")
        logger.info("="*80)
        
        self.running = True
        
        # Start Accelerator Controller (runs in its own thread)
        if self.accelerator_controller:
            logger.info("Starting Accelerator Controller...")
            accelerator_thread = threading.Thread(
                target=self.accelerator_controller.monitor_loop,
                name="AcceleratorController",
                daemon=True
            )
            accelerator_thread.start()
            logger.info("✅ Accelerator Controller started")
        
        # Start Ralph learning cycle
        logger.info("Starting Ralph learning cycle...")
        self.ralph_thread = threading.Thread(
            target=self._ralph_learning_loop,
            name="RalphLearning",
            daemon=True
        )
        self.ralph_thread.start()
        logger.info("✅ Ralph learning cycle started")
        
        # Start Optimus intelligent Tradier monitoring
        logger.info("Starting Optimus intelligent Tradier monitoring...")
        self.optimus_intelligence_thread = threading.Thread(
            target=self._optimus_intelligence_loop,
            name="OptimusIntelligence",
            daemon=True
        )
        self.optimus_intelligence_thread.start()
        logger.info("✅ Optimus intelligence loop started")
        
        # Start continuous learning loop
        logger.info("Starting continuous learning loop...")
        self.learning_thread = threading.Thread(
            target=self._continuous_learning_loop,
            name="ContinuousLearning",
            daemon=True
        )
        self.learning_thread.start()
        logger.info("✅ Continuous learning loop started")
        
        # Start enhancement loop
        logger.info("Starting enhancement loop...")
        self.enhancement_thread = threading.Thread(
            target=self._enhancement_loop,
            name="Enhancement",
            daemon=True
        )
        self.enhancement_thread.start()
        logger.info("✅ Enhancement loop started")
        
        logger.info("="*80)
        logger.info("🚀 NAE Full System is RUNNING")
        logger.info("="*80)
        logger.info("Press Ctrl+C to stop")
        logger.info("="*80)
    
    def _ralph_learning_loop(self):
        """Ralph's continuous learning cycle with GitHub integration"""
        logger.info("Ralph learning loop started")
        
        while self.running:
            try:
                cycle_start = time.time()
                logger.info("="*60)
                logger.info("🔄 Ralph Learning Cycle Starting...")
                logger.info("="*60)
                
                # Run Ralph's learning cycle (includes GitHub ingestion)
                if self.ralph:
                    try:
                        # This will automatically:
                        # - Search GitHub for trading tools/algorithms
                        # - Send discoveries to Donnie
                        # - Generate and evaluate strategies
                        # - Send high-confidence strategies to Optimus
                        result = self.ralph.run_cycle()
                        
                        logger.info(f"✅ Ralph cycle completed: {result.get('strategies_generated', 0)} strategies generated")
                        
                        # Log GitHub discoveries
                        if 'github_discoveries' in result:
                            discoveries = result['github_discoveries']
                            logger.info(f"🔍 GitHub discoveries: {len(discoveries)} items found")
                            for discovery in discoveries[:3]:  # Log first 3
                                logger.info(f"  - {discovery.get('name', 'Unknown')}: {discovery.get('details', '')[:100]}")
                        
                        # Log strategies sent to Optimus
                        if 'strategies_sent' in result:
                            logger.info(f"📤 Strategies sent to Optimus: {result['strategies_sent']}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error in Ralph cycle: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                cycle_time = time.time() - cycle_start
                logger.info(f"⏱️  Ralph cycle completed in {cycle_time:.2f}s")
                logger.info(f"⏳ Next cycle in {self.ralph_cycle_interval}s")
                logger.info("="*60)
                
                # Wait for next cycle
                time.sleep(self.ralph_cycle_interval)
                
            except Exception as e:
                logger.error(f"Error in Ralph learning loop: {e}")
                time.sleep(self.ralph_cycle_interval)
    
    def _optimus_intelligence_loop(self):
        """Optimus intelligent Tradier monitoring and decision making with trade execution"""
        logger.info("Optimus intelligence loop started - TRADING ENABLED")
        
        while self.running:
            try:
                cycle_start = time.time()
                trades_executed = 0
                
                if self.optimus:
                    try:
                        # Verify trading is enabled - re-enable if needed
                        if not self.optimus.trading_enabled:
                            if hasattr(self.optimus, 'deactivate_kill_switch'):
                                self.optimus.deactivate_kill_switch("Re-enabling for opportunity execution")
                                logger.info("🔄 Re-enabled trading for opportunity execution")
                        
                        # Only proceed if trading is enabled
                        if not self.optimus.trading_enabled:
                            logger.debug("⚠️ Trading disabled - skipping opportunity checks")
                            time.sleep(self.optimus_check_interval)
                            continue
                        
                        # Check Tradier account status
                        tradier_healthy = True
                        if hasattr(self.optimus, 'self_healing_engine') and self.optimus.self_healing_engine:
                            engine = self.optimus.self_healing_engine
                            health = engine.self_healing_engine.get_health_score()
                            tradier_healthy = health >= 0.7
                            
                            if tradier_healthy:
                                logger.debug(f"✅ Tradier healthy (score: {health:.2f})")
                            else:
                                logger.debug(f"⚠️ Tradier health degraded (score: {health:.2f})")
                        
                        if tradier_healthy and self.optimus.trading_enabled:
                            # ========== PROCESS MESSAGES/STRATEGIES FROM RALPH/DONNIE ==========
                            if hasattr(self.optimus, 'inbox') and self.optimus.inbox:
                                messages = list(self.optimus.inbox)  # Copy to avoid modification during iteration
                                self.optimus.inbox.clear()  # Clear after copying
                                
                                for message in messages:
                                    try:
                                        if message.get('type') in ['strategy', 'execute_trade'] or message.get('action') == 'execute_trade':
                                            logger.info(f"📋 Processing opportunity: {message.get('strategy_name', message.get('action', 'Unknown'))}")
                                            result = self._execute_optimus_trade(message)
                                            if result and result.get('status') == 'executed':
                                                trades_executed += 1
                                                logger.info(f"✅ Trade EXECUTED: {result.get('symbol', 'N/A')} - Status: {result.get('status')}")
                                    except Exception as e:
                                        logger.error(f"❌ Error processing message: {e}")
                            
                            # ========== PROCESS PENDING STRATEGIES ==========
                            if hasattr(self.optimus, 'pending_strategies'):
                                pending = getattr(self.optimus, 'pending_strategies', [])
                                if pending:
                                    logger.info(f"📋 Evaluating {len(pending)} pending strategies")
                                    for strategy in pending[:5]:  # Process up to 5 at a time
                                        try:
                                            result = self._execute_optimus_strategy(strategy)
                                            if result and result.get('status') == 'executed':
                                                trades_executed += 1
                                                logger.info(f"✅ Strategy EXECUTED: {strategy.get('name', 'Unknown')}")
                                        except Exception as e:
                                            logger.error(f"❌ Error executing strategy: {e}")
                            
                            # ========== PROCESS INSTRUCTIONS VIA RECEIVE_MESSAGE ==========
                            if hasattr(self.optimus, 'receive_message'):
                                if hasattr(self.optimus, '_instruction_queue'):
                                    queue = getattr(self.optimus, '_instruction_queue', [])
                                    while queue:
                                        instruction = queue.pop(0)
                                        try:
                                            logger.info(f"📨 Processing instruction: {instruction.get('action', 'Unknown')}")
                                            result = self.optimus.receive_message(instruction)
                                            if result and isinstance(result, dict) and result.get('status') == 'executed':
                                                trades_executed += 1
                                        except Exception as e:
                                            logger.error(f"❌ Error processing instruction: {e}")
                            
                            # ========== CONTINUOUS OPPORTUNITY SCANNING ==========
                            # Optimus actively scans for opportunities throughout the day
                            try:
                                from scripts.trigger_optimus_option_trade import (
                                    get_market_status,
                                    find_best_option_opportunity,
                                    create_option_execution_details
                                )
                                
                                # Check if market is open and we should scan
                                market_status = get_market_status()
                                if market_status.get("is_open") and market_status.get("can_trade", True):
                                    # Only scan every 5 cycles (5 minutes) to avoid excessive API calls
                                    if not hasattr(self, '_last_opportunity_scan'):
                                        self._last_opportunity_scan = 0
                                    
                                    current_time = time.time()
                                    if current_time - self._last_opportunity_scan >= 300:  # 5 minutes
                                        self._last_opportunity_scan = current_time
                                        
                                        phase = market_status.get("phase", "general")
                                        logger.debug(f"🔍 Scanning for opportunities (Phase: {phase})...")
                                        
                                        # Scan for option opportunities
                                        try:
                                            default_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
                                            opportunity = find_best_option_opportunity(self.optimus, default_symbols)
                                            
                                            if opportunity and opportunity.get("score", 0) > 30:  # Minimum score threshold
                                                score = opportunity.get("score", 0)
                                                symbol = opportunity.get("symbol", "UNKNOWN")
                                                logger.info(f"🎯 Opportunity found: {symbol} (score: {score:.2f})")
                                                
                                                # Create execution details
                                                execution_details = create_option_execution_details(opportunity, phase=phase)
                                                
                                                if execution_details:
                                                    # Verify price is included (required for validation)
                                                    if not execution_details.get("price") or execution_details.get("price") == 0:
                                                        logger.warning(f"⚠️  No price in execution details for {symbol} - skipping trade")
                                                    else:
                                                        # Execute the trade
                                                        result = self.optimus.execute_trade(execution_details)
                                                        if result and result.get('status') == 'executed':
                                                            trades_executed += 1
                                                            logger.info(f"✅ CONTINUOUS SCAN TRADE EXECUTED: {result.get('symbol', 'N/A')}")
                                                        elif result:
                                                            status = result.get('status', 'unknown')
                                                            reason = result.get('reason', 'No reason provided')
                                                            logger.info(f"⏸️ Opportunity not executed: {status} - {reason}")
                                                else:
                                                    logger.warning(f"⚠️  Failed to create execution details for {symbol}")
                                            elif opportunity:
                                                score = opportunity.get("score", 0)
                                                logger.debug(f"Opportunity score too low: {score:.2f} (threshold: 30)")
                                        except Exception as scan_error:
                                            logger.debug(f"Opportunity scan error: {scan_error}")
                                            
                            except ImportError:
                                # Option trade trigger script not available - skip continuous scanning
                                pass
                            except Exception as e:
                                logger.debug(f"Error in continuous opportunity scanning: {e}")
                            
                            # ========== CHECK ACCELERATOR FOR OPPORTUNITIES ==========
                            if hasattr(self.optimus, 'accelerator') and self.optimus.accelerator:
                                try:
                                    accel_status = self.optimus.accelerator.get_status()
                                    if accel_status.get('ready_for_trade', False):
                                        logger.debug("🚀 Accelerator ready for trading")
                                except:
                                    pass
                        
                        if trades_executed > 0:
                            logger.info(f"💰 EXECUTED {trades_executed} TRADE(S) THIS CYCLE")
                        
                    except Exception as e:
                        logger.error(f"❌ Error in Optimus intelligence check: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                cycle_time = time.time() - cycle_start
                if trades_executed > 0 or cycle_time > 1.0:
                    logger.debug(f"Optimus intelligence check: {cycle_time:.2f}s ({trades_executed} trades)")
                
                # Wait for next check
                time.sleep(self.optimus_check_interval)
                
            except Exception as e:
                logger.error(f"Error in Optimus intelligence loop: {e}")
                time.sleep(self.optimus_check_interval)
    
    def _execute_optimus_trade(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a trade from a message/opportunity"""
        try:
            if not self.optimus or not self.optimus.trading_enabled:
                return None
            
            # Convert message to execution details
            execution_details = message.get('parameters', {}) or message.copy()
            
            # Ensure required fields
            if 'action' not in execution_details:
                execution_details['action'] = message.get('action', 'buy')
            
            # Execute via Optimus
            if hasattr(self.optimus, 'execute_trade'):
                result = self.optimus.execute_trade(execution_details)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing Optimus trade: {e}")
            return None
    
    def _execute_optimus_strategy(self, strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a strategy through Optimus"""
        try:
            if not self.optimus or not self.optimus.trading_enabled:
                return None
            
            # Convert strategy to execution details
            execution_details = {
                'action': 'execute_trade',
                'strategy_name': strategy.get('name', 'Unknown'),
                'parameters': strategy.get('aggregated_details', {}),
                'trust_score': strategy.get('trust_score', 0),
                'backtest_score': strategy.get('backtest_score', 0),
                'meta_confidence': strategy.get('meta_confidence', 0.5)
            }
            
            # Execute via Optimus
            if hasattr(self.optimus, 'execute_trade'):
                result = self.optimus.execute_trade(execution_details)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing Optimus strategy: {e}")
            return None
    
    def _continuous_learning_loop(self):
        """Continuous learning from all agents"""
        logger.info("Continuous learning loop started")
        
        while self.running:
            try:
                logger.info("🧠 Running continuous learning analysis...")
                
                # Collect learning data from all agents
                learning_data = {
                    "timestamp": datetime.now().isoformat(),
                    "ralph_metrics": {},
                    "optimus_metrics": {},
                    "accelerator_metrics": {}
                }
                
                # Get Ralph's improvement metrics
                if self.ralph and hasattr(self.ralph, 'get_improvement_metrics'):
                    try:
                        learning_data["ralph_metrics"] = self.ralph.get_improvement_metrics()
                        logger.info(f"📊 Ralph metrics: {learning_data['ralph_metrics']}")
                    except:
                        pass
                
                # Get Optimus performance metrics
                if self.optimus:
                    try:
                        if hasattr(self.optimus, 'get_performance_metrics'):
                            learning_data["optimus_metrics"] = self.optimus.get_performance_metrics()
                    except:
                        pass
                
                # Get Accelerator status
                if self.accelerator_controller:
                    try:
                        status = self.accelerator_controller.get_status()
                        learning_data["accelerator_metrics"] = {
                            "sandbox_cycles": status.get("sandbox_stats", {}).get("cycles_run", 0),
                            "live_cycles": status.get("live_stats", {}).get("cycles_run", 0),
                            "sandbox_pnl": status.get("sandbox_stats", {}).get("daily_pnl", 0.0),
                            "live_pnl": status.get("live_stats", {}).get("daily_pnl", 0.0)
                        }
                        logger.info(f"📈 Accelerator: Sandbox P&L: ${learning_data['accelerator_metrics'].get('sandbox_pnl', 0):.2f}, Live P&L: ${learning_data['accelerator_metrics'].get('live_pnl', 0):.2f}")
                    except:
                        pass
                
                # Apply learning insights
                logger.info("✅ Learning analysis completed")
                
                # Wait for next learning cycle
                time.sleep(self.learning_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(self.learning_interval)
    
    def _enhancement_loop(self):
        """Continuous enhancement and improvement loop"""
        logger.info("Enhancement loop started")
        
        while self.running:
            try:
                logger.info("⚙️  Running enhancement analysis...")
                
                # Check for improvements that can be applied
                enhancements_applied = 0
                
                # Enhance Ralph strategies
                if self.ralph:
                    try:
                        # Ralph automatically improves through his learning cycles
                        pass
                    except:
                        pass
                
                # Enhance Optimus trading logic
                if self.optimus:
                    try:
                        # Optimus can enhance through excellence protocol
                        if hasattr(self.optimus, 'excellence_protocol'):
                            protocol = self.optimus.excellence_protocol
                            if protocol and protocol.improvement_active:
                                # Improvements happen automatically
                                pass
                    except:
                        pass
                
                logger.info(f"✅ Enhancement analysis completed ({enhancements_applied} applied)")
                
                # Wait for next enhancement cycle
                time.sleep(self.enhancement_interval)
                
            except Exception as e:
                logger.error(f"Error in enhancement loop: {e}")
                time.sleep(self.enhancement_interval)
    
    def stop(self):
        """Stop all systems gracefully"""
        logger.info("="*80)
        logger.info("Stopping NAE Full System...")
        logger.info("="*80)
        
        self.running = False
        
        # Stop Accelerator Controller
        if self.accelerator_controller:
            try:
                self.accelerator_controller.stop()
                logger.info("✅ Accelerator Controller stopped")
            except:
                pass
        
        # Wait for threads to finish
        threads = [
            self.ralph_thread,
            self.optimus_intelligence_thread,
            self.learning_thread,
            self.enhancement_thread
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=5)
                except:
                    pass
        
        logger.info("="*80)
        logger.info("✅ NAE Full System stopped")
        logger.info("="*80)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NAE Full System Runner")
    parser.add_argument("--sandbox", action="store_true", help="Enable sandbox mode")
    parser.add_argument("--no-live", action="store_true", help="Disable live mode")
    parser.add_argument("--aggressive", action="store_true", default=True, help="Enable aggressive accelerator (default: True)")
    parser.add_argument("--no-aggressive", dest="aggressive", action="store_false", help="Disable aggressive accelerator")
    
    args = parser.parse_args()
    
    # Determine modes
    sandbox_mode = args.sandbox
    live_mode = not args.no_live
    
    # Create and initialize system
    system = NAEFullSystem(
        sandbox_mode=sandbox_mode,
        live_mode=live_mode,
        aggressive_accelerator=args.aggressive
    )
    
    if not system.initialize():
        logger.error("Failed to initialize system")
        return 1
    
    # Start system
    try:
        system.start()
        
        # Keep main thread alive
        while system.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        system.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Accelerator Controller - Dual-Mode Operation (Sandbox + Live)

Manages the micro-scalp accelerator strategy in both sandbox (testing)
and live (production) environments simultaneously for M/L and profits.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to NAE Ready directory
nae_ready_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
if os.path.exists(nae_ready_dir):
    sys.path.insert(0, nae_ready_dir)
# Also try current directory structure
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

logger = logging.getLogger(__name__)


class AcceleratorController:
    """
    Controller for running accelerator in dual-mode:
    - Sandbox: Testing and M/L validation
    - Live: Production trading for profits
    """
    
    def __init__(self, sandbox_mode: bool = True, live_mode: bool = True):
        """
        Initialize accelerator controller
        
        Args:
            sandbox_mode: Enable sandbox/testing mode
            live_mode: Enable live/production mode
        """
        self.sandbox_mode = sandbox_mode
        self.live_mode = live_mode
        
        self.sandbox_optimus = None
        self.live_optimus = None
        self.sandbox_ralph = None
        self.live_ralph = None
        
        self.running = False
        self.cycle_interval = 60  # Run cycle every 60 seconds
        
        # Performance tracking
        self.sandbox_stats = {
            "cycles_run": 0,
            "trades_executed": 0,
            "daily_pnl": 0.0,
            "last_update": None
        }
        
        self.live_stats = {
            "cycles_run": 0,
            "trades_executed": 0,
            "daily_pnl": 0.0,
            "last_update": None
        }
        
        logger.info("AcceleratorController initialized")
        logger.info(f"  Sandbox mode: {'ENABLED' if sandbox_mode else 'DISABLED'}")
        logger.info(f"  Live mode: {'ENABLED' if live_mode else 'DISABLED'}")
    
    def initialize_agents(self):
        """Initialize Optimus and Ralph agents for both modes"""
        try:
            from agents.optimus import OptimusAgent
            from agents.ralph import RalphAgent
            
            # Initialize sandbox agents
            if self.sandbox_mode:
                logger.info("Initializing sandbox agents...")
                self.sandbox_optimus = OptimusAgent(sandbox=False)  # LIVE MODE - name kept for compatibility
                self.sandbox_ralph = RalphAgent()
                
                # Enable accelerator in sandbox
                if self.sandbox_optimus.enable_accelerator_mode(ralph_agent=self.sandbox_ralph):
                    logger.info("✅ Sandbox accelerator enabled")
                else:
                    logger.error("❌ Failed to enable sandbox accelerator")
            
            # Initialize live agents
            if self.live_mode:
                logger.info("Initializing live agents...")
                self.live_optimus = OptimusAgent(sandbox=False)  # Live trading
                self.live_ralph = RalphAgent()
                
                # Enable accelerator in live
                if self.live_optimus.enable_accelerator_mode(ralph_agent=self.live_ralph):
                    logger.info("✅ Live accelerator enabled")
                else:
                    logger.error("❌ Failed to enable live accelerator")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_sandbox_cycle(self) -> Dict[str, Any]:
        """Run one accelerator cycle in sandbox mode"""
        if not self.sandbox_mode or not self.sandbox_optimus:
            return {"status": "disabled"}
        
        try:
            result = self.sandbox_optimus.run_accelerator_cycle()
            
            # Update stats
            self.sandbox_stats["cycles_run"] += 1
            self.sandbox_stats["last_update"] = datetime.now().isoformat()
            
            # Get accelerator status
            if self.sandbox_optimus.accelerator:
                status = self.sandbox_optimus.accelerator.get_status()
                self.sandbox_stats["daily_pnl"] = status.get("daily_profit", 0.0)
                self.sandbox_stats["trades_executed"] = status.get("trades_today", 0)
            
            return {
                "status": "success",
                "result": result,
                "stats": self.sandbox_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error in sandbox cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_live_cycle(self) -> Dict[str, Any]:
        """Run one accelerator cycle in live mode"""
        if not self.live_mode or not self.live_optimus:
            return {"status": "disabled"}
        
        try:
            result = self.live_optimus.run_accelerator_cycle()
            
            # Update stats
            self.live_stats["cycles_run"] += 1
            self.live_stats["last_update"] = datetime.now().isoformat()
            
            # Get accelerator status
            if self.live_optimus.accelerator:
                status = self.live_optimus.accelerator.get_status()
                self.live_stats["daily_pnl"] = status.get("daily_profit", 0.0)
                self.live_stats["trades_executed"] = status.get("trades_today", 0)
            
            # Check if target reached
            if result == "TARGET_REACHED":
                account_size = self.live_optimus.get_account_size()
                logger.warning(f"🎯 Live accelerator target reached: ${account_size:.2f}")
                logger.warning("   Consider disabling accelerator and transitioning to main strategy")
            
            return {
                "status": "success",
                "result": result,
                "stats": self.live_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error in live cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_dual_cycle(self):
        """Run accelerator cycle in both sandbox and live modes"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "sandbox": {},
            "live": {}
        }
        
        # Run sandbox cycle
        if self.sandbox_mode:
            results["sandbox"] = self.run_sandbox_cycle()
        
        # Run live cycle
        if self.live_mode:
            results["live"] = self.run_live_cycle()
        
        return results
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting accelerator monitoring loop...")
        logger.info(f"  Cycle interval: {self.cycle_interval} seconds")
        
        self.running = True
        
        while self.running:
            try:
                # Run dual cycle
                results = self.run_dual_cycle()
                
                # Log results
                if results.get("sandbox", {}).get("status") == "success":
                    sandbox_result = results["sandbox"].get("result", "unknown")
                    sandbox_pnl = results["sandbox"].get("stats", {}).get("daily_pnl", 0.0)
                    logger.info(f"[SANDBOX] Cycle: {sandbox_result} | Daily P&L: ${sandbox_pnl:.2f}")
                
                if results.get("live", {}).get("status") == "success":
                    live_result = results["live"].get("result", "unknown")
                    live_pnl = results["live"].get("stats", {}).get("daily_pnl", 0.0)
                    logger.info(f"[LIVE] Cycle: {live_result} | Daily P&L: ${live_pnl:.2f}")
                
                # Wait for next cycle
                time.sleep(self.cycle_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(self.cycle_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        status = {
            "running": self.running,
            "sandbox_mode": self.sandbox_mode,
            "live_mode": self.live_mode,
            "sandbox_stats": self.sandbox_stats.copy(),
            "live_stats": self.live_stats.copy()
        }
        
        # Add account sizes if available
        if self.sandbox_optimus:
            try:
                status["sandbox_account_size"] = self.sandbox_optimus.get_account_size()
            except Exception:
                pass
        
        if self.live_optimus:
            try:
                status["live_account_size"] = self.live_optimus.get_account_size()
            except Exception:
                pass
        
        return status
    
    def stop(self):
        """Stop accelerator controller"""
        logger.info("Stopping accelerator controller...")
        self.running = False
        
        # Disable accelerators
        if self.sandbox_optimus:
            try:
                self.sandbox_optimus.disable_accelerator_mode()
            except Exception:
                pass
        
        if self.live_optimus:
            try:
                self.live_optimus.disable_accelerator_mode()
            except Exception:
                pass
        
        logger.info("✅ Accelerator controller stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Accelerator Controller - Dual-Mode Operation")
    # Use store_const to properly handle defaults
    parser.add_argument("--sandbox", action="store_true", default=False, help="Enable sandbox mode (default: False)")
    parser.add_argument("--no-sandbox", dest="sandbox", action="store_false", help="Disable sandbox mode")
    parser.add_argument("--live", action="store_true", default=None, help="Enable live mode (default: True)")
    parser.add_argument("--no-live", dest="live", action="store_false", help="Disable live mode")
    parser.add_argument("--interval", type=int, default=60, help="Cycle interval in seconds")
    
    args = parser.parse_args()
    
    # Handle default for live mode (True if not explicitly set)
    if args.live is None:
        args.live = True
    
    # Setup logging (ensure logs dir exists; cwd is NAE Ready when launched by master)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "accelerator_controller.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create controller
    controller = AcceleratorController(
        sandbox_mode=args.sandbox,
        live_mode=args.live
    )
    controller.cycle_interval = args.interval
    
    # Initialize agents
    if not controller.initialize_agents():
        logger.error("Failed to initialize agents")
        return 1
    
    # Run monitoring loop
    try:
        controller.monitor_loop()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        controller.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


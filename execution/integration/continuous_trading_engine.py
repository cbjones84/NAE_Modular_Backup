#!/usr/bin/env python3
"""
Continuous Trading Engine
Ensures trading operations run continuously and autonomously
"""

import os
import sys
import time
import logging
import threading
import json
import signal
import atexit
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging FIRST before any imports that might fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../..'))
execution_dir = os.path.abspath(os.path.join(script_dir, '..'))

sys.path.insert(0, nae_root)
sys.path.insert(0, execution_dir)

# Import with error handling and timeouts
TradierBalanceMonitor = None
DayTradingPrevention = None

try:
    from execution.monitoring.tradier_balance_monitor import TradierBalanceMonitor
    logger.info("✅ Imported TradierBalanceMonitor")
except ImportError as e:
    try:
        from monitoring.tradier_balance_monitor import TradierBalanceMonitor
        logger.info("✅ Imported TradierBalanceMonitor (alternative path)")
    except ImportError as e2:
        logger.error(f"❌ Failed to import TradierBalanceMonitor: {e}, {e2}")
        TradierBalanceMonitor = None

try:
    from execution.compliance.day_trading_prevention import DayTradingPrevention
    logger.info("✅ Imported DayTradingPrevention")
except ImportError as e:
    try:
        from compliance.day_trading_prevention import DayTradingPrevention
        logger.info("✅ Imported DayTradingPrevention (alternative path)")
    except ImportError as e2:
        logger.error(f"❌ Failed to import DayTradingPrevention: {e}, {e2}")
        DayTradingPrevention = None

# Import diagnostics
TradierDiagnostics = None
try:
    from execution.diagnostics.nae_tradier_diagnostics import TradierDiagnostics
    logger.info("✅ Imported TradierDiagnostics")
except ImportError as e:
    try:
        from diagnostics.nae_tradier_diagnostics import TradierDiagnostics
        logger.info("✅ Imported TradierDiagnostics (alternative path)")
    except ImportError as e2:
        logger.warning(f"⚠️ TradierDiagnostics not available: {e}, {e2}")
        TradierDiagnostics = None


class ContinuousTradingEngine:
    """
    Continuous trading engine that runs autonomously
    """
    
    def __init__(self):
        """Initialize continuous trading engine"""
        self.running = True
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # Logging setup
        self.log_file = "logs/continuous_trading_engine.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.log_action("Initializing Continuous Trading Engine...")
        
        # Initialize diagnostics
        self.diagnostics = None
        if TradierDiagnostics is not None:
            try:
                self.diagnostics = TradierDiagnostics(live=True)
                self.log_action("✅ Tradier diagnostics available")
            except Exception as e:
                self.log_action(f"⚠️ Could not initialize diagnostics: {e}")
        
        # Initialize Tradier balance monitor (with timeout protection)
        self.tradier_monitor = None
        if TradierBalanceMonitor is not None:
            try:
                self.log_action("Initializing TradierBalanceMonitor...")
                # Use threading to prevent hanging
                init_result = [None]
                init_error = [None]
                
                def init_monitor():
                    try:
                        init_result[0] = TradierBalanceMonitor(
                            check_interval=300,  # 5 minutes
                            funds_threshold=100.0,
                            sandbox=False
                        )
                    except Exception as e:
                        init_error[0] = e
                
                init_thread = threading.Thread(target=init_monitor, daemon=True)
                init_thread.start()
                init_thread.join(timeout=10)  # 10 second timeout
                
                if init_thread.is_alive():
                    logger.error("❌ TradierBalanceMonitor initialization timed out after 10 seconds")
                    self.tradier_monitor = None
                elif init_error[0]:
                    logger.error(f"❌ Failed to initialize TradierBalanceMonitor: {init_error[0]}")
                    self.tradier_monitor = None
                else:
                    self.tradier_monitor = init_result[0]
                    self.log_action("✅ TradierBalanceMonitor initialized")
            except Exception as e:
                logger.error(f"❌ Exception initializing TradierBalanceMonitor: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.tradier_monitor = None
        else:
            logger.warning("⚠️ TradierBalanceMonitor not available (import failed)")
        
        # Initialize day trading prevention
        self.day_trading_prevention = None
        if DayTradingPrevention is not None:
            try:
                self.log_action("Initializing DayTradingPrevention...")
                # DayTradingPrevention uses file-based storage (no Redis required)
                self.day_trading_prevention = DayTradingPrevention()
                self.log_action("✅ DayTradingPrevention initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize DayTradingPrevention: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.day_trading_prevention = None
        else:
            logger.warning("⚠️ DayTradingPrevention not available (import failed)")
        
        # Trading state
        self.trading_active = False
        self.last_trade_time = None
        self.trading_thread = None
        
        self.log_action("✅ Continuous Trading Engine initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        signal_name = signal.Signals(signum).name
        self.log_action(f"Received {signal_name} signal - shutting down gracefully")
        self.stop()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup on exit"""
        if self.running:
            self.stop()
    
    def log_action(self, message: str):
        """Log action"""
        ts = datetime.now().isoformat()
        log_msg = f"[{ts}] {message}"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log: {e}")
        
        logger.info(message)
        print(f"[Continuous Trading] {message}")
    
    def trading_loop(self):
        """Main trading loop"""
        self.log_action("Starting continuous trading loop")
        
        while self.running:
            try:
                # Check for funds
                if not self.tradier_monitor:
                    self.log_action("⚠️ Tradier monitor not initialized - waiting...")
                    time.sleep(300)
                    continue
                
                # Check balance with timeout protection
                balance = None
                try:
                    # Use threading to prevent hanging on API call
                    balance_result = [None]
                    balance_error = [None]
                    
                    def check_balance():
                        try:
                            balance_result[0] = self.tradier_monitor.check_balance()
                        except Exception as e:
                            balance_error[0] = e
                    
                    balance_thread = threading.Thread(target=check_balance, daemon=True)
                    balance_thread.start()
                    balance_thread.join(timeout=30)  # 30 second timeout for API call
                    
                    if balance_thread.is_alive():
                        self.log_action("⚠️ Balance check timed out after 30 seconds")
                        balance = None
                    elif balance_error[0]:
                        self.log_action(f"⚠️ Balance check error: {balance_error[0]}")
                        balance = None
                    else:
                        balance = balance_result[0]
                
                except Exception as e:
                    self.log_action(f"⚠️ Exception checking balance: {e}")
                    balance = None
                
                if balance and hasattr(balance, 'has_funds') and balance.has_funds:
                    if not self.trading_active:
                        self.log_action("💰 Funds available - Activating trading")
                        self.activate_trading(balance)
                    
                    # Trading is active - perform trading operations
                    self.execute_trading_cycle()
                else:
                    if self.trading_active:
                        self.log_action("⚠️ No funds available - Trading paused")
                        self.trading_active = False
                    elif balance is None:
                        self.log_action("⚠️ Could not check balance - will retry")
                
                # Sleep before next cycle
                time.sleep(300)  # 5 minutes
            
            except Exception as e:
                self.log_action(f"Error in trading loop: {e}")
                logger.exception("Trading loop error")
                time.sleep(300)
    
    def activate_trading(self, balance):
        """Activate trading"""
        self.trading_active = True
        self.log_action("🚀 Trading activated")
        self.log_action(f"   Cash: ${balance.cash:,.2f}")
        self.log_action(f"   Cash Available: ${balance.cash_available:,.2f}")
        self.log_action(f"   Pending Cash: ${balance.pending_cash:,.2f}")
        if balance.margin_balance != 0:
            self.log_action(f"   Margin Balance: ${balance.margin_balance:,.2f}")
        self.log_action(f"   Buying Power: ${balance.buying_power:,.2f}")
        self.log_action(f"   Total Available: ${balance.total_available:,.2f}")
    
    def run_diagnostics(self, test_symbol: str = "SPY250117C00500000") -> Dict[str, Any]:
        """
        Run Tradier diagnostics to identify why trades aren't being placed
        
        Args:
            test_symbol: Option symbol to test
        
        Returns:
            Diagnostics results
        """
        if not self.diagnostics:
            self.log_action("⚠️ Diagnostics not available")
            return {}
        
        self.log_action("🔍 Running Tradier diagnostics...")
        try:
            results = self.diagnostics.run_full_diagnostics(test_symbol=test_symbol)
            self.log_action("✅ Diagnostics complete - Check logs for details")
            return results
        except Exception as e:
            self.log_action(f"❌ Diagnostics error: {e}")
            logger.exception("Diagnostics error")
            return {}
    
    def execute_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Check compliance if available
            if self.day_trading_prevention:
                try:
                    compliance = self.day_trading_prevention.get_compliance_status()
                    
                    if not compliance.get("can_day_trade", True) and compliance.get("day_trades_in_period", 0) >= 3:
                        self.log_action("⚠️ Day trading limit reached - Waiting for reset")
                        return
                except Exception as e:
                    self.log_action(f"⚠️ Compliance check error (continuing): {e}")
            
            # Trading logic would go here
            # For now, just log that we're ready to trade
            self.log_action("✅ Trading cycle ready - Compliance OK")
            
        except Exception as e:
            self.log_action(f"Error in trading cycle: {e}")
            logger.exception("Trading cycle error")
    
    def start(self):
        """Start continuous trading"""
        self.log_action("Starting Continuous Trading Engine")
        
        # Start Tradier monitor if available (non-blocking)
        if self.tradier_monitor:
            try:
                # Don't start monitoring thread - we'll check balance manually in trading_loop
                # This prevents hanging if start_monitoring() blocks
                self.log_action("✅ Tradier monitor ready (manual balance checks)")
            except Exception as e:
                self.log_action(f"⚠️ Failed to prepare monitoring: {e}")
        else:
            self.log_action("⚠️ Tradier monitor not available - running in limited mode")
        
        # Start trading loop in thread
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
        self.log_action("✅ Trading loop thread started")
        
        # Keep main thread alive with periodic heartbeat
        try:
            heartbeat_interval = 300  # Log heartbeat every 5 minutes
            last_heartbeat_log = time.time()
            
            while self.running:
                time.sleep(60)
                
                # Periodic heartbeat to show we're alive
                current_time = time.time()
                if current_time - last_heartbeat_log >= heartbeat_interval:
                    uptime = datetime.now() - self.start_time
                    self.log_action(f"💓 Heartbeat - Uptime: {uptime}, Status: Running")
                    last_heartbeat_log = current_time
                    self.last_heartbeat = datetime.now()
                    
                    # Verify trading thread is still alive
                    if self.trading_thread and not self.trading_thread.is_alive():
                        self.log_action("⚠️ Trading thread died - restarting...")
                        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
                        self.trading_thread.start()
                        self.log_action("✅ Trading thread restarted")
        
        except KeyboardInterrupt:
            self.log_action("Received KeyboardInterrupt - shutting down")
            self.stop()
        except Exception as e:
            self.log_action(f"Error in main loop: {e}")
            logger.exception("Main loop error")
            # Don't exit - keep trying
            time.sleep(60)
    
    def stop(self):
        """Stop continuous trading"""
        self.log_action("Stopping Continuous Trading Engine")
        self.running = False
        if self.tradier_monitor:
            try:
                self.tradier_monitor.stop_monitoring()
            except Exception as e:
                logger.error(f"Error stopping monitor: {e}")


if __name__ == "__main__":
    try:
        engine = ContinuousTradingEngine()
        engine.start()
    except Exception as e:
        logger.error(f"Fatal error in Continuous Trading Engine: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


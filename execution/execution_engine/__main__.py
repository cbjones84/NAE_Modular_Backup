"""
Execution Engine Main Entry Point

Starts execution manager with primary engine (LEAN) and backup engines.
"""

import os
import sys
import logging
import signal
import time
from execution.execution_engine.execution_manager import get_execution_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global execution manager
execution_manager = None


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received, stopping execution manager...")
    if execution_manager:
        execution_manager.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    global execution_manager
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("="*60)
    logger.info("NAE Execution Engine Starting")
    logger.info("="*60)
    logger.info(f"Primary Engine: LEAN Self-Hosted")
    logger.info(f"Backup Engines: QuantTrader/PyBroker, NautilusTrader")
    logger.info("="*60)
    
    try:
        # Initialize execution manager
        execution_manager = get_execution_manager()
        
        # Start with primary engine
        if not execution_manager.start():
            logger.error("Failed to start primary execution engine")
            sys.exit(1)
        
        logger.info("Execution engine started successfully")
        
        # Monitor and check for primary recovery periodically
        while True:
            time.sleep(60)  # Check every minute
            
            # Check if primary engine has recovered
            execution_manager.check_primary_recovery()
            
            # Log status periodically
            status = execution_manager.get_status()
            if status["active_engine"] != status["primary_engine"]:
                logger.info(
                    f"Current engine: {status['active_engine']} "
                    f"(Primary: {status['primary_engine']})"
                )
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if execution_manager:
            execution_manager.stop()
        logger.info("Execution engine stopped")


if __name__ == "__main__":
    main()


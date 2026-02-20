"""
Universal restart wrapper for all NAE agents.
Ensures agents NEVER stop running - automatically restarts on any error or exit.
"""
import time
import traceback
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def run_forever(agent_name: str, main_function: Callable, restart_delay: int = 5):
    """
    Run an agent function forever with automatic restart on any exit.
    
    Args:
        agent_name: Name of the agent (for logging)
        main_function: Function to run continuously
        restart_delay: Initial delay before restart (increases exponentially)
    
    This function NEVER returns - it ensures the agent runs forever.
    """
    restart_count = 0
    max_restart_delay = 3600  # Max 1 hour delay
    
    while True:  # Outer infinite loop - NEVER EXIT
        try:
            logger.info("=" * 70)
            logger.info(f"🚀 Starting {agent_name} (Restart #{restart_count})")
            logger.info("=" * 70)
            
            # Run the main function
            main_function()
            
            # If function returns normally, restart immediately
            restart_count += 1
            logger.warning(f"⚠️  {agent_name} exited normally - RESTARTING (Restart #{restart_count})")
            logger.info("Restarting in 5 seconds...")
            time.sleep(5)
            
        except KeyboardInterrupt:
            restart_count += 1
            logger.warning(f"⚠️  KeyboardInterrupt received for {agent_name} - RESTARTING (Restart #{restart_count})")
            logger.info("Restarting in 5 seconds...")
            time.sleep(5)
            # Continue outer loop - NEVER STOP
            
        except SystemExit:
            restart_count += 1
            logger.warning(f"⚠️  SystemExit received for {agent_name} - RESTARTING (Restart #{restart_count})")
            logger.info("Restarting in 10 seconds...")
            time.sleep(10)
            # Continue outer loop - NEVER STOP
            
        except Exception as e:
            restart_count += 1
            delay = min(restart_delay * restart_count, max_restart_delay)  # Exponential backoff, max 1 hour
            logger.error(f"❌ Fatal error in {agent_name} (Restart #{restart_count}): {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            logger.info(f"🔄 Restarting in {delay} seconds...")
            time.sleep(delay)
            # Continue outer loop - NEVER STOP


def continuous_operation_loop(agent_name: str, operation_function: Callable, 
                               interval: int = 60, max_errors: int = 1000):
    """
    Run an operation function continuously in a loop that never stops.
    
    Args:
        agent_name: Name of the agent
        operation_function: Function to call repeatedly
        interval: Seconds between operations
        max_errors: Maximum consecutive errors before exponential backoff
    
    This function NEVER returns - it ensures continuous operation.
    """
    error_count = 0
    cycle_count = 0
    
    while True:  # Infinite loop - NEVER EXIT
        try:
            cycle_count += 1
            logger.debug(f"[{agent_name}] Cycle #{cycle_count}")
            
            # Execute the operation
            operation_function()
            
            # Reset error count on success
            error_count = 0
            
            # Wait before next cycle
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.warning(f"⚠️  KeyboardInterrupt for {agent_name} - Continuing operation...")
            time.sleep(interval)
            # Continue loop - NEVER STOP
            
        except SystemExit:
            logger.warning(f"⚠️  SystemExit for {agent_name} - Continuing operation...")
            time.sleep(interval)
            # Continue loop - NEVER STOP
            
        except Exception as e:
            error_count += 1
            delay = min(interval * (2 ** min(error_count // 10, 5)), 3600)  # Exponential backoff
            
            logger.error(f"❌ Error in {agent_name} cycle #{cycle_count} (Error #{error_count}): {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            
            if error_count >= max_errors:
                logger.critical(f"⚠️  {agent_name} has {error_count} consecutive errors - using extended delay")
            
            logger.info(f"🔄 Retrying in {delay} seconds...")
            time.sleep(delay)
            # Continue loop - NEVER STOP


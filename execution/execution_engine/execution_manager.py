"""
Execution Manager - Routes to appropriate execution engine with failover

Primary: LEAN self-hosted
Secondary backups: QuantTrader/PyBroker, NautilusTrader
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ExecutionEngineType(Enum):
    LEAN_SELF_HOSTED = "lean_self_hosted"
    QUANTTRADER_PYBROKER = "quanttrader_pybroker"
    NAUTILUS_TRADER = "nautilus_trader"


class ExecutionManager:
    """
    Manages execution engine selection and routing with automatic failover
    
    Primary: LEAN self-hosted
    Secondary: QuantTrader/PyBroker, NautilusTrader
    """
    
    def __init__(self):
        """Initialize execution manager with primary and backup engines"""
        # Engine priority order
        self.engine_priority = [
            ExecutionEngineType.LEAN_SELF_HOSTED,  # Primary
            ExecutionEngineType.QUANTTRADER_PYBROKER,  # Secondary 1
            ExecutionEngineType.NAUTILUS_TRADER  # Secondary 2
        ]
        
        # Current active engine
        self.active_engine_type: Optional[ExecutionEngineType] = None
        self.active_engine = None
        
        # All initialized engines
        self.engines: Dict[ExecutionEngineType, Any] = {}
        
        # Failover tracking
        self.failure_counts: Dict[ExecutionEngineType, int] = {}
        self.last_failure_time: Dict[ExecutionEngineType, Optional[datetime]] = {}
        self.failover_threshold = int(os.getenv("EXECUTION_FAILOVER_THRESHOLD", "5"))
        self.failover_timeout = int(os.getenv("EXECUTION_FAILOVER_TIMEOUT", "300"))  # 5 minutes
        
        # Initialize all engines
        self._initialize_all_engines()
        
        # Start with primary engine
        self._switch_to_engine(self.engine_priority[0])
        
        logger.info(f"Execution Manager initialized - Primary: {self.engine_priority[0].value}")
    
    def _initialize_all_engines(self):
        """Initialize all execution engines"""
        for engine_type in self.engine_priority:
            try:
                engine = self._create_engine(engine_type)
                self.engines[engine_type] = engine
                logger.info(f"Initialized {engine_type.value} engine")
            except Exception as e:
                logger.warning(f"Failed to initialize {engine_type.value}: {e}")
                self.engines[engine_type] = None
    
    def _create_engine(self, engine_type: ExecutionEngineType):
        """Create execution engine instance"""
        if engine_type == ExecutionEngineType.LEAN_SELF_HOSTED:
            from execution.execution_engine.lean_self_hosted import LEANSelfHostedEngine
            return LEANSelfHostedEngine()
        
        elif engine_type == ExecutionEngineType.QUANTTRADER_PYBROKER:
            from execution.execution_engine.quanttrader_adapter import QuantTraderPyBrokerAdapter
            return QuantTraderPyBrokerAdapter()
        
        elif engine_type == ExecutionEngineType.NAUTILUS_TRADER:
            from execution.execution_engine.nautilus_adapter import NautilusTraderAdapter
            return NautilusTraderAdapter()
        
        else:
            raise ValueError(f"Unsupported execution engine type: {engine_type}")
    
    def _switch_to_engine(self, engine_type: ExecutionEngineType):
        """Switch to specified execution engine"""
        if engine_type not in self.engines or self.engines[engine_type] is None:
            logger.error(f"Cannot switch to {engine_type.value}: engine not available")
            return False
        
        # Stop current engine if running
        if self.active_engine and hasattr(self.active_engine, 'stop'):
            try:
                self.active_engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping current engine: {e}")
        
        # Start new engine
        self.active_engine_type = engine_type
        self.active_engine = self.engines[engine_type]
        
        # Start engine if needed
        if hasattr(self.active_engine, 'start'):
            success = self.active_engine.start()
            if not success:
                logger.error(f"Failed to start {engine_type.value}")
                return False
        
        logger.info(f"Switched to execution engine: {engine_type.value}")
        return True
    
    def start(self):
        """Start execution manager with primary engine"""
        return self._switch_to_engine(self.engine_priority[0])
    
    def stop(self):
        """Stop all execution engines"""
        if self.active_engine and hasattr(self.active_engine, 'stop'):
            self.active_engine.stop()
    
    def record_failure(self, error: str = None):
        """Record execution engine failure and trigger failover if needed"""
        if not self.active_engine_type:
            return
        
        # Increment failure count
        if self.active_engine_type not in self.failure_counts:
            self.failure_counts[self.active_engine_type] = 0
        
        self.failure_counts[self.active_engine_type] += 1
        self.last_failure_time[self.active_engine_type] = datetime.now()
        
        logger.warning(
            f"Execution engine {self.active_engine_type.value} failure "
            f"(count: {self.failure_counts[self.active_engine_type]})"
        )
        
        # Check if failover needed
        if self.failure_counts[self.active_engine_type] >= self.failover_threshold:
            self._trigger_failover()
    
    def record_success(self):
        """Record successful execution"""
        if self.active_engine_type:
            # Reset failure count on success
            self.failure_counts[self.active_engine_type] = 0
            self.last_failure_time[self.active_engine_type] = None
    
    def _trigger_failover(self):
        """Trigger failover to next available engine"""
        current_index = self.engine_priority.index(self.active_engine_type)
        
        # Try next engines in priority order
        for i in range(current_index + 1, len(self.engine_priority)):
            next_engine = self.engine_priority[i]
            
            if next_engine in self.engines and self.engines[next_engine] is not None:
                logger.critical(
                    f"FAILOVER: Switching from {self.active_engine_type.value} "
                    f"to {next_engine.value}"
                )
                
                if self._switch_to_engine(next_engine):
                    # Reset failure count for new engine
                    self.failure_counts[next_engine] = 0
                    return True
        
        logger.error("FAILOVER FAILED: No backup engines available")
        return False
    
    def check_primary_recovery(self):
        """Check if primary engine has recovered and switch back"""
        primary = self.engine_priority[0]
        
        # Only check if not currently on primary
        if self.active_engine_type == primary:
            return
        
        # Check if primary is available and failure timeout has passed
        if primary in self.engines and self.engines[primary] is not None:
            last_failure = self.last_failure_time.get(primary)
            if last_failure:
                elapsed = (datetime.now() - last_failure).total_seconds()
                if elapsed > self.failover_timeout:
                    logger.info(f"Primary engine {primary.value} recovered, switching back")
                    self._switch_to_engine(primary)
                    self.failure_counts[primary] = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution manager status"""
        status = {
            "active_engine": self.active_engine_type.value if self.active_engine_type else None,
            "primary_engine": self.engine_priority[0].value,
            "backup_engines": [e.value for e in self.engine_priority[1:]],
            "failure_counts": {
                k.value: v for k, v in self.failure_counts.items()
            },
            "engines_available": {
                k.value: v is not None for k, v in self.engines.items()
            }
        }
        
        if self.active_engine and hasattr(self.active_engine, 'get_status'):
            engine_status = self.active_engine.get_status()
            status["engine_status"] = engine_status
        
        return status


def get_execution_manager() -> ExecutionManager:
    """Get execution manager instance (singleton pattern)"""
    global _execution_manager_instance
    if '_execution_manager_instance' not in globals():
        _execution_manager_instance = ExecutionManager()
    return _execution_manager_instance


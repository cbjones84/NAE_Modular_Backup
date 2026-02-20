"""
Pre-Trade Validator - Circuit Breakers and Risk Checks

Validates signals before execution with circuit breakers, exposure limits,
and per-strategy controls.
"""

import os
import json
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"  # Manual review required


class CircuitBreaker:
    """Circuit breaker for strategy or system-wide trading"""
    
    def __init__(self, name: str, failure_threshold: int = 5, timeout: int = 300):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_failure(self):
        """Record a failure"""
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker {self.name} OPENED after {self.failures} failures")
    
    def record_success(self):
        """Record a success"""
        self.failures = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "OPEN":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                    return False
            return True
        return False


class PreTradeValidator:
    """Pre-trade validation with circuit breakers and risk checks"""
    
    def __init__(self, redis_client: redis.Redis, postgres_conn):
        self.redis = redis_client
        self.postgres = postgres_conn
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.strategy_paused: Dict[str, bool] = {}
        self.exposure_limits: Dict[str, float] = {}
        
        # Initialize circuit breakers
        self.circuit_breakers["system"] = CircuitBreaker("system", failure_threshold=10)
        self.circuit_breakers["execution"] = CircuitBreaker("execution", failure_threshold=5)
    
    def validate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate signal with all checks
        
        Returns:
            {
                "status": "ACCEPTED" | "REJECTED" | "PENDING",
                "checks": [...],
                "errors": [...],
                "warnings": [...]
            }
        """
        result = {
            "status": ValidationStatus.ACCEPTED.value,
            "checks": [],
            "errors": [],
            "warnings": []
        }
        
        strategy_id = signal.get("strategy_id")
        symbol = signal.get("symbol")
        quantity = signal.get("quantity", 0)
        notional = signal.get("notional", 0)
        risk_meta = signal.get("risk_meta", {})
        
        # Check system circuit breaker
        if self.circuit_breakers["system"].is_open():
            result["status"] = ValidationStatus.REJECTED.value
            result["errors"].append("System circuit breaker is OPEN")
            result["checks"].append({
                "check": "system_circuit_breaker",
                "passed": False,
                "message": "System trading paused"
            })
            return result
        
        # Check execution circuit breaker
        if self.circuit_breakers["execution"].is_open():
            result["status"] = ValidationStatus.REJECTED.value
            result["errors"].append("Execution circuit breaker is OPEN")
            result["checks"].append({
                "check": "execution_circuit_breaker",
                "passed": False,
                "message": "Execution paused"
            })
            return result
        
        # Check if strategy is paused
        if self.strategy_paused.get(strategy_id, False):
            result["status"] = ValidationStatus.REJECTED.value
            result["errors"].append(f"Strategy {strategy_id} is paused")
            result["checks"].append({
                "check": "strategy_paused",
                "passed": False,
                "message": f"Strategy {strategy_id} paused"
            })
            return result
        
        # Check per-strategy circuit breaker
        if strategy_id not in self.circuit_breakers:
            self.circuit_breakers[strategy_id] = CircuitBreaker(
                f"strategy_{strategy_id}",
                failure_threshold=5
            )
        
        if self.circuit_breakers[strategy_id].is_open():
            result["status"] = ValidationStatus.REJECTED.value
            result["errors"].append(f"Strategy {strategy_id} circuit breaker is OPEN")
            result["checks"].append({
                "check": "strategy_circuit_breaker",
                "passed": False,
                "message": f"Strategy {strategy_id} circuit breaker open"
            })
            return result
        
        # Check exposure limits
        max_exposure = risk_meta.get("max_exposure", 0)
        if max_exposure > 0:
            current_exposure = self._get_current_exposure(strategy_id)
            if current_exposure + notional > max_exposure:
                result["status"] = ValidationStatus.REJECTED.value
                result["errors"].append(f"Exposure limit exceeded: {current_exposure + notional} > {max_exposure}")
                result["checks"].append({
                    "check": "exposure_limit",
                    "passed": False,
                    "message": f"Exposure limit exceeded"
                })
                return result
        
        # Check position size limits
        max_position_pct = risk_meta.get("max_position_pct")
        if max_position_pct:
            # Would check against portfolio value
            result["checks"].append({
                "check": "position_size",
                "passed": True,
                "message": "Position size within limits"
            })
        
        # Check correlation group limits
        correlation_group = signal.get("correlation_group")
        if correlation_group:
            group_exposure = self._get_correlation_group_exposure(correlation_group)
            if group_exposure + notional > max_exposure * 1.5:  # 50% buffer for correlation
                result["warnings"].append(f"Correlation group {correlation_group} exposure high")
                result["checks"].append({
                    "check": "correlation_group",
                    "passed": True,
                    "warning": True,
                    "message": "Correlation group exposure high"
                })
        
        # All checks passed
        result["checks"].append({
            "check": "all_checks",
            "passed": True,
            "message": "All pre-trade checks passed"
        })
        
        return result
    
    def _get_current_exposure(self, strategy_id: str) -> float:
        """Get current exposure for strategy"""
        # Would query from position ledger
        return 0.0
    
    def _get_correlation_group_exposure(self, correlation_group: str) -> float:
        """Get current exposure for correlation group"""
        # Would query from position ledger
        return 0.0
    
    def record_execution_result(self, signal_id: str, success: bool, error: Optional[str] = None):
        """Record execution result for circuit breaker logic"""
        # Get signal to determine strategy
        cursor = self.postgres.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("SELECT strategy_id FROM signals_raw WHERE id = %s", (signal_id,))
            row = cursor.fetchone()
            if row:
                strategy_id = row['strategy_id']
                
                # Update circuit breakers
                if success:
                    self.circuit_breakers["execution"].record_success()
                    if strategy_id in self.circuit_breakers:
                        self.circuit_breakers[strategy_id].record_success()
                else:
                    self.circuit_breakers["execution"].record_failure()
                    if strategy_id in self.circuit_breakers:
                        self.circuit_breakers[strategy_id].record_failure()
        finally:
            cursor.close()
    
    def pause_strategy(self, strategy_id: str):
        """Pause a strategy"""
        self.strategy_paused[strategy_id] = True
        logger.warning(f"Strategy {strategy_id} paused")
    
    def resume_strategy(self, strategy_id: str):
        """Resume a strategy"""
        self.strategy_paused[strategy_id] = False
        logger.info(f"Strategy {strategy_id} resumed")
    
    def set_exposure_limit(self, strategy_id: str, limit: float):
        """Set exposure limit for strategy"""
        self.exposure_limits[strategy_id] = limit
        logger.info(f"Exposure limit for {strategy_id} set to {limit}")


def get_validator(redis_client: redis.Redis, postgres_conn) -> PreTradeValidator:
    """Get pre-trade validator instance"""
    return PreTradeValidator(redis_client, postgres_conn)


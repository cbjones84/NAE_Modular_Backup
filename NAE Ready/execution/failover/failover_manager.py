"""
Failover Manager - Secondary Broker Routing

Handles failover to secondary broker (IBKR/Tradier) when primary (Schwab) fails.
"""

import os
import json
import redis
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class BrokerStatus(Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    FAILED = "FAILED"
    STANDBY = "STANDBY"


class FailoverManager:
    """
    Manages broker failover and routing
    
    Also coordinates with execution engine failover
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.primary_broker = os.getenv("PRIMARY_BROKER", "schwab")
        
        # Broker priority list (can include Tradier)
        broker_priority = os.getenv("BROKER_PRIORITY", "schwab,ibkr,tradier").split(",")
        self.broker_priority = [b.strip() for b in broker_priority]
        
        self.secondary_broker = self.broker_priority[1] if len(self.broker_priority) > 1 else "ibkr"
        self.tertiary_broker = self.broker_priority[2] if len(self.broker_priority) > 2 else "tradier"
        
        self.failover_threshold = int(os.getenv("FAILOVER_THRESHOLD", "5"))  # failures
        self.failover_timeout = int(os.getenv("FAILOVER_TIMEOUT", "300"))  # seconds
        
        # Execution engine failover coordination
        self.execution_engine_failover_enabled = os.getenv("EXECUTION_ENGINE_FAILOVER", "true").lower() == "true"
        
        # Initialize broker status for all brokers
        self.broker_status: Dict[str, BrokerStatus] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, Optional[datetime]] = {}
        
        for i, broker in enumerate(self.broker_priority):
            if i == 0:
                self.broker_status[broker] = BrokerStatus.PRIMARY
            else:
                self.broker_status[broker] = BrokerStatus.STANDBY
            self.failure_counts[broker] = 0
            self.last_failure_time[broker] = None
    
    def record_broker_failure(self, broker: str, error_type: str):
        """Record broker failure"""
        if broker not in self.failure_counts:
            return
        
        self.failure_counts[broker] += 1
        self.last_failure_time[broker] = datetime.now()
        
        logger.warning(f"Broker {broker} failure recorded: {error_type} (count: {self.failure_counts[broker]})")
        
        # Check if failover needed
        if broker == self.primary_broker and self.failure_counts[broker] >= self.failover_threshold:
            self.trigger_failover()
    
    def record_broker_success(self, broker: str):
        """Record broker success"""
        if broker not in self.failure_counts:
            return
        
        self.failure_counts[broker] = 0
        self.last_failure_time[broker] = None
        
        # If primary recovered, switch back
        if broker == self.primary_broker and self.broker_status[broker] == BrokerStatus.FAILED:
            if self._check_primary_recovery():
                self.switch_to_primary()
    
    def trigger_failover(self):
        """Trigger failover to next available broker"""
        current_broker = self.get_active_broker()
        current_index = self.broker_priority.index(current_broker) if current_broker in self.broker_priority else 0
        
        # Try next brokers in priority order
        for i in range(current_index + 1, len(self.broker_priority)):
            next_broker = self.broker_priority[i]
            
            if next_broker in self.broker_status and self.broker_status[next_broker] == BrokerStatus.STANDBY:
                logger.critical(
                    f"FAILOVER TRIGGERED: Switching from {current_broker} to {next_broker}"
                )
                
                # Mark current broker as failed
                self.broker_status[current_broker] = BrokerStatus.FAILED
                
                # Activate next broker
                self.broker_status[next_broker] = BrokerStatus.PRIMARY
                
                # Pause failed broker, activate new broker
                self.redis.set(f"broker:{current_broker}:paused", "true")
                self.redis.set(f"broker:{next_broker}:paused", "false")
                
                # Notify monitoring
                self._notify_failover(current_broker, next_broker)
                return True
        
        logger.error("FAILOVER FAILED: No backup brokers available")
        return False
    
    def switch_to_primary(self):
        """Switch back to primary broker"""
        primary = self.broker_priority[0]
        current_broker = self.get_active_broker()
        
        if current_broker == primary:
            logger.info("Already on primary broker")
            return
        
        logger.info(f"Switching back to primary broker: {primary}")
        
        # Mark current broker as standby
        self.broker_status[current_broker] = BrokerStatus.STANDBY
        
        # Activate primary
        self.broker_status[primary] = BrokerStatus.PRIMARY
        
        # Resume primary broker routing, pause current
        self.redis.set(f"broker:{primary}:paused", "false")
        self.redis.set(f"broker:{current_broker}:paused", "true")
        
        # Reset failure counts
        self.failure_counts[primary] = 0
    
    def get_active_broker(self) -> str:
        """Get currently active broker"""
        for broker, status in self.broker_status.items():
            if status == BrokerStatus.PRIMARY:
                return broker
        # Fallback to first available broker
        return self.broker_priority[0] if self.broker_priority else "schwab"
    
    def is_broker_available(self, broker: str) -> bool:
        """Check if broker is available"""
        status = self.broker_status.get(broker)
        return status in [BrokerStatus.PRIMARY, BrokerStatus.STANDBY]
    
    def route_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Route signal to appropriate broker"""
        active_broker = self.get_active_broker()
        
        # Check if broker is paused
        paused = self.redis.get(f"broker:{active_broker}:paused")
        if paused == "true":
            logger.warning(f"Broker {active_broker} is paused, using secondary")
            active_broker = self.secondary_broker
        
        return {
            "broker": active_broker,
            "status": self.broker_status[active_broker].value,
            "routing_decision": "primary" if active_broker == self.primary_broker else "failover"
        }
    
    def _check_primary_recovery(self) -> bool:
        """Check if primary broker has recovered"""
        if self.last_failure_time[self.primary_broker]:
            elapsed = (datetime.now() - self.last_failure_time[self.primary_broker]).total_seconds()
            return elapsed > self.failover_timeout
        return False
    
    def _notify_failover(self, from_broker: str, to_broker: str):
        """Notify monitoring system of failover"""
        failover_event = {
            "event": "failover_triggered",
            "timestamp": datetime.now().isoformat(),
            "from_broker": from_broker,
            "to_broker": to_broker,
            "reason": f"Failure count exceeded threshold: {self.failure_counts[from_broker]}"
        }
        
        # Push to monitoring queue
        self.redis.lpush("monitoring.events", json.dumps(failover_event))
        
        # Also log critical alert
        logger.critical(f"FAILOVER EVENT: {json.dumps(failover_event)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get failover manager status"""
        return {
            "broker_priority": self.broker_priority,
            "primary_broker": self.broker_priority[0] if self.broker_priority else None,
            "broker_status": {k: v.value for k, v in self.broker_status.items()},
            "failure_counts": self.failure_counts,
            "active_broker": self.get_active_broker()
        }


def get_failover_manager(redis_client: redis.Redis) -> FailoverManager:
    """Get failover manager instance"""
    return FailoverManager(redis_client)


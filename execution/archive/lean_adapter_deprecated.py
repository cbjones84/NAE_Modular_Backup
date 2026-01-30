"""
LEAN Execution Adapter

Connects LEAN algorithm to NAE signal queue and Schwab brokerage.
"""

import os
import json
import redis
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from quantconnect.algorithm import QCAlgorithm
from quantconnect.orders import OrderRequest, SubmitOrderRequest, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class NAESignalConsumer(QCAlgorithm):
    """
    LEAN algorithm that consumes NAE signals from Redis queue
    and executes orders via Schwab brokerage adapter
    """
    
    def initialize(self):
        """Initialize algorithm"""
        # Configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.queue_name = "execution.signals"
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True
        )
        
        # Set up brokerage (Schwab via QC adapter)
        self.set_brokerage_model("SchwabBrokerageModel")
        
        # Strategy router for multiple NAE strategies
        self.strategy_router = StrategyRouter()
        
        # Execution tracking
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.execution_ledger: List[Dict[str, Any]] = []
        
        # Schedule signal consumption
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(seconds=1)),
            self.consume_signals
        )
        
        logger.info("NAE Signal Consumer initialized")
    
    def consume_signals(self):
        """Consume signals from Redis queue"""
        try:
            # Pop signal from queue (blocking with timeout)
            signal_json = self.redis_client.brpop(self.queue_name, timeout=1)
            
            if signal_json:
                signal_data = json.loads(signal_json[1])
                self.process_signal(signal_data)
        
        except Exception as e:
            logger.error(f"Error consuming signals: {e}")
    
    def process_signal(self, signal: Dict[str, Any]):
        """Process a signal and execute order"""
        try:
            signal_id = signal.get("signal_id")
            strategy_id = signal.get("strategy_id")
            symbol = signal.get("symbol")
            action = signal.get("action")
            quantity = signal.get("quantity")
            notional = signal.get("notional")
            order_type = signal.get("order_type", "MARKET")
            limit_price = signal.get("limit_price")
            
            # Route through strategy router
            routing_result = self.strategy_router.route(signal)
            
            if routing_result["status"] != "ACCEPTED":
                logger.warning(f"Signal {signal_id} rejected by router: {routing_result['reason']}")
                return
            
            # Convert symbol to LEAN format
            lean_symbol = self.get_lean_symbol(symbol)
            
            # Calculate quantity if notional provided
            if notional and not quantity:
                current_price = self.securities[lean_symbol].price
                quantity = int(notional / current_price)
            
            # Determine order direction
            if action in ["BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE"]:
                order_quantity = quantity
            else:
                order_quantity = -quantity
            
            # Create order request
            if order_type == "MARKET":
                order_request = SubmitOrderRequest(
                    OrderType.MARKET,
                    lean_symbol,
                    order_quantity,
                    None,  # limit price
                    None,  # stop price
                    None,  # tag
                    f"NAE_{strategy_id}_{signal_id}"
                )
            elif order_type == "LIMIT":
                order_request = SubmitOrderRequest(
                    OrderType.LIMIT,
                    lean_symbol,
                    order_quantity,
                    limit_price,
                    None,
                    None,
                    f"NAE_{strategy_id}_{signal_id}"
                )
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return
            
            # Submit order
            order_ticket = self.submit_order_request(order_request)
            
            # Track order
            self.pending_orders[signal_id] = {
                "signal": signal,
                "order_ticket": order_ticket,
                "submitted_at": datetime.now().isoformat()
            }
            
            logger.info(f"Order submitted for signal {signal_id}: {order_ticket}")
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def get_lean_symbol(self, symbol: str):
        """Convert symbol to LEAN format"""
        # Add to universe if not already
        if symbol not in self.securities:
            self.add_equity(symbol)
        return self.symbol(symbol)
    
    def on_order_event(self, order_event):
        """Handle order events"""
        order = order_event.order
        order_tag = order.tag
        
        # Extract signal_id from tag
        if order_tag and order_tag.startswith("NAE_"):
            parts = order_tag.split("_")
            if len(parts) >= 3:
                signal_id = parts[2]
                
                # Record execution
                execution_record = {
                    "signal_id": signal_id,
                    "order_id": order.id,
                    "status": order.status.name,
                    "filled_quantity": order_event.fill_quantity,
                    "fill_price": order_event.fill_price,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.execution_ledger.append(execution_record)
                
                # Send execution event back to NAE
                self.send_execution_event(execution_record)
                
                # Remove from pending if filled
                if order.status == OrderStatus.FILLED:
                    self.pending_orders.pop(signal_id, None)
    
    def send_execution_event(self, execution_record: Dict[str, Any]):
        """Send execution event back to NAE"""
        try:
            # Push to Redis queue for NAE consumption
            self.redis_client.lpush(
                "execution.events",
                json.dumps(execution_record)
            )
        except Exception as e:
            logger.error(f"Error sending execution event: {e}")


class StrategyRouter:
    """
    Routes signals from multiple NAE strategies
    
    Handles:
    - Strategy priority
    - Risk allocation per strategy
    - Signal aggregation (e.g., merge opposing signals)
    """
    
    def __init__(self):
        self.strategy_budgets: Dict[str, float] = {}
        self.strategy_priorities: Dict[str, int] = {}
        self.correlation_groups: Dict[str, List[str]] = {}
    
    def route(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Route signal through strategy router"""
        strategy_id = signal.get("strategy_id")
        
        # Check strategy budget
        if strategy_id in self.strategy_budgets:
            budget = self.strategy_budgets[strategy_id]
            notional = signal.get("notional", 0)
            if notional > budget:
                return {
                    "status": "REJECTED",
                    "reason": f"Strategy budget exceeded: {notional} > {budget}"
                }
        
        # Check for opposing signals in correlation group
        correlation_group = signal.get("correlation_group")
        if correlation_group:
            # Would check for opposing signals and aggregate
            pass
        
        return {
            "status": "ACCEPTED",
            "routing": {
                "strategy_id": strategy_id,
                "priority": self.strategy_priorities.get(strategy_id, 0)
            }
        }
    
    def set_strategy_budget(self, strategy_id: str, budget: float):
        """Set risk budget for strategy"""
        self.strategy_budgets[strategy_id] = budget
    
    def set_strategy_priority(self, strategy_id: str, priority: int):
        """Set priority for strategy"""
        self.strategy_priorities[strategy_id] = priority


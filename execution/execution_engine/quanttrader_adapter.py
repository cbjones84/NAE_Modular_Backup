"""
QuantTrader + PyBroker Execution Adapter

Alternative execution engine using QuantTrader/PyBroker.
Good for simpler strategies and research-to-live pipeline.
"""

import os
import json
import redis
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pybroker
from pybroker import Strategy, YFinance

logger = logging.getLogger(__name__)


class QuantTraderPyBrokerAdapter:
    """
    QuantTrader/PyBroker execution adapter
    
    Uses PyBroker for strategy execution and order management.
    """
    
    def __init__(self):
        """Initialize QuantTrader/PyBroker adapter"""
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.queue_name = "execution.signals"
        
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True
        )
        
        # Broker configuration
        self.broker = self._initialize_broker()
        
        # Strategy tracking
        self.active_strategies: Dict[str, Any] = {}
        
        logger.info("QuantTrader/PyBroker adapter initialized")
    
    def _initialize_broker(self):
        """Initialize PyBroker broker"""
        # PyBroker supports multiple brokers
        # For Schwab, would use custom adapter or Alpaca if compatible
        broker_name = os.getenv("PRIMARY_BROKER", "alpaca")  # PyBroker has Alpaca support
        
        if broker_name == "alpaca":
            return pybroker.Alpaca(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_API_SECRET"),
                paper=True if os.getenv("PAPER_TRADING", "true").lower() == "true" else False
            )
        else:
            # For Schwab, would need custom broker implementation
            logger.warning(f"Broker {broker_name} not directly supported, using Alpaca")
            return pybroker.Alpaca(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_API_SECRET"),
                paper=True
            )
    
    def consume_signals(self):
        """Consume signals from Redis queue"""
        try:
            # Pop signal from queue
            signal_json = self.redis_client.brpop(self.queue_name, timeout=1)
            
            if signal_json:
                signal_data = json.loads(signal_json[1])
                self.process_signal(signal_data)
        
        except Exception as e:
            logger.error(f"Error consuming signals: {e}")
    
    def process_signal(self, signal: Dict[str, Any]):
        """Process signal and execute order"""
        try:
            signal_id = signal.get("signal_id")
            strategy_id = signal.get("strategy_id")
            symbol = signal.get("symbol")
            action = signal.get("action")
            quantity = signal.get("quantity")
            notional = signal.get("notional")
            order_type = signal.get("order_type", "MARKET")
            limit_price = signal.get("limit_price")
            
            # Convert action to PyBroker order side
            if action in ["BUY", "BUY_TO_OPEN", "BUY_TO_CLOSE"]:
                side = "buy"
            else:
                side = "sell"
            
            # Calculate quantity if notional provided
            if notional and not quantity:
                # Would get current price
                current_price = self._get_current_price(symbol)
                quantity = int(notional / current_price) if current_price > 0 else 0
            
            # Execute order via PyBroker
            if order_type == "MARKET":
                order_result = self.broker.buy_market(
                    symbol=symbol,
                    shares=quantity if side == "buy" else 0
                ) if side == "buy" else self.broker.sell_market(
                    symbol=symbol,
                    shares=quantity
                )
            elif order_type == "LIMIT":
                order_result = self.broker.buy_limit(
                    symbol=symbol,
                    shares=quantity if side == "buy" else 0,
                    limit_price=limit_price
                ) if side == "buy" else self.broker.sell_limit(
                    symbol=symbol,
                    shares=quantity,
                    limit_price=limit_price
                )
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return
            
            # Report execution back to NAE
            self._report_execution(signal_id, order_result)
            
            logger.info(f"Order executed for signal {signal_id}: {order_result}")
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            self._report_execution_error(signal.get("signal_id"), str(e))
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        # Would use PyBroker data provider
        try:
            data = YFinance(symbols=[symbol])
            # Get latest price
            return 100.0  # Placeholder
        except:
            return 0.0
    
    def _report_execution(self, signal_id: str, order_result: Any):
        """Report execution back to NAE"""
        execution_event = {
            "signal_id": signal_id,
            "order_id": getattr(order_result, "id", None),
            "status": "filled" if getattr(order_result, "filled", False) else "submitted",
            "filled_quantity": getattr(order_result, "filled_qty", 0),
            "fill_price": getattr(order_result, "avg_fill_price", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Push to execution events queue
        self.redis_client.lpush("execution.events", json.dumps(execution_event))
    
    def _report_execution_error(self, signal_id: str, error: str):
        """Report execution error"""
        error_event = {
            "signal_id": signal_id,
            "status": "error",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis_client.lpush("execution.events", json.dumps(error_event))
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        return {
            "engine": "quanttrader_pybroker",
            "status": "running",
            "broker": self.broker.__class__.__name__ if self.broker else None,
            "queue": self.queue_name
        }


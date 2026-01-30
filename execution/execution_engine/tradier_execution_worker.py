"""
Tradier Execution Worker

Consumes signals from queue and executes via Tradier adapter.
Handles Tradier-specific logic and streaming updates.
"""

import os
import json
import redis
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from execution.broker_adapters.tradier_adapter import TradierBrokerAdapter
from execution.pre_trade_validator.tradier_validator import TradierPreTradeValidator
from execution.order_handlers.tradier_order_handler import TradierOrderHandler

logger = logging.getLogger(__name__)


class TradierExecutionWorker:
    """
    Execution worker for Tradier
    
    Consumes signals from Redis queue and executes via Tradier API
    """
    
    def __init__(self):
        """Initialize Tradier execution worker"""
        # Redis configuration
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
        
        # Initialize Tradier adapter (supports both OAuth and API key)
        # Fix: Ensure correct endpoint (live vs sandbox)
        sandbox_env = os.getenv("TRADIER_SANDBOX", "").lower()
        # Default to live (False) if not explicitly set to "true"
        use_sandbox = sandbox_env == "true"
        
        if not use_sandbox:
            logger.info("🔴 Using LIVE Tradier endpoint")
        else:
            logger.info("🟡 Using SANDBOX Tradier endpoint")
        
        self.tradier = TradierBrokerAdapter(
            client_id=os.getenv("TRADIER_CLIENT_ID"),
            client_secret=os.getenv("TRADIER_CLIENT_SECRET"),
            api_key=os.getenv("TRADIER_API_KEY"),  # API key from env (vault is checked automatically)
            account_id=os.getenv("TRADIER_ACCOUNT_ID"),
            sandbox=use_sandbox
        )
        
        # Initialize Tradier validator
        self.validator = TradierPreTradeValidator(self.tradier)
        
        # Initialize enhanced order handler (fixes all common issues)
        self.order_handler = TradierOrderHandler(self.tradier)
        
        # Authenticate
        if not self.tradier.authenticate():
            logger.error("Failed to authenticate with Tradier")
        
        # Connect streaming
        self.tradier.connect_streaming()
        
        logger.info("Tradier execution worker initialized")
    
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
        """Process signal and execute order via Tradier"""
        try:
            signal_id = signal.get("signal_id")
            strategy_id = signal.get("strategy_id")
            symbol = signal.get("symbol")
            action = signal.get("action")
            quantity = signal.get("quantity")
            notional = signal.get("notional")
            order_type = signal.get("order_type", "MARKET")
            limit_price = signal.get("limit_price")
            stop_price = signal.get("stop_price")
            option_symbol = signal.get("option_symbol")
            
            logger.info(f"Processing signal {signal_id}: {action} {quantity} {symbol}")
            
            # Tradier-specific validation
            validation_result = self.validator.validate_order(signal)
            
            if not validation_result["passed"]:
                logger.warning(f"Signal {signal_id} failed Tradier validation: {validation_result['errors']}")
                self._report_rejection(signal_id, validation_result)
                return
            
            # Convert NAE action to Tradier side
            tradier_side = self._convert_action_to_tradier_side(action)
            
            # Convert order type
            tradier_order_type = order_type.lower()
            if tradier_order_type == "STOP":
                tradier_order_type = "stop"
            elif tradier_order_type == "STOP_LIMIT":
                tradier_order_type = "stop_limit"
            
            # Determine duration based on trading hours
            duration = signal.get("duration", "day")
            
            # Calculate quantity if notional provided
            if notional and not quantity:
                # Would get current price
                quantity = int(notional / 100.0)  # Placeholder
            
            # Build Tradier order
            tradier_order = {
                "symbol": symbol if not option_symbol else None,
                "option_symbol": option_symbol,
                "side": tradier_side,
                "quantity": quantity,
                "order_type": tradier_order_type,
                "duration": duration,
                "price": limit_price,
                "stop": stop_price,
                "tag": f"NAE_{strategy_id}_{signal_id}",
                "preview": signal.get("preview_order", False),
                "strategy_id": strategy_id  # For strategy condition checking
            }
            
            # Submit order using enhanced handler (fixes all issues)
            result = self.order_handler.submit_order_safe(tradier_order)
            
            # Check result
            if result.get("status") == "error":
                # Report errors
                errors = result.get("errors", [])
                logger.error(f"Signal {signal_id} failed: {', '.join(errors)}")
                self._report_execution_error(signal_id, "; ".join(errors))
            elif result.get("status") == "submitted":
                # Report successful execution
                execution_result = {
                    "order_id": result.get("order_id"),
                    "status": "submitted",
                    "broker": "tradier",
                    "fixes_applied": result.get("fixes_applied", []),
                    "warnings": result.get("warnings", [])
                }
                self._report_execution(signal_id, execution_result)
                logger.info(f"Order submitted for signal {signal_id}: {result.get('order_id')}")
                
                # Log any fixes that were applied
                if result.get("fixes_applied"):
                    logger.info(f"Fixes applied for signal {signal_id}: {', '.join(result['fixes_applied'])}")
                
                # Log warnings
                if result.get("warnings"):
                    logger.warning(f"Warnings for signal {signal_id}: {', '.join(result['warnings'])}")
            else:
                # Unknown status
                logger.warning(f"Unknown order status for signal {signal_id}: {result.get('status')}")
                self._report_execution_error(signal_id, f"Unknown status: {result.get('status')}")
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            self._report_execution_error(signal.get("signal_id"), str(e))
    
    def _convert_action_to_tradier_side(self, action: str) -> str:
        """Convert NAE action to Tradier side"""
        action_map = {
            "BUY": "buy",
            "SELL": "sell",
            "BUY_TO_OPEN": "buy",
            "SELL_TO_OPEN": "sell_short",
            "BUY_TO_CLOSE": "buy_to_cover",
            "SELL_TO_CLOSE": "sell"
        }
        return action_map.get(action.upper(), "buy")
    
    def _report_execution(self, signal_id: str, result: Dict[str, Any]):
        """Report execution back to NAE"""
        execution_event = {
            "signal_id": signal_id,
            "order_id": result.get("order_id"),
            "status": result.get("status"),
            "broker": "tradier",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        # Push to execution events queue
        self.redis_client.lpush("execution.events", json.dumps(execution_event))
    
    def _report_rejection(self, signal_id: str, validation_result: Dict[str, Any]):
        """Report rejection"""
        rejection_event = {
            "signal_id": signal_id,
            "status": "rejected",
            "broker": "tradier",
            "reason": validation_result.get("errors", []),
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis_client.lpush("execution.events", json.dumps(rejection_event))
    
    def _report_execution_error(self, signal_id: str, error: str):
        """Report execution error"""
        error_event = {
            "signal_id": signal_id,
            "status": "error",
            "broker": "tradier",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis_client.lpush("execution.events", json.dumps(error_event))
    
    def run(self):
        """Run execution worker loop"""
        logger.info("Tradier execution worker started")
        
        while True:
            try:
                self.consume_signals()
                
                # Check OAuth token expiry
                if self.tradier.oauth.is_token_expired():
                    logger.warning("Tradier OAuth token expired, attempting refresh")
                    self.tradier.oauth.refresh_access_token()
                
                # Check WebSocket connection
                if not self.tradier.ws_client.connected:
                    logger.warning("Tradier WebSocket disconnected, reconnecting")
                    self.tradier.connect_streaming()
            
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in execution worker loop: {e}")
                time.sleep(5)  # Wait before retrying


if __name__ == "__main__":
    worker = TradierExecutionWorker()
    worker.run()


"""
NAE Integration - Connects NAE to Signal Middleware

This module integrates NAE agents (Optimus) with the execution layer.
"""

import os
import json
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NAEExecutionClient:
    """Client for sending signals from NAE to execution middleware"""
    
    def __init__(self, middleware_url: str = None, hmac_secret: str = None):
        self.middleware_url = middleware_url or os.getenv("EXECUTION_MIDDLEWARE_URL", "http://localhost:8001")
        self.hmac_secret = (hmac_secret or os.getenv("HMAC_SECRET", "")).encode()
        self.session = requests.Session()
    
    def send_signal(
        self,
        strategy_id: str,
        symbol: str,
        action: str,
        quantity: Optional[int] = None,
        notional: Optional[float] = None,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        risk_meta: Optional[Dict[str, Any]] = None,
        correlation_group: Optional[str] = None,
        model_id: Optional[str] = None,
        confidence: Optional[float] = None,
        expected_pnl: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send trade signal to execution middleware
        
        Returns response from middleware
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = f"{strategy_id}_{symbol}_{datetime.now().timestamp()}"
        
        # Build signal payload
        signal = {
            "strategy_id": strategy_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "notional": notional,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "risk_meta": risk_meta or {
                "max_slippage": 0.01,
                "max_exposure": 10000.0,
                "max_position_pct": 0.1
            },
            "correlation_group": correlation_group,
            "request_id": request_id,
            "model_id": model_id,
            "confidence": confidence,
            "expected_pnl": expected_pnl
        }
        
        # Calculate HMAC signature
        payload_json = json.dumps(signal, sort_keys=True)
        signature = hmac.new(
            self.hmac_secret,
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Send request
        try:
            response = self.session.post(
                f"{self.middleware_url}/v1/signals",
                json=signal,
                headers={
                    "X-Signature": signature,
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending signal to middleware: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "request_id": request_id
            }


def integrate_with_optimus(optimus_agent):
    """
    Integrate execution client with Optimus agent
    
    Adds send_signal method to Optimus for easy signal sending
    """
    execution_client = NAEExecutionClient()
    
    def send_execution_signal(self, execution_details: Dict[str, Any]) -> Dict[str, Any]:
        """Send execution signal from Optimus"""
        return execution_client.send_signal(
            strategy_id=execution_details.get("strategy_id", "optimus_default"),
            symbol=execution_details.get("symbol", ""),
            action=execution_details.get("side", "BUY").upper(),
            quantity=execution_details.get("quantity"),
            notional=execution_details.get("notional"),
            order_type=execution_details.get("order_type", "MARKET"),
            limit_price=execution_details.get("limit_price"),
            stop_price=execution_details.get("stop_price"),
            risk_meta={
                "max_slippage": execution_details.get("max_slippage", 0.01),
                "max_exposure": execution_details.get("max_exposure", 10000.0),
                "max_position_pct": execution_details.get("max_position_pct", 0.1)
            },
            correlation_group=execution_details.get("correlation_group"),
            model_id=execution_details.get("model_id"),
            confidence=execution_details.get("confidence"),
            expected_pnl=execution_details.get("expected_pnl"),
            request_id=execution_details.get("request_id")
        )
    
    # Attach method to Optimus agent
    optimus_agent.send_execution_signal = send_execution_signal.__get__(optimus_agent, type(optimus_agent))
    
    logger.info("Execution client integrated with Optimus agent")
    return execution_client


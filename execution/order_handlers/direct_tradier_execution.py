#!/usr/bin/env python3
"""
Direct Tradier Trade Execution - Simplified Path
Bypasses order handler for immediate execution
"""
import os
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def execute_trade_direct(
    tradier_adapter,
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    price: Optional[float] = None,
    duration: str = "day",
    option_symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute trade directly through Tradier adapter, bypassing order handler
    
    Args:
        tradier_adapter: TradierBrokerAdapter instance
        symbol: Equity symbol (or underlying for options)
        side: buy, sell, buy_to_cover, sell_short
        quantity: Number of shares/contracts
        order_type: market, limit, stop, stop_limit
        price: Limit price (required for limit orders)
        duration: day, gtc, pre, post
        option_symbol: Option symbol in OCC format (for options)
    
    Returns:
        Result dictionary with status and order details
    """
    result = {
        "status": "pending",
        "order_id": None,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Build order dictionary
        order = {
            "symbol": symbol if not option_symbol else None,
            "option_symbol": option_symbol,
            "side": side.lower(),
            "quantity": quantity,
            "order_type": order_type.lower(),
            "duration": duration.lower()
        }
        
        # Add price for limit orders
        if order_type.lower() in ["limit", "stop_limit"] and price:
            order["price"] = price
        
        # Add stop price for stop orders
        if order_type.lower() in ["stop", "stop_limit"] and price:
            order["stop"] = price
        
        # Set class for options
        if option_symbol:
            order["class"] = "option"
        
        # Execute order directly
        logger.info(f"Executing direct trade: {order}")
        submit_result = tradier_adapter.submit_order(order)
        
        if submit_result.get("status") == "error":
            result["status"] = "error"
            result["errors"].append(submit_result.get("error", "Unknown error"))
        else:
            result["status"] = "submitted" if submit_result.get("order_id") else "pending"
            result["order_id"] = submit_result.get("order_id")
            result["result"] = submit_result
            logger.info(f"✅ Trade submitted successfully: Order ID {result['order_id']}")
    
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Direct execution failed: {str(e)}")
        logger.exception("Direct trade execution error")
    
    return result


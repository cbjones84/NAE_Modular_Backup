# NAE/adapters/mock.py
"""
Mock Broker Adapter
For testing and simulation without hitting real broker APIs
"""

from typing import Dict, List, Any
from .base import BrokerAdapter

class MockAdapter(BrokerAdapter):
    """Mock broker adapter for testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize mock adapter
        
        Args:
            config: Optional config dict (unused for mock)
        """
        self.config = config or {}
        self._orders = {}
        self._next_order_id = 1
        self._positions = []
        self._account = {
            "cash": 100000.0,
            "buying_power": 100000.0,
            "portfolio_value": 100000.0
        }
    
    def name(self) -> str:
        return "mock"
    
    def auth(self) -> bool:
        """Mock always authenticates successfully"""
        return True
    
    def get_account(self) -> Dict[str, Any]:
        """Get mock account"""
        return {
            "account_id": "mock-123",
            "cash": self._account["cash"],
            "buying_power": self._account["buying_power"],
            "portfolio_value": self._account["portfolio_value"],
            "equity": self._account["portfolio_value"],
            "currency": "USD"
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get mock positions"""
        return self._positions.copy()
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get mock quote (simulated prices)"""
        # Simple mock pricing
        base_price = 100.0
        return {
            "symbol": symbol,
            "bid": base_price - 0.01,
            "ask": base_price + 0.01,
            "last": base_price,
            "volume": 1000000,
            "high": base_price * 1.05,
            "low": base_price * 0.95,
            "open": base_price
        }
    
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place mock order (simulated execution)"""
        symbol = order.get('symbol', '')
        quantity = order.get('quantity', 0)
        side = order.get('side', 'buy')
        order_type = order.get('type', 'market')
        price = order.get('price', self.get_quote(symbol).get('last', 100.0))
        
        order_id = str(self._next_order_id)
        self._next_order_id += 1
        
        # Simulate order execution
        order_response = {
            "order_id": order_id,
            "status": "filled",
            "broker": "mock",
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "type": order_type,
            "filled_price": price,
            "timestamp": "2025-01-01T12:00:00Z"
        }
        
        self._orders[order_id] = order_response
        
        # Update positions
        if side == 'buy':
            # Add to positions
            existing_pos = None
            for pos in self._positions:
                if pos.get('symbol') == symbol:
                    existing_pos = pos
                    break
            
            if existing_pos:
                existing_pos['quantity'] += quantity
                existing_pos['cost_basis'] = (existing_pos['cost_basis'] * (existing_pos['quantity'] - quantity) + price * quantity) / existing_pos['quantity']
            else:
                self._positions.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": price,
                    "cost_basis": price * quantity,
                    "market_value": price * quantity
                })
            
            # Update cash
            self._account["cash"] -= price * quantity
            
        else:  # sell
            # Remove from positions
            for pos in self._positions:
                if pos.get('symbol') == symbol:
                    if pos['quantity'] >= quantity:
                        pos['quantity'] -= quantity
                        if pos['quantity'] == 0:
                            self._positions.remove(pos)
                        
                        # Update cash
                        self._account["cash"] += price * quantity
                        break
        
        return order_response
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel mock order"""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return {
                "order_id": order_id,
                "status": "cancelled"
            }
        
        return {
            "error": "Order not found",
            "status": "failed"
        }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get mock order status"""
        return self._orders.get(order_id, {})



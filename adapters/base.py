# NAE/adapters/base.py
"""
Base Broker Adapter Interface
Defines common methods that all broker adapters must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BrokerAdapter(ABC):
    """Abstract base class for broker adapters"""
    
    @abstractmethod
    def name(self) -> str:
        """Return broker name (e.g., 'etrade', 'alpaca')"""
        pass
    
    @abstractmethod
    def auth(self) -> bool:
        """Authenticate with broker. Returns True if successful."""
        pass
    
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information
        Returns: Dict with account details (cash, buying_power, etc.)
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        Returns: List of position dicts with symbol, quantity, etc.
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get quote for symbol
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        Returns: Dict with bid, ask, last price, etc.
        """
        pass
    
    @abstractmethod
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place an order
        Args:
            order: Dict with symbol, quantity, side ('buy'/'sell'), type, etc.
        Returns: Dict with order_id, status, etc.
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        Args:
            order_id: Order ID to cancel
        Returns: Dict with cancellation status
        """
        pass
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status (optional, may raise NotImplementedError)
        Args:
            order_id: Order ID to check
        Returns: Dict with order status
        """
        raise NotImplementedError("get_order_status not implemented for this adapter")
    
    def subscribe_market(self, symbols: List[str]) -> bool:
        """
        Subscribe to market data (optional, may raise NotImplementedError)
        Args:
            symbols: List of symbols to subscribe to
        Returns: True if successful
        """
        raise NotImplementedError("subscribe_market not implemented for this adapter")



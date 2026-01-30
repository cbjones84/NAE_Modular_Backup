# NAE Broker Adapters
"""
Modular broker adapter architecture for NAE
Supports multiple brokers (E*TRADE, Alpaca, etc.) through a common interface
"""

from .base import BrokerAdapter
from .manager import AdapterManager

__all__ = ['BrokerAdapter', 'AdapterManager']



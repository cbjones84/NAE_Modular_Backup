# NAE/adapters/ibkr.py
"""
Interactive Brokers (IBKR) Broker Adapter
Implements IBKR TWS API using ibapi (EWrapper/EClient pattern)
Supports stocks and options trading with market/limit orders and multi-leg options

Requirements:
- TWS (Trader Workstation) or IB Gateway must be running
- API access must be enabled in TWS settings
- Default ports: 7497 (paper trading), 7496 (live trading)
"""

import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from queue import Queue, Empty

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract, ComboLeg
    from ibapi.order import Order
    from ibapi.common import TickerId, BarData
    IBAPI_AVAILABLE = True
except ImportError:
    IBAPI_AVAILABLE = False
    # Fallback for type hints
    EClient = None
    EWrapper = None
    Contract = None
    Order = None

from .base import BrokerAdapter


class IBClient(EWrapper, EClient):
    """
    IB API client using EWrapper/EClient for synchronous trading
    
    Handles all IB API callbacks and manages connection state
    """
    
    def __init__(self):
        if not IBAPI_AVAILABLE:
            raise ImportError("ibapi package is required. Install TWS API or: pip install ibapi")
        
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        
        self.nextOrderId = None
        self.connected = False
        self.error_queue = Queue()
        self.account_data = {}
        self.positions_data = []
        self.quote_data = {}
        self.order_data = {}
        self.contract_details = {}
        self._lock = threading.Lock()
        
    def nextValidId(self, orderId: int):
        """Callback once connection is established to receive the next valid order ID"""
        super().nextValidId(orderId)
        with self._lock:
            self.nextOrderId = orderId
            self.connected = True
        print(f"[IBKR] Connected. Next valid order ID: {self.nextOrderId}")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """
        Basic error handler
        
        Args:
            reqId: Request ID (-1 for general errors)
            errorCode: Error code (2104-2106 are informational messages)
            errorString: Error message
            advancedOrderRejectJson: JSON string for advanced order rejections
        """
        # Error codes 2104-2106 are informational (market data subscription, etc.)
        if errorCode not in [2104, 2105, 2106]:
            error_msg = f"IB API Error {errorCode}: {errorString}"
            print(f"[IBKR] {error_msg}")
            self.error_queue.put({
                "reqId": reqId,
                "errorCode": errorCode,
                "errorString": errorString
            })
    
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """Callback for account value updates"""
        with self._lock:
            if accountName not in self.account_data:
                self.account_data[accountName] = {}
            self.account_data[accountName][key] = val
    
    def updatePortfolio(self, contract: Contract, position: float, marketPrice: float, 
                       marketValue: float, averageCost: float, unrealizedPNL: float, 
                       realizedPNL: float, accountName: str):
        """Callback for position updates"""
        with self._lock:
            pos_data = {
                "symbol": contract.symbol,
                "secType": contract.secType,
                "quantity": position,
                "market_price": marketPrice,
                "market_value": marketValue,
                "avg_cost": averageCost,
                "unrealized_pnl": unrealizedPNL,
                "realized_pnl": realizedPNL,
                "account": accountName
            }
            # Update or add position
            self.positions_data = [p for p in self.positions_data if p.get("symbol") != contract.symbol]
            if position != 0:  # Only add non-zero positions
                self.positions_data.append(pos_data)
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib: Any):
        """Callback for market data price updates"""
        with self._lock:
            if reqId not in self.quote_data:
                self.quote_data[reqId] = {}
            
            # Map tick types to quote fields
            # 1=Bid, 2=Ask, 4=Last, 6=High, 7=Low, 9=Close
            tick_map = {1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "close"}
            if tickType in tick_map:
                self.quote_data[reqId][tick_map[tickType]] = price
    
    def tickSize(self, reqId: TickerId, tickType: int, size: int):
        """Callback for market data size updates"""
        with self._lock:
            if reqId not in self.quote_data:
                self.quote_data[reqId] = {}
            
            # 0=Bid Size, 3=Ask Size, 5=Last Size
            size_map = {0: "bid_size", 3: "ask_size", 5: "last_size"}
            if tickType in size_map:
                self.quote_data[reqId][size_map[tickType]] = size
    
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                   avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                   clientId: int, whyHeld: str, mktCapPrice: float):
        """Callback for order status updates"""
        with self._lock:
            self.order_data[orderId] = {
                "order_id": orderId,
                "status": status,
                "filled": filled,
                "remaining": remaining,
                "avg_fill_price": avgFillPrice,
                "last_fill_price": lastFillPrice
            }
    
    def contractDetails(self, reqId: int, contractDetails: Any):
        """Callback for contract details (used for conId lookup)"""
        with self._lock:
            contract = contractDetails.contract
            self.contract_details[reqId] = {
                "conId": contract.conId,
                "symbol": contract.symbol,
                "secType": contract.secType,
                "exchange": contract.exchange,
                "currency": contract.currency
            }


class IBKRAdapter(BrokerAdapter):
    """
    Interactive Brokers adapter using TWS API
    
    Features:
    - Stock market/limit orders
    - Options trading (single and multi-leg)
    - Account and position management
    - Real-time market data
    
    Note: Requires TWS or IB Gateway to be running with API access enabled
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize IBKR adapter
        
        Args:
            config: Dict with:
                - host: TWS/Gateway host (default: "127.0.0.1")
                - port: TWS/Gateway port (default: 7497 for paper, 7496 for live)
                - client_id: Client ID for connection (default: 1)
                - paper_trading: bool, use paper trading (default: True)
        """
        if not IBAPI_AVAILABLE:
            raise ImportError(
                "ibapi package is required. "
                "Install TWS API from Interactive Brokers or: pip install ibapi"
            )
        
        self.config = config
        
        # Connection settings
        self.host = config.get("host", "127.0.0.1")
        self.paper_trading = config.get("paper_trading", True)
        self.port = config.get("port", 7497 if self.paper_trading else 7496)
        self.client_id = config.get("client_id", 1)
        
        # Initialize IB client
        self.client = None
        self.api_thread = None
        self._connected = False
        self._request_id = 1000  # Starting request ID for market data
        
        # Account name (will be set after connection)
        self.account_name = None
        
        # Connect to TWS/Gateway
        self._connect()
    
    def _connect(self):
        """Connect to TWS or IB Gateway"""
        try:
            self.client = IBClient()
            self.client.connect(self.host, self.port, self.client_id)
            
            # Launch the socket listening in a separate thread
            self.api_thread = threading.Thread(target=self.client.run, daemon=True)
            self.api_thread.start()
            
            # Wait until we get nextOrderId from IB (meaning connection is ready)
            timeout = 10  # 10 second timeout
            start_time = time.time()
            
            while self.client.nextOrderId is None:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Failed to connect to IB TWS/Gateway on {self.host}:{self.port}")
                time.sleep(0.1)
            
            self._connected = True
            print(f"[IBKR] Connected to TWS/Gateway on {self.host}:{self.port}")
            
            # Request account updates
            self.client.reqAccountUpdates(True, "")
            
            # Get account name from account data
            time.sleep(1)  # Wait for account data
            if self.client.account_data:
                self.account_name = list(self.client.account_data.keys())[0]
            
        except Exception as e:
            print(f"[IBKR] Connection error: {e}")
            self._connected = False
            raise
    
    def name(self) -> str:
        return "ibkr"
    
    def auth(self) -> bool:
        """Check if connected to TWS/Gateway"""
        return self._connected and self.client is not None and self.client.connected
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.auth():
            return {}
        
        try:
            # Request account updates if not already done
            if not self.account_name:
                self.client.reqAccountUpdates(True, "")
                time.sleep(1)
            
            account_data = self.client.account_data.get(self.account_name or "", {})
            
            # Extract key account values
            return {
                "account_id": self.account_name or "",
                "cash": float(account_data.get("CashBalance", 0)),
                "buying_power": float(account_data.get("BuyingPower", 0)),
                "portfolio_value": float(account_data.get("TotalCashValue", 0)),
                "equity": float(account_data.get("NetLiquidation", 0)),
                "currency": account_data.get("BaseCurrency", "USD"),
                "trading_blocked": False,  # IBKR doesn't provide this directly
                "pattern_day_trader": account_data.get("DayTradingStatus", "").startswith("Pattern")
            }
        except Exception as e:
            print(f"[IBKR] Error getting account: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if not self.auth():
            return []
        
        try:
            # Request position updates
            self.client.reqPositions()
            time.sleep(1)  # Wait for position data
            
            positions = []
            for pos in self.client.positions_data:
                positions.append({
                    "symbol": pos.get("symbol", ""),
                    "quantity": pos.get("quantity", 0),
                    "avg_price": pos.get("avg_cost", 0),
                    "market_value": pos.get("market_value", 0),
                    "cost_basis": pos.get("avg_cost", 0) * abs(pos.get("quantity", 0)),
                    "unrealized_pl": pos.get("unrealized_pnl", 0),
                    "unrealized_plpc": (pos.get("unrealized_pnl", 0) / (pos.get("cost_basis", 1) or 1)) * 100,
                    "side": "long" if pos.get("quantity", 0) > 0 else "short"
                })
            
            return positions
        except Exception as e:
            print(f"[IBKR] Error getting positions: {e}")
            return []
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote for symbol"""
        if not self.auth():
            return {}
        
        try:
            # Create contract
            contract = self._create_stock_contract(symbol)
            
            # Request market data
            req_id = self._get_next_request_id()
            self.client.reqMktData(req_id, contract, "", False, False, [])
            
            # Wait for quote data
            time.sleep(0.5)
            
            quote = self.client.quote_data.get(req_id, {})
            
            # Cancel market data subscription
            self.client.cancelMktData(req_id)
            
            return {
                "symbol": symbol,
                "bid": quote.get("bid", 0.0),
                "ask": quote.get("ask", 0.0),
                "last": quote.get("last", quote.get("close", 0.0)),
                "volume": None,  # Not available in tick data
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[IBKR] Error getting quote: {e}")
            return {}
    
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place an order (supports stocks and options)
        
        Args:
            order: Dict with:
                - symbol: Stock or option symbol
                - quantity: float, number of shares/contracts
                - side: 'buy' or 'sell'
                - type: 'market' or 'limit'
                - price: float (required for limit orders)
                - secType: 'STK' for stocks, 'OPT' for options (default: 'STK')
                - For options: lastTradeDate, strike, right ('C' or 'P')
        """
        if not self.auth():
            return {"error": "Not connected to TWS/Gateway", "status": "rejected"}
        
        try:
            # Create contract
            sec_type = order.get("secType", "STK")
            if sec_type == "STK":
                contract = self._create_stock_contract(order.get("symbol", ""))
            elif sec_type == "OPT":
                contract = self._create_option_contract(
                    order.get("symbol", ""),
                    order.get("lastTradeDate", ""),
                    order.get("strike", 0.0),
                    order.get("right", "C")
                )
            else:
                return {"error": f"Unsupported secType: {sec_type}", "status": "rejected"}
            
            # Create order
            ib_order = Order()
            ib_order.action = order.get("side", "buy").upper()
            ib_order.totalQuantity = float(order.get("quantity", 0))
            
            order_type = order.get("type", "market").lower()
            if order_type == "limit":
                ib_order.orderType = "LMT"
                ib_order.lmtPrice = float(order.get("price", 0.0))
            else:
                ib_order.orderType = "MKT"
            
            # Place order
            order_id = self.client.nextOrderId
            self.client.placeOrder(order_id, contract, ib_order)
            self.client.nextOrderId += 1
            
            # Wait briefly for order status
            time.sleep(0.5)
            order_status = self.client.order_data.get(order_id, {})
            
            return {
                "order_id": str(order_id),
                "status": order_status.get("status", "submitted"),
                "broker": "ibkr",
                "paper_trading": self.paper_trading,
                "timestamp": datetime.now().isoformat(),
                "filled": order_status.get("filled", 0),
                "remaining": order_status.get("remaining", ib_order.totalQuantity)
            }
            
        except Exception as e:
            return {
                "error": f"Order submission error: {e}",
                "status": "rejected"
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not self.auth():
            return {"error": "Not connected to TWS/Gateway", "status": "failed"}
        
        try:
            self.client.cancelOrder(int(order_id), "")
            
            return {
                "order_id": order_id,
                "status": "cancelled"
            }
        except Exception as e:
            return {
                "error": f"Cancel error: {e}",
                "status": "failed"
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if not self.auth():
            return {}
        
        try:
            order_data = self.client.order_data.get(int(order_id), {})
            return order_data
        except Exception as e:
            print(f"[IBKR] Error getting order status: {e}")
            return {}
    
    # ----------------------
    # Contract Construction
    # ----------------------
    
    def _create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """Create and return a stock Contract object"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def _create_option_contract(self, symbol: str, lastTradeDate: str, strike: float, right: str,
                                exchange: str = "SMART", currency: str = "USD", multiplier: int = 100) -> Contract:
        """
        Create and return an option Contract
        
        Args:
            symbol: Underlying symbol
            lastTradeDate: Expiration date (e.g., '20231215')
            strike: Strike price
            right: 'C' for calls or 'P' for puts
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = exchange
        contract.currency = currency
        contract.lastTradeDateOrContractMonth = lastTradeDate
        contract.strike = strike
        contract.right = right
        contract.multiplier = str(multiplier)
        return contract
    
    def create_combo_contract(self, symbol: str, legs: List[tuple]) -> Contract:
        """
        Create a multi-leg (BAG) Contract for an options strategy
        
        Args:
            symbol: Underlying symbol (ignored for options combos)
            legs: List of tuples (conId, ratio, action, exchange)
                - conId: Contract ID (must be resolved first)
                - ratio: Ratio quantity for this leg
                - action: 'BUY' or 'SELL'
                - exchange: Exchange for this leg
        """
        combo = Contract()
        combo.symbol = symbol
        combo.secType = "BAG"
        combo.currency = "USD"
        combo.exchange = "SMART"
        combo.comboLegs = []
        
        for (conId, ratio, action, exch) in legs:
            leg = ComboLeg()
            leg.conId = conId
            leg.ratio = ratio
            leg.action = action
            leg.exchange = exch
            combo.comboLegs.append(leg)
        
        return combo
    
    # ----------------------
    # Helper Methods
    # ----------------------
    
    def _get_next_request_id(self) -> int:
        """Get next available request ID"""
        req_id = self._request_id
        self._request_id += 1
        return req_id
    
    # ----------------------
    # Enhanced Trading Methods
    # ----------------------
    
    def buy_stock_market(self, symbol: str, qty: float) -> Dict[str, Any]:
        """Place a market buy order for a stock"""
        return self.place_order({
            "symbol": symbol,
            "quantity": qty,
            "side": "buy",
            "type": "market",
            "secType": "STK"
        })
    
    def sell_stock_limit(self, symbol: str, qty: float, limit_price: float) -> Dict[str, Any]:
        """Place a limit sell order for a stock"""
        return self.place_order({
            "symbol": symbol,
            "quantity": qty,
            "side": "sell",
            "type": "limit",
            "price": limit_price,
            "secType": "STK"
        })
    
    def buy_option_market(self, symbol: str, lastTradeDate: str, strike: float, 
                         right: str, qty: float) -> Dict[str, Any]:
        """Place a market buy order for a single-leg option"""
        return self.place_order({
            "symbol": symbol,
            "quantity": qty,
            "side": "buy",
            "type": "market",
            "secType": "OPT",
            "lastTradeDate": lastTradeDate,
            "strike": strike,
            "right": right
        })
    
    def place_multi_leg_order(self, combo: Contract, action: str, quantity: float) -> Dict[str, Any]:
        """
        Place a multi-leg options order using a BAG contract
        
        Args:
            combo: BAG Contract with ComboLegs
            action: 'BUY' or 'SELL'
            quantity: Base quantity for the multi-leg order
        """
        if not self.auth():
            return {"error": "Not connected to TWS/Gateway", "status": "rejected"}
        
        try:
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = "MKT"
            
            order_id = self.client.nextOrderId
            self.client.placeOrder(order_id, combo, order)
            self.client.nextOrderId += 1
            
            time.sleep(0.5)
            order_status = self.client.order_data.get(order_id, {})
            
            return {
                "order_id": str(order_id),
                "status": order_status.get("status", "submitted"),
                "broker": "ibkr",
                "paper_trading": self.paper_trading,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": f"Error placing multi-leg order: {e}",
                "status": "rejected"
            }


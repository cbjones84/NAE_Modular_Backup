# NAE/adapters/alpaca.py
"""
Alpaca Broker Adapter
Implements Alpaca API using official Python SDK (alpaca-py)
Supports stocks and options trading with market/limit orders and multi-leg options
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderStatus
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OptionLegRequest
    from alpaca.common.exceptions import APIError
    ALPACA_SDK_AVAILABLE = True
except ImportError:
    ALPACA_SDK_AVAILABLE = False
    # Fallback imports for type hints
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    OrderClass = None

from .base import BrokerAdapter


class AlpacaAdapter(BrokerAdapter):
    """
    Alpaca broker adapter using official Python SDK
    
    Features:
    - Stock market/limit orders
    - Options trading (single and multi-leg)
    - Paper and live trading support
    - Real-time account and position data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca adapter
        
        Args:
            config: Dict with:
                - API_KEY: Alpaca API key (or use APCA_API_KEY_ID env)
                - API_SECRET: Alpaca API secret (or use APCA_API_SECRET_KEY env)
                - paper_trading: bool, use paper trading (default: True)
        """
        if not ALPACA_SDK_AVAILABLE:
            raise ImportError(
                "alpaca-py package is required. Install with: pip install alpaca-py"
            )
        
        self.config = config
        
        # Get credentials (check multiple sources in order of security)
        # 1. Config dict (lowest priority)
        # 2. Environment variables
        # 3. Secure vault (highest priority for production)
        self.api_key = config.get("API_KEY")
        self.api_secret = config.get("API_SECRET")
        
        # Check environment variables
        if not self.api_key:
            self.api_key = os.environ.get("APCA_API_KEY_ID")
        if not self.api_secret:
            self.api_secret = os.environ.get("APCA_API_SECRET_KEY")
        
        # Check secure vault if still not found
        if not self.api_key or not self.api_secret:
            try:
                from secure_vault import get_vault
                vault = get_vault()
                if not self.api_key:
                    self.api_key = vault.get_secret("alpaca", "api_key")
                if not self.api_secret:
                    self.api_secret = vault.get_secret("alpaca", "api_secret")
            except Exception:
                pass  # Vault not available, continue with other sources
        
        # Check api_keys.json as last resort
        if not self.api_key or not self.api_secret:
            try:
                import json
                api_keys_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "api_keys.json")
                if os.path.exists(api_keys_path):
                    with open(api_keys_path, 'r') as f:
                        api_keys = json.load(f)
                        alpaca_config = api_keys.get("alpaca", {})
                        if not self.api_key:
                            self.api_key = alpaca_config.get("api_key")
                        if not self.api_secret:
                            self.api_secret = alpaca_config.get("api_secret")
            except Exception:
                pass  # Config file not available
        
        self.paper_trading = config.get("paper_trading", True)
        
        # Handle placeholder values
        if self.api_key and (self.api_key.startswith("YOUR_") or self.api_key == "FROM_VAULT"):
            self.api_key = None
        if self.api_secret and (self.api_secret.startswith("YOUR_") or self.api_secret == "FROM_VAULT"):
            self.api_secret = None
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API_KEY and API_SECRET are required. "
                "Set them via: config, environment variables (APCA_API_KEY_ID, APCA_API_SECRET_KEY), "
                "secure vault (alpaca.api_key, alpaca.api_secret), or config/api_keys.json"
            )
        
        # Initialize TradingClient (paper=False for production/live trading)
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper_trading
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Alpaca TradingClient: {e}")
    
    def name(self) -> str:
        return "alpaca"
    
    def auth(self) -> bool:
        """Authenticate with Alpaca"""
        try:
            # Test authentication by getting account info
            account = self.trading_client.get_account()
            return account is not None
        except Exception as e:
            print(f"Alpaca auth error: {e}")
            return False
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            
            return {
                "account_id": str(account.id) if account.id else "",
                "cash": float(account.cash) if account.cash else 0.0,
                "buying_power": float(account.buying_power) if account.buying_power else 0.0,
                "portfolio_value": float(account.portfolio_value) if account.portfolio_value else 0.0,
                "equity": float(account.equity) if account.equity else 0.0,
                "currency": account.currency if hasattr(account, 'currency') else 'USD',
                "pattern_day_trader": account.pattern_day_trader if hasattr(account, 'pattern_day_trader') else False,
                "trading_blocked": account.trading_blocked if hasattr(account, 'trading_blocked') else False,
                "account_blocked": account.account_blocked if hasattr(account, 'account_blocked') else False
            }
        except Exception as e:
            print(f"Error getting Alpaca account: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "avg_price": float(pos.avg_entry_price) if pos.avg_entry_price else 0.0,
                    "market_value": float(pos.market_value) if pos.market_value else 0.0,
                    "cost_basis": float(pos.cost_basis) if pos.cost_basis else 0.0,
                    "unrealized_pl": float(pos.unrealized_pl) if pos.unrealized_pl else 0.0,
                    "unrealized_plpc": float(pos.unrealized_plpc) if pos.unrealized_plpc else 0.0,
                    "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                })
            
            return result
        except Exception as e:
            print(f"Error getting Alpaca positions: {e}")
            return []
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote for symbol"""
        try:
            # Use market data client for quotes
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import LatestQuoteRequest
            
            # Create data client (uses same credentials)
            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Get latest quote
            quote_request = LatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = data_client.get_latest_quote(quote_request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bp) if quote.bp else 0.0,
                    "ask": float(quote.ap) if quote.ap else 0.0,
                    "last": float(quote.bp) if quote.bp else 0.0,  # Use bid as last if available
                    "volume": None,  # Not available in quote
                    "timestamp": quote.timestamp.isoformat() if quote.timestamp else ""
                }
            
            return {}
        except Exception as e:
            print(f"Error getting Alpaca quote: {e}")
            return {}
    
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place an order (supports stocks and single-leg options)
        
        Args:
            order: Dict with:
                - symbol: Stock or option symbol
                - quantity: float, number of shares/contracts
                - side: 'buy' or 'sell'
                - type: 'market' or 'limit'
                - price: float (required for limit orders)
                - time_in_force: 'day', 'gtc', 'ioc', 'fok' (default: 'day')
        """
        try:
            symbol = order.get('symbol', '')
            qty = float(order.get('quantity', 0))
            side_str = order.get('side', 'buy').lower()
            order_type = order.get('type', 'market').lower()
            time_in_force_str = order.get('time_in_force', 'day').lower()
            
            # Convert side
            side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
            
            # Convert time in force
            tif_map = {
                'day': TimeInForce.DAY,
                'gtc': TimeInForce.GTC,
                'ioc': TimeInForce.IOC,
                'fok': TimeInForce.FOK
            }
            time_in_force = tif_map.get(time_in_force_str, TimeInForce.DAY)
            
            # Create order request
            if order_type == 'limit':
                limit_price = float(order.get('price', 0.0))
                if limit_price <= 0:
                    return {
                        "error": "Limit price must be greater than 0",
                        "status": "rejected"
                    }
                
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=time_in_force,
                    limit_price=limit_price
                )
            else:  # market order
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=time_in_force
                )
            
            # Submit order
            submitted_order = self.trading_client.submit_order(order_data=order_data)
            
            return {
                "order_id": str(submitted_order.id),
                "status": submitted_order.status.value if hasattr(submitted_order.status, 'value') else str(submitted_order.status),
                "broker": "alpaca",
                "paper_trading": self.paper_trading,
                "timestamp": submitted_order.submitted_at.isoformat() if submitted_order.submitted_at else datetime.now().isoformat(),
                "client_order_id": submitted_order.client_order_id if hasattr(submitted_order, 'client_order_id') else None
            }
            
        except APIError as e:
            return {
                "error": f"Alpaca API error: {e}",
                "status": "rejected",
                "status_code": getattr(e, 'status_code', None)
            }
        except Exception as e:
            return {
                "error": f"Order submission error: {e}",
                "status": "rejected"
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            
            return {
                "order_id": order_id,
                "status": "cancelled"
            }
        except APIError as e:
            return {
                "error": f"Alpaca API error: {e}",
                "status": "failed",
                "status_code": getattr(e, 'status_code', None)
            }
        except Exception as e:
            return {
                "error": f"Cancel error: {e}",
                "status": "failed"
            }
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open orders"""
        try:
            cancelled_orders = None
            if hasattr(self.trading_client, "cancel_orders"):
                cancelled_orders = self.trading_client.cancel_orders()
            elif hasattr(self.trading_client, "cancel_all_orders"):
                cancelled_orders = self.trading_client.cancel_all_orders()
            status = {
                "status": "cancelled",
                "broker": "alpaca",
            }
            if cancelled_orders is not None:
                try:
                    status["orders_cancelled"] = len(cancelled_orders)
                except TypeError:
                    pass
            return status
        except APIError as e:
            return {
                "error": f"Alpaca API error: {e}",
                "status": "failed",
                "status_code": getattr(e, 'status_code', None)
            }
        except Exception as e:
            return {
                "error": f"Cancel all orders error: {e}",
                "status": "failed"
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            return {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "symbol": order.symbol,
                "quantity": float(order.qty) if order.qty else 0.0,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None
            }
        except Exception as e:
            print(f"Error getting Alpaca order status: {e}")
            return {}
    
    # ----------------------
    # Enhanced Options Trading Methods
    # ----------------------
    
    def buy_stock_market(self, symbol: str, qty: float) -> Dict[str, Any]:
        """Place a market buy order for a stock"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data=order_data)
            
            return {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "symbol": symbol,
                "quantity": qty,
                "side": "buy",
                "type": "market"
            }
        except Exception as e:
            return {
                "error": f"Error placing market buy order: {e}",
                "status": "rejected"
            }
    
    def sell_stock_limit(self, symbol: str, qty: float, limit_price: float) -> Dict[str, Any]:
        """Place a limit sell order for a stock"""
        try:
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                limit_price=limit_price,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data=order_data)
            
            return {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "symbol": symbol,
                "quantity": qty,
                "side": "sell",
                "type": "limit",
                "limit_price": limit_price
            }
        except Exception as e:
            return {
                "error": f"Error placing limit sell order: {e}",
                "status": "rejected"
            }
    
    def buy_option_market(self, option_symbol: str, qty: float) -> Dict[str, Any]:
        """
        Place a market buy order for a single-leg option
        
        Args:
            option_symbol: Full option contract symbol (e.g., "AAPL240119C00190000")
            qty: Number of contracts
        """
        try:
            order_data = MarketOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data=order_data)
            
            return {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "option_symbol": option_symbol,
                "quantity": qty,
                "side": "buy",
                "type": "market"
            }
        except Exception as e:
            return {
                "error": f"Error placing option market order: {e}",
                "status": "rejected"
            }
    
    def multi_leg_option_order(self, legs: List[tuple], qty: float) -> Dict[str, Any]:
        """
        Place a multi-leg options order (e.g., straddle, spread)
        
        Args:
            legs: List of tuples (symbol, side, ratio_qty)
                - symbol: Option contract symbol
                - side: OrderSide.BUY or OrderSide.SELL (or OptionSide for more control)
                - ratio_qty: Ratio quantity for this leg
            qty: Base quantity for the multi-leg order
            
        Example:
            legs = [
                ("AAPL240119C00190000", OrderSide.BUY, 1),
                ("AAPL240119P00190000", OrderSide.SELL, 1)
            ]
            
        Note: For options, you can also use OptionSide enum values:
            - OptionSide.BUY_TO_OPEN
            - OptionSide.SELL_TO_OPEN
            - OptionSide.BUY_TO_CLOSE
            - OptionSide.SELL_TO_CLOSE
        """
        try:
            # Try to import OptionSide if available (for more granular control)
            try:
                from alpaca.trading.enums import OptionSide
                HAS_OPTION_SIDE = True
            except ImportError:
                HAS_OPTION_SIDE = False
                OptionSide = None
            
            order_legs = []
            for leg_tuple in legs:
                if len(leg_tuple) != 3:
                    raise ValueError(f"Each leg must be a tuple of (symbol, side, ratio_qty), got: {leg_tuple}")
                
                symbol, side, ratio = leg_tuple
                
                # For options legs, OptionLegRequest accepts OrderSide directly
                # But if OptionSide is available and user wants more control, we support that too
                # Most commonly, OrderSide.BUY/SELL works fine for options
                leg_side = side  # Use side as-is (OrderSide or OptionSide)
                
                # Construct each OptionLegRequest
                leg = OptionLegRequest(
                    symbol=symbol,
                    side=leg_side,
                    ratio_qty=ratio
                )
                order_legs.append(leg)
            
            # Create a MarketOrderRequest with order_class MLEG for multi-leg
            mleg_order = MarketOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=order_legs
            )
            
            order = self.trading_client.submit_order(order_data=mleg_order)
            
            return {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "order_type": "multi_leg",
                "legs_count": len(legs),
                "quantity": qty
            }
        except Exception as e:
            return {
                "error": f"Error placing multi-leg options order: {e}",
                "status": "rejected"
            }

# NAE/adapters/kalshi.py
"""
Kalshi Adapter - CFTC-Regulated Prediction Market Integration for NAE
=====================================================================
Connects NAE to Kalshi for prediction market trading.

Kalshi is the FIRST federally regulated (CFTC) exchange for event contracts
in the United States. Users trade on real-world outcomes like economics,
elections, weather, and more.

ENHANCED using official Kalshi resources:
- Official kalshi-python SDK
- Kalshi API documentation (docs.kalshi.com)
- github.com/Kalshi/tools-and-analysis

Key Features:
- Full API integration with RSA-PSS authentication
- Market discovery and search
- Order placement (limit and market orders)
- Position tracking and portfolio management
- Balance and P&L monitoring
- High-probability bonding strategy support
- Cross-platform arbitrage detection (with Polymarket)

Requirements:
- kalshi-python (Official Kalshi Python SDK)
- Kalshi account with API credentials (RSA key pair)
- USD balance on Kalshi

ALIGNED WITH NAE GOALS:
1. Achieve generational wealth
2. Generate $6,243,561+ within 8 years
3. Diversify income streams with regulated prediction markets

REGULATORY ADVANTAGE:
- CFTC regulated = legal certainty in US
- USD-denominated (no crypto required)
- Proper tax reporting (1099 forms)
"""

import os
import json
import time
import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import base64

# =============================================================================
# EXTERNAL IMPORTS - Official Kalshi SDK
# =============================================================================

# Try to import Kalshi SDK
try:
    import kalshi_python
    from kalshi_python import Configuration
    from kalshi_python.models import (
        CreateOrderRequest,
        GetMarketsResponse,
        GetMarketResponse,
        GetEventsResponse,
        GetBalanceResponse,
        GetPositionsResponse,
        UserOrdersGetResponse,
    )
    KALSHI_SDK_AVAILABLE = True
except ImportError:
    KALSHI_SDK_AVAILABLE = False
    kalshi_python = None
    Configuration = None

# Try to import cryptography for RSA signing (backup if SDK doesn't handle it)
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================

class KalshiMarketStatus(Enum):
    """Kalshi market status"""
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    ACTIVE = "active"
    INACTIVE = "inactive"


class KalshiOrderSide(Enum):
    """Order side"""
    YES = "yes"
    NO = "no"


class KalshiOrderType(Enum):
    """Order type"""
    LIMIT = "limit"
    MARKET = "market"


class KalshiOrderStatus(Enum):
    """Order status"""
    RESTING = "resting"
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELED = "canceled"


@dataclass
class KalshiMarket:
    """
    Represents a Kalshi prediction market (event contract)
    """
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    category: str
    status: str
    yes_bid: float  # Best bid price for YES (in cents)
    yes_ask: float  # Best ask price for YES (in cents)
    no_bid: float   # Best bid price for NO
    no_ask: float   # Best ask price for NO
    last_price: float
    volume: int
    volume_24h: int
    open_interest: int
    close_time: Optional[datetime.datetime]
    expiration_time: Optional[datetime.datetime]
    result: Optional[str] = None
    rules_primary: str = ""
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    
    @property
    def yes_price(self) -> float:
        """Get YES price as decimal (0-1)"""
        return self.yes_ask / 100 if self.yes_ask else 0.5
    
    @property
    def no_price(self) -> float:
        """Get NO price as decimal (0-1)"""
        return self.no_ask / 100 if self.no_ask else 0.5
    
    @property
    def spread(self) -> float:
        """Get bid-ask spread for YES side"""
        return (self.yes_ask - self.yes_bid) / 100 if self.yes_ask and self.yes_bid else 0.0
    
    @property
    def days_to_close(self) -> int:
        """Days until market closes"""
        if not self.close_time:
            return 365
        delta = self.close_time - datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        return max(0, delta.days)
    
    @property
    def is_high_probability(self) -> bool:
        """Check if this is a high-probability opportunity"""
        return self.yes_price >= 0.95 or self.no_price >= 0.95
    
    @property
    def best_bonding_side(self) -> Tuple[str, float]:
        """Get the best side for bonding strategy"""
        if self.yes_price >= self.no_price:
            return ("yes", self.yes_price)
        return ("no", self.no_price)


@dataclass
class KalshiEvent:
    """Represents a Kalshi event (collection of markets)"""
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: str
    category: str
    status: str
    mutually_exclusive: bool
    markets: List[str] = field(default_factory=list)


@dataclass
class KalshiOrder:
    """Represents a Kalshi order"""
    order_id: str
    ticker: str
    side: str
    type: str
    status: str
    yes_price: int  # In cents
    no_price: int   # In cents
    count: int      # Number of contracts
    remaining_count: int
    created_time: str
    expiration_time: Optional[str] = None


@dataclass
class KalshiPosition:
    """Represents a position in a Kalshi market"""
    ticker: str
    market_title: str
    position: int  # Positive = long YES, Negative = long NO
    market_exposure: int  # In cents
    realized_pnl: int     # In cents
    total_cost: int       # In cents
    resting_orders_count: int = 0


# =============================================================================
# KALSHI API CLIENT
# =============================================================================

class KalshiApiClient:
    """
    Low-level Kalshi API client with RSA-PSS authentication
    Based on official Kalshi API documentation
    """
    
    # API Endpoints
    PROD_API = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_API = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        demo: bool = False
    ):
        self.api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID")
        self.private_key_pem = private_key_pem or os.environ.get("KALSHI_PRIVATE_KEY")
        self.base_url = self.DEMO_API if demo else self.PROD_API
        self.demo = demo
        
        self.logger = logging.getLogger("KalshiApiClient")
        self.session = requests.Session()
        
        # Load private key for signing
        self.private_key = None
        if CRYPTO_AVAILABLE and self.private_key_pem:
            try:
                self.private_key = serialization.load_pem_private_key(
                    self.private_key_pem.encode(),
                    password=None,
                    backend=default_backend()
                )
            except Exception as e:
                self.logger.warning(f"Could not load private key: {e}")
    
    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """
        Sign request using RSA-PSS
        Based on Kalshi API authentication requirements
        """
        if not self.private_key or not CRYPTO_AVAILABLE:
            return ""
        
        # Create message to sign
        message = f"{timestamp}{method}{path}"
        
        # Sign with RSA-PSS
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated API request"""
        url = f"{self.base_url}{path}"
        
        timestamp = str(int(time.time() * 1000))
        signature = self._sign_request(method, path, timestamp)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key_id and signature:
            headers["KALSHI-ACCESS-KEY"] = self.api_key_id
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get markets"""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        
        return self._request("GET", "/markets", params=params)
    
    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get specific market by ticker"""
        return self._request("GET", f"/markets/{ticker}")
    
    def get_events(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = True
    ) -> Dict[str, Any]:
        """Get events"""
        params = {
            "limit": limit,
            "with_nested_markets": with_nested_markets
        }
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        
        return self._request("GET", "/events", params=params)
    
    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """Get specific event"""
        return self._request("GET", f"/events/{event_ticker}")
    
    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        """Get orderbook for a market"""
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
    
    def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        return self._request("GET", "/portfolio/balance")
    
    def get_positions(self, limit: int = 100) -> Dict[str, Any]:
        """Get current positions"""
        return self._request("GET", "/portfolio/positions", params={"limit": limit})
    
    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get orders"""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        
        return self._request("GET", "/portfolio/orders", params=params)
    
    def create_order(
        self,
        ticker: str,
        side: str,
        count: int,
        type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an order
        
        Args:
            ticker: Market ticker
            side: "yes" or "no"
            count: Number of contracts
            type: "limit" or "market"
            yes_price: Price in cents (1-99) for YES
            no_price: Price in cents (1-99) for NO
            expiration_ts: Optional expiration timestamp
            client_order_id: Optional client-specified order ID
        """
        data = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "count": count,
            "type": type
        }
        
        if yes_price is not None:
            data["yes_price"] = yes_price
        if no_price is not None:
            data["no_price"] = no_price
        if expiration_ts:
            data["expiration_ts"] = expiration_ts
        if client_order_id:
            data["client_order_id"] = client_order_id
        
        return self._request("POST", "/portfolio/orders", data=data)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")


# =============================================================================
# MAIN KALSHI ADAPTER
# =============================================================================

class KalshiAdapter:
    """
    Kalshi API Adapter for NAE
    
    Provides interface to Kalshi's CFTC-regulated prediction markets:
    - Market discovery and analysis
    - Order execution
    - Position management
    - Strategy support (bonding, arbitrage)
    """
    
    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        demo: bool = False
    ):
        """
        Initialize Kalshi adapter
        
        Credentials loaded from (in order):
        1. Constructor arguments
        2. Environment variables (KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY)
        3. Secure vault (kalshi.api_key_id, kalshi.private_key)
        4. Config file (config/kalshi_private_key.pem for private key)
        
        Args:
            api_key_id: Kalshi API Key ID
            private_key_pem: RSA private key in PEM format
            demo: Use demo environment (for testing)
        """
        # Load credentials with fallback chain
        self.api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID") or self._load_from_vault("api_key_id")
        self.private_key_pem = private_key_pem or os.environ.get("KALSHI_PRIVATE_KEY") or self._load_private_key()
        self.demo = demo
        
        self.logger = logging.getLogger("KalshiAdapter")
        self.log_file = "logs/kalshi.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Initialize API client
        self.api_client: Optional[KalshiApiClient] = None
        self.sdk_client: Optional[Any] = None
        self._init_client()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests
        
        # Cache
        self._markets_cache: Dict[str, List[KalshiMarket]] = {}
        self._cache_expiry = 60
        self._last_cache_update = 0
        
        self.log_action(f"KalshiAdapter initialized ({'DEMO' if demo else 'PROD'} mode)")
    
    def _load_from_vault(self, key: str) -> Optional[str]:
        """Load a credential from the secure vault"""
        try:
            from secure_vault import get_vault
            vault = get_vault()
            value = vault.get_secret("kalshi", key)
            if value:
                self.log_action(f"Loaded {key} from secure vault")
            return value
        except Exception as e:
            self.log_action(f"Could not load {key} from vault: {e}")
            return None
    
    def _load_private_key(self) -> Optional[str]:
        """Load RSA private key from vault or file"""
        # Try vault first
        pem = self._load_from_vault("private_key")
        if pem:
            return pem
        
        # Try config file
        key_file = "config/kalshi_private_key.pem"
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r') as f:
                    pem = f.read()
                self.log_action("Loaded private key from config file")
                return pem
            except Exception as e:
                self.log_action(f"Could not load private key from file: {e}")
        
        return None
    
    def _init_client(self):
        """Initialize Kalshi client"""
        # Try official SDK first
        if KALSHI_SDK_AVAILABLE and kalshi_python and Configuration:
            try:
                config = Configuration()
                if self.demo:
                    config.host = "https://demo-api.kalshi.co/trade-api/v2"
                else:
                    config.host = "https://api.elections.kalshi.com/trade-api/v2"
                
                if self.api_key_id:
                    config.api_key_id = self.api_key_id
                if self.private_key_pem:
                    config.private_key_pem = self.private_key_pem
                
                self.sdk_client = kalshi_python.ApiInstance(configuration=config)
                self.log_action("✅ Kalshi SDK client initialized")
                return
            except Exception as e:
                self.log_action(f"⚠️ SDK init failed: {e}")
        
        # Fallback to custom API client
        if self.api_key_id and self.private_key_pem:
            try:
                self.api_client = KalshiApiClient(
                    api_key_id=self.api_key_id,
                    private_key_pem=self.private_key_pem,
                    demo=self.demo
                )
                self.log_action("✅ Kalshi API client initialized")
            except Exception as e:
                self.log_action(f"⚠️ API client init failed: {e}")
        else:
            self.log_action("⚠️ No API credentials provided. Read-only mode.")
    
    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.datetime.now().isoformat()
        log_entry = f"[{ts}] {message}"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
        except Exception:
            pass
        
        print(f"[Kalshi] {message}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    # =========================================================================
    # MARKET DISCOVERY
    # =========================================================================
    
    def get_markets(
        self,
        status: str = "open",
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[KalshiMarket]:
        """
        Get available prediction markets
        
        Args:
            status: Market status filter (open, closed, settled)
            category: Category filter (economics, politics, etc.)
            limit: Maximum number of markets
            
        Returns:
            List of KalshiMarket objects
        """
        cache_key = f"{status}_{category}_{limit}"
        if (cache_key in self._markets_cache and
            time.time() - self._last_cache_update < self._cache_expiry):
            return self._markets_cache[cache_key]
        
        self._rate_limit()
        
        markets = []
        
        # Use API client
        if self.api_client:
            try:
                response = self.api_client.get_markets(
                    limit=limit,
                    status=status
                )
                
                if "markets" in response:
                    for m in response["markets"]:
                        market = self._parse_market(m)
                        if market:
                            if category is None or category.lower() in market.category.lower():
                                markets.append(market)
                
            except Exception as e:
                self.log_action(f"Error fetching markets: {e}")
        
        # Update cache
        self._markets_cache[cache_key] = markets
        self._last_cache_update = time.time()
        
        self.log_action(f"Fetched {len(markets)} markets")
        return markets
    
    def _parse_market(self, data: Dict[str, Any]) -> Optional[KalshiMarket]:
        """Parse market data into KalshiMarket object"""
        try:
            # Parse close time
            close_time = None
            if data.get("close_time"):
                try:
                    close_time = datetime.datetime.fromisoformat(
                        data["close_time"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    pass
            
            # Parse expiration time
            expiration_time = None
            if data.get("expiration_time"):
                try:
                    expiration_time = datetime.datetime.fromisoformat(
                        data["expiration_time"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    pass
            
            return KalshiMarket(
                ticker=data.get("ticker", ""),
                event_ticker=data.get("event_ticker", ""),
                title=data.get("title", ""),
                subtitle=data.get("subtitle", ""),
                category=data.get("category", ""),
                status=data.get("status", "unknown"),
                yes_bid=float(data.get("yes_bid", 0) or 0),
                yes_ask=float(data.get("yes_ask", 0) or 0),
                no_bid=float(data.get("no_bid", 0) or 0),
                no_ask=float(data.get("no_ask", 0) or 0),
                last_price=float(data.get("last_price", 0) or 0),
                volume=int(data.get("volume", 0) or 0),
                volume_24h=int(data.get("volume_24h", 0) or 0),
                open_interest=int(data.get("open_interest", 0) or 0),
                close_time=close_time,
                expiration_time=expiration_time,
                result=data.get("result"),
                rules_primary=data.get("rules_primary", ""),
                floor_strike=data.get("floor_strike"),
                cap_strike=data.get("cap_strike")
            )
        except Exception as e:
            self.logger.debug(f"Error parsing market: {e}")
            return None
    
    def get_market_by_ticker(self, ticker: str) -> Optional[KalshiMarket]:
        """Get a specific market by ticker"""
        self._rate_limit()
        
        if self.api_client:
            try:
                response = self.api_client.get_market(ticker)
                if "market" in response:
                    return self._parse_market(response["market"])
            except Exception as e:
                self.log_action(f"Error fetching market {ticker}: {e}")
        
        return None
    
    def get_events(
        self,
        status: str = "open",
        limit: int = 100
    ) -> List[KalshiEvent]:
        """Get Kalshi events (collections of markets)"""
        self._rate_limit()
        
        events = []
        
        if self.api_client:
            try:
                response = self.api_client.get_events(
                    limit=limit,
                    status=status
                )
                
                if "events" in response:
                    for e in response["events"]:
                        event = self._parse_event(e)
                        if event:
                            events.append(event)
            except Exception as e:
                self.log_action(f"Error fetching events: {e}")
        
        return events
    
    def _parse_event(self, data: Dict[str, Any]) -> Optional[KalshiEvent]:
        """Parse event data"""
        try:
            market_tickers = []
            if "markets" in data:
                for m in data["markets"]:
                    if isinstance(m, dict):
                        market_tickers.append(m.get("ticker", ""))
                    else:
                        market_tickers.append(str(m))
            
            return KalshiEvent(
                event_ticker=data.get("event_ticker", ""),
                series_ticker=data.get("series_ticker", ""),
                title=data.get("title", ""),
                subtitle=data.get("subtitle", ""),
                category=data.get("category", ""),
                status=data.get("status", ""),
                mutually_exclusive=data.get("mutually_exclusive", False),
                markets=market_tickers
            )
        except Exception as e:
            self.logger.debug(f"Error parsing event: {e}")
            return None
    
    def search_markets(self, query: str, limit: int = 20) -> List[KalshiMarket]:
        """Search markets by keyword"""
        all_markets = self.get_markets(limit=500)
        
        query_lower = query.lower()
        matching = [
            m for m in all_markets
            if query_lower in m.title.lower() or query_lower in m.subtitle.lower()
        ]
        
        return matching[:limit]
    
    # =========================================================================
    # ACCOUNT / PORTFOLIO
    # =========================================================================
    
    def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        self._rate_limit()
        
        if not self.api_client:
            return {"error": "API client not initialized", "balance": 0, "available": 0}
        
        try:
            response = self.api_client.get_balance()
            
            # Kalshi returns balance in cents
            balance = response.get("balance", 0)
            available = response.get("portfolio_value", balance)
            
            return {
                "balance_cents": balance,
                "balance_usd": balance / 100,
                "available_cents": available,
                "available_usd": available / 100,
                "currency": "USD"
            }
        except Exception as e:
            self.log_action(f"Error fetching balance: {e}")
            return {"error": str(e), "balance": 0, "available": 0}
    
    def get_positions(self) -> List[KalshiPosition]:
        """Get current positions"""
        self._rate_limit()
        
        positions = []
        
        if not self.api_client:
            return positions
        
        try:
            response = self.api_client.get_positions()
            
            if "market_positions" in response:
                for p in response["market_positions"]:
                    position = KalshiPosition(
                        ticker=p.get("ticker", ""),
                        market_title=p.get("market_title", ""),
                        position=p.get("position", 0),
                        market_exposure=p.get("market_exposure", 0),
                        realized_pnl=p.get("realized_pnl", 0),
                        total_cost=p.get("total_cost", 0),
                        resting_orders_count=p.get("resting_orders_count", 0)
                    )
                    positions.append(position)
        except Exception as e:
            self.log_action(f"Error fetching positions: {e}")
        
        return positions
    
    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[KalshiOrder]:
        """Get orders"""
        self._rate_limit()
        
        orders = []
        
        if not self.api_client:
            return orders
        
        try:
            response = self.api_client.get_orders(ticker=ticker, status=status)
            
            if "orders" in response:
                for o in response["orders"]:
                    order = KalshiOrder(
                        order_id=o.get("order_id", ""),
                        ticker=o.get("ticker", ""),
                        side=o.get("side", ""),
                        type=o.get("type", ""),
                        status=o.get("status", ""),
                        yes_price=o.get("yes_price", 0),
                        no_price=o.get("no_price", 0),
                        count=o.get("count", 0),
                        remaining_count=o.get("remaining_count", 0),
                        created_time=o.get("created_time", ""),
                        expiration_time=o.get("expiration_time")
                    )
                    orders.append(order)
        except Exception as e:
            self.log_action(f"Error fetching orders: {e}")
        
        return orders
    
    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================
    
    def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        count: int,
        price: int,  # Price in cents (1-99)
        order_type: str = "limit"
    ) -> Dict[str, Any]:
        """
        Place an order on Kalshi
        
        Args:
            ticker: Market ticker
            side: "yes" or "no"
            count: Number of contracts
            price: Price in cents (1-99)
            order_type: "limit" or "market"
            
        Returns:
            Order result
        """
        if not self.api_client:
            return {"error": "API client not initialized"}
        
        # Validate inputs
        side = side.lower()
        if side not in ["yes", "no"]:
            return {"error": f"Invalid side: {side}. Must be 'yes' or 'no'."}
        
        if not 1 <= price <= 99:
            return {"error": f"Invalid price: {price}. Must be 1-99 cents."}
        
        if count <= 0:
            return {"error": f"Invalid count: {count}. Must be positive."}
        
        self._rate_limit()
        
        try:
            # Set price based on side
            yes_price = price if side == "yes" else None
            no_price = price if side == "no" else None
            
            response = self.api_client.create_order(
                ticker=ticker,
                side=side,
                count=count,
                type=order_type,
                yes_price=yes_price,
                no_price=no_price
            )
            
            if "error" in response:
                self.log_action(f"❌ Order failed: {response['error']}")
                return response
            
            order_id = response.get("order", {}).get("order_id", "unknown")
            self.log_action(f"✅ Order placed: {side.upper()} {count}x @ {price}¢ on {ticker}")
            
            return {
                "success": True,
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": price,
                "price_usd": price / 100,
                "status": "resting"
            }
            
        except Exception as e:
            self.log_action(f"❌ Order placement failed: {e}")
            return {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not self.api_client:
            return {"error": "API client not initialized"}
        
        self._rate_limit()
        
        try:
            response = self.api_client.cancel_order(order_id)
            
            if "error" in response:
                return response
            
            self.log_action(f"✅ Order cancelled: {order_id}")
            return {"success": True, "order_id": order_id}
            
        except Exception as e:
            self.log_action(f"❌ Cancel failed: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # BONDING STRATEGY
    # =========================================================================
    
    def find_bonding_opportunities(
        self,
        min_probability: float = 0.95,
        max_days_to_close: int = 14,
        min_volume: int = 100,
        min_annualized_return: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Find high-probability bonding opportunities on Kalshi
        
        Args:
            min_probability: Minimum probability threshold (default 95%)
            max_days_to_close: Maximum days until close
            min_volume: Minimum trading volume
            min_annualized_return: Minimum annualized return percentage
            
        Returns:
            List of bonding opportunities sorted by annualized return
        """
        markets = self.get_markets(status="open", limit=500)
        opportunities = []
        
        for market in markets:
            # Skip if not high probability
            if not market.is_high_probability:
                continue
            
            # Skip if too far from close
            if market.days_to_close > max_days_to_close:
                continue
            
            # Skip if low volume
            if market.volume < min_volume:
                continue
            
            # Get best bonding side
            side, price = market.best_bonding_side
            
            # Calculate potential return
            days = max(1, market.days_to_close)
            profit_per_contract = 100 - (price * 100)  # In cents
            profit_pct = (profit_per_contract / (price * 100)) * 100
            annualized = profit_pct * (365 / days)
            
            if annualized < min_annualized_return:
                continue
            
            opportunities.append({
                "ticker": market.ticker,
                "title": market.title,
                "side": side,
                "price_cents": int(price * 100),
                "price_decimal": price,
                "days_to_close": days,
                "profit_per_contract_cents": profit_per_contract,
                "profit_pct": round(profit_pct, 2),
                "annualized_return": round(annualized, 2),
                "volume": market.volume,
                "open_interest": market.open_interest,
                "spread": market.spread,
                "category": market.category,
                "risk_level": "LOW" if price >= 0.98 else "MEDIUM" if price >= 0.95 else "HIGH"
            })
        
        opportunities.sort(key=lambda x: x["annualized_return"], reverse=True)
        
        self.log_action(f"Found {len(opportunities)} bonding opportunities")
        return opportunities
    
    # =========================================================================
    # ARBITRAGE DETECTION (with Polymarket)
    # =========================================================================
    
    def find_cross_platform_arbitrage(
        self,
        polymarket_prices: Dict[str, float],
        min_spread: float = 0.03
    ) -> List[Dict[str, Any]]:
        """
        Find arbitrage opportunities between Kalshi and Polymarket
        
        Args:
            polymarket_prices: Dict mapping questions to Polymarket YES prices
            min_spread: Minimum price spread to consider
            
        Returns:
            List of arbitrage opportunities
        """
        kalshi_markets = self.get_markets(status="open")
        opportunities = []
        
        for market in kalshi_markets:
            title_lower = market.title.lower()
            
            for poly_question, poly_price in polymarket_prices.items():
                if self._is_similar_market(title_lower, poly_question.lower()):
                    kalshi_price = market.yes_price
                    spread = abs(kalshi_price - poly_price)
                    
                    if spread >= min_spread:
                        opportunities.append({
                            "kalshi_ticker": market.ticker,
                            "kalshi_title": market.title,
                            "kalshi_price": kalshi_price,
                            "polymarket_price": poly_price,
                            "spread": round(spread, 4),
                            "spread_pct": round(spread * 100, 2),
                            "action": {
                                "kalshi": "BUY" if kalshi_price < poly_price else "SELL",
                                "polymarket": "SELL" if kalshi_price < poly_price else "BUY"
                            },
                            "potential_profit_per_contract_cents": int(spread * 100)
                        })
        
        opportunities.sort(key=lambda x: x["spread"], reverse=True)
        return opportunities
    
    def _is_similar_market(self, title1: str, title2: str) -> bool:
        """Check if two market titles are about the same event"""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        stop_words = {"will", "the", "a", "an", "be", "is", "in", "on", "at", "to", "?", "by"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap >= 0.5
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of Kalshi markets"""
        markets = self.get_markets(limit=500)
        
        total_volume = sum(m.volume for m in markets)
        total_open_interest = sum(m.open_interest for m in markets)
        
        # Category breakdown
        categories: Dict[str, Dict[str, Any]] = {}
        for m in markets:
            cat = m.category or "Other"
            if cat not in categories:
                categories[cat] = {"count": 0, "volume": 0}
            categories[cat]["count"] += 1
            categories[cat]["volume"] += m.volume
        
        # Bonding opportunities
        bonding = self.find_bonding_opportunities()
        
        # Balance
        balance = self.get_balance()
        
        return {
            "total_markets": len(markets),
            "total_volume": total_volume,
            "total_open_interest": total_open_interest,
            "categories": categories,
            "bonding_opportunities": len(bonding),
            "top_bonding": bonding[:5] if bonding else [],
            "balance": balance,
            "sdk_available": KALSHI_SDK_AVAILABLE,
            "api_connected": self.api_client is not None,
            "environment": "DEMO" if self.demo else "PRODUCTION",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Kalshi API"""
        results = {
            "api_connected": False,
            "balance": None,
            "sample_market": None,
            "errors": []
        }
        
        # Test market fetching
        try:
            markets = self.get_markets(limit=1)
            results["api_connected"] = len(markets) > 0
            if markets:
                results["sample_market"] = markets[0].title
        except Exception as e:
            results["errors"].append(f"Markets: {e}")
        
        # Test balance (requires auth)
        if self.api_client:
            try:
                balance = self.get_balance()
                if "error" not in balance:
                    results["balance"] = balance
            except Exception as e:
                results["errors"].append(f"Balance: {e}")
        
        results["status"] = "connected" if results["api_connected"] else "error"
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_adapter_instance: Optional[KalshiAdapter] = None

def get_kalshi_adapter(demo: bool = False) -> KalshiAdapter:
    """Get global Kalshi adapter instance"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = KalshiAdapter(demo=demo)
    return _adapter_instance


# =============================================================================
# MAIN - Test the adapter
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("KALSHI ADAPTER - CFTC-Regulated Prediction Markets")
    print("="*70)
    
    # Use demo mode for testing
    adapter = KalshiAdapter(demo=True)
    
    print("\n=== Testing Connection ===")
    status = adapter.test_connection()
    print(f"API Connected: {'✅' if status['api_connected'] else '❌'}")
    print(f"Sample Market: {status.get('sample_market', 'N/A')}")
    
    print("\n=== Market Summary ===")
    summary = adapter.get_market_summary()
    print(f"Total Markets: {summary['total_markets']}")
    print(f"Total Volume: {summary['total_volume']:,} contracts")
    print(f"Bonding Opportunities: {summary['bonding_opportunities']}")
    
    print("\n=== Top 5 Bonding Opportunities ===")
    for i, opp in enumerate(summary.get("top_bonding", [])[:5], 1):
        print(f"\n{i}. {opp['title'][:60]}...")
        print(f"   Ticker: {opp['ticker']}")
        print(f"   Side: {opp['side'].upper()} @ {opp['price_cents']}¢")
        print(f"   Days to Close: {opp['days_to_close']}")
        print(f"   Annualized Return: {opp['annualized_return']}%")
        print(f"   Risk Level: {opp['risk_level']}")
    
    print("\n=== Markets by Category ===")
    for cat, data in sorted(summary.get("categories", {}).items(), 
                           key=lambda x: x[1]["volume"], reverse=True)[:5]:
        print(f"  {cat}: {data['count']} markets, {data['volume']:,} volume")
    
    print("\n" + "="*70)
    print("Adapter Test Complete")
    print("="*70)


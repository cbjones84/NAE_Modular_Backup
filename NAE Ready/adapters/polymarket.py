# NAE/adapters/polymarket.py
"""
Polymarket Adapter - Enhanced Prediction Market Integration for NAE
====================================================================
Connects NAE to Polymarket for prediction market trading.

This adapter is ENHANCED using official code from:
- github.com/Polymarket/py-clob-client (Official Python SDK)
- github.com/Polymarket/agents (Official AI agents framework)

Polymarket is a decentralized prediction market on Polygon where users
bet USDC on real-world outcomes (elections, sports, crypto, etc.).

Features:
- Full CLOB (Central Limit Order Book) integration
- Gamma API for market discovery
- Web3 wallet integration for Polygon
- Market and limit order execution
- Position tracking
- High-probability bonding strategy support
- Cross-platform arbitrage detection
- AI-powered market analysis

Requirements:
- py-clob-client (Polymarket's official Python SDK): pip install py-clob-client
- web3 (Ethereum/Polygon interface): pip install web3
- Polygon wallet with USDC
- Polymarket API credentials

ALIGNED WITH NAE GOALS:
1. Achieve generational wealth
2. Generate $6,243,561+ within 8 years
3. Diversify income streams beyond traditional trading
"""

import os
import json
import time
import datetime
import requests
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# =============================================================================
# EXTERNAL IMPORTS - Official Polymarket SDK
# =============================================================================

# Try to import Polymarket SDK (py-clob-client)
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        OrderArgs,
        MarketOrderArgs,
        OrderType,
        OrderBookSummary,
        BookParams,
        TradeParams,
        OpenOrderParams,
    )
    from py_clob_client.constants import POLYGON  # Chain ID 137
    POLYMARKET_SDK_AVAILABLE = True
except ImportError:
    POLYMARKET_SDK_AVAILABLE = False
    ClobClient = None
    ApiCreds = None
    OrderArgs = None
    MarketOrderArgs = None
    OrderType = None
    OrderBookSummary = None
    BookParams = None
    POLYGON = 137

# Try to import Web3 for wallet operations
try:
    from web3 import Web3
    from web3.constants import MAX_INT
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None

# Try to import httpx for faster async requests
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================

class MarketStatus(Enum):
    """Polymarket market status"""
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class PolymarketMarket:
    """
    Represents a Polymarket prediction market
    Based on official Polymarket API response structure
    """
    id: str
    question: str
    description: str
    outcomes: List[str]
    outcome_prices: List[float]
    end_date: datetime.datetime
    volume: float
    liquidity: float
    yes_price: float
    no_price: float
    status: MarketStatus
    category: str
    slug: str
    clob_token_ids: List[str] = field(default_factory=list)
    condition_id: str = ""
    rewards_min_size: float = 0.0
    rewards_max_spread: float = 0.0
    spread: float = 0.0
    active: bool = True
    funded: bool = True
    
    @property
    def days_to_resolution(self) -> int:
        """Days until market resolves"""
        delta = self.end_date - datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        return max(0, delta.days)
    
    @property
    def is_high_probability(self) -> bool:
        """Check if this is a high-probability bonding opportunity"""
        return self.yes_price >= 0.95 or self.no_price >= 0.95
    
    @property
    def best_bonding_side(self) -> Tuple[str, float]:
        """Get the best side for bonding strategy"""
        if self.yes_price >= self.no_price:
            return ("YES", self.yes_price)
        return ("NO", self.no_price)
    
    @property
    def implied_probability_yes(self) -> float:
        """Get implied probability for YES outcome"""
        return self.yes_price
    
    @property
    def implied_probability_no(self) -> float:
        """Get implied probability for NO outcome"""
        return self.no_price


@dataclass
class PolymarketOrder:
    """Represents a Polymarket order"""
    id: str
    market_id: str
    side: str
    outcome: str  # YES or NO
    price: float
    size: float
    filled: float
    status: str
    created_at: datetime.datetime
    token_id: str = ""


@dataclass
class PolymarketPosition:
    """Represents a position in a Polymarket market"""
    market_id: str
    market_question: str
    outcome: str
    shares: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    token_id: str = ""


@dataclass
class PolymarketEvent:
    """Represents a Polymarket event (collection of markets)"""
    id: str
    ticker: str
    slug: str
    title: str
    description: str
    end_date: str
    active: bool
    closed: bool
    archived: bool
    restricted: bool
    new: bool
    featured: bool
    markets: List[str] = field(default_factory=list)


# =============================================================================
# GAMMA API CLIENT
# Based on github.com/Polymarket/agents/agents/polymarket/gamma.py
# =============================================================================

class GammaMarketClient:
    """
    Polymarket Gamma API Client for market discovery
    
    This is based on the official Polymarket agents repository:
    https://github.com/Polymarket/agents/blob/main/agents/polymarket/gamma.py
    """
    
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.gamma_markets_endpoint = self.gamma_url + "/markets"
        self.gamma_events_endpoint = self.gamma_url + "/events"
        self.logger = logging.getLogger("GammaMarketClient")
    
    def _request(self, url: str, params: Optional[Dict] = None) -> Any:
        """Make HTTP request with error handling"""
        try:
            if HTTPX_AVAILABLE:
                response = httpx.get(url, params=params, timeout=30)
            else:
                response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API error: HTTP {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None
    
    def get_markets(
        self,
        querystring_params: Optional[Dict] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get markets from Gamma API
        
        Args:
            querystring_params: Optional query parameters
            limit: Maximum number of markets
            
        Returns:
            List of market data dictionaries
        """
        params = querystring_params or {}
        params["limit"] = limit
        
        data = self._request(self.gamma_markets_endpoint, params=params)
        return data if data else []
    
    def get_events(
        self,
        querystring_params: Optional[Dict] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get events from Gamma API"""
        params = querystring_params or {}
        params["limit"] = limit
        
        data = self._request(self.gamma_events_endpoint, params=params)
        return data if data else []
    
    def get_current_markets(self, limit: int = 100) -> List[Dict]:
        """Get currently active markets"""
        return self.get_markets(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
            },
            limit=limit
        )
    
    def get_all_current_markets(self, limit: int = 100) -> List[Dict]:
        """Get ALL currently active markets with pagination"""
        offset = 0
        all_markets = []
        
        while True:
            params = {
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit,
                "offset": offset,
            }
            market_batch = self.get_markets(querystring_params=params)
            
            if not market_batch:
                break
                
            all_markets.extend(market_batch)
            
            if len(market_batch) < limit:
                break
            offset += limit
        
        return all_markets
    
    def get_current_events(self, limit: int = 100) -> List[Dict]:
        """Get current active events"""
        return self.get_events(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
            },
            limit=limit
        )
    
    def get_clob_tradable_markets(self, limit: int = 100) -> List[Dict]:
        """Get markets that are tradable on CLOB"""
        return self.get_markets(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
                "enableOrderBook": True,
            },
            limit=limit
        )
    
    def get_market(self, market_id: Union[str, int]) -> Optional[Dict]:
        """Get a specific market by ID"""
        url = f"{self.gamma_markets_endpoint}/{market_id}"
        return self._request(url)
    
    def get_event(self, event_id: Union[str, int]) -> Optional[Dict]:
        """Get a specific event by ID"""
        url = f"{self.gamma_events_endpoint}/{event_id}"
        return self._request(url)


# =============================================================================
# MAIN POLYMARKET ADAPTER
# Enhanced with official Polymarket code patterns
# =============================================================================

class PolymarketAdapter:
    """
    Enhanced Polymarket API Adapter for NAE
    
    Based on official Polymarket repositories:
    - py-clob-client: CLOB trading interface
    - agents: Market discovery and trading patterns
    
    Provides interface to Polymarket's prediction markets including:
    - Market discovery and analysis via Gamma API
    - CLOB order execution with Level 1/2 authentication
    - Position management
    - Web3 wallet integration
    - Strategy support (bonding, arbitrage, AI-powered)
    """
    
    # Polymarket API endpoints
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    # Polygon network
    CHAIN_ID = 137  # Polygon mainnet
    POLYGON_RPC = "https://polygon-rpc.com"
    
    # Contract addresses (from official Polymarket code)
    EXCHANGE_ADDRESS = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
    NEG_RISK_EXCHANGE_ADDRESS = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        sandbox: bool = False
    ):
        """
        Initialize Polymarket adapter
        
        Authentication Levels (from official py-clob-client):
        - Level 0: Only host URL - access to public endpoints
        - Level 1: Host + chain_id + private_key - L1 authenticated endpoints
        - Level 2: Host + chain_id + private_key + API creds - all endpoints
        
        Args:
            api_key: Polymarket CLOB API key
            api_secret: Polymarket CLOB API secret
            api_passphrase: Polymarket CLOB API passphrase
            wallet_address: Polygon wallet address
            private_key: Wallet private key (for signing transactions)
            sandbox: Use sandbox/testnet mode (Mumbai testnet)
        """
        self.api_key = api_key or os.environ.get("POLYMARKET_API_KEY")
        self.api_secret = api_secret or os.environ.get("POLYMARKET_API_SECRET")
        self.api_passphrase = api_passphrase or os.environ.get("POLYMARKET_API_PASSPHRASE")
        self.wallet_address = wallet_address or os.environ.get("POLYMARKET_WALLET")
        self.private_key = private_key or os.environ.get("POLYGON_WALLET_PRIVATE_KEY") or os.environ.get("POLYMARKET_PRIVATE_KEY")
        self.sandbox = sandbox
        
        # Chain configuration
        self.chain_id = 80001 if sandbox else self.CHAIN_ID  # Mumbai testnet or Polygon mainnet
        
        # Initialize logging
        self.logger = logging.getLogger("PolymarketAdapter")
        self.log_file = "logs/polymarket.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Initialize Gamma client for market discovery
        self.gamma = GammaMarketClient()
        
        # Initialize CLOB client for trading
        self.clob_client: Optional[Any] = None
        self.api_creds: Optional[Any] = None
        self._init_clob_client()
        
        # Initialize Web3 for wallet operations
        self.web3: Optional[Any] = None
        self.usdc_contract: Optional[Any] = None
        self.ctf_contract: Optional[Any] = None
        self._init_web3()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
        # Cache
        self._markets_cache: Dict[str, List[PolymarketMarket]] = {}
        self._cache_expiry = 60  # 60 seconds
        self._last_cache_update = 0
        
        self.log_action("PolymarketAdapter initialized (Enhanced with official Polymarket code)")
    
    def _init_clob_client(self):
        """
        Initialize CLOB client with authentication
        Based on official py-clob-client initialization pattern
        """
        if not POLYMARKET_SDK_AVAILABLE or ClobClient is None:
            self.log_action("⚠️ py-clob-client not installed. Install with: pip install py-clob-client")
            return
        
        if not self.private_key:
            self.log_action("⚠️ No private key provided. CLOB trading disabled.")
            return
        
        try:
            # Initialize client with private key (Level 1 auth)
            self.clob_client = ClobClient(
                host=self.CLOB_API,
                key=self.private_key,
                chain_id=self.chain_id
            )
            
            # Try to get or create API credentials (Level 2 auth)
            if self.api_key and self.api_secret and self.api_passphrase and ApiCreds is not None:
                # Use provided credentials
                self.api_creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase
                )
                if self.clob_client is not None:
                    self.clob_client.set_api_creds(self.api_creds)
                self.log_action("✅ CLOB client initialized with provided API credentials (Level 2)")
            else:
                # Try to derive or create credentials
                try:
                    if self.clob_client is not None:
                        self.api_creds = self.clob_client.create_or_derive_api_creds()
                        self.clob_client.set_api_creds(self.api_creds)
                        self.log_action("✅ CLOB client initialized with derived API credentials (Level 2)")
                except Exception as e:
                    self.log_action(f"⚠️ Could not derive API creds: {e}. Using Level 1 auth only.")
            
        except Exception as e:
            self.log_action(f"❌ CLOB client initialization failed: {e}")
            self.clob_client = None
    
    def _init_web3(self):
        """
        Initialize Web3 for Polygon wallet operations
        Based on official Polymarket agents Web3 setup
        """
        if not WEB3_AVAILABLE or Web3 is None:
            self.log_action("⚠️ web3 not installed. Install with: pip install web3")
            return
        
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.POLYGON_RPC))
            
            # ERC20 ABI for USDC balance queries
            erc20_abi = json.loads('[{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}]')
            
            if self.web3 is not None:
                self.usdc_contract = self.web3.eth.contract(
                    address=Web3.to_checksum_address(self.USDC_ADDRESS),
                    abi=erc20_abi
                )
            
            self.log_action("✅ Web3 initialized for Polygon")
            
        except Exception as e:
            self.log_action(f"❌ Web3 initialization failed: {e}")
            self.web3 = None
    
    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.datetime.now().isoformat()
        log_entry = f"[{ts}] {message}"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
        except Exception:
            pass
        
        print(f"[Polymarket] {message}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    # =========================================================================
    # WALLET OPERATIONS
    # Based on official Polymarket agents wallet integration
    # =========================================================================
    
    def get_wallet_address(self) -> Optional[str]:
        """Get wallet address from private key"""
        if self.wallet_address:
            return self.wallet_address
        
        if self.web3 is not None and self.private_key:
            try:
                account = self.web3.eth.account.from_key(self.private_key)
                return account.address
            except Exception as e:
                self.log_action(f"Error deriving wallet address: {e}")
        
        return None
    
    def get_usdc_balance(self) -> float:
        """
        Get USDC balance from Polygon wallet
        Based on official Polymarket agents balance check
        """
        if self.web3 is None or self.usdc_contract is None or Web3 is None:
            return 0.0
        
        wallet = self.get_wallet_address()
        if not wallet:
            return 0.0
        
        try:
            balance_wei = self.usdc_contract.functions.balanceOf(
                Web3.to_checksum_address(wallet)
            ).call()
            # USDC has 6 decimals
            return float(balance_wei) / 1e6
        except Exception as e:
            self.log_action(f"Error getting USDC balance: {e}")
            return 0.0
    
    def get_balance(self) -> Dict[str, Any]:
        """Get comprehensive balance info"""
        usdc_balance = self.get_usdc_balance()
        wallet = self.get_wallet_address()
        
        result: Dict[str, Any] = {
            "usdc": usdc_balance,
            "wallet_address": wallet,
            "chain": "Polygon" if not self.sandbox else "Mumbai Testnet",
            "chain_id": self.chain_id
        }
        
        # Get MATIC balance if Web3 available
        if self.web3 is not None and wallet and Web3 is not None:
            try:
                matic_wei = self.web3.eth.get_balance(Web3.to_checksum_address(wallet))
                result["matic"] = float(self.web3.from_wei(matic_wei, 'ether'))
            except Exception:
                result["matic"] = 0.0
        
        return result
    
    # =========================================================================
    # MARKET DISCOVERY
    # Based on official Polymarket agents Gamma API patterns
    # =========================================================================
    
    def get_markets(
        self,
        active_only: bool = True,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[PolymarketMarket]:
        """
        Get available prediction markets using Gamma API
        
        Args:
            active_only: Only return active markets
            category: Filter by category (politics, sports, crypto, etc.)
            limit: Maximum number of markets to return
            
        Returns:
            List of PolymarketMarket objects
        """
        # Check cache
        cache_key = f"{active_only}_{category}_{limit}"
        if (cache_key in self._markets_cache and 
            time.time() - self._last_cache_update < self._cache_expiry):
            return self._markets_cache[cache_key]
        
        # Build query params
        params: Dict[str, Any] = {}
        if active_only:
            params["active"] = True
            params["closed"] = False
            params["archived"] = False
        if category:
            params["tag"] = category
        
        # Fetch from Gamma API
        raw_markets = self.gamma.get_markets(querystring_params=params, limit=limit)
        
        # Parse markets
        markets = []
        for m in raw_markets:
            try:
                market = self._parse_market(m)
                if market:
                    markets.append(market)
            except Exception as e:
                self.logger.debug(f"Error parsing market: {e}")
        
        # Update cache
        self._markets_cache[cache_key] = markets
        self._last_cache_update = time.time()
        
        self.log_action(f"Fetched {len(markets)} markets")
        return markets
    
    def get_all_tradeable_markets(self) -> List[PolymarketMarket]:
        """Get all markets that are tradeable on CLOB"""
        raw_markets = self.gamma.get_clob_tradable_markets(limit=500)
        
        markets = []
        for m in raw_markets:
            try:
                market = self._parse_market(m)
                if market and market.active:
                    markets.append(market)
            except Exception:
                pass
        
        return markets
    
    def get_events(self, active_only: bool = True, limit: int = 100) -> List[PolymarketEvent]:
        """Get Polymarket events (collections of markets)"""
        params: Dict[str, Any] = {}
        if active_only:
            params["active"] = True
            params["closed"] = False
            params["archived"] = False
        
        raw_events = self.gamma.get_events(querystring_params=params, limit=limit)
        
        events = []
        for e in raw_events:
            try:
                event = self._parse_event(e)
                if event:
                    events.append(event)
            except Exception:
                pass
        
        return events
    
    def _parse_market(self, data: Dict[str, Any]) -> Optional[PolymarketMarket]:
        """
        Parse market data into PolymarketMarket object
        Based on official Polymarket API response structure
        """
        try:
            # Extract outcomes and prices
            outcomes = data.get("outcomes", ["Yes", "No"])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            
            prices = data.get("outcomePrices", [0.5, 0.5])
            if isinstance(prices, str):
                prices = json.loads(prices)
            
            prices = [float(p) for p in prices]
            yes_price = prices[0] if len(prices) > 0 else 0.5
            no_price = prices[1] if len(prices) > 1 else 1 - yes_price
            
            # Extract CLOB token IDs
            clob_token_ids = data.get("clobTokenIds", [])
            if isinstance(clob_token_ids, str):
                clob_token_ids = json.loads(clob_token_ids)
            
            # Parse end date
            end_date_str = data.get("endDate") or data.get("end_date_iso") or data.get("endDateIso")
            if end_date_str:
                try:
                    end_date = datetime.datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    end_date = datetime.datetime.now() + datetime.timedelta(days=365)
            else:
                end_date = datetime.datetime.now() + datetime.timedelta(days=365)
            
            # Determine status
            if data.get("resolved"):
                status = MarketStatus.RESOLVED
            elif data.get("closed"):
                status = MarketStatus.CLOSED
            elif data.get("archived"):
                status = MarketStatus.ARCHIVED
            elif data.get("active", True):
                status = MarketStatus.ACTIVE
            else:
                status = MarketStatus.UNKNOWN
            
            # Extract category from tags
            tags = data.get("tags", [])
            if isinstance(tags, list) and len(tags) > 0:
                category = tags[0].get("label", "") if isinstance(tags[0], dict) else str(tags[0])
            else:
                category = data.get("category", "")
            
            return PolymarketMarket(
                id=str(data.get("id") or data.get("condition_id", "")),
                question=data.get("question", "Unknown"),
                description=data.get("description", ""),
                outcomes=outcomes if isinstance(outcomes, list) else ["Yes", "No"],
                outcome_prices=prices,
                end_date=end_date,
                volume=float(data.get("volume", 0) or data.get("volumeNum", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or data.get("liquidityNum", 0) or 0),
                yes_price=yes_price,
                no_price=no_price,
                status=status,
                category=category,
                slug=data.get("slug", ""),
                clob_token_ids=clob_token_ids,
                condition_id=data.get("conditionId", ""),
                rewards_min_size=float(data.get("rewardsMinSize", 0) or 0),
                rewards_max_spread=float(data.get("rewardsMaxSpread", 0) or 0),
                spread=float(data.get("spread", 0) or 0),
                active=data.get("active", True),
                funded=data.get("funded", True)
            )
        except Exception as e:
            self.logger.debug(f"Error parsing market data: {e}")
            return None
    
    def _parse_event(self, data: Dict[str, Any]) -> Optional[PolymarketEvent]:
        """Parse event data into PolymarketEvent object"""
        try:
            # Extract market IDs
            markets_data = data.get("markets", [])
            market_ids = []
            for m in markets_data:
                if isinstance(m, dict):
                    market_ids.append(str(m.get("id", "")))
                else:
                    market_ids.append(str(m))
            
            return PolymarketEvent(
                id=str(data.get("id", "")),
                ticker=data.get("ticker", ""),
                slug=data.get("slug", ""),
                title=data.get("title", ""),
                description=data.get("description", ""),
                end_date=data.get("endDate", ""),
                active=data.get("active", True),
                closed=data.get("closed", False),
                archived=data.get("archived", False),
                restricted=data.get("restricted", False),
                new=data.get("new", False),
                featured=data.get("featured", False),
                markets=market_ids
            )
        except Exception as e:
            self.logger.debug(f"Error parsing event data: {e}")
            return None
    
    def get_market_by_id(self, market_id: str) -> Optional[PolymarketMarket]:
        """Get a specific market by ID"""
        data = self.gamma.get_market(market_id)
        if data:
            return self._parse_market(data)
        return None
    
    def search_markets(self, query: str, limit: int = 20) -> List[PolymarketMarket]:
        """Search markets by keyword"""
        params = {"_q": query}
        raw_markets = self.gamma.get_markets(querystring_params=params, limit=limit)
        
        markets = []
        for m in raw_markets:
            market = self._parse_market(m)
            if market:
                markets.append(market)
        
        return markets
    
    # =========================================================================
    # ORDER EXECUTION
    # Based on official py-clob-client order patterns
    # =========================================================================
    
    def get_orderbook(self, token_id: str) -> Optional[Any]:
        """
        Get orderbook for a token
        Based on official py-clob-client get_order_book
        """
        if not self.clob_client:
            return None
        
        try:
            return self.clob_client.get_order_book(token_id)
        except Exception as e:
            self.log_action(f"Error fetching orderbook: {e}")
            return None
    
    def get_midpoint_price(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token"""
        if not self.clob_client:
            return None
        
        try:
            result = self.clob_client.get_midpoint(token_id)
            return float(result.get("mid", 0))
        except Exception as e:
            self.log_action(f"Error fetching midpoint: {e}")
            return None
    
    def place_order(
        self,
        market_id: str,
        outcome: str,  # "YES" or "NO"
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float
    ) -> Dict[str, Any]:
        """
        Place a limit order on Polymarket
        Based on official py-clob-client order placement
        
        Args:
            market_id: Market ID or condition ID
            outcome: "YES" or "NO"
            side: "BUY" or "SELL"
            price: Limit price (0.01 to 0.99)
            size: Number of shares
            
        Returns:
            Order result with order ID or error
        """
        if not self.clob_client:
            return {"error": "CLOB client not initialized. Check API credentials."}
        
        if not OrderArgs:
            return {"error": "py-clob-client SDK not installed"}
        
        try:
            # Validate inputs
            outcome = outcome.upper()
            side = side.upper()
            
            if outcome not in ["YES", "NO"]:
                return {"error": f"Invalid outcome: {outcome}. Must be YES or NO."}
            
            if side not in ["BUY", "SELL"]:
                return {"error": f"Invalid side: {side}. Must be BUY or SELL."}
            
            if not 0.01 <= price <= 0.99:
                return {"error": f"Invalid price: {price}. Must be between 0.01 and 0.99."}
            
            if size <= 0:
                return {"error": f"Invalid size: {size}. Must be positive."}
            
            # Get token ID for the outcome
            token_id = self._get_token_id(market_id, outcome)
            if not token_id:
                return {"error": f"Could not find token ID for market {market_id} outcome {outcome}"}
            
            # Create order using official SDK pattern
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                fee_rate_bps=0  # Polymarket has 0 maker fees
            )
            
            # Create and post order
            signed_order = self.clob_client.create_order(order_args)
            result = self.clob_client.post_order(signed_order)
            
            self.log_action(f"✅ Order placed: {side} {size} {outcome} @ {price} for market {market_id}")
            
            return {
                "success": True,
                "order_id": result.get("orderID") or result.get("id"),
                "market_id": market_id,
                "outcome": outcome,
                "side": side,
                "price": price,
                "size": size,
                "token_id": token_id,
                "status": "open"
            }
            
        except Exception as e:
            self.log_action(f"❌ Order placement failed: {e}")
            return {"error": str(e)}
    
    def place_market_order(
        self,
        market_id: str,
        outcome: str,
        side: str,
        amount: float
    ) -> Dict[str, Any]:
        """
        Place a market order (FOK - Fill or Kill)
        Based on official py-clob-client market order pattern
        
        Args:
            market_id: Market ID
            outcome: "YES" or "NO"
            side: "BUY" or "SELL"
            amount: USDC amount to spend
            
        Returns:
            Order result
        """
        if not self.clob_client:
            return {"error": "CLOB client not initialized"}
        
        if not MarketOrderArgs or not OrderType:
            return {"error": "py-clob-client SDK not installed"}
        
        try:
            outcome = outcome.upper()
            side = side.upper()
            
            token_id = self._get_token_id(market_id, outcome)
            if not token_id:
                return {"error": f"Could not find token ID for market {market_id}"}
            
            # Create market order
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side
            )
            
            signed_order = self.clob_client.create_market_order(order_args)
            result = self.clob_client.post_order(signed_order, orderType=OrderType.FOK)
            
            self.log_action(f"✅ Market order executed: {side} ${amount} {outcome}")
            
            return {
                "success": True,
                "order_id": result.get("orderID") or result.get("id"),
                "market_id": market_id,
                "outcome": outcome,
                "side": side,
                "amount": amount,
                "status": "filled"
            }
            
        except Exception as e:
            self.log_action(f"❌ Market order failed: {e}")
            return {"error": str(e)}
    
    def _get_token_id(self, market_id: str, outcome: str) -> Optional[str]:
        """
        Get CLOB token ID for a market outcome
        """
        # First try to get from market data
        market = self.get_market_by_id(market_id)
        if market and market.clob_token_ids:
            # YES is typically index 0, NO is index 1
            idx = 0 if outcome.upper() == "YES" else 1
            if len(market.clob_token_ids) > idx:
                return market.clob_token_ids[idx]
        
        # Fallback: query directly
        market_data = self.gamma.get_market(market_id)
        if market_data:
            tokens = market_data.get("tokens", [])
            for token in tokens:
                if token.get("outcome", "").upper() == outcome.upper():
                    return token.get("token_id")
            
            # Try clobTokenIds
            clob_ids = market_data.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                clob_ids = json.loads(clob_ids)
            
            idx = 0 if outcome.upper() == "YES" else 1
            if len(clob_ids) > idx:
                return clob_ids[idx]
        
        return None
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order"""
        if not self.clob_client:
            return {"error": "CLOB client not initialized"}
        
        try:
            result = self.clob_client.cancel(order_id)
            self.log_action(f"✅ Order cancelled: {order_id}")
            return {"success": True, "order_id": order_id}
        except Exception as e:
            self.log_action(f"❌ Order cancellation failed: {e}")
            return {"error": str(e)}
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open orders"""
        if not self.clob_client:
            return {"error": "CLOB client not initialized"}
        
        try:
            result = self.clob_client.cancel_all()
            self.log_action("✅ All orders cancelled")
            return {"success": True, "result": result}
        except Exception as e:
            self.log_action(f"❌ Cancel all orders failed: {e}")
            return {"error": str(e)}
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        if not self.clob_client:
            return []
        
        try:
            orders = self.clob_client.get_orders()
            return orders if isinstance(orders, list) else []
        except Exception as e:
            self.log_action(f"Error fetching orders: {e}")
            return []
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        if not self.clob_client:
            return []
        
        try:
            trades = self.clob_client.get_trades()
            return trades if isinstance(trades, list) else []
        except Exception as e:
            self.log_action(f"Error fetching trades: {e}")
            return []
    
    # =========================================================================
    # POSITION TRACKING
    # =========================================================================
    
    def get_positions(self) -> List[PolymarketPosition]:
        """Get current positions - requires integration with subgraph or API"""
        # Note: Polymarket doesn't have a direct "positions" endpoint
        # Positions are tracked via trades and resolved outcomes
        # This would typically use the Polymarket subgraph
        
        positions = []
        
        # Get trades and aggregate positions
        trades = self.get_trades()
        position_map: Dict[str, Dict[str, float]] = {}
        
        for trade in trades:
            asset_id = trade.get("asset_id", "")
            side = trade.get("side", "")
            size = float(trade.get("size", 0))
            price = float(trade.get("price", 0))
            
            if asset_id not in position_map:
                position_map[asset_id] = {"shares": 0, "cost": 0}
            
            if side == "BUY":
                position_map[asset_id]["shares"] += size
                position_map[asset_id]["cost"] += size * price
            else:
                position_map[asset_id]["shares"] -= size
                position_map[asset_id]["cost"] -= size * price
        
        for asset_id, pos_data in position_map.items():
            if pos_data["shares"] > 0.001:  # Filter dust positions
                avg_price = pos_data["cost"] / pos_data["shares"] if pos_data["shares"] > 0 else 0
                
                positions.append(PolymarketPosition(
                    market_id=asset_id,
                    market_question="",  # Would need to fetch
                    outcome="",
                    shares=pos_data["shares"],
                    avg_price=avg_price,
                    current_price=0,  # Would need orderbook
                    unrealized_pnl=0,
                    realized_pnl=0,
                    token_id=asset_id
                ))
        
        return positions
    
    # =========================================================================
    # BONDING STRATEGY
    # High-probability bonding opportunities
    # =========================================================================
    
    def find_bonding_opportunities(
        self,
        min_probability: float = 0.95,
        max_days_to_resolution: int = 14,
        min_liquidity: float = 1000,
        min_annualized_return: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Find high-probability bonding opportunities
        
        Bonding strategy: Buy shares priced at 95%+ that resolve soon
        for consistent, low-risk returns.
        
        Args:
            min_probability: Minimum probability threshold (default 95%)
            max_days_to_resolution: Maximum days until resolution
            min_liquidity: Minimum market liquidity in USDC
            min_annualized_return: Minimum annualized return percentage
            
        Returns:
            List of bonding opportunities sorted by annualized return
        """
        markets = self.get_markets(active_only=True, limit=500)
        opportunities = []
        
        for market in markets:
            # Skip if not high probability
            if not market.is_high_probability:
                continue
            
            # Skip if too far from resolution
            if market.days_to_resolution > max_days_to_resolution:
                continue
            
            # Skip if too low liquidity
            if market.liquidity < min_liquidity:
                continue
            
            # Get best bonding side
            side, price = market.best_bonding_side
            
            # Calculate potential return
            days = max(1, market.days_to_resolution)
            
            # Profit per share if wins
            profit_per_share = 1.0 - price
            profit_pct = (profit_per_share / price) * 100
            
            # Annualized return
            annualized = profit_pct * (365 / days)
            
            # Skip if return too low
            if annualized < min_annualized_return:
                continue
            
            opportunities.append({
                "market_id": market.id,
                "question": market.question,
                "side": side,
                "price": price,
                "token_id": market.clob_token_ids[0 if side == "YES" else 1] if len(market.clob_token_ids) > 1 else "",
                "days_to_resolution": days,
                "profit_pct": round(profit_pct, 2),
                "annualized_return": round(annualized, 2),
                "liquidity": market.liquidity,
                "volume": market.volume,
                "category": market.category,
                "risk_level": "LOW" if price >= 0.98 else "MEDIUM" if price >= 0.95 else "HIGH"
            })
        
        # Sort by annualized return (highest first)
        opportunities.sort(key=lambda x: x["annualized_return"], reverse=True)
        
        self.log_action(f"Found {len(opportunities)} bonding opportunities")
        return opportunities
    
    # =========================================================================
    # ARBITRAGE DETECTION
    # =========================================================================
    
    def find_arbitrage_opportunities(
        self,
        other_platform_prices: Dict[str, float],
        min_spread: float = 0.03
    ) -> List[Dict[str, Any]]:
        """
        Find arbitrage opportunities between Polymarket and other platforms
        
        Args:
            other_platform_prices: Dict mapping market questions to prices
            min_spread: Minimum price spread to consider (default 3%)
            
        Returns:
            List of arbitrage opportunities
        """
        markets = self.get_markets(active_only=True)
        opportunities = []
        
        for market in markets:
            question_lower = market.question.lower()
            
            for other_question, other_price in other_platform_prices.items():
                if self._is_similar_market(question_lower, other_question.lower()):
                    poly_price = market.yes_price
                    spread = abs(poly_price - other_price)
                    
                    if spread >= min_spread:
                        opportunities.append({
                            "polymarket_question": market.question,
                            "polymarket_price": poly_price,
                            "other_platform_price": other_price,
                            "spread": round(spread, 4),
                            "spread_pct": round(spread * 100, 2),
                            "action": "BUY_POLYMARKET" if poly_price < other_price else "SELL_POLYMARKET",
                            "potential_profit_per_share": spread
                        })
        
        opportunities.sort(key=lambda x: x["spread"], reverse=True)
        return opportunities
    
    def _is_similar_market(self, question1: str, question2: str) -> bool:
        """Check if two market questions are about the same event"""
        words1 = set(question1.split())
        words2 = set(question2.split())
        
        stop_words = {"will", "the", "a", "an", "be", "is", "in", "on", "at", "to", "?"}
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
        """Get summary of Polymarket activity"""
        markets = self.get_markets(active_only=True, limit=500)
        
        total_volume = sum(m.volume for m in markets)
        total_liquidity = sum(m.liquidity for m in markets)
        
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
        
        # Balance info
        balance = self.get_balance()
        
        return {
            "total_markets": len(markets),
            "total_volume": total_volume,
            "total_liquidity": total_liquidity,
            "categories": categories,
            "bonding_opportunities": len(bonding),
            "top_bonding": bonding[:5] if bonding else [],
            "balance": balance,
            "sdk_available": POLYMARKET_SDK_AVAILABLE,
            "web3_available": WEB3_AVAILABLE,
            "clob_connected": self.clob_client is not None,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Polymarket API"""
        results = {
            "gamma_api": False,
            "clob_api": False,
            "web3": False,
            "wallet": None,
            "balance": None,
            "errors": []
        }
        
        # Test Gamma API
        try:
            markets = self.get_markets(limit=1)
            results["gamma_api"] = len(markets) > 0
            results["sample_market"] = markets[0].question if markets else None
        except Exception as e:
            results["errors"].append(f"Gamma API: {e}")
        
        # Test CLOB API
        if self.clob_client:
            try:
                ok = self.clob_client.get_ok()
                results["clob_api"] = ok is not None
            except Exception as e:
                results["errors"].append(f"CLOB API: {e}")
        
        # Test Web3
        if self.web3:
            try:
                results["web3"] = self.web3.is_connected()
                results["wallet"] = self.get_wallet_address()
                results["balance"] = self.get_balance()
            except Exception as e:
                results["errors"].append(f"Web3: {e}")
        
        results["status"] = "connected" if results["gamma_api"] else "error"
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_adapter_instance: Optional[PolymarketAdapter] = None

def get_polymarket_adapter() -> PolymarketAdapter:
    """Get global Polymarket adapter instance"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = PolymarketAdapter()
    return _adapter_instance


# =============================================================================
# MAIN - Test the adapter
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("POLYMARKET ADAPTER - Enhanced with Official Polymarket Code")
    print("="*70)
    
    adapter = PolymarketAdapter()
    
    print("\n=== Testing Connection ===")
    status = adapter.test_connection()
    print(f"Gamma API: {'✅' if status['gamma_api'] else '❌'}")
    print(f"CLOB API: {'✅' if status['clob_api'] else '❌'}")
    print(f"Web3: {'✅' if status['web3'] else '❌'}")
    print(f"Wallet: {status.get('wallet', 'Not configured')}")
    
    if status.get('balance'):
        print(f"USDC Balance: ${status['balance'].get('usdc', 0):.2f}")
    
    print("\n=== Market Summary ===")
    summary = adapter.get_market_summary()
    print(f"Total Markets: {summary['total_markets']}")
    print(f"Total Volume: ${summary['total_volume']:,.2f}")
    print(f"Total Liquidity: ${summary['total_liquidity']:,.2f}")
    print(f"Bonding Opportunities: {summary['bonding_opportunities']}")
    
    print("\n=== Top 5 Bonding Opportunities ===")
    for i, opp in enumerate(summary.get("top_bonding", [])[:5], 1):
        print(f"\n{i}. {opp['question'][:60]}...")
        print(f"   Side: {opp['side']} @ ${opp['price']:.2f}")
        print(f"   Days to Resolution: {opp['days_to_resolution']}")
        print(f"   Annualized Return: {opp['annualized_return']}%")
        print(f"   Risk Level: {opp['risk_level']}")
    
    print("\n=== Active Markets by Category ===")
    for cat, data in sorted(summary.get("categories", {}).items(), key=lambda x: x[1]["volume"], reverse=True)[:5]:
        print(f"  {cat}: {data['count']} markets, ${data['volume']:,.2f} volume")
    
    print("\n" + "="*70)
    print("Adapter Test Complete")
    print("="*70)

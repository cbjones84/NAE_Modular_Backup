"""
Tradier Broker Adapter

Full Tradier API integration with OAuth 2.0, REST API, and WebSocket streaming.
Supports equity, options, and multileg orders.
"""

import os
import re
import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import websocket
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from urllib.parse import urlencode
import base64
import sys

# Add parent directory to path for vault access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    from secure_vault import get_vault
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradierError(Exception):
    """Custom exception for Tradier API errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, endpoint: Optional[str] = None):
        """
        Initialize Tradier error
        
        Args:
            message: Error message
            status_code: HTTP status code (if available)
            endpoint: API endpoint that failed (if available)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.endpoint = endpoint
        self.timestamp = datetime.now().isoformat()
    
    def __str__(self) -> str:
        """Format error message consistently"""
        parts = [f"TradierError: {self.message}"]
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        if self.endpoint:
            parts.append(f"Endpoint: {self.endpoint}")
        parts.append(f"Time: {self.timestamp}")
        return " | ".join(parts)


class TradierOAuth:
    """Tradier OAuth 2.0 authentication handler with API key support - LIVE MODE"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, api_key: str = None, sandbox: bool = False):
        """
        Initialize Tradier OAuth or API key authentication
        
        Args:
            client_id: Tradier OAuth client ID (optional if using API key)
            client_secret: Tradier OAuth client secret (optional if using API key)
            api_key: Tradier API key for Bearer token auth (optional if using OAuth)
            sandbox: Use sandbox environment
        """
        self.client_id = client_id or os.getenv("TRADIER_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("TRADIER_CLIENT_SECRET")
        self.sandbox = sandbox
        
        # Try to get API key from vault first, then env var, then parameter
        if api_key:
            self.api_key = api_key
        else:
            # Initialize api_key to None first to avoid AttributeError
            self.api_key = None
            
            # Try vault first (most secure)
            if VAULT_AVAILABLE:
                try:
                    vault = get_vault()
                    self.api_key = vault.get_secret("tradier", "api_key")
                except Exception as e:
                    logger.debug(f"Could not access vault: {e}")
                    self.api_key = None
            
            # Fall back to environment variable
            if not self.api_key:
                self.api_key = os.getenv("TRADIER_API_KEY")
        
        # Determine authentication method
        self.use_api_key = bool(self.api_key)
        self.use_oauth = bool(self.client_id and self.client_secret)
        
        if not self.use_api_key and not self.use_oauth:
            logger.warning("No Tradier authentication credentials provided. Need either API key or OAuth credentials.")
        
        if self.sandbox:
            self.auth_url = "https://sandbox.tradier.com/v1/oauth/authorize"
            self.token_url = "https://sandbox.tradier.com/v1/oauth/accesstoken"
            self.api_base = "https://sandbox.tradier.com/v1"
        else:
            self.auth_url = "https://api.tradier.com/v1/oauth/authorize"
            self.token_url = "https://api.tradier.com/v1/oauth/accesstoken"
            self.api_base = "https://api.tradier.com/v1"
        
        # For API key auth, use API key as access token
        if self.use_api_key:
            self.access_token = self.api_key
            self.token_expires_at = None  # API keys don't expire
        else:
            self.access_token: Optional[str] = None
            self.token_expires_at: Optional[datetime] = None
        
        self.refresh_token: Optional[str] = None
        self.account_id: Optional[str] = None
    
    def get_authorization_url(self, redirect_uri: str, state: str = None) -> str:
        """
        Get authorization URL for OAuth flow
        
        Args:
            redirect_uri: Redirect URI after authorization
            state: Optional state parameter
        
        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri
        }
        
        if state:
            params["state"] = state
        
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token
        
        Args:
            code: Authorization code from redirect
            redirect_uri: Redirect URI used in authorization
        
        Returns:
            Token response
        """
        # Basic auth header
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri
        }
        
        response = requests.post(
            self.token_url,
            data=data,
            headers={
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )
        
        response.raise_for_status()
        token_data = response.json()
        
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        
        # Tokens expire in 24 hours (or as specified by Tradier)
        expires_in = token_data.get("expires_in", 86400)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        return token_data
    
    def refresh_access_token(self) -> bool:
        """
        Refresh access token using refresh token
        
        Returns:
            True if successful
        """
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False
        
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        
        try:
            response = requests.post(
                self.token_url,
                data=data,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data.get("access_token")
            if "refresh_token" in token_data:
                self.refresh_token = token_data.get("refresh_token")
            
            expires_in = token_data.get("expires_in", 86400)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info("Access token refreshed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return False
    
    def is_token_expired(self) -> bool:
        """Check if access token is expired or expiring soon"""
        # API keys don't expire
        if self.use_api_key:
            return False
        
        if not self.token_expires_at:
            return True
        
        # Consider expired if less than 5 minutes remaining
        return datetime.now() >= (self.token_expires_at - timedelta(minutes=5))
    
    def ensure_valid_token(self) -> bool:
        """Ensure access token is valid, refresh if needed"""
        # API keys don't need refresh
        if self.use_api_key:
            return bool(self.access_token)
        
        if not self.access_token or self.is_token_expired():
            if self.refresh_token:
                return self.refresh_access_token()
            else:
                logger.error("Token expired and no refresh token available")
                return False
        return True


class TradierRESTClient:
    """Tradier REST API client with retries and error handling"""
    
    def __init__(self, oauth: TradierOAuth):
        """
        Initialize REST client with retry strategy
        
        Args:
            oauth: TradierOAuth instance
        """
        self.oauth = oauth
        self.api_base = oauth.api_base
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth"""
        if not self.oauth.ensure_valid_token():
            raise TradierError("Invalid or expired access token", endpoint="authentication")
        
        # Use Bearer token authentication (works for both OAuth tokens and API keys)
        return {
            "Authorization": f"Bearer {self.oauth.access_token}",
            "Accept": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make API request with retries and error handling
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional request arguments
        
        Returns:
            Response JSON data
        
        Raises:
            Exception: On API errors (4xx/5xx)
        """
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=30,
                **kwargs
            )
            
            # Raise on HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for Tradier API errors in response
            if "errors" in data:
                error_msg = str(data["errors"])
                logger.error(f"Tradier API error: {error_msg}")
                raise TradierError(
                    f"Tradier API error: {error_msg}",
                    status_code=response.status_code,
                    endpoint=endpoint
                )
            
            return data
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            
            # CRITICAL: Capture response body for debugging 400/4xx errors
            response_body = ""
            if e.response is not None:
                try:
                    response_body = e.response.text
                    logger.error(f"Tradier API error response body: {response_body}")
                except Exception:
                    pass
            
            # Handle rate limiting (429)
            if status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                time.sleep(retry_after)
                # Retry once after waiting
                try:
                    response = self.session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        timeout=30,
                        **kwargs
                    )
                    response.raise_for_status()
                    data = response.json()
                    if "errors" in data:
                        raise TradierError(
                            f"Tradier API error: {data['errors']}",
                            status_code=response.status_code,
                            endpoint=endpoint
                        )
                    return data
                except Exception as retry_error:
                    raise TradierError(
                        f"Request failed after rate limit retry: {str(retry_error)}",
                        status_code=status_code,
                        endpoint=endpoint
                    ) from retry_error
            
            error_msg = f"HTTP {status_code} | Endpoint: {endpoint} | Body: {response_body[:500]} | Details: {str(e)}"
            logger.error(error_msg)
            raise TradierError(error_msg, status_code=status_code, endpoint=endpoint) from e
            
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout after 30s | Endpoint: {endpoint} | Details: {str(e)}"
            logger.error(error_msg)
            raise TradierError(error_msg, endpoint=endpoint) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed | Endpoint: {endpoint} | Details: {str(e)}"
            logger.error(error_msg)
            raise TradierError(error_msg, endpoint=endpoint) from e
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts"""
        # Try /user/profile first to get account information
        try:
            response = self.session.get(
                f"{self.api_base}/user/profile",
                headers=self._get_headers()
            )
            response.raise_for_status()
            profile_data = response.json()
            
            # Check if profile contains account info
            if "profile" in profile_data:
                profile = profile_data["profile"]
                if isinstance(profile, dict) and "account" in profile:
                    accounts = profile["account"]
                    return accounts if isinstance(accounts, list) else [accounts]
        except Exception as e:
            logger.debug(f"Failed to get accounts from /user/profile: {e}")
        
        # Fallback to /accounts endpoint
        try:
            response = self.session.get(
                f"{self.api_base}/accounts",
                headers=self._get_headers()
            )
            response.raise_for_status()
            accounts_data = response.json()
            
            # Handle different response formats
            if "accounts" in accounts_data:
                accounts = accounts_data["accounts"]
                if isinstance(accounts, dict) and "account" in accounts:
                    account_list = accounts["account"]
                    return account_list if isinstance(account_list, list) else [account_list]
                elif isinstance(accounts, list):
                    return accounts
            elif "account" in accounts_data:
                account = accounts_data["account"]
                return account if isinstance(account, list) else [account]
            
            return []
        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            # If both fail, try to extract account from error or return empty
            raise
    
    def get_account_details(self, account_id: str) -> Dict[str, Any]:
        """Get account details"""
        # Try /user/profile first (contains account details)
        try:
            response = self.session.get(
                f"{self.api_base}/user/profile",
                headers=self._get_headers()
            )
            response.raise_for_status()
            profile_data = response.json()
            
            if "profile" in profile_data:
                profile = profile_data["profile"]
                if isinstance(profile, dict):
                    # Check if account info is in profile
                    if "account" in profile:
                        accounts = profile["account"]
                        account_list = accounts if isinstance(accounts, list) else [accounts]
                        # Find matching account
                        for acc in account_list:
                            if isinstance(acc, dict):
                                acc_id = acc.get("account_number") or acc.get("id")
                                if acc_id == account_id:
                                    return acc
        except Exception as e:
            logger.debug(f"Failed to get account details from /user/profile: {e}")
        
        # Fallback to /accounts/{account_id} endpoint
        try:
            response = self.session.get(
                f"{self.api_base}/accounts/{account_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            account_data = response.json()
            return account_data.get("account", account_data)
        except TradierError:
            # Re-raise TradierError
            raise
        except Exception as e:
            logger.error(f"Failed to get account details from /accounts/{account_id}: {e}")
            # Raise exception instead of returning fake data
            raise TradierError(
                f"Failed to get account details: {str(e)}",
                endpoint=f"accounts/{account_id}"
            ) from e
    
    def get_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """Get account positions"""
        data = self._request("GET", f"accounts/{account_id}/positions")
        positions = data.get("positions", {})
        if isinstance(positions, dict) and "position" in positions:
            pos_list = positions["position"]
            return pos_list if isinstance(pos_list, list) else [pos_list]
        return []
    
    def get_orders(self, account_id: str) -> List[Dict[str, Any]]:
        """Get account orders"""
        data = self._request("GET", f"accounts/{account_id}/orders")
        orders = data.get("orders", {})
        if isinstance(orders, dict) and "order" in orders:
            order_list = orders["order"]
            return order_list if isinstance(order_list, list) else [order_list]
        return []
    
    def preview_order(
        self,
        account_id: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        duration: str = "day",
        price: Optional[float] = None,
        stop: Optional[float] = None,
        option_symbol: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Preview order (validate before submitting)
        
        Args:
            account_id: Account ID
            symbol: Symbol (equity) or option_symbol for options
            side: buy, buy_to_cover, sell, sell_short
            quantity: Quantity
            order_type: market, limit, stop, stop_limit
            duration: day, gtc, pre, post
            price: Limit price (if limit order)
            stop: Stop price (if stop order)
            option_symbol: Option symbol (for options)
            tag: Order tag
        
        Returns:
            Preview response with warnings and cost
        """
        data = {
            "class": "equity" if not option_symbol else "option",
            "symbol": symbol,  # Always the underlying symbol
            "side": side,
            "quantity": str(quantity),
            "type": order_type,
            "duration": duration,
            "preview": "true"
        }
        
        # For options: include option_symbol as separate field (Tradier requires both)
        if option_symbol:
            data["option_symbol"] = option_symbol
        
        # Only include price for limit/stop_limit orders
        if price and order_type in ("limit", "stop_limit"):
            data["price"] = str(price)
        if stop and order_type in ("stop", "stop_limit"):
            data["stop"] = str(stop)
        
        # Use centralized request method with retries
        return self._request("POST", f"accounts/{account_id}/orders", data=data)
    
    def submit_order(
        self,
        account_id: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        duration: str = "day",
        price: Optional[float] = None,
        stop: Optional[float] = None,
        option_symbol: Optional[str] = None,
        tag: Optional[str] = None,
        preview: bool = False
    ) -> Dict[str, Any]:
        """
        Submit order
        
        Args:
            account_id: Account ID
            symbol: Symbol (equity) or option_symbol for options
            side: buy, buy_to_cover, sell, sell_short
            quantity: Quantity
            order_type: market, limit, stop, stop_limit
            duration: day, gtc, pre, post
            price: Limit price (if limit order)
            stop: Stop price (if stop order)
            option_symbol: Option symbol (for options)
            tag: Order tag
            preview: Preview order first
        
        Returns:
            Order response
        """
        # Preview first if requested
        if preview:
            preview_result = self.preview_order(
                account_id, symbol, side, quantity, order_type,
                duration, price, stop, option_symbol, tag
            )
            
            # Check for warnings
            if "warnings" in preview_result:
                logger.warning(f"Order preview warnings: {preview_result['warnings']}")
        
        data = {
            "class": "equity" if not option_symbol else "option",
            "symbol": symbol,  # Always the underlying symbol
            "side": side,
            "quantity": str(quantity),  # Tradier expects string for quantity
            "type": order_type,
            "duration": duration
        }
        
        # For options: include option_symbol as separate field (Tradier requires both)
        if option_symbol:
            data["option_symbol"] = option_symbol
        
        # CRITICAL FIX: Only include price for limit/stop_limit orders
        # Market orders MUST NOT have a price field (causes 400 error)
        if price and order_type in ("limit", "stop_limit"):
            data["price"] = str(price)
        if stop and order_type in ("stop", "stop_limit"):
            data["stop"] = str(stop)
        
        logger.info(f"Submitting order to Tradier: {data}")
        
        # Use centralized request method with retries
        return self._request("POST", f"accounts/{account_id}/orders", data=data)
    
    def submit_multileg_order(
        self,
        account_id: str,
        legs: List[Dict[str, Any]],
        order_type: str = "market",
        duration: str = "day",
        price: Optional[float] = None,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit multileg (combo) order
        
        Args:
            account_id: Account ID
            legs: List of leg dictionaries with symbol, side, quantity
            order_type: market, limit, stop, stop_limit
            duration: day, gtc
            price: Limit price
            tag: Order tag
        
        Returns:
            Order response
        """
        data = {
            "class": "multileg",
            "type": order_type,
            "duration": duration,
            "leg": legs
        }
        
        if price:
            data["price"] = price
        if tag:
            data["tag"] = tag
        
        # Use centralized request method with retries
        return self._request("POST", f"accounts/{account_id}/orders", data=data)
    
    def get_order_status(self, account_id: str, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        data = self._request("GET", f"accounts/{account_id}/orders/{order_id}")
        return data.get("order", {})
    
    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        return self._request("DELETE", f"accounts/{account_id}/orders/{order_id}")


class TradierWebSocketClient:
    """Tradier WebSocket streaming client"""
    
    def __init__(self, oauth: TradierOAuth, account_id: str):
        """
        Initialize WebSocket client
        
        Args:
            oauth: TradierOAuth instance
            account_id: Account ID for streaming
        """
        self.oauth = oauth
        self.account_id = account_id
        self.sandbox = oauth.sandbox
        
        if self.sandbox:
            self.ws_url = "wss://stream.sandbox.tradier.com/v1/markets/events"
        else:
            self.ws_url = "wss://stream.tradier.com/v1/markets/events"
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        
        # Event handlers
        self.on_message: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_close: Optional[Callable] = None
    
    def connect(self):
        """Connect to WebSocket"""
        if self.connected:
            logger.warning("WebSocket already connected")
            return
        
        if not self.oauth.ensure_valid_token():
            raise Exception("Invalid or expired access token")
        
        def on_open(ws):
            logger.info("Tradier WebSocket connected")
            self.connected = True
            self.reconnect_attempts = 0
            
            # Subscribe to account events
            subscribe_msg = {
                "sessionid": self.oauth.access_token,
                "filter": f"account_{self.account_id}"
            }
            ws.send(json.dumps(subscribe_msg))
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if self.on_message:
                    self.on_message(data)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.connected = False
            if self.on_error:
                self.on_error(error)
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning("WebSocket closed")
            self.connected = False
            if self.on_close:
                self.on_close(close_status_code, close_msg)
            
            # Attempt reconnection
            self._attempt_reconnect()
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            header=[f"Authorization: Bearer {self.oauth.access_token}"]
        )
        
        # Start WebSocket in separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()
    
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
        self.connected = False
    
    def _attempt_reconnect(self):
        """Attempt to reconnect WebSocket"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        time.sleep(self.reconnect_delay)
        
        try:
            # Refresh token if needed
            self.oauth.ensure_valid_token()
            self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")


class TradierBrokerAdapter:
    """
    Tradier broker adapter for NAE execution - LIVE MODE ONLY
    
    Handles OAuth, REST API, WebSocket streaming, and order execution.
    This is the ONLY broker for NAE - Alpaca and IBKR have been removed.
    """
    
    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        api_key: str = None,
        account_id: str = None,
        sandbox: bool = False  # LIVE MODE by default
    ):
        """
        Initialize Tradier adapter
        
        Args:
            client_id: Tradier OAuth client ID (optional if using API key)
            client_secret: Tradier OAuth client secret (optional if using API key)
            api_key: Tradier API key for Bearer token auth (optional if using OAuth)
            account_id: Tradier account ID
            sandbox: Use sandbox environment
        """
        self.account_id = account_id or os.getenv("TRADIER_ACCOUNT_ID")
        self.sandbox = sandbox
        
        # Initialize OAuth (supports both OAuth and API key auth)
        self.oauth = TradierOAuth(client_id, client_secret, api_key, sandbox)
        
        # Initialize REST client
        self.rest_client = TradierRESTClient(self.oauth)
        
        # Initialize WebSocket client
        self.ws_client = TradierWebSocketClient(self.oauth, self.account_id)
        
        # Setup WebSocket event handlers
        self.ws_client.on_message = self._handle_ws_message
        self.ws_client.on_error = self._handle_ws_error
        self.ws_client.on_close = self._handle_ws_close
        
        # Order tracking
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Tradier adapter initialized (sandbox={sandbox})")
    
    def authenticate(self):
        """Authenticate with Tradier"""
        if not self.oauth.access_token:
            if self.oauth.use_api_key:
                logger.error("No API key available. Check vault or environment variable TRADIER_API_KEY.")
            else:
                logger.error("No access token. Use OAuth flow to obtain token or provide API key.")
            return False
        
        return self.oauth.ensure_valid_token()
    
    def connect_streaming(self):
        """Connect to WebSocket streaming"""
        try:
            self.ws_client.connect()
            logger.info("Tradier WebSocket streaming connected")
            return True
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            return False
    
    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit order to Tradier
        
        Args:
            order: Order details {
                "symbol": str,
                "side": "buy" | "sell" | "buy_to_cover" | "sell_short",
                "quantity": int,
                "order_type": "market" | "limit" | "stop" | "stop_limit",
                "duration": "day" | "gtc" | "pre" | "post",
                "price": float (optional),
                "stop": float (optional),
                "option_symbol": str (optional),
                "tag": str (optional),
                "preview": bool (optional)
            }
        
        Returns:
            Order submission result
        """
        try:
            # Ensure authenticated
            if not self.authenticate():
                return {
                    "status": "error",
                    "error": "Authentication failed",
                    "broker": "tradier"
                }
            
            # Submit order
            result = self.rest_client.submit_order(
                account_id=self.account_id,
                symbol=order.get("symbol", ""),
                side=order.get("side", "buy"),
                quantity=order.get("quantity", 0),
                order_type=order.get("order_type", "market"),
                duration=order.get("duration", "day"),
                price=order.get("price"),
                stop=order.get("stop"),
                option_symbol=order.get("option_symbol"),
                tag=order.get("tag"),
                preview=order.get("preview", False)
            )
            
            # Extract order ID
            order_id = None
            if "order" in result:
                order_data = result["order"]
                if isinstance(order_data, dict):
                    order_id = order_data.get("id")
                elif isinstance(order_data, list) and len(order_data) > 0:
                    order_id = order_data[0].get("id")
            
            # Track order
            if order_id:
                self.pending_orders[order_id] = {
                    "order": order,
                    "order_id": order_id,
                    "submitted_at": datetime.now().isoformat(),
                    "status": "submitted"
                }
            
            return {
                "status": "submitted",
                "order_id": order_id,
                "broker": "tradier",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error submitting order to Tradier: {e}")
            return {
                "status": "error",
                "error": str(e),
                "broker": "tradier"
            }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            if not self.authenticate():
                return []
            
            return self.rest_client.get_positions(self.account_id)
        except Exception as e:
            logger.error(f"Error getting positions from Tradier: {e}")
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not self.authenticate():
                return {}
            
            return self.rest_client.get_account_details(self.account_id)
        except Exception as e:
            logger.error(f"Error getting account info from Tradier: {e}")
            return {}
    
    def _handle_ws_message(self, data: Dict[str, Any]):
        """Handle WebSocket message"""
        # Process account events (fills, order status)
        if "event" in data:
            event_type = data.get("event")
            
            if event_type == "order":
                # Order status update
                order_data = data.get("order", {})
                order_id = order_data.get("id")
                
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]["status"] = order_data.get("status")
                    self.pending_orders[order_id]["last_update"] = datetime.now().isoformat()
            
            elif event_type == "fill":
                # Fill event
                fill_data = data.get("fill", {})
                order_id = fill_data.get("order_id")
                
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]["fills"] = self.pending_orders[order_id].get("fills", [])
                    self.pending_orders[order_id]["fills"].append(fill_data)
    
    def _handle_ws_error(self, error):
        """Handle WebSocket error"""
        logger.error(f"Tradier WebSocket error: {error}")
    
    def _handle_ws_close(self, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.warning(f"Tradier WebSocket closed: {close_status_code} - {close_msg}")
    
    def get_account_balance(self) -> float:
        """
        Get account balance (total equity).
        
        Returns:
            Account balance as float, or 0.0 if unavailable
        """
        try:
            account_info = self.get_account_info()
            if isinstance(account_info, dict):
                # Try common balance fields
                balance = account_info.get("total_equity") or \
                         account_info.get("equity") or \
                         account_info.get("cash") or \
                         account_info.get("total_cash") or \
                         account_info.get("account_value")
                if balance:
                    return float(balance)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_buying_power(self) -> float:
        """
        Get buying power (available cash for trading).
        
        Returns:
            Buying power as float, or 0.0 if unavailable
        """
        try:
            account_info = self.get_account_info()
            if isinstance(account_info, dict):
                buying_power = account_info.get("buying_power") or \
                              account_info.get("day_trading_buying_power") or \
                              account_info.get("cash") or \
                              account_info.get("total_cash")
                if buying_power:
                    return float(buying_power)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return 0.0
    
    def get_settled_cash(self) -> Optional[float]:
        """
        Get settled cash available (cash that has cleared settlement).
        
        For Tradier cash accounts, this is typically the cash balance
        minus any unsettled funds from recent trades.
        
        Returns:
            Settled cash as float, or None if unavailable
        """
        try:
            account_info = self.get_account_info()
            if isinstance(account_info, dict):
                # Tradier may provide settled_cash field
                settled = account_info.get("settled_cash") or \
                         account_info.get("cash") or \
                         account_info.get("total_cash")
                if settled:
                    return float(settled)
            return None
        except Exception as e:
            logger.error(f"Error getting settled cash: {e}")
            return None
    
    def get_unsettled_cash(self) -> Optional[float]:
        """
        Get unsettled cash (funds pending settlement).
        
        Returns:
            Unsettled cash as float, or None if unavailable
        """
        try:
            account_info = self.get_account_info()
            if isinstance(account_info, dict):
                unsettled = account_info.get("unsettled_cash") or \
                           account_info.get("pending_cash")
                if unsettled:
                    return float(unsettled)
            return None
        except Exception as e:
            logger.error(f"Error getting unsettled cash: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        return {
            "broker": "tradier",
            "sandbox": self.sandbox,
            "account_id": self.account_id,
            "authenticated": self.oauth.access_token is not None,
            "token_expired": self.oauth.is_token_expired() if self.oauth.access_token else True,
            "websocket_connected": self.ws_client.connected,
            "pending_orders": len(self.pending_orders)
        }
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get quote for a symbol
        
        Args:
            symbol: Stock/ETF symbol
        
        Returns:
            Quote data dict or None
        """
        try:
            # Use REST client to get quote
            data = self.rest_client._request("GET", f"markets/quotes?symbols={symbol}")
            return data
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_option_expirations(self, symbol: str) -> List[str]:
        """
        Get available option expiration dates for a symbol.
        
        Args:
            symbol: Underlying stock/ETF symbol
        
        Returns:
            List of expiration dates (YYYY-MM-DD format)
        """
        try:
            data = self.rest_client._request("GET", f"markets/options/expirations?symbol={symbol}")
            expirations = data.get("expirations", {})
            if isinstance(expirations, dict):
                date_list = expirations.get("date", [])
                return date_list if isinstance(date_list, list) else [date_list]
            return []
        except Exception as e:
            logger.error(f"Error getting option expirations for {symbol}: {e}")
            return []
    
    def get_option_chain(self, symbol: str, expiration: str, option_type: str = None) -> List[Dict[str, Any]]:
        """
        Get option chain for a symbol and expiration.
        
        Args:
            symbol: Underlying stock/ETF symbol
            expiration: Expiration date (YYYY-MM-DD)
            option_type: 'call', 'put', or None for both
        
        Returns:
            List of option contracts with greeks, bid/ask, etc.
        """
        try:
            url = f"markets/options/chains?symbol={symbol}&expiration={expiration}&greeks=true"
            if option_type:
                url += f"&option_type={option_type}"
            
            data = self.rest_client._request("GET", url)
            options = data.get("options", {})
            if isinstance(options, dict):
                option_list = options.get("option", [])
                return option_list if isinstance(option_list, list) else [option_list]
            return []
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol} exp {expiration}: {e}")
            return []
    
    def get_option_strikes(self, symbol: str, expiration: str) -> List[float]:
        """
        Get available strike prices for a symbol and expiration.
        
        Args:
            symbol: Underlying stock/ETF symbol
            expiration: Expiration date (YYYY-MM-DD)
        
        Returns:
            List of strike prices
        """
        try:
            data = self.rest_client._request(
                "GET", f"markets/options/strikes?symbol={symbol}&expiration={expiration}"
            )
            strikes = data.get("strikes", {})
            if isinstance(strikes, dict):
                strike_list = strikes.get("strike", [])
                return strike_list if isinstance(strike_list, list) else [strike_list]
            return []
        except Exception as e:
            logger.error(f"Error getting option strikes for {symbol}: {e}")
            return []
    
    def get_balances(self) -> Optional[Dict[str, Any]]:
        """
        Get account balances
        
        Returns:
            Balances dict or None
        """
        try:
            data = self.rest_client._request("GET", f"accounts/{self.account_id}/balances")
            return data
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
            return None


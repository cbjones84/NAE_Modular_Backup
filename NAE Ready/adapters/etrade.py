# NAE/adapters/etrade.py
"""
E*TRADE Broker Adapter
Implements E*TRADE API with OAuth 1.0a authentication
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from urllib.parse import quote

from .base import BrokerAdapter

try:
    from requests_oauthlib import OAuth1Session
    from agents.etrade_oauth import ETradeOAuth
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    print("[Warning] requests-oauthlib not available. Install: pip install requests-oauthlib oauthlib")

class EtradeAdapter(BrokerAdapter):
    """E*TRADE broker adapter with OAuth 1.0a"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize E*TRADE adapter
        
        Args:
            config: Dict with:
                - consumer_key: E*TRADE consumer key (or use ETRADE_CONSUMER_KEY env)
                - consumer_secret: E*TRADE consumer secret (or use ETRADE_CONSUMER_SECRET env)
                - sandbox: bool, use sandbox (default: True)
                - token_file: Path to saved tokens file (optional)
        """
        self.config = config
        
        # Get credentials
        self.consumer_key = config.get("consumer_key") or os.environ.get("ETRADE_CONSUMER_KEY")
        self.consumer_secret = config.get("consumer_secret") or os.environ.get("ETRADE_CONSUMER_SECRET")
        self.sandbox = config.get("sandbox", True)
        self.token_file = config.get("token_file", f"config/etrade_tokens_{'sandbox' if self.sandbox else 'prod'}.json")
        
        # Base URL
        self.base_url = "https://apisb.etrade.com" if self.sandbox else "https://api.etrade.com"
        
        # OAuth handler
        self.oauth = None
        self.oauth_session = None
        
        # Rate limiting
        self.rate_limit_remaining = 120  # E*TRADE: 120 calls/minute
        self.last_request_time = 0
        
        if not OAUTH_AVAILABLE:
            raise ImportError("requests-oauthlib is required for E*TRADE adapter")
        
        if not self.consumer_key or not self.consumer_secret:
            raise ValueError("E*TRADE consumer_key and consumer_secret are required")
        
        # Initialize OAuth
        self.oauth = ETradeOAuth(self.consumer_key, self.consumer_secret, self.sandbox)
        
        # Try to load existing tokens
        if not self.oauth.load_tokens(self.token_file):
            print(f"⚠️  No saved OAuth tokens. Complete OAuth flow first:")
            print(f"   python3 scripts/setup_etrade_oauth.py {'--sandbox' if self.sandbox else '--prod'}")
    
    def name(self) -> str:
        return "etrade"
    
    def auth(self) -> bool:
        """Authenticate with E*TRADE"""
        try:
            # Check if we have access tokens
            if not self.oauth.access_token or not self.oauth.access_token_secret:
                # Try to load from file
                if not self.oauth.load_tokens(self.token_file):
                    return False
            
            # Create authenticated session
            if not self.oauth.oauth_session:
                self.oauth_session = self.oauth.create_authenticated_session()
            
            if not self.oauth_session:
                return False
            
            # Test authentication with a simple API call
            try:
                url = f"{self.base_url}/v1/accounts/list"
                response = self.oauth_session.get(url, timeout=10)
                return response.status_code == 200
            except Exception:
                return False
                
        except Exception as e:
            print(f"E*TRADE auth error: {e}")
            return False
    
    def _rate_limit_check(self):
        """Enforce rate limiting (120 calls/minute)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / 120.0  # 0.5 seconds between calls
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _ensure_authenticated(self) -> bool:
        """Ensure we have an authenticated session"""
        if not self.auth():
            return False
        return self.oauth_session is not None
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        if not self._ensure_authenticated():
            return {}
        
        self._rate_limit_check()
        
        try:
            url = f"{self.base_url}/v1/accounts/list"
            response = self.oauth_session.get(url, timeout=10)
            
            if response.status_code != 200:
                return {}
            
            data = response.json()
            accounts = data.get('AccountListResponse', {}).get('Accounts', {}).get('Account', [])
            
            if not accounts:
                return {}
            
            # Get first account's balance
            account_id_key = accounts[0].get('accountId', {}).get('accountIdKey', '')
            if account_id_key:
                return self.get_account_balance(account_id_key)
            
            return {}
            
        except Exception as e:
            print(f"Error getting E*TRADE account: {e}")
            return {}
    
    def get_account_balance(self, account_id_key: str) -> Dict[str, Any]:
        """Get balance for specific account"""
        if not self._ensure_authenticated():
            return {}
        
        self._rate_limit_check()
        
        try:
            url = f"{self.base_url}/v1/accounts/{account_id_key}/balance"
            response = self.oauth_session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                balance_response = data.get('BalanceResponse', {})
                
                return {
                    "account_id": account_id_key,
                    "account_type": balance_response.get('accountType', ''),
                    "cash": balance_response.get('Cash', {}).get('cashAvailable', 0.0),
                    "buying_power": balance_response.get('Computed', {}).get('RealTimeValues', {}).get('totalAccountValue', 0.0),
                    "market_value": balance_response.get('Computed', {}).get('RealTimeValues', {}).get('netValue', 0.0),
                    "day_trading_buying_power": balance_response.get('Computed', {}).get('dayTradingBuyingPower', 0.0)
                }
            
            return {}
            
        except Exception as e:
            print(f"Error getting E*TRADE balance: {e}")
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if not self._ensure_authenticated():
            return []
        
        self._rate_limit_check()
        
        try:
            # Get account list first
            accounts_url = f"{self.base_url}/v1/accounts/list"
            accounts_response = self.oauth_session.get(accounts_url, timeout=10)
            
            if accounts_response.status_code != 200:
                return []
            
            accounts_data = accounts_response.json()
            accounts = accounts_data.get('AccountListResponse', {}).get('Accounts', {}).get('Account', [])
            
            all_positions = []
            
            for account in accounts:
                account_id_key = account.get('accountId', {}).get('accountIdKey', '')
                if not account_id_key:
                    continue
                
                # Get positions for this account
                positions_url = f"{self.base_url}/v1/accounts/{account_id_key}/positions"
                positions_response = self.oauth_session.get(positions_url, timeout=10)
                
                if positions_response.status_code == 200:
                    positions_data = positions_response.json()
                    position_list = positions_data.get('PortfolioResponse', {}).get('AccountPortfolio', [])
                    
                    for account_portfolio in position_list:
                        for position in account_portfolio.get('Position', []):
                            all_positions.append({
                                "symbol": position.get('symbol', {}).get('symbol', ''),
                                "quantity": position.get('quantity', 0),
                                "cost_basis": position.get('costBasis', 0.0),
                                "market_value": position.get('marketValue', 0.0),
                                "gain_loss": position.get('gainLoss', 0.0),
                                "days_gain_loss": position.get('daysGainLoss', 0.0),
                                "account_id": account_id_key
                            })
            
            return all_positions
            
        except Exception as e:
            print(f"Error getting E*TRADE positions: {e}")
            return []
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote for symbol"""
        if not self._ensure_authenticated():
            return {}
        
        self._rate_limit_check()
        
        try:
            # E*TRADE quote endpoint
            url = f"{self.base_url}/v1/market/quote/{symbol}"
            params = {"detailFlag": "ALL"}
            
            response = self.oauth_session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                quote_response = data.get('QuoteResponse', {})
                quote_data = quote_response.get('QuoteData', [])[0] if quote_response.get('QuoteData') else {}
                
                if quote_data:
                    all_quotes = quote_data.get('All', {})
                    return {
                        "symbol": symbol,
                        "bid": all_quotes.get('bid', 0.0),
                        "ask": all_quotes.get('ask', 0.0),
                        "last": all_quotes.get('lastTrade', 0.0),
                        "volume": all_quotes.get('totalVolume', 0),
                        "high": all_quotes.get('high', 0.0),
                        "low": all_quotes.get('low', 0.0),
                        "open": all_quotes.get('open', 0.0)
                    }
            
            return {}
            
        except Exception as e:
            print(f"Error getting E*TRADE quote: {e}")
            return {}
    
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order"""
        if not self._ensure_authenticated():
            return {
                "error": "Not authenticated",
                "status": "rejected"
            }
        
        self._rate_limit_check()
        
        try:
            account_id_key = order.get('account_id_key') or order.get('account_id')
            if not account_id_key:
                return {
                    "error": "account_id_key is required",
                    "status": "rejected"
                }
            
            # Build E*TRADE order structure
            symbol = order.get('symbol', '')
            quantity = order.get('quantity', 0)
            side = order.get('side', 'buy').upper()
            order_type = order.get('type', 'market').upper()
            price = order.get('price', 0.0)
            
            etrade_order = {
                "PlaceOrderRequest": {
                    "orderType": order_type,
                    "clientOrderId": f"NAE_{int(time.time())}",
                    "Order": [{
                        "Instrument": [{
                            "Product": {
                                "securityType": "EQ",
                                "symbol": symbol
                            },
                            "orderAction": side,
                            "quantityType": "QUANTITY",
                            "quantity": quantity
                        }],
                        "orderTerm": "GOOD_FOR_DAY",
                        "priceType": order_type
                    }]
                }
            }
            
            # Add limit price if limit order
            if order_type == "LIMIT":
                etrade_order["PlaceOrderRequest"]["Order"][0]["limitPrice"] = price
            
            url = f"{self.base_url}/v1/accounts/{account_id_key}/orders/place"
            response = self.oauth_session.post(
                url,
                json=etrade_order,
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                order_response = result.get('PlaceOrderResponse', {})
                order_ids = order_response.get('OrderIds', {}).get('OrderId', [])
                
                return {
                    "order_id": order_ids[0].get('orderId', '') if order_ids else f"etrade_{int(time.time())}",
                    "status": "submitted",
                    "broker": "etrade",
                    "sandbox": self.sandbox,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "client_order_id": order_response.get('clientOrderId', '')
                }
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get('Error', {}).get('message', error_msg)
                except:
                    pass
                
                return {
                    "error": f"Order submission failed: {error_msg}",
                    "status": "rejected",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "error": f"Order submission error: {e}",
                "status": "rejected"
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not self._ensure_authenticated():
            return {
                "error": "Not authenticated",
                "status": "failed"
            }
        
        self._rate_limit_check()
        
        try:
            # Note: E*TRADE cancel endpoint requires account_id_key
            # This is a simplified version - may need account_id_key in order_id or separate param
            return {
                "error": "Cancel requires account_id_key. Use get_order_status then cancel.",
                "status": "failed"
            }
            
        except Exception as e:
            return {
                "error": f"Cancel error: {e}",
                "status": "failed"
            }
    
    def get_order_status(self, account_id_key: str, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if not self._ensure_authenticated():
            return {}
        
        self._rate_limit_check()
        
        try:
            url = f"{self.base_url}/v1/accounts/{account_id_key}/orders/{order_id}"
            response = self.oauth_session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            print(f"Error getting E*TRADE order status: {e}")
            return {}



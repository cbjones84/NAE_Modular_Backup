# NAE/agents/etrade_oauth.py
"""
E*Trade OAuth 1.0a Implementation
Handles OAuth authentication flow for E*Trade API
"""

import os
import time
import json
import hashlib
import requests
from typing import Dict, Any, Optional, Tuple
from urllib.parse import parse_qsl, urlencode
from requests_oauthlib import OAuth1Session
from requests.auth import HTTPBasicAuth

class ETradeOAuth:
    """E*Trade OAuth 1.0a authentication handler"""
    
    def __init__(self, consumer_key: str, consumer_secret: str, sandbox: bool = True):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.sandbox = sandbox
        self.base_url = "https://apisb.etrade.com" if sandbox else "https://api.etrade.com"
        
        # OAuth tokens
        self.request_token = None
        self.request_token_secret = None
        self.access_token = None
        self.access_token_secret = None
        self.oauth_verifier = None
        
        # OAuth session
        self.oauth_session = None
        
    def start_oauth(self) -> Dict[str, Any]:
        """
        Step 1: Start OAuth flow (get request token)
        Improved version based on E*TRADE OAuth 1.0a best practices
        
        Returns: Dict with authorize_url, resource_owner_key, resource_owner_secret
        """
        try:
            request_token_url = f"{self.base_url}/oauth/request_token"
            
            # Create OAuth1 session with callback
            # E*TRADE sandbox uses 'oob' (out-of-band) for callback
            callback_uri = 'oob'  # Out-of-band for manual entry
            if not self.sandbox:
                # Production might use a callback URL
                callback_uri = os.environ.get("ETRADE_CALLBACK_URI", "oob")
            
            oauth = OAuth1Session(
                self.consumer_key,
                client_secret=self.consumer_secret,
                callback_uri=callback_uri,
                signature_type='AUTH_HEADER',
                signature_method='HMAC-SHA1'
            )
            
            # Fetch request token
            fetch_response = oauth.fetch_request_token(request_token_url)
            
            resource_owner_key = fetch_response.get('oauth_token')
            resource_owner_secret = fetch_response.get('oauth_token_secret')
            
            if not resource_owner_key or not resource_owner_secret:
                raise ValueError("Failed to obtain request token")
            
            # Store for later use
            self.request_token = resource_owner_key
            self.request_token_secret = resource_owner_secret
            
            # Build authorization URL
            authorize_url = oauth.authorization_url(f"{self.base_url}/oauth/authorize")
            
            return {
                "authorize_url": authorize_url,
                "resource_owner_key": resource_owner_key,
                "resource_owner_secret": resource_owner_secret
            }
            
        except Exception as e:
            print(f"Error starting OAuth: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "authorize_url": None,
                "resource_owner_key": None,
                "resource_owner_secret": None
            }
    
    def get_request_token(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Step 1: Get request token (legacy method, use start_oauth() for new code)
        Returns: (request_token, request_token_secret, authorization_url)
        """
        result = self.start_oauth()
        return (
            result.get("resource_owner_key"),
            result.get("resource_owner_secret"),
            result.get("authorize_url")
        )
    
    def finish_oauth(self, resource_owner_key: str, resource_owner_secret: str, 
                     oauth_verifier: str) -> Dict[str, Any]:
        """
        Step 2: Finish OAuth flow (exchange request token for access token)
        Improved version based on E*TRADE OAuth 1.0a best practices
        
        Args:
            resource_owner_key: Request token (from start_oauth)
            resource_owner_secret: Request token secret (from start_oauth)
            oauth_verifier: Verification code from user authorization
        
        Returns: Dict with oauth_token (access token) and oauth_token_secret
        """
        try:
            access_token_url = f"{self.base_url}/oauth/access_token"
            
            # Create OAuth1 session with request token and verifier
            oauth = OAuth1Session(
                self.consumer_key,
                client_secret=self.consumer_secret,
                resource_owner_key=resource_owner_key,
                resource_owner_secret=resource_owner_secret,
                verifier=oauth_verifier,
                signature_type='AUTH_HEADER',
                signature_method='HMAC-SHA1'
            )
            
            # Fetch access token
            tokens = oauth.fetch_access_token(access_token_url)
            
            # Store access tokens
            self.access_token = tokens.get('oauth_token')
            self.access_token_secret = tokens.get('oauth_token_secret')
            self.oauth_verifier = oauth_verifier
            
            # Also update request tokens if they match
            if resource_owner_key == self.request_token:
                # Already stored
                pass
            else:
                self.request_token = resource_owner_key
                self.request_token_secret = resource_owner_secret
            
            return {
                "oauth_token": self.access_token,
                "oauth_token_secret": self.access_token_secret
            }
            
        except Exception as e:
            print(f"Error finishing OAuth: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "oauth_token": None,
                "oauth_token_secret": None
            }
    
    def get_access_token(self, oauth_verifier: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Step 2: Exchange request token for access token (legacy method, use finish_oauth() for new code)
        Args:
            oauth_verifier: Verifier code from user authorization
        Returns: (access_token, access_token_secret)
        """
        if not self.request_token or not self.request_token_secret:
            raise ValueError("Request token not obtained. Call start_oauth() or get_request_token() first.")
        
        result = self.finish_oauth(self.request_token, self.request_token_secret, oauth_verifier)
        return result.get("oauth_token"), result.get("oauth_token_secret")
    
    def create_authenticated_session(self) -> Optional[OAuth1Session]:
        """
        Create authenticated OAuth session for API calls
        Returns: OAuth1Session ready for API calls
        """
        try:
            if not self.access_token or not self.access_token_secret:
                raise ValueError("Access token not obtained. Complete OAuth flow first.")
            
            # Create authenticated session
            oauth = OAuth1Session(
                self.consumer_key,
                client_secret=self.consumer_secret,
                resource_owner_key=self.access_token,
                resource_owner_secret=self.access_token_secret,
                signature_type='AUTH_HEADER',
                signature_method='HMAC-SHA1'
            )
            
            self.oauth_session = oauth
            return oauth
            
        except Exception as e:
            print(f"Error creating authenticated session: {e}")
            return None
    
    def save_tokens(self, filepath: str = "config/etrade_tokens.json"):
        """Save access tokens to file (encrypted in production)"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            tokens = {
                'access_token': self.access_token,
                'access_token_secret': self.access_token_secret,
                'sandbox': self.sandbox,
                'timestamp': time.time()
            }
            with open(filepath, 'w') as f:
                json.dump(tokens, f)
            print(f"Tokens saved to {filepath}")
        except Exception as e:
            print(f"Error saving tokens: {e}")
    
    def load_tokens(self, filepath: str = "config/etrade_tokens.json") -> bool:
        """Load access tokens from file"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                tokens = json.load(f)
            
            # Check if tokens are for same environment
            if tokens.get('sandbox') != self.sandbox:
                print(f"Warning: Saved tokens are for {'sandbox' if tokens.get('sandbox') else 'production'}, but using {'sandbox' if self.sandbox else 'production'}")
                return False
            
            self.access_token = tokens.get('access_token')
            self.access_token_secret = tokens.get('access_token_secret')
            
            # Create authenticated session
            self.create_authenticated_session()
            
            return True
            
        except Exception as e:
            print(f"Error loading tokens: {e}")
            return False


#!/usr/bin/env python3
"""
Ralph GitHub Continuous Research with Trading Safety Controls
Runs continuously with comprehensive pre-trade checks, position sizing, and circuit breakers
"""

import time
import logging
import os
import sys
import json  # pyright: ignore[reportUnusedImport]  # pyright: ignore[reportUnusedImport]
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path  # pyright: ignore[reportUnusedImport]
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Any, Optional, List, Tuple, Union  # pyright: ignore[reportUnusedImport]
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../NAE Ready'))

try:
    from ralph_github_research import RalphGitHubResearch
except ImportError:
    RalphGitHubResearch = None

try:
    from execution.compliance.day_trading_prevention import DayTradingPrevention  # type: ignore
except ImportError:
    DayTradingPrevention = None  # type: ignore

# fractional_kelly is defined locally in this file, no import needed

logging.basicConfig(level=logging.INFO)
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


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is triggered"""
    pass


@dataclass
class TradingState:
    """Tracks trading state for circuit breaker"""
    consecutive_errors: int = 0
    initial_equity: float = 0.0
    current_equity: float = 0.0
    daily_loss_pct: float = 0.0
    trading_paused: bool = False
    paused_until: Optional[datetime] = None  # Auto-resume after 1 hour
    last_error_time: Optional[datetime] = None


class NotificationService:
    """Notification service for sending alerts via multiple channels"""
    
    def __init__(self):
        """Initialize notification service"""
        # Try to load from .env.notifications file first
        env_file = os.path.join(os.path.dirname(__file__), ".env.notifications")
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ.setdefault(key.strip(), value.strip())
            except Exception as e:
                logger.warning(f"Could not load .env.notifications: {e}")
        
        # Email configuration - Default to cbjones84@yahoo.com
        self.email_enabled = os.getenv("NOTIFICATION_EMAIL_ENABLED", "true").lower() == "true"
        self.email_to = os.getenv("NOTIFICATION_EMAIL_TO", "cbjones84@yahoo.com")
        self.email_from = os.getenv("NOTIFICATION_EMAIL_FROM", "cbjones84@yahoo.com")
        self.smtp_server = os.getenv("NOTIFICATION_SMTP_SERVER", "smtp.mail.yahoo.com")
        self.smtp_port = int(os.getenv("NOTIFICATION_SMTP_PORT", "587"))
        self.smtp_username = os.getenv("NOTIFICATION_SMTP_USERNAME", "cbjones84@yahoo.com")
        self.smtp_password = os.getenv("NOTIFICATION_SMTP_PASSWORD", "lbtfwssbsieswkft")
        self.smtp_use_tls = os.getenv("NOTIFICATION_SMTP_USE_TLS", "true").lower() == "true"
        
        # SMS configuration
        self.sms_enabled = os.getenv("NOTIFICATION_SMS_ENABLED", "false").lower() == "true"
        self.sms_to = os.getenv("NOTIFICATION_SMS_TO")
        
        # Webhook configuration
        self.webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")
        
        # Rate limiting tracking
        self.last_rate_limit_time: Optional[datetime] = None
        self.rate_limit_retry_after: int = 60
        
        logger.info(f"NotificationService initialized - Email: {self.email_enabled} -> {self.email_to}")
    
    def send(self, title: str, message: str, priority: str = "normal"):
        """
        Send notification via configured channels
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level (normal, high, critical)
        """
        try:
            # Email notification
            if self.email_enabled and self.email_to:
                self._send_email(title, message, priority)
            
            # SMS notification (only for critical)
            if self.sms_enabled and self.sms_to and priority == "critical":
                self._send_sms(message, priority)
            
            # Webhook notification (Slack/Discord)
            if self.webhook_url:
                self._send_webhook(title, message, priority)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _send_email(self, title: str, message: str, priority: str):
        """Send email notification via SMTP"""
        if not self.email_enabled or not self.email_to:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"[NAE {priority.upper()}] {title}"
            
            # Email body with HTML formatting for better readability
            html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .header {{ background-color: {'#ff4444' if priority == 'critical' else '#ffaa00' if priority == 'high' else '#4444ff'}; 
                  color: white; padding: 10px; border-radius: 5px; }}
        .content {{ padding: 20px; background-color: #f9f9f9; border-radius: 5px; margin-top: 10px; }}
        .footer {{ margin-top: 20px; font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🚨 NAE Trading System Alert</h2>
    </div>
    <div class="content">
        <p><strong>Priority:</strong> {priority.upper()}</p>
        <p><strong>Title:</strong> {title}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <hr>
        <p><strong>Message:</strong></p>
        <pre style="background-color: #fff; padding: 10px; border-left: 3px solid #444; white-space: pre-wrap;">{message}</pre>
    </div>
    <div class="footer">
        <p>---</p>
        <p>Neural Agency Engine Trading System</p>
        <p>This is an automated notification.</p>
    </div>
</body>
</html>
"""
            
            plain_body = f"""
NAE Trading System Alert

Priority: {priority.upper()}
Title: {title}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}

---
Neural Agency Engine Trading System
This is an automated notification.
"""
            
            # Attach both HTML and plain text versions
            msg.attach(MIMEText(plain_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            if self.smtp_username and self.smtp_password:
                # Authenticated SMTP
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                if self.smtp_use_tls:
                    server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                server.quit()
                logger.info(f"✅ Email sent to {self.email_to}: {title}")
            else:
                # For Yahoo Mail, we can try using app password or OAuth
                # Log warning but still attempt to send
                logger.warning("SMTP credentials not configured. Attempting unauthenticated send (may fail).")
                logger.info(f"📧 Email notification ({priority}): {title} - {message[:100]}")
                logger.info(f"   To configure SMTP, set NOTIFICATION_SMTP_USERNAME and NOTIFICATION_SMTP_PASSWORD")
                
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            # Fallback: log the notification
            logger.info(f"📧 Email notification ({priority}): {title} - {message[:100]}")
            logger.info(f"   Email would be sent to: {self.email_to}")
    
    def _send_sms(self, message: str, priority: str):
        """Send SMS notification"""
        # TODO: Implement SMS sending (Twilio, etc.)
        logger.info(f"📱 SMS notification ({priority}): {message[:100]}")
    
    def _send_webhook(self, title: str, message: str, priority: str):
        """Send webhook notification (Slack/Discord)"""
        try:
            payload = {
                "title": title,
                "message": message,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"✅ Webhook notification sent: {title}")
        except Exception as e:
            logger.warning(f"Failed to send webhook notification: {e}")


class TradierClient:
    """
    Central Tradier API client with retries and error handling
    All broker calls go through this client to prevent silent failures
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        account_id: Optional[str] = None, 
        sandbox: bool = True,
        timeout: int = 30
    ):
        """
        Initialize Tradier client
        
        Args:
            api_key: Tradier API key (from env or vault)
            account_id: Tradier account ID
            sandbox: Use sandbox environment
            timeout: Request timeout in seconds (default: 30)
        
        Raises:
            TradierError: If API key or account ID not configured
        """
        self.api_key = api_key or os.getenv("TRADIER_API_KEY")
        self.account_id = account_id or os.getenv("TRADIER_ACCOUNT_ID")
        self.sandbox = sandbox
        self.timeout = int(os.getenv("TRADIER_API_TIMEOUT", str(timeout)))
        
        # Validate API key at startup
        if not self.api_key:
            raise TradierError(
                "TRADIER_API_KEY not configured. Set environment variable TRADIER_API_KEY or pass api_key parameter.",
                endpoint="initialization"
            )
        
        # Validate account ID at startup
        if not self.account_id:
            raise TradierError(
                "TRADIER_ACCOUNT_ID not configured. Set environment variable TRADIER_ACCOUNT_ID or pass account_id parameter.",
                endpoint="initialization"
            )
        
        if self.sandbox:
            self.api_base = "https://sandbox.tradier.com/v1"
        else:
            self.api_base = "https://api.tradier.com/v1"
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],  # Added 408 (Request Timeout)
            allowed_methods=["GET", "POST", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)  # type: ignore
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting tracking
        self.last_rate_limit_time: Optional[datetime] = None
        self.rate_limit_retry_after: int = 60
        
        logger.info(f"TradierClient initialized (sandbox={sandbox}, timeout={self.timeout}s)")
        
        # Optional: Test connectivity at startup (can be disabled)
        if os.getenv("TRADIER_VALIDATE_ON_STARTUP", "false").lower() == "true":
            try:
                self.health_check()
                logger.info("✅ API connectivity validated")
            except TradierError as e:
                logger.warning(f"⚠️ API connectivity test failed: {e}. System will attempt to continue.")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
    
    def health_check(self) -> bool:
        """
        Perform lightweight health check to verify API connectivity
        
        Returns:
            True if API is accessible
        
        Raises:
            TradierError: If health check fails
        """
        try:
            # Use a lightweight endpoint for health check
            data = self._request("GET", "accounts", timeout=5)  # pyright: ignore[reportUnusedVariable]
            return True
        except TradierError as e:
            raise TradierError(f"Health check failed: {e.message}", endpoint="health_check") from e
    
    def _request(self, method: str, endpoint: str, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Make API request with retries and error handling
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (without base URL)
            timeout: Request timeout (uses instance default if not provided)
            **kwargs: Additional request arguments
        
        Returns:
            Response JSON data
        
        Raises:
            TradierError: On API errors (4xx/5xx)
        """
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        request_timeout = timeout or self.timeout
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=request_timeout,
                **kwargs
            )
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", self.rate_limit_retry_after))
                self.last_rate_limit_time = datetime.now()
                self.rate_limit_retry_after = retry_after
                
                logger.warning(f"Rate limited. Waiting {retry_after} seconds before retry.")
                time.sleep(retry_after)
                
                # Retry once after waiting
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=request_timeout,
                    **kwargs
                )
            
            # Raise on HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
            except ValueError as e:
                raise TradierError(
                    f"Invalid JSON response: {str(e)}",
                    status_code=response.status_code,
                    endpoint=endpoint
                )
            
            # Check for Tradier API errors in response
            if "errors" in data:
                error_msg = str(data["errors"])
                logger.error(f"Tradier API error: {error_msg}")
                raise TradierError(
                    f"Tradier API error: {error_msg}",
                    status_code=response.status_code,
                    endpoint=endpoint
                )
            
            # Reset rate limit tracking on success
            if response.status_code != 429:
                self.last_rate_limit_time = None
            
            return data
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            
            # Format error message consistently
            error_msg = self._format_error_message(
                f"HTTP request failed",
                status_code=status_code,
                endpoint=endpoint,
                details=str(e)
            )
            
            logger.error(error_msg)
            raise TradierError(error_msg, status_code=status_code, endpoint=endpoint) from e
            
        except requests.exceptions.Timeout as e:
            error_msg = self._format_error_message(
                f"Request timeout after {request_timeout}s",
                endpoint=endpoint,
                details=str(e)
            )
            logger.error(error_msg)
            raise TradierError(error_msg, endpoint=endpoint) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = self._format_error_message(
                f"Request failed",
                endpoint=endpoint,
                details=str(e)
            )
            logger.error(error_msg)
            raise TradierError(error_msg, endpoint=endpoint) from e
    
    def _format_error_message(self, base_msg: str, status_code: Optional[int] = None, 
                             endpoint: Optional[str] = None, details: Optional[str] = None) -> str:
        """
        Format error message consistently
        
        Args:
            base_msg: Base error message
            status_code: HTTP status code (optional)
            endpoint: API endpoint (optional)
            details: Additional details (optional)
        
        Returns:
            Formatted error message
        """
        parts = [base_msg]
        if status_code:
            parts.append(f"Status: {status_code}")
        if endpoint:
            parts.append(f"Endpoint: {endpoint}")
        if details:
            parts.append(f"Details: {details}")
        return " | ".join(parts)
    
    def get_balances(self) -> Dict[str, float]:
        """
        Get account balances
        
        Returns:
            Dict with equity, buying_power, cash, etc.
        
        Raises:
            TradierError: On API failure
        """
        # Use /balances endpoint which is more reliable
        data = self._request("GET", f"accounts/{self.account_id}/balances")
        balances_data = data.get("balances", {})
        
        if not balances_data:
            raise TradierError(
                "Balance data not found in API response",
                endpoint=f"accounts/{self.account_id}/balances"
            )
        
        # Extract cash details
        cash_data = balances_data.get("cash", {})
        
        return {
            "equity": float(balances_data.get("total_equity", 0) or balances_data.get("equity", 0) or 0),
            "buying_power": float(balances_data.get("buying_power", 0) or balances_data.get("day_trading_buying_power", 0) or balances_data.get("cash_available", 0) or 0),
            "cash": float(cash_data.get("cash_available", 0) if isinstance(cash_data, dict) else balances_data.get("total_cash", 0) or balances_data.get("cash", 0) or 0),
            "settled_cash": float(cash_data.get("cash_available", 0) if isinstance(cash_data, dict) else balances_data.get("total_cash", 0) or 0),
            "market_value": float(balances_data.get("market_value", 0) or 0),
            "open_pl": float(balances_data.get("open_pl", 0) or 0),
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get account positions
        
        Returns:
            List of position dictionaries
        
        Raises:
            TradierError: On API failure
        """
        data = self._request("GET", f"accounts/{self.account_id}/positions")
        positions = data.get("positions", {})
        
        if isinstance(positions, dict) and "position" in positions:
            pos_list = positions["position"]
            return pos_list if isinstance(pos_list, list) else [pos_list]
        return []
    
    def submit_order(
        self,
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
        Submit order to Tradier
        
        Args:
            symbol: Stock symbol
            side: buy, sell, buy_to_cover, sell_short
            quantity: Number of shares/contracts
            order_type: market, limit, stop, stop_limit
            duration: day, gtc, pre, post
            price: Limit price (optional)
            stop: Stop price (optional)
            option_symbol: Option symbol (optional)
            tag: Order tag (optional)
        
        Returns:
            Order response
        
        Raises:
            TradierError: On API failure
        """
        data = {
            "class": "equity" if not option_symbol else "option",
            "symbol": symbol if not option_symbol else option_symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "duration": duration
        }
        
        if price:
            data["price"] = price
        if stop:
            data["stop"] = stop
        if tag:
            data["tag"] = tag
        
        return self._request("POST", f"accounts/{self.account_id}/orders", data=data)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            Cancellation response
        
        Raises:
            TradierError: On API failure
        """
        return self._request("DELETE", f"accounts/{self.account_id}/orders/{order_id}")
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get account orders
        
        Returns:
            List of order dictionaries
        
        Raises:
            TradierError: On API failure
        """
        data = self._request("GET", f"accounts/{self.account_id}/orders")
        orders = data.get("orders", {})
        
        if isinstance(orders, dict) and "order" in orders:
            order_list = orders["order"]
            return order_list if isinstance(order_list, list) else [order_list]
        return []


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float, 
                     fraction: float = 0.90, max_pct: float = 0.25) -> float:
    """
    Calculate fractional Kelly position size percentage
    
    EXTREME AGGRESSIVE MODE: Maximum risk for maximum returns
    - Using 90% of full Kelly (near full Kelly - EXTREME)
    - Max position size 25% of equity (was 2% - EXTREME RISK)
    
    Args:
        win_rate: Win probability (0.0 to 1.0)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive number)
        fraction: Fraction of full Kelly to use (default 0.90 = 90% - EXTREME)
        max_pct: Maximum position size percentage (default 0.25 = 25% - EXTREME RISK)
    
    Returns:
        Position size as percentage of equity (0.0 to max_pct)
    """
    if avg_loss <= 0:
        return 0.0
    
    # Calculate win odds (avg_win / avg_loss)
    win_odds = avg_win / avg_loss
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win_rate, q = 1 - win_rate, b = win_odds
    p = win_rate
    q = 1.0 - p
    b = win_odds
    
    if b <= 0:
        return 0.0
    
    # Full Kelly fraction
    full_kelly = (p * b - q) / b
    
    # Apply fractional Kelly and cap at max_pct
    kelly_pct = max(0.0, min(full_kelly * fraction, max_pct))
    
    return kelly_pct


class TradingSafetyManager:
    """
    Manages pre-trade checks, position sizing, time filters, and circuit breakers
    
    EXTREME AGGRESSIVE MODE ENABLED: Maximum risk parameters for maximum returns
    - Position sizing: 90% Kelly fraction, 25% max position size (was 20%, 2%)
    - Daily loss limit: 35% (was 5% - EXTREME)
    - Circuit breaker: 50% drawdown (was 10% - EXTREME)
    - Error tolerance: 10 consecutive errors (was 3)
    
    WARNING: These settings represent EXTREME RISK. Large losses are possible.
    """
    
    def __init__(self, client: TradierClient, pdt_checker: Optional[Any] = None):
        """
        Initialize safety manager
        
        Args:
            client: TradierClient instance
            pdt_checker: DayTradingPrevention instance (optional)
        """
        self.client = client
        self.pdt_checker = pdt_checker
        self.state = TradingState()
        
        # Initialize notification service
        self.notification_service = NotificationService()
        
        # Configuration - EXTREME AGGRESSIVE MODE: Maximum risk for maximum returns
        self.MIN_BUYING_POWER = 25.0  # Very low minimum buying power floor (was $100)
        self.DAILY_MAX_LOSS_PCT = 0.35  # 35% daily loss limit (was 5% - EXTREME RISK)
        self.PDT_THRESHOLD = 25000.0  # $25k PDT threshold
        self.MAX_CONSECUTIVE_ERRORS = 10  # Very high error tolerance (was 3)
        self.MAX_DRAWDOWN_PCT = 0.50  # 50% intraday drawdown limit (was 10% - EXTREME RISK)
        
        # Time filters
        self.MARKET_OPEN_TIME = dt_time(9, 30)  # 9:30 AM ET
        self.MARKET_CLOSE_TIME = dt_time(16, 0)  # 4:00 PM ET
        self.SKIP_FIRST_MINUTES = 10  # Skip first 10 minutes
        self.SKIP_LAST_MINUTES = 20  # Skip last 20 minutes
        
        # Cycle interval
        self.CYCLE_INTERVAL_MIN = 30  # Minimum seconds between cycles
        self.CYCLE_INTERVAL_MAX = 60  # Maximum seconds between cycles
        
        logger.info("TradingSafetyManager initialized - EXTREME AGGRESSIVE MODE")
        logger.warning("⚠️  EXTREME RISK SETTINGS ENABLED - Maximum risk for maximum returns")
        logger.info(f"  Position sizing: 90% Kelly, 25% max position size (EXTREME)")
        logger.info(f"  Daily loss limit: {self.DAILY_MAX_LOSS_PCT*100:.0f}% (EXTREME)")
        logger.info(f"  Circuit breaker drawdown: {self.MAX_DRAWDOWN_PCT*100:.0f}% (EXTREME)")
        logger.warning("  WARNING: These settings allow for very large losses. Monitor closely.")
    
    def _is_market_hours(self) -> Tuple[bool, str]:
        """
        Check if current time is within trading hours (excluding filtered periods)
        
        Returns:
            (is_allowed, reason)
        """
        now = datetime.now()
        current_time = now.time()
        
        # Check if weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Market closed (weekend)"
        
        # Check market hours
        if current_time < self.MARKET_OPEN_TIME:
            return False, f"Before market open ({self.MARKET_OPEN_TIME})"
        
        if current_time >= self.MARKET_CLOSE_TIME:
            return False, f"After market close ({self.MARKET_CLOSE_TIME})"
        
        # Check skip periods
        open_dt = datetime.combine(now.date(), self.MARKET_OPEN_TIME)
        close_dt = datetime.combine(now.date(), self.MARKET_CLOSE_TIME)
        
        skip_start_end = open_dt + timedelta(minutes=self.SKIP_FIRST_MINUTES)
        skip_end_start = close_dt - timedelta(minutes=self.SKIP_LAST_MINUTES)
        
        if now < skip_start_end:
            return False, f"Within first {self.SKIP_FIRST_MINUTES} minutes (filtered)"
        
        if now >= skip_end_start:
            return False, f"Within last {self.SKIP_LAST_MINUTES} minutes (filtered)"
        
        return True, "Market hours"
    
    def _check_buying_power(self) -> Tuple[bool, str]:
        """
        Check if buying power is above safe floor
        
        Returns:
            (is_sufficient, reason)
        """
        try:
            balances = self.client.get_balances()
            buying_power = balances.get("buying_power", 0)
            
            if buying_power < self.MIN_BUYING_POWER:
                return False, f"Buying power ${buying_power:.2f} below floor ${self.MIN_BUYING_POWER:.2f}"
            
            return True, f"Buying power sufficient: ${buying_power:.2f}"
        except TradierError as e:
            return False, f"Failed to check buying power: {e}"
    
    def _check_daily_loss(self) -> Tuple[bool, str]:
        """
        Check if daily loss exceeds limit
        
        Returns:
            (is_allowed, reason)
        """
        try:
            balances = self.client.get_balances()
            current_equity = balances.get("equity", 0)
            
            # Initialize initial equity if not set
            if self.state.initial_equity == 0:
                self.state.initial_equity = current_equity
                self.state.current_equity = current_equity
            
            # Update current equity
            self.state.current_equity = current_equity
            
            # Calculate daily loss percentage
            if self.state.initial_equity > 0:
                self.state.daily_loss_pct = (
                    (self.state.initial_equity - current_equity) / self.state.initial_equity
                )
            else:
                self.state.daily_loss_pct = 0.0
            
            if self.state.daily_loss_pct >= self.DAILY_MAX_LOSS_PCT:
                reason = (
                    f"Daily loss {self.state.daily_loss_pct*100:.2f}% "
                    f"exceeds limit {self.DAILY_MAX_LOSS_PCT*100:.2f}%"
                )
                self.pause_trading(reason)
                return False, reason
            
            return True, f"Daily loss {self.state.daily_loss_pct*100:.2f}% within limit"
        except TradierError as e:
            return False, f"Failed to check daily loss: {e}"
    
    def _check_pdt(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Check PDT compliance
        
        Args:
            symbol: Stock symbol
            side: Order side (buy/sell)
        
        Returns:
            (is_allowed, reason)
        """
        if not self.pdt_checker:
            return True, "PDT checker not available"
        
        try:
            balances = self.client.get_balances()
            equity = balances.get("equity", 0)
            
            # If account < $25k, check PDT
            if equity < self.PDT_THRESHOLD:
                allowed, reason = self.pdt_checker.check_day_trade_allowed(symbol, side)
                if not allowed:
                    return False, f"PDT violation: {reason}"
            
            return True, "PDT compliant"
        except TradierError as e:
            return False, f"Failed to check PDT: {e}"
    
    def _check_circuit_breaker(self) -> Tuple[bool, str]:
        """
        Check circuit breaker conditions
        
        Returns:
            (is_allowed, reason)
        """
        # Check consecutive errors
        if self.state.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            reason = (
                f"Circuit breaker: {self.state.consecutive_errors} consecutive errors "
                f"(limit: {self.MAX_CONSECUTIVE_ERRORS})"
            )
            self.pause_trading(reason)
            return False, reason
        
        # Check drawdown
        if self.state.initial_equity > 0:
            drawdown_pct = (
                (self.state.initial_equity - self.state.current_equity) / self.state.initial_equity
            )
            if drawdown_pct >= self.MAX_DRAWDOWN_PCT:
                reason = (
                    f"Circuit breaker: Drawdown {drawdown_pct*100:.2f}% "
                    f"exceeds limit {self.MAX_DRAWDOWN_PCT*100:.2f}%"
                )
                self.pause_trading(reason)
                return False, reason
        
        return True, "Circuit breaker OK"
    
    def pre_trade_check(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Run all pre-trade checks before submitting order
        
        Args:
            symbol: Stock symbol
            side: Order side (buy/sell)
        
        Returns:
            (is_allowed, reason)
        """
        # Check if trading is paused - auto-resume after 1 hour
        if self.state.trading_paused:
            if self.state.paused_until and datetime.now() >= self.state.paused_until:
                self.state.trading_paused = False
                self.state.paused_until = None
                logger.info("✅ Circuit breaker cooldown expired (1hr) - Trading resumed")
            else:
                return False, "Trading paused (circuit breaker or daily loss limit) - resumes in 1 hour"
        
        # Check market hours
        allowed, reason = self._is_market_hours()
        if not allowed:
            return False, reason
        
        # Check buying power
        allowed, reason = self._check_buying_power()
        if not allowed:
            return False, reason
        
        # Check daily loss
        allowed, reason = self._check_daily_loss()
        if not allowed:
            return False, reason
        
        # Check PDT
        allowed, reason = self._check_pdt(symbol, side)
        if not allowed:
            return False, reason
        
        # Check circuit breaker
        allowed, reason = self._check_circuit_breaker()
        if not allowed:
            return False, reason
        
        return True, "All pre-trade checks passed"
    
    def calculate_position_size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        price: float
    ) -> Tuple[int, float]:
        """
        Calculate position size using fractional Kelly
        
        Args:
            equity: Account equity
            win_rate: Win probability (0.0 to 1.0)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            price: Stock price
        
        Returns:
            (quantity, notional_value)
        """
        # Calculate position percentage - EXTREME AGGRESSIVE MODE
        # Using 90% Kelly fraction and 25% max position size for maximum returns
        pct = fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.90, max_pct=0.25)
        
        # Calculate notional value
        notional = equity * pct
        
        # Calculate quantity (round down to whole shares)
        quantity = int(notional / price) if price > 0 else 0
        
        return quantity, notional
    
    def record_error(self, error: Exception):
        """Record an error for circuit breaker"""
        self.state.consecutive_errors += 1
        self.state.last_error_time = datetime.now()
        logger.warning(f"Error recorded ({self.state.consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS}): {error}")
    
    def record_success(self):
        """Reset error counter on successful operation"""
        if self.state.consecutive_errors > 0:
            logger.info(f"Success - resetting error counter (was {self.state.consecutive_errors})")
        self.state.consecutive_errors = 0
    
    def pause_trading(self, reason: str):
        """Pause trading and send alert"""
        self.state.trading_paused = True
        alert_msg = f"🚨 CIRCUIT BREAKER TRIGGERED: Trading paused - {reason}"
        logger.error(alert_msg)
        print(f"\n{'='*60}")
        print(alert_msg)
        print(f"{'='*60}\n")
        
        # Send notification via all configured channels
        self.notification_service.send(
            title="Circuit Breaker Triggered",
            message=alert_msg,
            priority="critical"
        )
    
    def reset_daily_state(self):
        """Reset daily state (call at start of trading day)"""
        self.state.initial_equity = 0.0
        self.state.current_equity = 0.0
        self.state.daily_loss_pct = 0.0
        self.state.trading_paused = False
        self.state.paused_until = None
        logger.info("Daily state reset")


def continuous_research_loop():
    """Continuous research loop with trading safety controls"""
    
    # Initialize components
    research = RalphGitHubResearch() if RalphGitHubResearch else None
    
    # Initialize Tradier client
    client = TradierClient()
    
    # Initialize PDT checker if available
    pdt_checker = None
    if DayTradingPrevention:
        try:
            pdt_checker = DayTradingPrevention()
        except Exception as e:
            logger.warning(f"Could not initialize PDT checker: {e}")
    
    # Initialize safety manager
    safety_manager = TradingSafetyManager(client, pdt_checker)
    
    logger.info("🔄 Starting continuous GitHub research with trading safety controls")
    logger.info(f"Market hours filter: Skip first {safety_manager.SKIP_FIRST_MINUTES}min, last {safety_manager.SKIP_LAST_MINUTES}min")
    logger.info(f"Cycle interval: {safety_manager.CYCLE_INTERVAL_MIN}-{safety_manager.CYCLE_INTERVAL_MAX}s")
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            logger.info("=" * 60)
            logger.info(f"Starting research cycle #{cycle_count}")
            logger.info("=" * 60)
            
            # Reset daily state at start of trading day (9:30 AM)
            now = datetime.now()
            if now.hour == 9 and now.minute == 30:
                safety_manager.reset_daily_state()
            
            # Check market hours
            allowed, reason = safety_manager._is_market_hours()
            if not allowed:
                logger.info(f"⏸️  Outside trading hours: {reason}")
                # Wait until next market open
                wait_seconds = 3600  # Check every hour
                logger.info(f"⏳ Waiting {wait_seconds}s until next check...")
                time.sleep(wait_seconds)
                continue
            
            # Run research if available
            if research:
                results = research.run_full_research()
                total_repos = sum(len(r) for r in results.values())
                logger.info(f"✅ Research complete: {total_repos} repositories analyzed")
            
            # Example: Pre-trade check before any order
            # This would be called before submitting orders
            example_symbol = "AAPL"
            example_side = "buy"
            allowed, reason = safety_manager.pre_trade_check(example_symbol, example_side)
            logger.info(f"Pre-trade check for {example_symbol} {example_side}: {allowed} - {reason}")
            
            # Calculate cycle interval (random between min and max)
            import random
            cycle_interval = random.randint(
                safety_manager.CYCLE_INTERVAL_MIN,
                safety_manager.CYCLE_INTERVAL_MAX
            )
            
            logger.info(f"⏳ Waiting {cycle_interval}s until next cycle...")
            time.sleep(cycle_interval)
            
        except KeyboardInterrupt:
            logger.warning("⚠️  KeyboardInterrupt received - RESTARTING instead of stopping")
            logger.info("Restarting in 5 seconds...")
            time.sleep(5)
            # Continue loop - NEVER STOP
        except TradierError as e:
            safety_manager.record_error(e)
            logger.error(f"Tradier error in research loop: {e}")
            logger.info("Retrying in 1 hour...")
            time.sleep(3600)
            # Continue loop - NEVER STOP
        except Exception as e:
            safety_manager.record_error(e)
            logger.error(f"Error in research loop: {e}")
            logger.info("Retrying in 1 hour...")
            time.sleep(3600)
            # Continue loop - NEVER STOP


def run_forever_with_restart():
    """
    Run continuous_research_loop forever with automatic restart on any exit.
    This ensures NAE NEVER stops running.
    """
    restart_count = 0
    max_restart_delay = 3600  # Max 1 hour delay
    
    while True:  # Outer infinite loop - NEVER EXIT
        try:
            logger.info("=" * 70)
            logger.info(f"🚀 Starting NAE Trading System (Restart #{restart_count})")
            logger.info("=" * 70)
            continuous_research_loop()
        except KeyboardInterrupt:
            restart_count += 1
            logger.warning(f"⚠️  KeyboardInterrupt received - RESTARTING (Restart #{restart_count})")
            logger.info("Restarting in 5 seconds...")
            time.sleep(5)
            # Continue outer loop - NEVER STOP
        except SystemExit:
            restart_count += 1
            logger.warning(f"⚠️  SystemExit received - RESTARTING (Restart #{restart_count})")
            logger.info("Restarting in 10 seconds...")
            time.sleep(10)
            # Continue outer loop - NEVER STOP
        except Exception as e:
            restart_count += 1
            delay = min(60 * restart_count, max_restart_delay)  # Exponential backoff, max 1 hour
            logger.error(f"❌ Fatal error in NAE (Restart #{restart_count}): {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            logger.info(f"🔄 Restarting in {delay} seconds...")
            time.sleep(delay)
            # Continue outer loop - NEVER STOP


if __name__ == '__main__':
    import traceback
    run_forever_with_restart()

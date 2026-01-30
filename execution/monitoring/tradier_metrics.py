"""
Tradier-Specific Metrics

Tracks Tradier OAuth token expiry, WebSocket status, and Tradier-specific metrics.
"""

from prometheus_client import Gauge, Counter
from execution.monitoring.metrics import metrics_collector

# Tradier-specific metrics
tradier_oauth_token_expires_at = Gauge(
    'nae_tradier_oauth_token_expires_at',
    'Tradier OAuth token expiration timestamp',
    ['account_id']
)

tradier_websocket_connected = Gauge(
    'nae_tradier_websocket_connected',
    'Tradier WebSocket connection status (1=connected, 0=disconnected)',
    ['account_id']
)

tradier_websocket_reconnects = Counter(
    'nae_tradier_websocket_reconnects_total',
    'Total Tradier WebSocket reconnection attempts',
    ['account_id']
)

tradier_order_previews = Counter(
    'nae_tradier_order_previews_total',
    'Total Tradier order previews',
    ['account_id']
)

tradier_pre_post_market_orders = Counter(
    'nae_tradier_pre_post_market_orders_total',
    'Total Tradier pre/post-market orders',
    ['account_id', 'session_type']
)


class TradierMetricsCollector:
    """Collects Tradier-specific metrics"""
    
    def __init__(self):
        self.metrics_collector = metrics_collector
    
    def update_oauth_expiry(self, account_id: str, expires_at: float):
        """Update OAuth token expiration"""
        tradier_oauth_token_expires_at.labels(account_id=account_id).set(expires_at)
    
    def update_websocket_status(self, account_id: str, connected: bool):
        """Update WebSocket connection status"""
        tradier_websocket_connected.labels(account_id=account_id).set(1 if connected else 0)
    
    def record_websocket_reconnect(self, account_id: str):
        """Record WebSocket reconnection"""
        tradier_websocket_reconnects.labels(account_id=account_id).inc()
    
    def record_order_preview(self, account_id: str):
        """Record order preview"""
        tradier_order_previews.labels(account_id=account_id).inc()
    
    def record_pre_post_market_order(self, account_id: str, session_type: str):
        """Record pre/post-market order"""
        tradier_pre_post_market_orders.labels(
            account_id=account_id,
            session_type=session_type
        ).inc()


# Global Tradier metrics collector
tradier_metrics = TradierMetricsCollector()


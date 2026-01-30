"""
Monitoring and Metrics - Prometheus Integration

Exposes metrics for orders, executions, latency, and system health.
"""

import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
orders_submitted_total = Counter(
    'nae_orders_submitted_total',
    'Total orders submitted',
    ['strategy_id', 'symbol', 'order_type']
)

orders_filled_total = Counter(
    'nae_orders_filled_total',
    'Total orders filled',
    ['strategy_id', 'symbol', 'order_type']
)

orders_rejected_total = Counter(
    'nae_orders_rejected_total',
    'Total orders rejected',
    ['strategy_id', 'reason']
)

order_latency_ms = Histogram(
    'nae_order_latency_ms',
    'Order execution latency in milliseconds',
    ['strategy_id'],
    buckets=[10, 50, 100, 500, 1000, 5000]
)

strategy_drawdown_percent = Gauge(
    'nae_strategy_drawdown_percent',
    'Strategy drawdown percentage',
    ['strategy_id', 'period']
)

pending_signals = Gauge(
    'nae_pending_signals',
    'Number of pending signals in queue',
    ['queue_name']
)

oauth_token_expires_at = Gauge(
    'nae_oauth_token_expires_at',
    'OAuth token expiration timestamp',
    ['broker', 'account_id']
)

execution_failures_total = Counter(
    'nae_execution_failures_total',
    'Total execution failures',
    ['broker', 'error_type']
)

reconciliation_errors_total = Counter(
    'nae_reconciliation_errors_total',
    'Total reconciliation errors',
    ['type']
)

broker_connectivity = Gauge(
    'nae_broker_connectivity',
    'Broker connectivity status (1=connected, 0=disconnected)',
    ['broker']
)


class MetricsCollector:
    """Collects and exposes metrics"""
    
    def __init__(self, port: int = 8002):
        self.port = port
        self.start_server()
    
    def start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
    
    def record_order_submitted(self, strategy_id: str, symbol: str, order_type: str):
        """Record order submission"""
        orders_submitted_total.labels(
            strategy_id=strategy_id,
            symbol=symbol,
            order_type=order_type
        ).inc()
    
    def record_order_filled(self, strategy_id: str, symbol: str, order_type: str):
        """Record order fill"""
        orders_filled_total.labels(
            strategy_id=strategy_id,
            symbol=symbol,
            order_type=order_type
        ).inc()
    
    def record_order_rejected(self, strategy_id: str, reason: str):
        """Record order rejection"""
        orders_rejected_total.labels(
            strategy_id=strategy_id,
            reason=reason
        ).inc()
    
    def record_order_latency(self, strategy_id: str, latency_ms: float):
        """Record order execution latency"""
        order_latency_ms.labels(strategy_id=strategy_id).observe(latency_ms)
    
    def update_strategy_drawdown(self, strategy_id: str, period: str, drawdown_pct: float):
        """Update strategy drawdown"""
        strategy_drawdown_percent.labels(
            strategy_id=strategy_id,
            period=period
        ).set(drawdown_pct)
    
    def update_pending_signals(self, queue_name: str, count: int):
        """Update pending signals count"""
        pending_signals.labels(queue_name=queue_name).set(count)
    
    def update_oauth_expiry(self, broker: str, account_id: str, expires_at: float):
        """Update OAuth token expiration"""
        oauth_token_expires_at.labels(
            broker=broker,
            account_id=account_id
        ).set(expires_at)
    
    def record_execution_failure(self, broker: str, error_type: str):
        """Record execution failure"""
        execution_failures_total.labels(
            broker=broker,
            error_type=error_type
        ).inc()
    
    def record_reconciliation_error(self, error_type: str):
        """Record reconciliation error"""
        reconciliation_errors_total.labels(type=error_type).inc()
    
    def update_broker_connectivity(self, broker: str, connected: bool):
        """Update broker connectivity status"""
        broker_connectivity.labels(broker=broker).set(1 if connected else 0)


# Global metrics collector instance
metrics_collector = MetricsCollector()


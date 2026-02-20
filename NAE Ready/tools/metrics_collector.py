# NAE/tools/metrics_collector.py
"""
Metrics Collection System for NAE
Implements Prometheus-compatible metrics with real-time dashboards

Key Metrics:
- PnL (daily/weekly)
- Realized volatility
- Max drawdown (30/90/365d)
- Sharpe, Sortino ratios
- Hit rate, average return per trade
- Latency per decision
- Model drift score
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
from enum import Enum

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Install with: pip install prometheus-client")


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    agent: str
    version: str
    model_id: Optional[str] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: str  # e.g., ">", "<", "=="
    threshold: float
    severity: str  # "critical", "warning", "info"
    enabled: bool = True


class MetricsCollector:
    """
    Centralized metrics collection system for NAE
    
    Collects metrics from all agents and provides:
    - Real-time metric storage
    - Prometheus-compatible export
    - Alert evaluation
    - Historical aggregation
    """
    
    def __init__(self, prometheus_port: int = 8000):
        self.prometheus_port = prometheus_port
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[AlertRule] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Prometheus metrics (if available)
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
            try:
                start_http_server(self.prometheus_port)
                print(f"✅ Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                print(f"⚠️  Could not start Prometheus server: {e}")
        
        # Default alert rules
        self._setup_default_alerts()
        
        # Start background thread for alert evaluation
        self.running = True
        self.alert_thread = threading.Thread(target=self._alert_evaluation_loop, daemon=True)
        self.alert_thread.start()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Trading metrics
        self.prometheus_metrics['pnl'] = Gauge('nae_pnl', 'Portfolio PnL', ['agent', 'period'])
        self.prometheus_metrics['realized_volatility'] = Gauge('nae_realized_volatility', 'Realized volatility', ['agent', 'period'])
        self.prometheus_metrics['max_drawdown'] = Gauge('nae_max_drawdown', 'Maximum drawdown', ['agent', 'period'])
        self.prometheus_metrics['sharpe_ratio'] = Gauge('nae_sharpe_ratio', 'Sharpe ratio', ['agent', 'period'])
        self.prometheus_metrics['sortino_ratio'] = Gauge('nae_sortino_ratio', 'Sortino ratio', ['agent', 'period'])
        self.prometheus_metrics['hit_rate'] = Gauge('nae_hit_rate', 'Trade hit rate', ['agent'])
        self.prometheus_metrics['avg_return_per_trade'] = Gauge('nae_avg_return_per_trade', 'Average return per trade', ['agent'])
        
        # Performance metrics
        self.prometheus_metrics['decision_latency'] = Histogram('nae_decision_latency_seconds', 'Decision latency', ['agent', 'model_id'])
        self.prometheus_metrics['model_drift_score'] = Gauge('nae_model_drift_score', 'Model drift score', ['agent', 'model_id'])
        
        # Risk metrics
        self.prometheus_metrics['position_size'] = Gauge('nae_position_size', 'Position size', ['agent', 'symbol'])
        self.prometheus_metrics['daily_loss'] = Gauge('nae_daily_loss', 'Daily loss', ['agent'])
        self.prometheus_metrics['consecutive_losses'] = Gauge('nae_consecutive_losses', 'Consecutive losses', ['agent'])
        
        # System metrics
        self.prometheus_metrics['data_feed_delay'] = Gauge('nae_data_feed_delay_seconds', 'Data feed delay', ['source'])
        self.prometheus_metrics['model_confidence'] = Gauge('nae_model_confidence', 'Model confidence', ['agent', 'model_id'])
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        self.alerts = [
            AlertRule("pnl_drawdown_24h", "daily_pnl", "<", -0.05, "critical"),  # >5% loss in 24h
            AlertRule("strategy_exposure_limit", "strategy_exposure", ">", 0.20, "warning"),  # >20% exposure
            AlertRule("model_confidence_low", "model_confidence", "<", 0.50, "warning"),
            AlertRule("data_feed_delay", "data_feed_delay", ">", 5.0, "critical"),  # >5s delay
            AlertRule("consecutive_losses", "consecutive_losses", ">", 5, "critical"),
        ]
    
    def record_metric(
        self,
        name: str,
        value: float,
        agent: str,
        tags: Optional[Dict[str, str]] = None,
        version: str = "1.0",
        model_id: Optional[str] = None
    ):
        """
        Record a metric
        
        Args:
            name: Metric name
            value: Metric value
            agent: Agent name
            tags: Additional tags
            version: Agent version
            model_id: Model identifier
        """
        tags = tags or {}
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags,
            agent=agent,
            version=version,
            model_id=model_id
        )
        
        with self.lock:
            self.metrics[name].append(metric_point)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and name in self.prometheus_metrics:
            try:
                metric = self.prometheus_metrics[name]
                if isinstance(metric, Gauge):
                    metric.labels(agent=agent, **tags).set(value)
                elif isinstance(metric, Histogram):
                    metric.labels(agent=agent, model_id=model_id or "default").observe(value)
            except Exception as e:
                pass  # Silently fail if Prometheus update fails
    
    def record_pnl(self, pnl: float, agent: str, period: str = "daily"):
        """Record PnL metric"""
        self.record_metric("pnl", pnl, agent, tags={"period": period})
        self.record_metric("daily_pnl", pnl, agent)  # Also record as daily_pnl for alerts
    
    def record_trade(
        self,
        agent: str,
        return_pct: float,
        latency_seconds: float,
        model_id: Optional[str] = None
    ):
        """Record trade metrics"""
        self.record_metric("trade_return", return_pct, agent, model_id=model_id)
        self.record_metric("decision_latency", latency_seconds, agent, model_id=model_id)
    
    def record_model_confidence(self, confidence: float, agent: str, model_id: str):
        """Record model confidence"""
        self.record_metric("model_confidence", confidence, agent, model_id=model_id)
    
    def record_data_feed_delay(self, delay_seconds: float, source: str):
        """Record data feed delay"""
        self.record_metric("data_feed_delay", delay_seconds, "system", tags={"source": source})
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        agent: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Get metrics with optional filtering"""
        with self.lock:
            if name:
                metrics = list(self.metrics[name])
            else:
                metrics = []
                for metric_list in self.metrics.values():
                    metrics.extend(metric_list)
            
            # Filter by agent
            if agent:
                metrics = [m for m in metrics if m.agent == agent]
            
            # Filter by time range
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return sorted(metrics, key=lambda x: x.timestamp)
    
    def calculate_sharpe_ratio(self, agent: str, period_days: int = 30) -> float:
        """Calculate Sharpe ratio for agent"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=period_days)
        
        returns = [m.value for m in self.get_metrics("trade_return", agent, start_time, end_time)]
        
        if len(returns) < 2:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)
    
    def calculate_max_drawdown(self, agent: str, period_days: int = 30) -> float:
        """Calculate maximum drawdown"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=period_days)
        
        pnl_metrics = self.get_metrics("pnl", agent, start_time, end_time)
        if not pnl_metrics:
            return 0.0
        
        cumulative_pnl = []
        running_sum = 0.0
        for metric in sorted(pnl_metrics, key=lambda x: x.timestamp):
            running_sum += metric.value
            cumulative_pnl.append(running_sum)
        
        if not cumulative_pnl:
            return 0.0
        
        peak = cumulative_pnl[0]
        max_dd = 0.0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def calculate_hit_rate(self, agent: str) -> float:
        """Calculate hit rate (percentage of profitable trades)"""
        returns = [m.value for m in self.get_metrics("trade_return", agent)]
        
        if not returns:
            return 0.0
        
        profitable = sum(1 for r in returns if r > 0)
        return profitable / len(returns)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alerts.append(rule)
    
    def _alert_evaluation_loop(self):
        """Background thread to evaluate alerts"""
        while self.running:
            try:
                self._evaluate_alerts()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Error in alert evaluation: {e}")
                time.sleep(60)
    
    def _evaluate_alerts(self):
        """Evaluate all alert rules"""
        for rule in self.alerts:
            if not rule.enabled:
                continue
            
            # Get latest metric value
            metrics = self.get_metrics(rule.metric)
            if not metrics:
                continue
            
            latest_value = metrics[-1].value
            
            # Check condition
            triggered = False
            if rule.condition == ">":
                triggered = latest_value > rule.threshold
            elif rule.condition == "<":
                triggered = latest_value < rule.threshold
            elif rule.condition == "==":
                triggered = abs(latest_value - rule.threshold) < 0.001
            elif rule.condition == ">=":
                triggered = latest_value >= rule.threshold
            elif rule.condition == "<=":
                triggered = latest_value <= rule.threshold
            
            if triggered:
                alert = {
                    "rule": rule.name,
                    "metric": rule.metric,
                    "value": latest_value,
                    "threshold": rule.threshold,
                    "severity": rule.severity,
                    "timestamp": datetime.now().isoformat(),
                    "agent": metrics[-1].agent
                }
                
                self.alert_history.append(alert)
                
                # Log alert
                self._log_alert(alert)
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert (can be extended to send notifications)"""
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🔵"
        }
        emoji = severity_emoji.get(alert["severity"], "⚪")
        
        print(f"{emoji} ALERT [{alert['severity'].upper()}]: {alert['rule']} - "
              f"{alert['metric']} = {alert['value']:.4f} (threshold: {alert['threshold']:.4f})")
        
        # Save to file
        alert_file = "logs/alerts.jsonl"
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)
        with open(alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard"""
        agents = set()
        for metrics_list in self.metrics.values():
            for metric in metrics_list:
                agents.add(metric.agent)
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "system_metrics": {},
            "alerts": self.alert_history[-10:]  # Last 10 alerts
        }
        
        for agent in agents:
            dashboard_data["agents"][agent] = {
                "pnl": self._get_latest_metric("pnl", agent, 0.0),
                "sharpe_ratio": self.calculate_sharpe_ratio(agent),
                "max_drawdown_30d": self.calculate_max_drawdown(agent, 30),
                "max_drawdown_90d": self.calculate_max_drawdown(agent, 90),
                "hit_rate": self.calculate_hit_rate(agent),
                "model_confidence": self._get_latest_metric("model_confidence", agent, 0.0),
            }
        
        return dashboard_data
    
    def _get_latest_metric(self, name: str, agent: str, default: float = 0.0) -> float:
        """Get latest metric value"""
        metrics = self.get_metrics(name, agent)
        if metrics:
            return metrics[-1].value
        return default
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        # Prometheus client handles this automatically via HTTP endpoint
        return f"# Prometheus metrics available at http://localhost:{self.prometheus_port}/metrics\n"
    
    def shutdown(self):
        """Shutdown metrics collector"""
        self.running = False
        if self.alert_thread.is_alive():
            self.alert_thread.join(timeout=5)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


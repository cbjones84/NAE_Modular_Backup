#!/bin/bash
# Setup Grafana Dashboard for NAE Monitoring

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "📊 Setting up Grafana Dashboard for NAE"
echo "=========================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Installing Grafana requires Docker."
    echo "   Please install Docker Desktop or use manual Grafana installation."
    exit 1
fi

# Create docker-compose for Grafana/Prometheus
cat > docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: nae_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: nae_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=nae_admin_2024
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana_dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana_datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    depends_on:
      - prometheus
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
EOF

# Create Prometheus config
mkdir -p config
cat > config/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nae'
    static_configs:
      - targets: ['host.docker.internal:8000']
        labels:
          instance: 'nae_local'
EOF

# Create Grafana datasource config
mkdir -p config/grafana_datasources
cat > config/grafana_datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create Grafana dashboard config
mkdir -p config/grafana_dashboards
cat > config/grafana_dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'NAE Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Create NAE dashboard JSON
cat > config/grafana_dashboards/nae_dashboard.json << 'EOFDASH'
{
  "dashboard": {
    "title": "NAE Trading Dashboard",
    "tags": ["nae", "trading"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Portfolio Value",
        "type": "graph",
        "targets": [
          {
            "expr": "nae_pnl{agent=\"Optimus\", period=\"daily\"}",
            "legendFormat": "Daily PnL"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Sharpe Ratio",
        "type": "stat",
        "targets": [
          {
            "expr": "nae_sharpe_ratio{agent=\"Optimus\", period=\"30d\"}",
            "legendFormat": "30-Day Sharpe"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Max Drawdown",
        "type": "stat",
        "targets": [
          {
            "expr": "nae_max_drawdown{agent=\"Optimus\", period=\"30d\"}",
            "legendFormat": "30-Day Max DD"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0}
      },
      {
        "id": 4,
        "title": "Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "nae_hit_rate{agent=\"Optimus\"}",
            "legendFormat": "Win Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8}
      },
      {
        "id": 5,
        "title": "Decision Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, nae_decision_latency_seconds_bucket{agent=\"Optimus\"})",
            "legendFormat": "P95 Latency"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8}
      },
      {
        "id": 6,
        "title": "Model Confidence",
        "type": "graph",
        "targets": [
          {
            "expr": "nae_model_confidence{agent=\"Optimus\"}",
            "legendFormat": "Confidence"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8}
      },
      {
        "id": 7,
        "title": "Circuit Breaker Status",
        "type": "stat",
        "targets": [
          {
            "expr": "nae_consecutive_losses{agent=\"Optimus\"}",
            "legendFormat": "Consecutive Losses"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8}
      },
      {
        "id": 8,
        "title": "Daily Loss",
        "type": "stat",
        "targets": [
          {
            "expr": "nae_daily_loss{agent=\"Optimus\"}",
            "legendFormat": "Daily Loss %"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 12}
      }
    ],
    "refresh": "10s",
    "schemaVersion": 27,
    "version": 1
  }
}
EOFDASH

echo "✅ Configuration files created"
echo ""
echo "To start monitoring stack:"
echo "  docker-compose -f docker-compose.monitoring.yml up -d"
echo ""
echo "Access Grafana at: http://localhost:3000"
echo "  Username: admin"
echo "  Password: nae_admin_2024"
echo ""
echo "Access Prometheus at: http://localhost:9090"
echo ""
echo "To stop:"
echo "  docker-compose -f docker-compose.monitoring.yml down"


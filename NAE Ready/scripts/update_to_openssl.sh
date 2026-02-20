#!/bin/bash
# Update everything to use OpenSSL Python once it's available
# This script checks for OpenSSL Python and updates the monitor

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔍 Checking for Python with OpenSSL..."
echo

# Find Python with OpenSSL
PYTHON_OPENSSL=""
for py in \
    "$(brew --prefix python@3.11 2>/dev/null)/bin/python3.11" \
    "/opt/homebrew/bin/python3.11" \
    "/usr/local/bin/python3.11" \
    "$(brew --prefix python@3.14 2>/dev/null)/bin/python3.14" \
    "/opt/homebrew/bin/python3.14" \
    "/usr/local/bin/python3.14" \
    "$(brew --prefix python@3.12 2>/dev/null)/bin/python3.12" \
    "/opt/homebrew/bin/python3.12" \
    "/usr/local/bin/python3.12"; do
    
    if [ -f "$py" ]; then
        SSL_VERSION=$("$py" -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null)
        if [[ "$SSL_VERSION" == *"OpenSSL"* ]]; then
            PYTHON_OPENSSL="$py"
            echo "✅ Found Python with OpenSSL:"
            echo "   Path: $py"
            echo "   SSL: $SSL_VERSION"
            "$py" --version
            break
        fi
    fi
done

if [ -z "$PYTHON_OPENSSL" ]; then
    echo "❌ No Python with OpenSSL found"
    echo
    echo "💡 Install Python with OpenSSL:"
    echo "   brew install python@3.11"
    echo
    echo "   Or wait for current installation to complete"
    exit 1
fi

echo
echo "🔄 Updating monitor to use OpenSSL Python..."

# Stop current monitor
cd "$NAE_DIR"
bash scripts/stop_etrade_monitor.sh >/dev/null 2>&1

# Restart with OpenSSL Python
echo "🚀 Restarting monitor..."
nohup "$PYTHON_OPENSSL" scripts/monitor_etrade_status.py --interval 60 > logs/etrade_monitor.log 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > logs/etrade_monitor.pid

sleep 2

# Verify
if ps -p $MONITOR_PID > /dev/null 2>&1; then
    echo "✅ Monitor restarted with OpenSSL Python"
    echo "   PID: $MONITOR_PID"
    echo "   Using: $PYTHON_OPENSSL"
    echo
    echo "📋 Verify OpenSSL in use:"
    echo "   bash scripts/python_openssl.sh -c \"import ssl; print(ssl.OPENSSL_VERSION)\""
else
    echo "❌ Failed to restart monitor"
    exit 1
fi

echo
echo "✅ Update complete! All scripts now use Python with OpenSSL"



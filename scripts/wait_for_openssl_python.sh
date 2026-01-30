#!/bin/bash
# Wait for Python with OpenSSL to become available, then restart monitor

echo "⏳ Waiting for Python with OpenSSL to be installed..."
echo "   (This happens automatically when brew install completes)"

CHECK_COUNT=0
MAX_CHECKS=60  # Check for up to 60 minutes

while [ $CHECK_COUNT -lt $MAX_CHECKS ]; do
    # Try to find Python with OpenSSL
    PYTHON_OPENSSL=$(bash "$(dirname "$0")/python_openssl.sh" -c "import ssl; print('OpenSSL' if 'OpenSSL' in ssl.OPENSSL_VERSION else 'LibreSSL')" 2>&1 | grep -i openssl)
    
    if [[ "$PYTHON_OPENSSL" == *"OpenSSL"* ]]; then
        echo "✅ Python with OpenSSL found!"
        
        # Stop current monitor if running
        bash "$(dirname "$0")/stop_etrade_monitor.sh" 2>/dev/null
        
        # Restart with OpenSSL Python
        echo "🚀 Restarting monitor with OpenSSL Python..."
        bash "$(dirname "$0")/start_etrade_monitor.sh"
        
        exit 0
    fi
    
    CHECK_COUNT=$((CHECK_COUNT + 1))
    
    if [ $((CHECK_COUNT % 5)) -eq 0 ]; then
        echo "   Checked $CHECK_COUNT times... (still waiting)"
    fi
    
    sleep 60  # Check every minute
done

echo "❌ Timeout: Python with OpenSSL not found after $MAX_CHECKS minutes"
echo "   Please install manually: brew install python@3.11"
exit 1



#!/bin/bash
# Find Python that uses OpenSSL (not LibreSSL)

echo "🔍 Searching for Python with OpenSSL..."
echo

# Check common locations
PYTHON_LOCATIONS=(
    "$(brew --prefix python@3.14)/bin/python3.14"
    "$(brew --prefix python@3.11)/bin/python3.11"
    "$(brew --prefix python@3.12)/bin/python3.12"
    "/opt/homebrew/bin/python3.14"
    "/opt/homebrew/bin/python3.11"
    "/opt/homebrew/bin/python3.12"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3.14"
    "/usr/local/bin/python3.11"
    "/usr/local/bin/python3.12"
    "/usr/local/bin/python3"
)

FOUND=""

for PYTHON in "${PYTHON_LOCATIONS[@]}"; do
    if [ -f "$PYTHON" ]; then
        SSL_VERSION=$("$PYTHON" -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null)
        
        if [[ "$SSL_VERSION" == *"OpenSSL"* ]]; then
            echo "✅ FOUND: $PYTHON"
            echo "   SSL Library: $SSL_VERSION"
            echo "   Version: $("$PYTHON" --version 2>&1)"
            FOUND="$PYTHON"
            break
        fi
    fi
done

if [ -z "$FOUND" ]; then
    echo "❌ No Python with OpenSSL found"
    echo
    echo "💡 To install Python with OpenSSL:"
    echo "   brew install python@3.11"
    echo "   (or wait for python@3.14 to finish installing)"
    exit 1
else
    echo
    echo "📋 Use this Python: $FOUND"
    exit 0
fi



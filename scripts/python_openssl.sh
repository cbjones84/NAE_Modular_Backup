#!/bin/bash
# Wrapper script to use Python with OpenSSL (not LibreSSL)
# This finds and uses a Python installation that uses OpenSSL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Try to find Python with OpenSSL
PYTHON_OPENSSL=""

# Method 1: Use the Python finder script
if [ -f "$SCRIPT_DIR/get_python_openssl.py" ]; then
    # Use any available Python to find the OpenSSL one
    PYTHON_OPENSSL=$(python3 "$SCRIPT_DIR/get_python_openssl.py" 2>/dev/null)
fi

# Method 2: Check common Homebrew locations (prioritize newest versions)
if [ -z "$PYTHON_OPENSSL" ] || [ "$PYTHON_OPENSSL" = "python3" ]; then
    for py in \
        "$(brew --prefix python@3.14 2>/dev/null)/bin/python3.14" \
        "/usr/local/opt/python@3.14/bin/python3.14" \
        "/opt/homebrew/bin/python3.14" \
        "/usr/local/bin/python3.14" \
        "$(brew --prefix python@3.11 2>/dev/null)/bin/python3.11" \
        "/usr/local/opt/python@3.11/bin/python3.11" \
        "/opt/homebrew/bin/python3.11" \
        "/usr/local/bin/python3.11" \
        "$(brew --prefix python@3.12 2>/dev/null)/bin/python3.12" \
        "/opt/homebrew/bin/python3.12" \
        "/usr/local/bin/python3.12"; do
        
        if [ -f "$py" ]; then
            SSL_VERSION=$("$py" -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null)
            if [[ "$SSL_VERSION" == *"OpenSSL"* ]]; then
                PYTHON_OPENSSL="$py"
                break
            fi
        fi
    done
fi

# Fallback to regular python3 if nothing found
if [ -z "$PYTHON_OPENSSL" ] || [ ! -f "$PYTHON_OPENSSL" ]; then
    PYTHON_OPENSSL="python3"
    echo "⚠️  Warning: Could not find Python with OpenSSL, using: $PYTHON_OPENSSL" >&2
    echo "   Install with: brew install python@3.11" >&2
fi

# Execute with all arguments
exec "$PYTHON_OPENSSL" "$@"


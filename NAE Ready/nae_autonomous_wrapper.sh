#!/bin/bash
# NAE Autonomous Wrapper Script
# Runs from its own directory so it works regardless of install location

cd "$(dirname "$0")"

# Load .env files (later files override; .env.prod has TRADIER credentials)
for envfile in .env ../.env .env.prod ../.env.prod ../NAE/.env.prod; do
    if [ -f "$envfile" ]; then
        export $(grep -v '^#' "$envfile" | grep -v '^$' | xargs) 2>/dev/null || true
    fi
done

# Activate virtual environment if it exists
if [ -d "venv_python311" ]; then
    source venv_python311/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the autonomous master (use python3 from PATH)
exec python3 nae_autonomous_master.py


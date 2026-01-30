#!/bin/bash
# Initialize Safety Systems for NAE

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "NAE Safety Systems Initialization"
echo "=========================================="
echo ""

# Load environment
if [ -f "$NAE_ROOT/.env" ]; then
    source "$NAE_ROOT/.env"
else
    echo "❌ .env file not found. Run configure_environments.sh first."
    exit 1
fi

# Create safety directories
mkdir -p "$NAE_ROOT/safety"
mkdir -p "$NAE_ROOT/safety/locks"
mkdir -p "$NAE_ROOT/safety/checks"
mkdir -p "$NAE_ROOT/safety/logs"

echo "✅ Created safety directories"
echo ""

# Create production lock file (HP only)
if [ "$PRODUCTION" == "true" ]; then
    LOCK_FILE="$NAE_ROOT/safety/locks/production.lock"
    cat > "$LOCK_FILE" << EOF
# Production Lock File
# This file must exist for live trading to be enabled
# Created: $(date)
# Machine: $NAE_MACHINE_NAME
# Device ID: $NAE_DEVICE_ID
# Branch: $NAE_BRANCH

LOCKED=true
CREATED_AT=$(date +%s)
MACHINE_TYPE=$NAE_MACHINE_TYPE
DEVICE_ID=$NAE_DEVICE_ID
EOF
    echo "✅ Created production lock file"
else
    echo "ℹ️  Production lock file not needed (dev machine)"
fi

# Create device ID file
DEVICE_ID_FILE="$NAE_ROOT/safety/device_id.txt"
echo "$NAE_DEVICE_ID" > "$DEVICE_ID_FILE"
chmod 600 "$DEVICE_ID_FILE"
echo "✅ Created device ID file"

# Create branch check file
BRANCH_CHECK_FILE="$NAE_ROOT/safety/checks/branch_check.py"
cat > "$BRANCH_CHECK_FILE" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Branch check for production safety"""

import os
import sys
import subprocess

def get_current_branch():
    """Get current git branch"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None

def check_branch():
    """Check if branch is correct for production"""
    production = os.getenv('PRODUCTION', 'false').lower() == 'true'
    required_branch = os.getenv('NAE_GIT_BRANCH', 'main')
    current_branch = get_current_branch()
    
    if production and current_branch != required_branch:
        print(f"❌ ERROR: Production mode requires branch '{required_branch}', but current branch is '{current_branch}'")
        return False
    
    return True

if __name__ == '__main__':
    if not check_branch():
        sys.exit(1)
PYTHON_EOF

chmod +x "$BRANCH_CHECK_FILE"
echo "✅ Created branch check script"

# Create anti-double-trade system
ANTI_DOUBLE_TRADE_FILE="$NAE_ROOT/safety/anti_double_trade.py"
cat > "$ANTI_DOUBLE_TRADE_FILE" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Anti-double-trade system"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class AntiDoubleTrade:
    """Prevents double trades across devices"""
    
    def __init__(self):
        self.safety_dir = Path(__file__).parent.parent / "safety"
        self.trade_log_file = self.safety_dir / "logs" / "trade_log.json"
        self.device_id = os.getenv("NAE_DEVICE_ID", "")
        self.time_window = int(os.getenv("NAE_TRADE_TIME_WINDOW", "60"))  # seconds
        
        # Ensure log file exists
        self.trade_log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.trade_log_file.exists():
            with open(self.trade_log_file, 'w') as f:
                json.dump({"trades": []}, f)
    
    def check_trade_allowed(self, trade_nonce: str) -> tuple[bool, str]:
        """Check if trade is allowed"""
        # Load trade log
        with open(self.trade_log_file, 'r') as f:
            log_data = json.load(f)
        
        trades = log_data.get("trades", [])
        current_time = time.time()
        
        # Check for recent trades
        recent_trades = [
            t for t in trades
            if current_time - t.get("timestamp", 0) < self.time_window
        ]
        
        # Check for duplicate nonce
        for trade in trades:
            if trade.get("nonce") == trade_nonce:
                return False, f"Duplicate trade nonce: {trade_nonce}"
        
        # Check for trades from different device
        for trade in recent_trades:
            if trade.get("device_id") != self.device_id:
                return False, f"Recent trade from different device: {trade.get('device_id')}"
        
        return True, "Trade allowed"
    
    def log_trade(self, trade_nonce: str, trade_details: dict):
        """Log a trade"""
        with open(self.trade_log_file, 'r') as f:
            log_data = json.load(f)
        
        trade_entry = {
            "nonce": trade_nonce,
            "device_id": self.device_id,
            "timestamp": time.time(),
            "details": trade_details
        }
        
        log_data["trades"].append(trade_entry)
        
        # Keep only last 1000 trades
        log_data["trades"] = log_data["trades"][-1000:]
        
        with open(self.trade_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def generate_nonce(self) -> str:
        """Generate a unique trade nonce"""
        import hashlib
        import secrets
        
        nonce_data = f"{self.device_id}:{time.time()}:{secrets.token_hex(16)}"
        return hashlib.sha256(nonce_data.encode()).hexdigest()[:32]

if __name__ == '__main__':
    anti_double = AntiDoubleTrade()
    nonce = anti_double.generate_nonce()
    allowed, message = anti_double.check_trade_allowed(nonce)
    print(f"Trade allowed: {allowed}")
    print(f"Message: {message}")
    print(f"Nonce: {nonce}")
PYTHON_EOF

chmod +x "$ANTI_DOUBLE_TRADE_FILE"
echo "✅ Created anti-double-trade system"

# Create production safety check
SAFETY_CHECK_FILE="$NAE_ROOT/safety/production_safety_check.py"
cat > "$SAFETY_CHECK_FILE" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Production safety check"""

import os
import sys
from pathlib import Path

def check_production_safety():
    """Check all production safety requirements"""
    errors = []
    warnings = []
    
    # Check production mode
    production = os.getenv('PRODUCTION', 'false').lower() == 'true'
    
    if production:
        # Check branch
        from safety.checks.branch_check import check_branch
        if not check_branch():
            errors.append("Branch check failed")
        
        # Check production lock file
        lock_file = Path(__file__).parent.parent / "safety" / "locks" / "production.lock"
        if not lock_file.exists():
            errors.append("Production lock file not found")
        
        # Check device ID
        device_id = os.getenv('NAE_DEVICE_ID', '')
        if not device_id:
            errors.append("Device ID not set")
        
        # Check safety enabled
        safety_enabled = os.getenv('NAE_SAFETY_ENABLED', 'false').lower() == 'true'
        if not safety_enabled:
            warnings.append("Safety systems not enabled")
    else:
        print("ℹ️  Development mode - safety checks relaxed")
    
    # Report results
    if errors:
        print("❌ Safety check FAILED:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("⚠️  Safety warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    if not errors and not warnings:
        print("✅ All safety checks passed")
    
    return len(errors) == 0

if __name__ == '__main__':
    if not check_production_safety():
        sys.exit(1)
PYTHON_EOF

chmod +x "$SAFETY_CHECK_FILE"
echo "✅ Created production safety check"

echo ""
echo "=========================================="
echo "✅ Safety systems initialized!"
echo "=========================================="
echo ""
echo "Safety systems created:"
echo "  - Production lock file"
echo "  - Device ID tracking"
echo "  - Branch verification"
echo "  - Anti-double-trade system"
echo "  - Production safety checks"
echo ""
echo "Next steps:"
echo "1. Test safety systems: python3 safety/production_safety_check.py"
echo "2. Set up Git branches (see dual_machine_setup.md)"
echo "3. Configure Master-Node communication (see master_node_setup.md)"
echo ""


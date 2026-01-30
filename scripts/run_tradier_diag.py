#!/usr/bin/env python3
"""
Quick script to run Tradier diagnostics
"""

import os
import sys

# Add NAE root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, nae_root)

from execution.diagnostics.nae_tradier_diagnostics import TradierDiagnostics

# Get credentials from environment
API_KEY = os.getenv("TRADIER_API_KEY")
ACCOUNT_ID = os.getenv("TRADIER_ACCOUNT_ID")

if not API_KEY:
    print("❌ ERROR: TRADIER_API_KEY not found in environment")
    print("   Set it with: export TRADIER_API_KEY='your_key'")
    sys.exit(1)

# Create diagnostics instance
diag = TradierDiagnostics(
    api_key=API_KEY,
    account_id=ACCOUNT_ID,
    live=True  # Set False if you want to test sandbox
)

# Run diagnostics
diag.run_full_diagnostics()


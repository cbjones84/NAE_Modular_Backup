#!/usr/bin/env python3
"""
Optimus Launcher - Runs Optimus agent with accelerator from NAE Ready

This script is started by nae_autonomous_master to ensure the Optimus trading agent
(including the micro-scalp accelerator strategy) runs in production.
"""

import os
import sys

# Load environment variables from .env.prod (ensure Tradier credentials are available)
def _load_env_file(filepath: str) -> bool:
    """Load environment variables from a file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        return True
    return False

# Load from NAE/.env.prod first
_script_dir = os.path.dirname(os.path.abspath(__file__))
_nae_dir = os.path.dirname(_script_dir)
_load_env_file(os.path.join(_nae_dir, ".env.prod")) or _load_env_file(os.path.join(_nae_dir, ".env"))

# Resolve NAE Ready path (sibling of NAE/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_nae_dir = os.path.dirname(_script_dir)
_nae_ready = os.path.join(os.path.dirname(_nae_dir), "NAE Ready")

if os.path.isdir(_nae_ready):
    sys.path.insert(0, _nae_ready)
    os.chdir(_nae_ready)
else:
    # Fallback: try workspace root
    _workspace = os.path.dirname(_nae_dir)
    _nae_ready_alt = os.path.join(_workspace, "NAE Ready")
    if os.path.isdir(_nae_ready_alt):
        sys.path.insert(0, _nae_ready_alt)
        os.chdir(_nae_ready_alt)
    else:
        sys.stderr.write(f"ERROR: NAE Ready directory not found at {_nae_ready} or {_nae_ready_alt}\n")
        sys.exit(1)

# Run Optimus main loop (includes day trading + accelerator cycles)
if __name__ == "__main__":
    from agents.optimus import optimus_main_loop
    optimus_main_loop()

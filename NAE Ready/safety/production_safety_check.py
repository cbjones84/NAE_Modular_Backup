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

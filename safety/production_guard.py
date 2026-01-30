#!/usr/bin/env python3
"""
Production Guard

Enforces production safety requirements before allowing live trading.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List
import subprocess
import json

class ProductionGuard:
    """Guards against unsafe production operations"""
    
    def __init__(self):
        self.nae_root = Path(__file__).parent.parent
        self.safety_dir = self.nae_root / "safety"
        self.lock_file = self.safety_dir / "locks" / "production.lock"
        
        # Load environment
        self.production = os.getenv('PRODUCTION', 'false').lower() == 'true'
        self.device_id = os.getenv('NAE_DEVICE_ID', '')
        self.required_branch = os.getenv('NAE_GIT_BRANCH', 'main')
        self.safety_enabled = os.getenv('NAE_SAFETY_ENABLED', 'true').lower() == 'true'
    
    def check_all(self) -> Tuple[bool, List[str]]:
        """Run all safety checks"""
        errors = []
        
        if not self.safety_enabled:
            return True, []  # Safety disabled
        
        # Check production mode
        if self.production:
            # Check branch
            branch_ok, branch_error = self._check_branch()
            if not branch_ok:
                errors.append(branch_error)
            
            # Check lock file
            lock_ok, lock_error = self._check_lock_file()
            if not lock_ok:
                errors.append(lock_error)
            
            # Check device ID
            device_ok, device_error = self._check_device_id()
            if not device_ok:
                errors.append(device_error)
        
        return len(errors) == 0, errors
    
    def _check_branch(self) -> Tuple[bool, str]:
        """Check if current branch matches required branch"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.nae_root
            )
            current_branch = result.stdout.strip()
            
            if current_branch != self.required_branch:
                return False, f"Branch mismatch: required '{self.required_branch}', current '{current_branch}'"
            
            return True, ""
        except Exception as e:
            return False, f"Branch check failed: {e}"
    
    def _check_lock_file(self) -> Tuple[bool, str]:
        """Check if production lock file exists"""
        if not self.lock_file.exists():
            return False, "Production lock file not found"
        
        # Verify lock file content
        try:
            with open(self.lock_file, 'r') as f:
                content = f.read()
                if 'LOCKED=true' not in content:
                    return False, "Production lock file invalid"
        except Exception as e:
            return False, f"Error reading lock file: {e}"
        
        return True, ""
    
    def _check_device_id(self) -> Tuple[bool, str]:
        """Check if device ID is set"""
        if not self.device_id:
            return False, "Device ID not set"
        
        # Verify device ID file
        device_id_file = self.safety_dir / "device_id.txt"
        if device_id_file.exists():
            with open(device_id_file, 'r') as f:
                stored_id = f.read().strip()
                if stored_id != self.device_id:
                    return False, "Device ID mismatch"
        
        return True, ""
    
    def allow_live_trading(self) -> bool:
        """Check if live trading is allowed"""
        if not self.production:
            return False  # Only production allows live trading
        
        ok, errors = self.check_all()
        if not ok:
            return False
        
        # Additional check: live trading must be explicitly enabled
        live_trading_enabled = os.getenv('NAE_LIVE_TRADING_ENABLED', 'false').lower() == 'true'
        return live_trading_enabled
    
    def get_status(self) -> dict:
        """Get safety status"""
        ok, errors = self.check_all()
        
        return {
            "production_mode": self.production,
            "safety_enabled": self.safety_enabled,
            "device_id": self.device_id,
            "required_branch": self.required_branch,
            "all_checks_passed": ok,
            "errors": errors,
            "live_trading_allowed": self.allow_live_trading()
        }


def require_production_safety(func):
    """Decorator to require production safety checks"""
    def wrapper(*args, **kwargs):
        guard = ProductionGuard()
        ok, errors = guard.check_all()
        
        if not ok:
            print("❌ Production safety check failed:")
            for error in errors:
                print(f"   - {error}")
            raise RuntimeError("Production safety check failed")
        
        return func(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    guard = ProductionGuard()
    status = guard.get_status()
    
    print("Production Guard Status:")
    print(json.dumps(status, indent=2))
    
    if not status["all_checks_passed"]:
        sys.exit(1)


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

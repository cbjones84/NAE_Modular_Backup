#!/usr/bin/env python3
"""
Find and return path to Python that uses OpenSSL (not LibreSSL)
This script itself uses whatever Python is available, but finds the OpenSSL one
"""

import sys
import os
import subprocess

def find_python_with_openssl():
    """Find Python executable that uses OpenSSL"""
    
    # Check common locations
    locations = []
    
    # Try brew Python versions
    try:
        brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
        locations.extend([
            f"{brew_prefix}/bin/python3.14",
            f"{brew_prefix}/bin/python3.11",
            f"{brew_prefix}/bin/python3.12",
            f"{brew_prefix}/bin/python3",
        ])
    except:
        pass
    
    # Check standard Homebrew locations
    locations.extend([
        "/opt/homebrew/bin/python3.14",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3.14",
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3",
    ])
    
    # Check each location
    for python_path in locations:
        if os.path.exists(python_path):
            try:
                result = subprocess.run(
                    [python_path, '-c', 'import ssl; print(ssl.OPENSSL_VERSION)'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    ssl_version = result.stdout.strip()
                    if 'OpenSSL' in ssl_version:
                        # Get Python version
                        version_result = subprocess.run(
                            [python_path, '--version'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
                        
                        return {
                            "path": python_path,
                            "ssl_version": ssl_version,
                            "python_version": version
                        }
            except:
                continue
    
    return None


if __name__ == "__main__":
    result = find_python_with_openssl()
    
    if result:
        print(result["path"])
        sys.exit(0)
    else:
        print("python3", file=sys.stderr)  # Fallback
        sys.exit(1)



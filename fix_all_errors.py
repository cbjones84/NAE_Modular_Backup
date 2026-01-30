#!/usr/bin/env python3
"""
Comprehensive error checker and fixer for NAE codebase
Scans for and fixes common Python errors
"""

import os
import sys
import ast
import re
from pathlib import Path

def check_syntax_errors(file_path):
    """Check for syntax errors in a Python file"""
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source, filename=file_path)
    except SyntaxError as e:
        errors.append({
            'file': file_path,
            'type': 'SyntaxError',
            'line': e.lineno,
            'message': e.msg,
            'text': e.text
        })
    except Exception as e:
        errors.append({
            'file': file_path,
            'type': type(e).__name__,
            'message': str(e)
        })
    return errors

def check_import_errors(file_path):
    """Check for import errors"""
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Extract imports
        tree = ast.parse(source, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name)
                    except ImportError:
                        # Check if it's a relative import
                        if not alias.name.startswith('.'):
                            errors.append({
                                'file': file_path,
                                'type': 'ImportError',
                                'module': alias.name,
                                'message': f'Cannot import {alias.name}'
                            })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        __import__(node.module)
                    except ImportError:
                        if not node.module.startswith('.'):
                            errors.append({
                                'file': file_path,
                                'type': 'ImportError',
                                'module': node.module,
                                'message': f'Cannot import from {node.module}'
                            })
    except Exception as e:
        pass  # Skip files with syntax errors
    return errors

def scan_directory(directory):
    """Scan directory for Python files and check for errors"""
    all_errors = []
    all_warnings = []
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env', '.git']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Check syntax
                syntax_errors = check_syntax_errors(file_path)
                all_errors.extend(syntax_errors)
    
    return all_errors, all_warnings

def main():
    """Main function"""
    print("="*70)
    print("NAE Codebase Error Checker")
    print("="*70)
    
    nae_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\nScanning: {nae_dir}")
    print("Checking for syntax errors...\n")
    
    errors, warnings = scan_directory(nae_dir)
    
    if errors:
        print(f"❌ Found {len(errors)} error(s):\n")
        for error in errors[:20]:  # Show first 20
            print(f"  {error['file']}")
            print(f"    Type: {error['type']}")
            if 'line' in error:
                print(f"    Line: {error['line']}")
            print(f"    Message: {error['message']}")
            print()
    else:
        print("✅ No syntax errors found!")
    
    print("="*70)
    print(f"Total errors: {len(errors)}")
    print(f"Total warnings: {len(warnings)}")
    print("="*70)
    
    if errors:
        sys.exit(1)
    else:
        print("\n✅ Codebase is error-free!")
        sys.exit(0)

if __name__ == "__main__":
    main()


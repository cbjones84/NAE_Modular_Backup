#!/usr/bin/env python3
"""
HP Push Update Script
Run this on HP (Dev) to push changes and update Mac (Prod)

Usage:
    python scripts/hp_push_update.py "commit message"
    python scripts/hp_push_update.py "commit message" --restart
"""

import os
import sys
import subprocess
import requests
import argparse
from datetime import datetime

# Configuration
MAC_IP = os.environ.get('NAE_MASTER_URL', 'http://192.168.132.36').replace(':8080', '')
UPDATE_PORT = 8081
API_KEY = os.environ.get('NAE_NODE_API_KEY', 'a07e9de261c6eb815fbcd9cb6263f0862534af1cd3cc3540c87ed70ce0e4438d')

def run_command(cmd, cwd=None):
    """Run a command and return result"""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()

def git_push(message):
    """Commit and push changes to GitHub"""
    print("\n📦 Committing and pushing changes...")
    
    # Add all changes
    success, out, err = run_command(['git', 'add', '.'])
    if not success:
        print(f"   ⚠️  git add warning: {err}")
    
    # Commit
    success, out, err = run_command(['git', 'commit', '-m', message])
    if not success and 'nothing to commit' not in err:
        print(f"   ❌ Commit failed: {err}")
        return False
    elif 'nothing to commit' in out or 'nothing to commit' in err:
        print("   ℹ️  No changes to commit")
    else:
        print(f"   ✅ Committed: {message}")
    
    # Push to dev
    success, out, err = run_command(['git', 'push', 'origin', 'dev'])
    if not success:
        print(f"   ❌ Push to dev failed: {err}")
        return False
    print("   ✅ Pushed to dev branch")
    
    # Merge to main
    run_command(['git', 'checkout', 'main'])
    success, out, err = run_command(['git', 'merge', 'dev'])
    if not success:
        print(f"   ❌ Merge to main failed: {err}")
        run_command(['git', 'checkout', 'dev'])
        return False
    print("   ✅ Merged to main")
    
    # Push main
    success, out, err = run_command(['git', 'push', 'origin', 'main'])
    if not success:
        print(f"   ❌ Push to main failed: {err}")
    else:
        print("   ✅ Pushed to main branch")
    
    # Switch back to dev
    run_command(['git', 'checkout', 'dev'])
    
    return True

def trigger_mac_pull(restart=False):
    """Trigger Mac to pull updates"""
    endpoint = 'restart' if restart else 'pull'
    url = f"{MAC_IP}:{UPDATE_PORT}/api/update/{endpoint}"
    headers = {'X-API-Key': API_KEY, 'Content-Type': 'application/json'}
    
    print(f"\n🍎 Triggering Mac to {'pull and restart' if restart else 'pull updates'}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.post(url, headers=headers, json={}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Mac updated successfully!")
            if 'current_commit' in data:
                print(f"   📌 Mac now at: {data.get('current_commit', 'unknown')}")
            return True
        elif response.status_code == 401:
            print("   ❌ Unauthorized - check API key")
            return False
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Could not connect to Mac updater service at {url}")
        print("   💡 Make sure mac_auto_updater.py is running on Mac")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_mac_status():
    """Check Mac updater service status"""
    url = f"{MAC_IP}:{UPDATE_PORT}/api/update/health"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Push updates from HP (Dev) to Mac (Prod)')
    parser.add_argument('message', nargs='?', default='Update from HP dev', help='Commit message')
    parser.add_argument('--restart', action='store_true', help='Restart NAE on Mac after pull')
    parser.add_argument('--pull-only', action='store_true', help='Only trigger Mac pull (no git push)')
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           HP → Mac Update Script                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check Mac connection
    print("🔍 Checking Mac updater service...")
    if check_mac_status():
        print("   ✅ Mac updater service is running")
    else:
        print("   ⚠️  Mac updater service not reachable")
        print(f"   💡 Start it on Mac: python3 scripts/mac_auto_updater.py")
        if not args.pull_only:
            print("   Continuing with git push anyway...")
    
    # Git push (unless pull-only)
    if not args.pull_only:
        if not git_push(args.message):
            print("\n❌ Git push failed")
            sys.exit(1)
    
    # Trigger Mac pull
    if check_mac_status():
        trigger_mac_pull(restart=args.restart)
    
    print(f"\n✅ Update complete! ({datetime.now().strftime('%H:%M:%S')})")

if __name__ == '__main__':
    main()



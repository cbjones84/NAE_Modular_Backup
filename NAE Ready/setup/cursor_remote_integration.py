#!/usr/bin/env python3
"""
Cursor Remote Integration

Enables Cursor to execute commands on HP OmniBook X (prod) from Mac (dev).
Provides seamless integration for dual-machine workflow with automatic updates.
"""

import os
import sys
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Optional: watchdog for auto-sync (install with: pip install watchdog)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

# Add NAE paths
script_dir = Path(__file__).parent
nae_root = script_dir.parent
sys.path.insert(0, str(nae_root))

from setup.remote_execution_bridge import RemoteExecutionBridge


class GitChangeHandler(FileSystemEventHandler):
    """Watch for git changes and trigger sync"""
    
    def __init__(self, integration):
        self.integration = integration
        self.last_sync = time.time()
        self.sync_cooldown = 30  # Minimum seconds between syncs
        self.changes_detected = False
    
    def on_modified(self, event):
        """Detect file modifications"""
        if event.is_directory:
            return
        
        # Ignore log files, cache, etc.
        ignored_patterns = ['.log', '.pyc', '__pycache__', '.git', 'node_modules']
        if any(pattern in str(event.src_path) for pattern in ignored_patterns):
            return
        
        # Check if it's a code file
        code_extensions = ['.py', '.json', '.md', '.yml', '.yaml', '.sh', '.txt']
        if any(event.src_path.endswith(ext) for ext in code_extensions):
            self.changes_detected = True
            current_time = time.time()
            
            # Debounce: wait for more changes, then sync
            if current_time - self.last_sync > self.sync_cooldown:
                self._schedule_sync()
    
    def _schedule_sync(self):
        """Schedule a sync after a short delay"""
        def delayed_sync():
            time.sleep(5)  # Wait 5 seconds for file to finish saving
            if self.changes_detected:
                print(f"\n🔄 Changes detected - syncing to HP OmniBook X...")
                self.integration.auto_sync_to_prod()
                self.changes_detected = False
                self.last_sync = time.time()
        
        threading.Thread(target=delayed_sync, daemon=True).start()


class CursorRemoteIntegration:
    """Integration for Cursor to control HP OmniBook X"""
    
    def __init__(self, auto_sync_enabled: bool = False):
        """Initialize Cursor remote integration"""
        self.bridge = RemoteExecutionBridge()
        self.config_file = nae_root / "config" / "remote_config.json"
        self.auto_sync_enabled = auto_sync_enabled
        self.observer = None
        self.watch_thread = None
        
        if auto_sync_enabled:
            self._start_auto_sync()
    
    def execute_on_prod(self, command: str, description: str = "") -> Dict[str, Any]:
        """Execute command on production (HP OmniBook X)"""
        print(f"🚀 Executing on HP OmniBook X (prod): {description or command}")
        
        result = self.bridge.execute_remote(command)
        
        if result.get("success"):
            print(f"✅ Success: {description or command}")
            if result.get("output"):
                print(f"Output:\n{result['output']}")
        else:
            print(f"❌ Failed: {description or command}")
            if result.get("error"):
                print(f"Error: {result['error']}")
        
        return result
    
    def sync_to_prod(self, branch: str = "prod") -> bool:
        """Sync changes from Mac (dev) to HP (prod)"""
        print(f"🔄 Syncing changes to HP OmniBook X (branch: {branch})...")
        
        try:
            # Check current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=nae_root
            )
            current_branch = branch_result.stdout.strip()
            
            # Stage any uncommitted changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=nae_root,
                capture_output=True
            )
            
            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=nae_root
            )
            
            has_changes = bool(status_result.stdout.strip())
            
            if has_changes:
                # Commit changes with timestamp
                commit_msg = f"Auto-sync: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                commit_result = subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    capture_output=True,
                    text=True,
                    cwd=nae_root
                )
                if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
                    print(f"⚠️  Commit warning: {commit_result.stdout}")
            
            # Push to GitHub (current branch and target branch)
            push_commands = [
                ["git", "push", "origin", current_branch],
            ]
            if branch != current_branch:
                push_commands.append(["git", "push", "origin", f"{current_branch}:{branch}"])
            
            for push_cmd in push_commands:
                result = subprocess.run(
                    push_cmd,
                    capture_output=True,
                    text=True,
                    cwd=nae_root
                )
                
                if result.returncode != 0:
                    print(f"⚠️  Git push warning: {result.stderr}")
                    # Continue anyway - may already be up to date
            
            # Pull on HP OmniBook X
            pull_cmd = f"cd {nae_root} && git pull origin {branch}"
            pull_result = self.bridge.execute_remote(pull_cmd)
            
            if pull_result.get("success"):
                print(f"✅ Changes synced to HP OmniBook X (branch: {branch})")
                return True
            else:
                print(f"⚠️  Pull on HP may have issues: {pull_result.get('error', 'Unknown')}")
                # Still return True if push succeeded - pull can be done manually
                return True
                
        except Exception as e:
            print(f"❌ Sync error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def auto_sync_to_prod(self) -> bool:
        """Auto-sync with minimal output (for background use)"""
        try:
            # Quick sync - commit, push, pull
            subprocess.run(["git", "add", "-A"], cwd=nae_root, capture_output=True)
            
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=nae_root
            )
            
            if status.stdout.strip():
                subprocess.run(
                    ["git", "commit", "-m", f"Auto-update: {time.strftime('%H:%M:%S')}"],
                    cwd=nae_root,
                    capture_output=True
                )
            
            # Push to prod branch
            subprocess.run(
                ["git", "push", "origin", "prod"],
                cwd=nae_root,
                capture_output=True
            )
            
            # Pull on HP
            self.bridge.execute_remote(f"cd {nae_root} && git pull origin prod")
            
            return True
        except:
            return False
    
    def _start_auto_sync(self):
        """Start watching for file changes and auto-syncing"""
        if not WATCHDOG_AVAILABLE:
            print("⚠️  Watchdog not installed - auto-sync disabled")
            print("   Install with: pip install watchdog")
            return
        
        print("🔍 Starting auto-sync watcher for HP OmniBook X...")
        
        event_handler = GitChangeHandler(self)
        self.observer = Observer()
        
        # Watch NAE Ready directory (exclude large dirs)
        watch_path = nae_root / "NAE Ready"
        self.observer.schedule(
            event_handler,
            str(watch_path),
            recursive=True
        )
        
        self.observer.start()
        print("✅ Auto-sync watcher started")
    
    def stop_auto_sync(self):
        """Stop auto-sync watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("🛑 Auto-sync watcher stopped")
    
    def get_prod_status(self) -> Dict[str, Any]:
        """Get production status"""
        print("📊 Getting HP OmniBook X status...")
        status = self.bridge.get_remote_status()
        
        print("\n=== HP OmniBook X Status ===")
        print(f"Branch: {status.get('branch', 'unknown')}")
        print(f"Production Mode: {status.get('production_mode', 'unknown')}")
        print(f"NAE Running: {status.get('nae_running', 'unknown')}")
        print(f"Uptime: {status.get('uptime', 'unknown')}")
        
        return status
    
    def start_prod_nae(self) -> bool:
        """Start NAE on production"""
        print("🚀 Starting NAE on HP OmniBook X...")
        result = self.bridge.start_nae_production()
        
        if result.get("success"):
            print("✅ NAE started on HP OmniBook X")
            return True
        else:
            print(f"❌ Failed to start NAE: {result.get('error', 'Unknown error')}")
            return False
    
    def stop_prod_nae(self) -> bool:
        """Stop NAE on production"""
        print("🛑 Stopping NAE on HP OmniBook X...")
        result = self.bridge.stop_nae_production()
        
        if result.get("success"):
            print("✅ NAE stopped on HP OmniBook X")
            return True
        else:
            print(f"⚠️  Stop command executed (may already be stopped)")
            return True
    
    def verify_connection(self) -> bool:
        """Verify connection to HP OmniBook X"""
        print("🔍 Verifying connection to HP OmniBook X...")
        
        result = self.bridge.execute_remote("echo 'Connection verified'")
        
        if result.get("success"):
            print("✅ Connection verified")
            return True
        else:
            print(f"❌ Connection failed: {result.get('error', 'Unknown error')}")
            print("\nTroubleshooting:")
            print("1. Check HP OmniBook X is powered on and connected to network")
            print("2. Verify SSH is enabled on HP")
            print("3. Check SSH key is copied to HP")
            print("4. Run: ./setup/configure_remote_connection.sh")
            return False


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cursor Remote Integration")
    parser.add_argument("--status", action="store_true", help="Get production status")
    parser.add_argument("--sync", action="store_true", help="Sync changes to production")
    parser.add_argument("--auto-sync", action="store_true", help="Start auto-sync watcher")
    parser.add_argument("--branch", type=str, default="prod", help="Target branch (default: prod)")
    parser.add_argument("--start", action="store_true", help="Start NAE on production")
    parser.add_argument("--stop", action="store_true", help="Stop NAE on production")
    parser.add_argument("--verify", action="store_true", help="Verify connection")
    parser.add_argument("--command", type=str, help="Execute command on production")
    
    args = parser.parse_args()
    
    integration = CursorRemoteIntegration(auto_sync_enabled=args.auto_sync)
    
    try:
        if args.verify:
            integration.verify_connection()
        elif args.status:
            integration.get_prod_status()
        elif args.sync:
            integration.sync_to_prod(branch=args.branch)
        elif args.auto_sync:
            print("✅ Auto-sync watcher running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                integration.stop_auto_sync()
        elif args.start:
            integration.start_prod_nae()
        elif args.stop:
            integration.stop_prod_nae()
        elif args.command:
            integration.execute_on_prod(args.command)
        else:
            parser.print_help()
    finally:
        if integration.observer:
            integration.stop_auto_sync()
        integration.bridge.close()


if __name__ == "__main__":
    main()


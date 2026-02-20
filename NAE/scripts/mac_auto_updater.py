#!/usr/bin/env python3
"""
Mac Auto-Updater Service
Automatically pulls updates from GitHub and manages NAE updates
Runs on port 8081
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get NAE root directory
NAE_ROOT = Path(__file__).parent.parent
CURRENT_BRANCH = "prod"  # Production branch


class AutoUpdater:
    """Handles automatic updates from GitHub"""
    
    def __init__(self, nae_root: Path, branch: str = "prod"):
        self.nae_root = nae_root
        self.branch = branch
        self.last_update_check = None
        self.last_update_time = None
        self.update_in_progress = False
        self.update_history = []
    
    def check_for_updates(self) -> dict:
        """Check if updates are available"""
        try:
            os.chdir(self.nae_root)
            
            # Fetch latest from remote
            result = subprocess.run(
                ['git', 'fetch', 'origin', self.branch],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Git fetch failed: {result.stderr}"
                }
            
            # Check if local is behind remote
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'HEAD..origin/{self.branch}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            self.last_update_check = datetime.now().isoformat()
            
            return {
                'success': True,
                'updates_available': commits_behind > 0,
                'commits_behind': commits_behind,
                'current_branch': self.branch,
                'last_check': self.last_update_check
            }
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def pull_updates(self, restart_nae: bool = False) -> dict:
        """Pull latest updates from GitHub"""
        if self.update_in_progress:
            return {
                'success': False,
                'error': 'Update already in progress'
            }
        
        self.update_in_progress = True
        
        try:
            os.chdir(self.nae_root)
            
            # Get current commit before update
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            old_commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Pull updates
            logger.info(f"Pulling updates from origin/{self.branch}...")
            result = subprocess.run(
                ['git', 'pull', 'origin', self.branch],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.update_in_progress = False
                return {
                    'success': False,
                    'error': f"Git pull failed: {result.stderr}"
                }
            
            # Get new commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            new_commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            self.last_update_time = datetime.now().isoformat()
            
            update_info = {
                'success': True,
                'old_commit': old_commit[:8],
                'new_commit': new_commit[:8],
                'updated': old_commit != new_commit,
                'timestamp': self.last_update_time,
                'output': result.stdout
            }
            
            # Restart NAE if requested and updated
            if restart_nae and update_info['updated']:
                logger.info("Restarting NAE after update...")
                restart_result = self._restart_nae()
                update_info['nae_restarted'] = restart_result.get('success', False)
            
            self.update_history.append(update_info)
            if len(self.update_history) > 50:
                self.update_history.pop(0)
            
            self.update_in_progress = False
            return update_info
            
        except Exception as e:
            logger.error(f"Error pulling updates: {e}")
            self.update_in_progress = False
            return {
                'success': False,
                'error': str(e)
            }
    
    def _restart_nae(self) -> dict:
        """Restart NAE production service"""
        try:
            stop_script = self.nae_root / 'stop_nae_production.sh'
            start_script = self.nae_root / 'start_nae_production.sh'
            
            if stop_script.exists():
                subprocess.run(['bash', str(stop_script)], timeout=30)
                time.sleep(2)
            
            if start_script.exists():
                subprocess.Popen(
                    ['bash', str(start_script)],
                    cwd=str(self.nae_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return {'success': True}
            
            return {'success': False, 'error': 'NAE scripts not found'}
            
        except Exception as e:
            logger.error(f"Error restarting NAE: {e}")
            return {'success': False, 'error': str(e)}


# Initialize updater
updater = AutoUpdater(NAE_ROOT, CURRENT_BRANCH)


@app.route('/api/update/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'mac_auto_updater',
        'branch': CURRENT_BRANCH,
        'nae_root': str(NAE_ROOT),
        'last_check': updater.last_update_check,
        'last_update': updater.last_update_time,
        'update_in_progress': updater.update_in_progress
    })


@app.route('/api/update/check', methods=['GET'])
def check_updates():
    """Check for available updates"""
    result = updater.check_for_updates()
    return jsonify(result)


@app.route('/api/update/pull', methods=['POST'])
def pull_updates():
    """Pull latest updates"""
    data = request.get_json() or {}
    restart_nae = data.get('restart_nae', False)
    
    result = updater.pull_updates(restart_nae=restart_nae)
    return jsonify(result)


@app.route('/api/update/status', methods=['GET'])
def update_status():
    """Get update status and history"""
    return jsonify({
        'last_check': updater.last_update_check,
        'last_update': updater.last_update_time,
        'update_in_progress': updater.update_in_progress,
        'current_branch': CURRENT_BRANCH,
        'recent_history': updater.update_history[-10:] if updater.update_history else []
    })


def main():
    """Main entry point"""
    port = int(os.getenv('AUTO_UPDATER_PORT', '8081'))
    host = os.getenv('AUTO_UPDATER_HOST', '0.0.0.0')
    
    logger.info("=" * 60)
    logger.info("Mac Auto-Updater Service")
    logger.info("=" * 60)
    logger.info(f"NAE Root: {NAE_ROOT}")
    logger.info(f"Branch: {CURRENT_BRANCH}")
    logger.info(f"Starting on {host}:{port}")
    logger.info("=" * 60)
    
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()


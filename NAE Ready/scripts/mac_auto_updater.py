#!/usr/bin/env python3
"""
Mac Auto-Updater Service
Runs alongside the main NAE system to enable remote updates from HP (Dev)

Features:
- /api/update/pull - Triggers git pull from main branch
- /api/update/status - Shows current git status
- /api/update/restart - Pulls and restarts NAE services
"""

import os
import subprocess
import json
from datetime import datetime
from flask import Flask, jsonify, request
from functools import wraps

app = Flask(__name__)

# Configuration
NAE_ROOT = os.environ.get('NAE_ROOT', '/Users/melissabishop/Documents/NAE/NAE')
API_KEY = os.environ.get('NAE_UPDATE_API_KEY', 'a07e9de261c6eb815fbcd9cb6263f0862534af1cd3cc3540c87ed70ce0e4438d')
UPDATE_PORT = int(os.environ.get('NAE_UPDATE_PORT', 8081))

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != API_KEY:
            return jsonify({'error': 'Unauthorized', 'message': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

def run_git_command(cmd, cwd=None):
    """Run a git command and return result"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or NAE_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Command timed out'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/api/update/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'mac-auto-updater',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/update/status', methods=['GET'])
@require_api_key
def git_status():
    """Get current git status"""
    status = run_git_command(['git', 'status', '--porcelain'])
    branch = run_git_command(['git', 'branch', '--show-current'])
    last_commit = run_git_command(['git', 'log', '-1', '--oneline'])
    remote_check = run_git_command(['git', 'fetch', '--dry-run'])
    
    return jsonify({
        'branch': branch.get('stdout', 'unknown'),
        'last_commit': last_commit.get('stdout', 'unknown'),
        'has_changes': bool(status.get('stdout', '')),
        'changes': status.get('stdout', '').split('\n') if status.get('stdout') else [],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/update/pull', methods=['POST'])
@require_api_key
def git_pull():
    """Pull latest changes from main branch"""
    # Fetch first
    fetch_result = run_git_command(['git', 'fetch', 'origin'])
    
    # Pull from main
    pull_result = run_git_command(['git', 'pull', 'origin', 'main'])
    
    # Get new commit info
    commit_result = run_git_command(['git', 'log', '-1', '--oneline'])
    
    return jsonify({
        'success': pull_result.get('success', False),
        'fetch': fetch_result,
        'pull': pull_result,
        'current_commit': commit_result.get('stdout', 'unknown'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/update/restart', methods=['POST'])
@require_api_key
def pull_and_restart():
    """Pull updates and restart NAE services"""
    # First pull
    pull_result = git_pull()
    
    # Find and restart NAE processes
    restart_commands = [
        ['pkill', '-f', 'nae_autonomous_master.py'],
        ['sleep', '2'],
    ]
    
    restart_results = []
    for cmd in restart_commands:
        result = run_git_command(cmd)
        restart_results.append({'cmd': ' '.join(cmd), 'result': result})
    
    # Start NAE again (in background)
    start_cmd = f'cd "{NAE_ROOT}" && nohup python3 nae_autonomous_master.py > /tmp/nae.log 2>&1 &'
    start_result = subprocess.run(['bash', '-c', start_cmd], capture_output=True, text=True)
    
    return jsonify({
        'pull_result': pull_result.get_json() if hasattr(pull_result, 'get_json') else pull_result,
        'restart_results': restart_results,
        'nae_started': start_result.returncode == 0,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Mac Auto-Updater Service                           ║
╠══════════════════════════════════════════════════════════════╣
║  Port: {UPDATE_PORT}                                                 ║
║  NAE Root: {NAE_ROOT[:45]}...  ║
║                                                              ║
║  Endpoints:                                                  ║
║    GET  /api/update/health  - Health check (no auth)         ║
║    GET  /api/update/status  - Git status                     ║
║    POST /api/update/pull    - Pull from main                 ║
║    POST /api/update/restart - Pull and restart NAE           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=UPDATE_PORT, debug=False)



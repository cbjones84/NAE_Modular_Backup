#!/usr/bin/env python3
"""
AutoGen Studio Bridge for NAE
Provides API endpoints for AutoGen Studio to communicate with Casey
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

# Add NAE to path
nae_root = Path(__file__).parent
sys.path.insert(0, str(nae_root))

from agents.casey import CaseyAgent
# Try new integration first, fallback to old
try:
    from autogen_studio_nae_integration import NAEStudioIntegration
    # Create wrapper from new integration
    _integration = NAEStudioIntegration()
    if "Casey" in _integration.agent_bridges:
        CaseyAutoGenAgent = type('CaseyAutoGenAgent', (), {
            '__init__': lambda self: setattr(self, 'casey', _integration.agent_bridges["Casey"].nae_agent),
            'process_message': lambda self, msg, sender="User": _integration.agent_bridges["Casey"].process_message(msg, {"sender": sender})
        })
    else:
        raise ImportError("Casey not in new integration")
except (ImportError, AttributeError):
    # Fallback to old integration
    from autogen_studio_integration import CaseyAutoGenAgent

app = Flask(__name__)
CORS(app)

# Initialize Casey wrapper
casey_wrapper = None

def get_casey():
    """Get or create Casey wrapper"""
    global casey_wrapper
    if casey_wrapper is None:
        casey_wrapper = CaseyAutoGenAgent()
    return casey_wrapper

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "NAE AutoGen Studio Bridge",
        "casey_available": True
    })

@app.route('/api/casey/process', methods=['POST'])
def process_command():
    """Process a command through Casey"""
    try:
        data = request.json
        message = data.get('message', '')
        sender = data.get('sender', 'User')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        casey = get_casey()
        response = casey.process_message(message, sender)
        
        return jsonify({
            'success': True,
            'response': response,
            'sender': 'Casey'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/casey/tools/read_file', methods=['POST'])
def tool_read_file():
    """Tool: Read file"""
    try:
        data = request.json
        file_path = data.get('file_path', '')
        
        if not file_path:
            return jsonify({'success': False, 'error': 'No file_path provided'}), 400
        
        casey = get_casey().casey
        result = casey.read_file(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/casey/tools/write_file', methods=['POST'])
def tool_write_file():
    """Tool: Write file"""
    try:
        data = request.json
        file_path = data.get('file_path', '')
        content = data.get('content', '')
        
        if not file_path:
            return jsonify({'success': False, 'error': 'No file_path provided'}), 400
        
        casey = get_casey().casey
        result = casey.write_file(file_path, content)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/casey/tools/list_directory', methods=['POST'])
def tool_list_directory():
    """Tool: List directory"""
    try:
        data = request.json
        directory = data.get('directory', '.')
        
        casey = get_casey().casey
        result = casey.list_directory(directory)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/casey/tools/search_codebase', methods=['POST'])
def tool_search_codebase():
    """Tool: Search codebase"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        casey = get_casey().casey
        result = casey.semantic_search(query)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/casey/tools/execute_python', methods=['POST'])
def tool_execute_python():
    """Tool: Execute Python code"""
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'success': False, 'error': 'No code provided'}), 400
        
        casey = get_casey().casey
        result = casey.execute_python_code(code)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/casey/tools/get_status', methods=['GET'])
def tool_get_status():
    """Tool: Get system status"""
    try:
        casey = get_casey().casey
        
        status = {
            'agents': [],
            'system_health': {}
        }
        
        if hasattr(casey, 'monitored_agents'):
            agents = casey.monitored_agents
            if isinstance(agents, dict):
                status['agents'] = list(agents.keys())
            elif isinstance(agents, list):
                status['agents'] = [str(a) for a in agents]
        
        try:
            import psutil
            status['system_health'] = {
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent
            }
        except:
            pass
        
        return jsonify({
            'success': True,
            'status': status
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("NAE AutoGen Studio Bridge")
    print("=" * 60)
    print("Starting bridge server on http://localhost:8080")
    print("AutoGen Studio can connect to this bridge to communicate with Casey")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=True)


# NAE/debug_tools.py
"""
Debugging Tools for NAE
Provides interactive debugging, logging, and inspection capabilities
"""

import os
import sys
import json
import traceback
import pdb
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

class DebugTools:
    """Comprehensive debugging tools for NAE"""
    
    def __init__(self, log_dir: str = "logs/debug"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup debug logging"""
        log_file = os.path.join(self.log_dir, "debug.log")
        
        # Create logger
        logger = logging.getLogger("NAE_DEBUG")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    def debug_agent(self, agent_name: str) -> Dict[str, Any]:
        """Debug an agent"""
        try:
            # Import agent
            module_name = f"agents.{agent_name.lower()}"
            module = __import__(module_name, fromlist=[agent_name])
            agent_class = getattr(module, f"{agent_name}Agent")
            
            # Initialize agent
            if agent_name.lower() == "optimus":
                agent = agent_class(sandbox=True)
            else:
                agent = agent_class()
            
            # Gather debug info
            debug_info = {
                "agent_name": agent_name,
                "class_name": agent_class.__name__,
                "module": module_name,
                "attributes": dir(agent),
                "methods": [m for m in dir(agent) if callable(getattr(agent, m)) and not m.startswith('_')],
                "has_goals": hasattr(agent, "goals"),
                "goals": getattr(agent, "goals", []),
                "log_file": getattr(agent, "log_file", None),
                "status": getattr(agent, "status", "unknown")
            }
            
            self.logger.info(f"Debugged agent: {agent_name}")
            return debug_info
            
        except Exception as e:
            error_info = {
                "agent_name": agent_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.logger.error(f"Error debugging agent {agent_name}: {e}")
            return error_info
    
    def inspect_method(self, agent_name: str, method_name: str) -> Dict[str, Any]:
        """Inspect an agent method"""
        try:
            # Import agent
            module_name = f"agents.{agent_name.lower()}"
            module = __import__(module_name, fromlist=[agent_name])
            agent_class = getattr(module, f"{agent_name}Agent")
            
            # Get method
            if not hasattr(agent_class, method_name):
                return {"error": f"Method {method_name} not found"}
            
            method = getattr(agent_class, method_name)
            
            # Inspect method
            import inspect
            sig = inspect.signature(method)
            
            inspect_info = {
                "agent_name": agent_name,
                "method_name": method_name,
                "signature": str(sig),
                "parameters": [str(p) for p in sig.parameters.values()],
                "docstring": inspect.getdoc(method),
                "source_file": inspect.getfile(method),
                "line_number": inspect.getsourcelines(method)[1]
            }
            
            self.logger.info(f"Inspected method: {agent_name}.{method_name}")
            return inspect_info
            
        except Exception as e:
            error_info = {
                "agent_name": agent_name,
                "method_name": method_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.logger.error(f"Error inspecting method: {e}")
            return error_info
    
    def inspect_logs(self, agent_name: str, lines: int = 100) -> Dict[str, Any]:
        """Inspect agent logs"""
        try:
            log_file = f"logs/{agent_name.lower()}.log"
            
            if not os.path.exists(log_file):
                return {"error": f"Log file not found: {log_file}"}
            
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            
            return {
                "agent_name": agent_name,
                "log_file": log_file,
                "total_lines": len(log_lines),
                "recent_lines": log_lines[-lines:] if len(log_lines) > lines else log_lines
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            import psutil
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids())
            }
            
            return status
            
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }


# Global debug tools instance
_debug_tools = None

def get_debug_tools() -> DebugTools:
    """Get global debug tools instance"""
    global _debug_tools
    if _debug_tools is None:
        _debug_tools = DebugTools()
    return _debug_tools


if __name__ == "__main__":
    # Test debug tools
    debug = DebugTools()
    
    # Debug an agent
    info = debug.debug_agent("Ralph")
    print(json.dumps(info, indent=2))
    
    # Inspect a method
    info = debug.inspect_method("Ralph", "generate_strategies")
    print(json.dumps(info, indent=2))
    
    # Get system status
    status = debug.get_system_status()
    print(json.dumps(status, indent=2))



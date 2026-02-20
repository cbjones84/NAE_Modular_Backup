# NAE/command_executor.py
"""
Safe Command Execution System for NAE
Provides secure execution of Python code and system commands
"""

import os
import sys
import subprocess
import tempfile
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import ast
import logging

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"

@dataclass
class ExecutionResult:
    """Command execution result"""
    status: ExecutionStatus
    output: str
    error: Optional[str] = None
    duration: float = 0.0
    return_code: Optional[int] = None

class SafeCommandExecutor:
    """Safe command and code execution system"""
    
    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.blocked_imports = [
            'subprocess', 'os.system', 'eval', 'exec', '__import__',
            'open', 'file', 'input', 'raw_input'
        ]
        self.blocked_commands = [
            'rm', 'del', 'format', 'mkfs', 'dd', 'shutdown',
            'reboot', 'sudo', 'su', 'passwd'
        ]
    
    def execute_python_code(self, code: str, context: Optional[Dict] = None) -> ExecutionResult:
        """Safely execute Python code"""
        import time
        start_time = time.time()
        
        try:
            # Validate code safety
            if not self._is_code_safe(code):
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    output="",
                    error="Code contains blocked operations",
                    duration=time.time() - start_time
                )
            
            # Create temporary namespace
            namespace = {"__builtins__": __builtins__}
            if context:
                namespace.update(context)
            
            # Execute in restricted environment
            exec_globals = {
                "__builtins__": {
                    k: v for k, v in __builtins__.items()
                    if k not in ['eval', 'exec', '__import__', 'open', 'file', 'input']
                }
            }
            exec_globals.update(context or {})
            
            # Capture output
            import io
            output_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            try:
                # Compile and execute
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, exec_globals)
                
                output = output_buffer.getvalue()
                duration = time.time() - start_time
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    output=output[:self.max_output_size],
                    duration=duration
                )
            finally:
                sys.stdout = old_stdout
                
        except SyntaxError as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Syntax error: {str(e)}",
                duration=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Execution error: {str(e)}\n{traceback.format_exc()}",
                duration=time.time() - start_time
            )
    
    def execute_system_command(self, command: str, cwd: Optional[str] = None) -> ExecutionResult:
        """Safely execute system command"""
        import time
        start_time = time.time()
        
        try:
            # Check if command is blocked
            if self._is_command_blocked(command):
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    output="",
                    error="Command is blocked for security",
                    duration=time.time() - start_time
                )
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd or os.getcwd()
            )
            
            duration = time.time() - start_time
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILED,
                output=result.stdout[:self.max_output_size],
                error=result.stderr[:self.max_output_size] if result.stderr else None,
                duration=duration,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                output="",
                error=f"Command timed out after {self.timeout} seconds",
                duration=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Command execution error: {str(e)}",
                duration=time.time() - start_time
            )
    
    def execute_agent_method(self, agent_name: str, method_name: str, 
                            args: Optional[List] = None, kwargs: Optional[Dict] = None) -> ExecutionResult:
        """Safely execute an agent method"""
        import time
        start_time = time.time()
        
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
            
            # Get method
            if not hasattr(agent, method_name):
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output="",
                    error=f"Method {method_name} not found on {agent_name}",
                    duration=time.time() - start_time
                )
            
            method = getattr(agent, method_name)
            
            # Execute method
            args = args or []
            kwargs = kwargs or {}
            result = method(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Convert result to string
            if isinstance(result, (dict, list)):
                import json
                output = json.dumps(result, indent=2)
            else:
                output = str(result)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output[:self.max_output_size],
                duration=duration
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Agent method execution error: {str(e)}\n{traceback.format_exc()}",
                duration=time.time() - start_time
            )
    
    def _is_code_safe(self, code: str) -> bool:
        """Check if code is safe to execute"""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.blocked_imports:
                            return False
                
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.blocked_imports:
                            return False
                
                if isinstance(node, ast.ImportFrom):
                    if node.module in self.blocked_imports:
                        return False
            
            return True
        except:
            return False
    
    def _is_command_blocked(self, command: str) -> bool:
        """Check if command is blocked"""
        command_lower = command.lower()
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return True
        return False


# Global executor instance
_executor = None

def get_executor() -> SafeCommandExecutor:
    """Get global command executor"""
    global _executor
    if _executor is None:
        _executor = SafeCommandExecutor()
    return _executor


if __name__ == "__main__":
    # Test command executor
    executor = SafeCommandExecutor()
    
    # Test Python code execution
    result = executor.execute_python_code("""
x = 10
y = 20
print(f"Sum: {x + y}")
""")
    print(f"Code execution: {result.status}")
    print(f"Output: {result.output}")
    
    # Test system command
    result = executor.execute_system_command("echo 'Hello NAE'")
    print(f"Command execution: {result.status}")
    print(f"Output: {result.output}")



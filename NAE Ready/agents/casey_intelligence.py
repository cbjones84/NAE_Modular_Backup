"""
Casey Intelligence Module - Advanced AI Reasoning and Context Awareness
Makes Casey as intelligent as Composer 1 and Cursor 2.0
"""

import os
import re
import ast
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import difflib

@dataclass
class Context:
    """Represents current context for understanding user intent"""
    current_files: List[str] = field(default_factory=list)
    recent_commands: List[str] = field(default_factory=list)
    active_agents: List[str] = field(default_factory=list)
    working_directory: str = ""
    project_structure: Dict[str, Any] = field(default_factory=dict)
    user_goals: List[str] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    code_patterns: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class Intent:
    """Represents user intent parsed from command"""
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    context_needed: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)

class CaseyIntelligence:
    """
    Advanced intelligence engine for Casey
    Provides Composer 1 and Cursor 2.0 level intelligence
    """
    
    def __init__(self, casey_agent):
        self.casey = casey_agent
        self.context = Context()
        self.command_history: List[Tuple[str, Intent, Any]] = []
        self.learning_patterns: Dict[str, float] = defaultdict(float)
        self.code_understanding_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize context
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize context from current system state"""
        # Get active agents
        if hasattr(self.casey, 'monitored_agents'):
            # Handle both dict and list types for monitored_agents
            if isinstance(self.casey.monitored_agents, dict):
                self.context.active_agents = list(self.casey.monitored_agents.keys())
            elif isinstance(self.casey.monitored_agents, list):
                # Extract agent names from list of tuples (name, process)
                self.context.active_agents = [name for name, _ in self.casey.monitored_agents if isinstance(name, str)]
            else:
                self.context.active_agents = []
        
        # Get working directory
        self.context.working_directory = os.getcwd()
        
        # Get user goals
        if hasattr(self.casey, 'goals'):
            self.context.user_goals = self.casey.goals if isinstance(self.casey.goals, list) else []
    
    def understand_intent(self, command: str) -> Intent:
        """
        Understand user intent from natural language command
        Uses advanced pattern matching and context awareness
        """
        command_lower = command.lower().strip()
        
        # Check for multi-step commands
        if self._is_multi_step(command):
            return self._parse_multi_step(command)
        
        # Pattern matching for common intents
        intent_patterns = {
            'read_file': [
                r'read\s+(?:file\s+)?(.+)',
                r'open\s+(?:file\s+)?(.+)',
                r'show\s+(?:me\s+)?(.+\.py)',
                r'view\s+(?:file\s+)?(.+)',
                r'display\s+(?:file\s+)?(.+)',
            ],
            'write_file': [
                r'write\s+(?:to\s+)?(.+)',
                r'create\s+(?:file\s+)?(.+)',
                r'edit\s+(?:file\s+)?(.+)',
                r'update\s+(?:file\s+)?(.+)',
                r'modify\s+(?:file\s+)?(.+)',
            ],
            'search_codebase': [
                r'search\s+(?:for\s+)?(.+)',
                r'find\s+(?:all\s+)?(.+)',
                r'grep\s+(?:for\s+)?(.+)',
                r'look\s+for\s+(.+)',
                r'where\s+is\s+(.+)',
            ],
            'execute_code': [
                r'run\s+(?:code\s+)?(.+)',
                r'execute\s+(.+)',
                r'run\s+(?:command\s+)?(.+)',
                r'do\s+(.+)',
            ],
            'debug': [
                r'debug\s+(.+)',
                r'fix\s+(.+)',
                r'error\s+in\s+(.+)',
                r'problem\s+with\s+(.+)',
                r'why\s+is\s+(.+)',
            ],
            'analyze': [
                r'analyze\s+(.+)',
                r'explain\s+(.+)',
                r'what\s+does\s+(.+)',
                r'how\s+does\s+(.+)',
                r'understand\s+(.+)',
            ],
            'create_agent': [
                r'create\s+(?:agent\s+)?(.+)',
                r'build\s+(?:agent\s+)?(.+)',
                r'make\s+(?:agent\s+)?(.+)',
                r'new\s+agent\s+(.+)',
            ],
            'monitor': [
                r'check\s+(?:status\s+)?(.+)',
                r'monitor\s+(.+)',
                r'watch\s+(.+)',
                r'status\s+of\s+(.+)',
            ],
            'improve': [
                r'improve\s+(.+)',
                r'optimize\s+(.+)',
                r'enhance\s+(.+)',
                r'better\s+(.+)',
            ],
        }
        
        # Match patterns
        best_match = None
        best_confidence = 0.0
        
        for action, patterns in intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower, re.IGNORECASE)
                if match:
                    confidence = self._calculate_confidence(command, action, match)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        target = match.group(1).strip() if match.lastindex else None
                        best_match = Intent(
                            action=action,
                            target=target,
                            confidence=confidence,
                            parameters=self._extract_parameters(command)
                        )
        
        # If no pattern match, use general understanding
        if not best_match:
            best_match = self._general_understanding(command)
        
        # Enhance with context
        best_match = self._enhance_with_context(best_match, command)
        
        # Add suggestions
        best_match.suggested_actions = self._generate_suggestions(best_match)
        
        return best_match
    
    def _is_multi_step(self, command: str) -> bool:
        """Check if command contains multiple steps"""
        multi_step_indicators = [
            'then', 'and', 'also', 'after', 'next', 'followed by',
            'first', 'second', 'finally', 'lastly'
        ]
        return any(indicator in command.lower() for indicator in multi_step_indicators)
    
    def _parse_multi_step(self, command: str) -> Intent:
        """Parse multi-step command into sequence of intents"""
        # Split by multi-step indicators
        steps = re.split(r'\s+(?:then|and|also|after|next|followed by|first|second|finally|lastly)\s+', 
                        command.lower(), flags=re.IGNORECASE)
        
        if len(steps) > 1:
            return Intent(
                action='multi_step',
                parameters={'steps': [self.understand_intent(step) for step in steps]},
                confidence=0.9
            )
        return self.understand_intent(command)
    
    def _calculate_confidence(self, command: str, action: str, match: re.Match) -> float:
        """Calculate confidence score for intent match"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for exact matches
        if match.group(0).lower() == command.lower().strip():
            confidence += 0.2
        
        # Increase confidence if target is clear
        if match.lastindex and match.group(1):
            confidence += 0.1
        
        # Increase confidence based on learning patterns
        pattern_key = f"{action}:{command[:20]}"
        if pattern_key in self.learning_patterns:
            confidence += min(self.learning_patterns[pattern_key], 0.2)
        
        return min(confidence, 1.0)
    
    def _extract_parameters(self, command: str) -> Dict[str, Any]:
        """Extract parameters from command"""
        params = {}
        
        # Extract file paths
        file_pattern = r'([\w/\.\-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|yaml|yml))'
        files = re.findall(file_pattern, command)
        if files:
            params['files'] = files
        
        # Extract numbers
        numbers = re.findall(r'\d+', command)
        if numbers:
            params['numbers'] = [int(n) for n in numbers]
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', command)
        if quoted:
            params['quoted'] = [q[0] or q[1] for q in quoted]
        
        # Extract flags/options
        flags = re.findall(r'--(\w+)|-(\w)', command)
        if flags:
            params['flags'] = [f[0] or f[1] for f in flags]
        
        return params
    
    def _general_understanding(self, command: str) -> Intent:
        """General understanding when no pattern matches"""
        command_lower = command.lower()
        
        # Question detection
        if any(word in command_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return Intent(
                action='question',
                target=command,
                confidence=0.6,
                parameters={'question': command}
            )
        
        # Action detection
        action_words = ['do', 'make', 'create', 'build', 'run', 'execute', 'start', 'stop']
        if any(word in command_lower for word in action_words):
            return Intent(
                action='general_action',
                target=command,
                confidence=0.5,
                parameters={'action': command}
            )
        
        # Default to search
        return Intent(
            action='search_codebase',
            target=command,
            confidence=0.4,
            parameters={'query': command}
        )
    
    def _enhance_with_context(self, intent: Intent, command: str) -> Intent:
        """Enhance intent with context awareness"""
        # Add context from recent files
        if intent.action == 'read_file' and not intent.target:
            if self.context.current_files:
                intent.target = self.context.current_files[-1]
                intent.confidence += 0.1
        
        # Add context from recent commands
        if self.context.recent_commands:
            last_command = self.context.recent_commands[-1]
            if 'continue' in command.lower() or 'same' in command.lower():
                # Continue from last command
                intent.parameters['continue_from'] = last_command
        
        # Add context from error history
        if intent.action == 'debug' and self.context.error_history:
            recent_error = self.context.error_history[-1]
            if not intent.target:
                intent.target = recent_error.get('file')
                intent.parameters['error'] = recent_error.get('error')
                intent.confidence += 0.2
        
        # Add context from project structure
        if intent.target and intent.target in self.context.project_structure:
            intent.parameters['file_info'] = self.context.project_structure[intent.target]
        
        return intent
    
    def _generate_suggestions(self, intent: Intent) -> List[str]:
        """Generate intelligent suggestions based on intent"""
        suggestions = []
        
        if intent.action == 'read_file':
            suggestions.extend([
                f"Would you like me to analyze {intent.target}?",
                f"Should I search for related files?",
                f"Would you like to edit {intent.target}?",
            ])
        
        elif intent.action == 'debug':
            suggestions.extend([
                "I can analyze the error and suggest fixes",
                "Would you like me to check similar code patterns?",
                "I can search for similar errors in the codebase",
            ])
        
        elif intent.action == 'create_agent':
            suggestions.extend([
                "I can create the agent with NAE goals integration",
                "Would you like me to add AutoGen communication?",
                "I can set up monitoring and logging",
            ])
        
        elif intent.action == 'search_codebase':
            suggestions.extend([
                "I can search semantically for related code",
                "Would you like me to find all usages?",
                "I can analyze code patterns and relationships",
            ])
        
        # Add context-aware suggestions
        if self.context.recent_commands:
            suggestions.append("Would you like to continue from your last action?")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def understand_code(self, code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Deep code understanding - like Cursor 2.0
        Analyzes code structure, dependencies, patterns, and intent
        """
        if file_path and file_path in self.code_understanding_cache:
            return self.code_understanding_cache[file_path]
        
        analysis = {
            'structure': {},
            'dependencies': [],
            'patterns': [],
            'intent': '',
            'complexity': 0,
            'suggestions': [],
            'potential_issues': [],
        }
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Analyze structure
            analysis['structure'] = self._analyze_structure(tree)
            
            # Find dependencies
            analysis['dependencies'] = self._find_dependencies(tree, code)
            
            # Detect patterns
            analysis['patterns'] = self._detect_patterns(tree, code)
            
            # Understand intent
            analysis['intent'] = self._understand_code_intent(tree, code)
            
            # Calculate complexity
            analysis['complexity'] = self._calculate_complexity(tree)
            
            # Generate suggestions
            analysis['suggestions'] = self._generate_code_suggestions(analysis)
            
            # Find potential issues
            analysis['potential_issues'] = self._find_potential_issues(tree, code)
            
        except SyntaxError as e:
            analysis['potential_issues'].append({
                'type': 'syntax_error',
                'message': str(e),
                'line': e.lineno,
                'severity': 'error'
            })
        
        # Cache result
        if file_path:
            self.code_understanding_cache[file_path] = analysis
        
        return analysis
    
    def _analyze_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code structure"""
        structure = {
            'classes': [],
            'functions': [],
            'imports': [],
            'decorators': [],
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                structure['classes'].append({
                    'name': node.name,
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    'bases': [self._get_name(base) for base in node.bases],
                })
            elif isinstance(node, ast.FunctionDef):
                structure['functions'].append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [self._get_name(d) for d in node.decorator_list],
                })
            elif isinstance(node, ast.Import):
                structure['imports'].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                structure['imports'].extend([f"{node.module}.{alias.name}" for alias in node.names])
        
        return structure
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _find_dependencies(self, tree: ast.AST, code: str) -> List[str]:
        """Find code dependencies"""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
        
        # Also check for string-based imports
        import_pattern = r'import\s+([\w\.]+)'
        dependencies.extend(re.findall(import_pattern, code))
        
        return list(set(dependencies))
    
    def _detect_patterns(self, tree: ast.AST, code: str) -> List[str]:
        """Detect common code patterns"""
        patterns = []
        
        # Check for design patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for singleton
                if len([n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__new__']) > 0:
                    patterns.append('singleton')
                
                # Check for factory
                if 'factory' in node.name.lower() or 'create' in node.name.lower():
                    patterns.append('factory')
        
        # Check for async patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                patterns.append('async')
                break
        
        # Check for decorator patterns
        decorator_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.decorator_list)
        if decorator_count > 3:
            patterns.append('decorator_heavy')
        
        return patterns
    
    def _understand_code_intent(self, tree: ast.AST, code: str) -> str:
        """Understand what the code is trying to do"""
        intent_keywords = {
            'api': ['request', 'response', 'endpoint', 'route', 'api'],
            'database': ['query', 'database', 'db', 'sql', 'orm'],
            'trading': ['trade', 'order', 'position', 'market', 'price'],
            'monitoring': ['monitor', 'watch', 'check', 'status', 'health'],
            'authentication': ['auth', 'login', 'token', 'password', 'user'],
        }
        
        code_lower = code.lower()
        intents = []
        
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in code_lower for keyword in keywords):
                intents.append(intent_type)
        
        # Analyze function names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()
                if 'get' in func_name or 'fetch' in func_name:
                    intents.append('data_retrieval')
                elif 'set' in func_name or 'update' in func_name:
                    intents.append('data_modification')
                elif 'create' in func_name or 'build' in func_name:
                    intents.append('creation')
        
        return ', '.join(set(intents)) if intents else 'general'
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate code complexity"""
        complexity = 0
        
        for node in ast.walk(tree):
            # Cyclomatic complexity factors
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _generate_code_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent code suggestions"""
        suggestions = []
        
        # Complexity suggestions
        if analysis['complexity'] > 10:
            suggestions.append("Consider refactoring to reduce complexity")
        
        # Pattern suggestions
        if 'async' in analysis['patterns']:
            suggestions.append("Consider using asyncio.gather for concurrent operations")
        
        # Dependency suggestions
        if len(analysis['dependencies']) > 10:
            suggestions.append("Consider splitting into smaller modules")
        
        # Structure suggestions
        if len(analysis['structure']['classes']) == 0 and len(analysis['structure']['functions']) > 5:
            suggestions.append("Consider organizing functions into classes")
        
        return suggestions
    
    def _find_potential_issues(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Find potential code issues"""
        issues = []
        
        # Check for common issues
        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    'type': 'bare_except',
                    'message': 'Bare except clause catches all exceptions',
                    'line': node.lineno,
                    'severity': 'warning'
                })
            
            # Check for unused variables
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Simple check - could be enhanced
                        pass
        
        # Check for TODO/FIXME comments
        todo_pattern = r'(TODO|FIXME|XXX|HACK):\s*(.+)'
        for match in re.finditer(todo_pattern, code, re.IGNORECASE):
            issues.append({
                'type': 'todo',
                'message': match.group(2),
                'line': code[:match.start()].count('\n') + 1,
                'severity': 'info'
            })
        
        return issues
    
    def learn_from_interaction(self, command: str, intent: Intent, result: Any):
        """Learn from user interactions to improve understanding"""
        # Store interaction
        self.command_history.append((command, intent, result))
        
        # Update learning patterns
        pattern_key = f"{intent.action}:{command[:20]}"
        if result and hasattr(result, 'get') and result.get('success'):
            self.learning_patterns[pattern_key] += 0.1
        else:
            self.learning_patterns[pattern_key] -= 0.05
        
        # Update context
        self.context.recent_commands.append(command)
        if len(self.context.recent_commands) > 10:
            self.context.recent_commands.pop(0)
        
        # Update current files if file operation
        if intent.action in ['read_file', 'write_file'] and intent.target:
            if intent.target not in self.context.current_files:
                self.context.current_files.append(intent.target)
            if len(self.context.current_files) > 5:
                self.context.current_files.pop(0)
    
    def get_proactive_suggestions(self) -> List[str]:
        """Generate proactive suggestions based on context"""
        suggestions = []
        
        # Suggest based on recent errors
        if self.context.error_history:
            recent_error = self.context.error_history[-1]
            suggestions.append(f"I noticed an error in {recent_error.get('file', 'your code')}. Would you like me to help fix it?")
        
        # Suggest based on file patterns
        if len(self.context.current_files) > 0:
            suggestions.append(f"You've been working with {self.context.current_files[-1]}. Would you like me to analyze it?")
        
        # Suggest based on agent status
        if self.context.active_agents:
            suggestions.append(f"I see {len(self.context.active_agents)} agents running. Would you like me to check their status?")
        
        # Suggest based on time patterns
        current_hour = datetime.datetime.now().hour
        if current_hour >= 9 and current_hour <= 17:
            suggestions.append("It's a good time to run system checks. Would you like me to do that?")
        
        return suggestions[:3]


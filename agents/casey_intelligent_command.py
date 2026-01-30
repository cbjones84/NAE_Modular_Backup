"""
Intelligent Command Processing for Casey
Provides Composer 1 and Cursor 2.0 level command understanding
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
from agents.casey_intelligence import CaseyIntelligence, Intent

class IntelligentCommandProcessor:
    """
    Processes commands with advanced understanding and context awareness
    """
    
    def __init__(self, casey_agent, intelligence: CaseyIntelligence):
        self.casey = casey_agent
        self.intelligence = intelligence
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a command with intelligent understanding
        Returns execution plan and suggestions
        """
        # Understand intent
        intent = self.intelligence.understand_intent(command)
        
        # Generate execution plan
        execution_plan = self._generate_execution_plan(intent, command)
        
        # Get proactive suggestions
        suggestions = self.intelligence.get_proactive_suggestions()
        
        return {
            'intent': {
                'action': intent.action,
                'target': intent.target,
                'confidence': intent.confidence,
                'parameters': intent.parameters
            },
            'execution_plan': execution_plan,
            'suggestions': suggestions + intent.suggested_actions,
            'context_needed': intent.context_needed
        }
    
    def _generate_execution_plan(self, intent: Intent, command: str) -> List[Dict[str, Any]]:
        """Generate step-by-step execution plan"""
        plan = []
        
        if intent.action == 'multi_step':
            # Handle multi-step commands
            for step_intent in intent.parameters.get('steps', []):
                plan.extend(self._generate_execution_plan(step_intent, ''))
            return plan
        
        # Single step execution plans
        if intent.action == 'read_file':
            plan.append({
                'step': 1,
                'action': 'read_file',
                'target': intent.target or self._extract_file_from_context(),
                'method': 'read_file',
                'description': f"Read file: {intent.target}"
            })
        
        elif intent.action == 'write_file':
            plan.append({
                'step': 1,
                'action': 'write_file',
                'target': intent.target,
                'method': 'write_file',
                'description': f"Write to file: {intent.target}",
                'needs_content': True
            })
        
        elif intent.action == 'search_codebase':
            plan.append({
                'step': 1,
                'action': 'semantic_search',
                'query': intent.target or command,
                'method': 'semantic_search',
                'description': f"Search codebase for: {intent.target or command}"
            })
        
        elif intent.action == 'execute_code':
            plan.append({
                'step': 1,
                'action': 'execute_python_code',
                'code': intent.target or command,
                'method': 'execute_python_code',
                'description': f"Execute code: {intent.target or command[:50]}"
            })
        
        elif intent.action == 'debug':
            plan.append({
                'step': 1,
                'action': 'debug_code',
                'target': intent.target,
                'error': intent.parameters.get('error'),
                'method': 'debug_code',
                'description': f"Debug: {intent.target}"
            })
            plan.append({
                'step': 2,
                'action': 'analyze_code',
                'target': intent.target,
                'method': 'understand_context',
                'description': f"Analyze code structure: {intent.target}"
            })
        
        elif intent.action == 'analyze':
            plan.append({
                'step': 1,
                'action': 'understand_code',
                'target': intent.target,
                'method': 'understand_context',
                'description': f"Analyze: {intent.target}"
            })
        
        elif intent.action == 'create_agent':
            plan.append({
                'step': 1,
                'action': 'build_agent',
                'agent_name': intent.target,
                'method': 'build_or_refine_agent',
                'description': f"Create agent: {intent.target}"
            })
        
        elif intent.action == 'monitor':
            plan.append({
                'step': 1,
                'action': 'check_status',
                'target': intent.target or 'all',
                'method': 'get_system_state',
                'description': f"Monitor: {intent.target or 'system'}"
            })
        
        elif intent.action == 'improve':
            plan.append({
                'step': 1,
                'action': 'analyze',
                'target': intent.target,
                'method': 'understand_context',
                'description': f"Analyze for improvements: {intent.target}"
            })
            plan.append({
                'step': 2,
                'action': 'suggest_improvements',
                'target': intent.target,
                'method': '_generate_improvements',
                'description': f"Generate improvement suggestions"
            })
        
        else:
            # Default: try to understand and execute
            plan.append({
                'step': 1,
                'action': 'general',
                'command': command,
                'method': 'semantic_search',
                'description': f"Process command: {command[:50]}"
            })
        
        return plan
    
    def _extract_file_from_context(self) -> Optional[str]:
        """Extract file from context if available"""
        if self.intelligence.context.current_files:
            return self.intelligence.context.current_files[-1]
        return None
    
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the generated plan"""
        results = []
        
        for step in execution_plan:
            method_name = step.get('method')
            if not method_name or not hasattr(self.casey, method_name):
                results.append({
                    'step': step.get('step'),
                    'success': False,
                    'error': f"Method {method_name} not found"
                })
                continue
            
            try:
                method = getattr(self.casey, method_name)
                
                # Prepare arguments
                kwargs = {}
                if step.get('target'):
                    kwargs['file_path'] = step['target']
                if step.get('query'):
                    kwargs['query'] = step['query']
                if step.get('code'):
                    kwargs['code'] = step['code']
                if step.get('error'):
                    kwargs['error_message'] = step['error']
                if step.get('agent_name'):
                    kwargs['agent_name'] = step['agent_name']
                
                # Execute
                result = method(**kwargs)
                
                # Learn from result
                if self.intelligence:
                    self.intelligence.learn_from_interaction(
                        step.get('description', ''),
                        Intent(action=step.get('action')),
                        result
                    )
                
                results.append({
                    'step': step.get('step'),
                    'success': result.get('success', True) if isinstance(result, dict) else True,
                    'result': result,
                    'description': step.get('description')
                })
                
            except Exception as e:
                results.append({
                    'step': step.get('step'),
                    'success': False,
                    'error': str(e),
                    'description': step.get('description')
                })
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'steps_completed': len([r for r in results if r.get('success')]),
            'total_steps': len(results)
        }


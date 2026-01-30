#!/usr/bin/env python3
"""
AutoGen Studio Integration for NAE
Connects Casey and other NAE agents to AutoGen Studio
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add NAE to path
nae_root = Path(__file__).parent
sys.path.insert(0, str(nae_root))

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("Warning: AutoGen not available. Install with: pip install pyautogen")

try:
    from autogen_agentchat.agents import AssistantAgent as StudioAssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    AUTOGEN_STUDIO_AVAILABLE = True
except ImportError:
    AUTOGEN_STUDIO_AVAILABLE = False
    print("Warning: AutoGen Studio agents not available. Install with: pip install autogenstudio")

from agents.casey import CaseyAgent

class CaseyAutoGenAgent:
    """
    Wrapper to make Casey compatible with AutoGen Studio
    """
    
    def __init__(self):
        self.casey = CaseyAgent()
        self.name = "Casey"
        
    def process_message(self, message: str, sender: str = "User") -> str:
        """
        Process a message from AutoGen Studio and return Casey's response
        """
        try:
            # Use Casey's intelligent command processing if available
            if hasattr(self.casey, 'intelligence') and self.casey.intelligence:
                from agents.casey_intelligent_command import IntelligentCommandProcessor
                processor = IntelligentCommandProcessor(self.casey, self.casey.intelligence)
                understanding = processor.process_command(message)
                
                # Execute the plan
                if understanding.get('execution_plan'):
                    result = processor.execute_plan(understanding['execution_plan'])
                    
                    # Format response
                    response_parts = []
                    response_parts.append(f"**Understanding:** {understanding['intent']['action']}")
                    if understanding['intent'].get('target'):
                        response_parts.append(f"**Target:** {understanding['intent']['target']}")
                    if understanding['intent'].get('confidence'):
                        response_parts.append(f"**Confidence:** {int(understanding['intent']['confidence'] * 100)}%")
                    
                    response_parts.append(f"\n**Execution:** {result.get('steps_completed', 0)}/{result.get('total_steps', 0)} steps completed")
                    
                    if result.get('success'):
                        response_parts.append("✅ **Success!**")
                        if result.get('results'):
                            for step_result in result['results']:
                                if step_result.get('description'):
                                    response_parts.append(f"- {step_result['description']}")
                    else:
                        response_parts.append("❌ **Failed**")
                        for step_result in result.get('results', []):
                            if not step_result.get('success') and step_result.get('error'):
                                response_parts.append(f"- Error: {step_result['error']}")
                    
                    # Add suggestions
                    if understanding.get('suggestions'):
                        response_parts.append("\n**Suggestions:**")
                        for suggestion in understanding['suggestions'][:3]:
                            response_parts.append(f"- {suggestion}")
                    
                    return "\n".join(response_parts)
            
            # Fallback to direct command execution
            return self._execute_direct_command(message)
            
        except Exception as e:
            return f"Error processing command: {str(e)}"
    
    def _execute_direct_command(self, message: str) -> str:
        """Execute command directly using Casey's methods"""
        message_lower = message.lower()
        
        try:
            if message_lower.startswith('read') or 'read file' in message_lower:
                # Extract file path
                parts = message.split()
                file_path = None
                for i, part in enumerate(parts):
                    if part.lower() == 'read' and i + 1 < len(parts):
                        file_path = ' '.join(parts[i+1:])
                        break
                    elif part.lower() == 'file' and i + 1 < len(parts):
                        file_path = ' '.join(parts[i+1:])
                        break
                
                if file_path:
                    result = self.casey.read_file(file_path)
                    if result.get('success'):
                        content = result.get('content', '')
                        lines = content.split('\n')
                        preview = '\n'.join(lines[:50])
                        if len(lines) > 50:
                            preview += f"\n\n... ({len(lines) - 50} more lines)"
                        return f"**File:** {file_path}\n\n```\n{preview}\n```"
                    else:
                        return f"Error reading file: {result.get('error', 'Unknown error')}"
            
            elif message_lower.startswith('list') or 'list directory' in message_lower:
                parts = message.split()
                directory = '.'
                if len(parts) > 1:
                    directory = ' '.join(parts[1:])
                
                result = self.casey.list_directory(directory)
                if result.get('success'):
                    files = result.get('files', [])
                    file_list = '\n'.join([f"- {f}" for f in files[:20]])
                    if len(files) > 20:
                        file_list += f"\n... ({len(files) - 20} more items)"
                    return f"**Directory:** {directory}\n\n{file_list}"
                else:
                    return f"Error listing directory: {result.get('error', 'Unknown error')}"
            
            elif message_lower.startswith('search') or 'grep' in message_lower:
                result = self.casey.semantic_search(message)
                if isinstance(result, dict) and result.get('success'):
                    matches = result.get('matches', [])
                    if matches:
                        match_list = '\n'.join([f"- {m.get('file', 'Unknown')}: {m.get('context', '')[:100]}" for m in matches[:10]])
                        return f"**Search Results:**\n\n{match_list}"
                    else:
                        return "No matches found"
                return f"Search completed: {message}"
            
            elif message_lower.startswith('status') or 'system status' in message_lower:
                # Get system status
                status_info = []
                status_info.append("**System Status:**")
                
                # Get agents
                if hasattr(self.casey, 'monitored_agents'):
                    agents = self.casey.monitored_agents
                    if isinstance(agents, dict):
                        status_info.append(f"**Active Agents:** {len(agents)}")
                        for name in agents.keys():
                            status_info.append(f"- {name}")
                    elif isinstance(agents, list):
                        status_info.append(f"**Active Agents:** {len(agents)}")
                
                # Get system health
                try:
                    import psutil
                    cpu = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory().percent
                    status_info.append(f"\n**System Health:**")
                    status_info.append(f"- CPU: {cpu}%")
                    status_info.append(f"- Memory: {memory}%")
                except:
                    pass
                
                return "\n".join(status_info)
            
            else:
                return f"I received your message: '{message}'\n\nI can help you with:\n- Reading files (e.g., 'read agents/optimus.py')\n- Listing directories (e.g., 'list agents')\n- Searching codebase (e.g., 'search trading strategies')\n- Checking system status (e.g., 'status')\n- Executing code\n- Debugging errors\n\nWhat would you like me to do?"
        
        except Exception as e:
            return f"Error: {str(e)}"

def create_autogen_agents():
    """
    Create AutoGen agents for Studio integration
    """
    if not AUTOGEN_AVAILABLE:
        raise ImportError("AutoGen not available. Install with: pip install pyautogen")
    
    # Create Casey wrapper
    casey_wrapper = CaseyAutoGenAgent()
    
    # Create AutoGen AssistantAgent for Casey
    casey_agent = AssistantAgent(
        name="Casey",
        system_message="""You are Casey, the AI-powered system orchestrator for NAE (Neural Agency Engine).

Your responsibilities:
- Build or refine all agents dynamically
- Monitor agent CPU/Memory usage
- Send email alerts on agent crashes or high resource usage
- Support AutoGen communication
- File Operations: Read, write, edit, delete, list files
- Codebase Search: Grep, glob, semantic search across codebase
- Code Execution: Execute Python code, terminal commands, agent methods
- Context Understanding: Analyze multiple files, understand relationships
- Debugging & Testing: Debug code, suggest fixes, run tests

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading

When receiving commands:
1. Understand the intent
2. Execute the appropriate action using available tools
3. Provide clear feedback
4. Suggest next steps

Always be helpful, clear, and focused on achieving NAE goals.""",
        llm_config=False,  # Will be configured by AutoGen Studio
        human_input_mode="NEVER",
    )
    
    # Create UserProxyAgent
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=0,
        code_execution_config={
            "work_dir": str(nae_root),
            "use_docker": False
        }
    )
    
    # Add custom function to Casey agent for processing commands
    def casey_process_command(message: str) -> str:
        """Process command through Casey"""
        return casey_wrapper.process_message(message)
    
    # Register function
    casey_agent.register_function(
        function_map={
            "process_command": casey_process_command
        }
    )
    
    return {
        "casey": casey_agent,
        "user": user_proxy,
        "casey_wrapper": casey_wrapper
    }

def create_group_chat():
    """
    Create a group chat for AutoGen Studio
    """
    agents = create_autogen_agents()
    
    groupchat = GroupChat(
        agents=[agents["casey"], agents["user"]],
        messages=[],
        max_round=10
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=False  # Will be configured by AutoGen Studio
    )
    
    return {
        "groupchat": groupchat,
        "manager": manager,
        "agents": agents
    }

# AutoGen Studio compatible functions
def casey_read_file(file_path: str) -> str:
    """Read a file using Casey"""
    casey = CaseyAgent()
    result = casey.read_file(file_path)
    if result.get('success'):
        return result.get('content', '')
    return f"Error: {result.get('error', 'Unknown error')}"

def casey_list_directory(directory: str = ".") -> str:
    """List directory using Casey"""
    casey = CaseyAgent()
    result = casey.list_directory(directory)
    if result.get('success'):
        files = result.get('files', [])
        return "\n".join(files)
    return f"Error: {result.get('error', 'Unknown error')}"

def casey_search_codebase(query: str) -> str:
    """Search codebase using Casey"""
    casey = CaseyAgent()
    result = casey.semantic_search(query)
    if isinstance(result, dict) and result.get('success'):
        matches = result.get('matches', [])
        return "\n".join([f"{m.get('file', 'Unknown')}: {m.get('context', '')}" for m in matches[:10]])
    return "No matches found"

def casey_execute_python(code: str) -> str:
    """Execute Python code using Casey"""
    casey = CaseyAgent()
    result = casey.execute_python_code(code)
    if result.get('success'):
        return result.get('output', 'Execution successful')
    return f"Error: {result.get('error', 'Unknown error')}"

def casey_get_system_status() -> str:
    """Get system status using Casey"""
    casey = CaseyAgent()
    status = []
    
    if hasattr(casey, 'monitored_agents'):
        agents = casey.monitored_agents
        if isinstance(agents, dict):
            status.append(f"Active Agents: {len(agents)}")
            status.extend([f"- {name}" for name in agents.keys()])
    
    try:
        import psutil
        status.append(f"CPU: {psutil.cpu_percent(interval=1)}%")
        status.append(f"Memory: {psutil.virtual_memory().percent}%")
    except:
        pass
    
    return "\n".join(status)

if __name__ == "__main__":
    print("AutoGen Studio Integration for NAE")
    print("=" * 50)
    
    if not AUTOGEN_AVAILABLE:
        print("Error: AutoGen not installed")
        print("Install with: pip install pyautogen")
        sys.exit(1)
    
    # Create agents
    agents = create_autogen_agents()
    print(f"✅ Created {len(agents)} agents")
    print(f"   - Casey: {agents['casey'].name}")
    print(f"   - User: {agents['user'].name}")
    
    # Create group chat
    group_chat = create_group_chat()
    print(f"✅ Created group chat with {len(group_chat['groupchat'].agents)} agents")
    
    print("\n" + "=" * 50)
    print("AutoGen Studio Integration Ready!")
    print("=" * 50)
    print("\nTo use with AutoGen Studio:")
    print("1. Start AutoGen Studio: autogenstudio ui")
    print("2. Import this configuration")
    print("3. Start chatting with Casey!")


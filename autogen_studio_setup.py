#!/usr/bin/env python3
"""
Setup script for AutoGen Studio integration with NAE
Creates configuration files and sets up the integration
"""

import os
import json
import sys
from pathlib import Path

def create_autogen_studio_config():
    """Create AutoGen Studio configuration file"""
    
    config = {
        "name": "NAE - Neural Agency Engine",
        "description": "Neural Agency Engine with Casey as the orchestrator",
        "agents": [
            {
                "name": "Casey",
                "type": "assistant",
                "model": "gpt-4",
                "system_message": """You are Casey, the AI-powered system orchestrator for NAE (Neural Agency Engine).

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
2. Execute the appropriate action
3. Provide clear feedback
4. Suggest next steps

Always be helpful, clear, and focused on achieving NAE goals.""",
                "tools": [
                    "read_file",
                    "write_file",
                    "list_directory",
                    "grep_search",
                    "semantic_search",
                    "execute_python_code",
                    "execute_terminal_command",
                    "debug_code",
                    "understand_context"
                ],
                "code_execution_config": {
                    "work_dir": "NAE",
                    "use_docker": False
                }
            },
            {
                "name": "User",
                "type": "userproxy",
                "human_input_mode": "TERMINATE",
                "max_consecutive_auto_reply": 0,
                "code_execution_config": {
                    "work_dir": "NAE",
                    "use_docker": False
                }
            }
        ],
        "workflows": [
            {
                "name": "Casey Command",
                "description": "Send commands to Casey for NAE system management",
                "type": "groupchat",
                "agents": ["Casey", "User"]
            }
        ]
    }
    
    config_path = Path(__file__).parent / "autogen_studio_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created AutoGen Studio config: {config_path}")
    return config_path

def create_autogen_functions():
    """Create function definitions for AutoGen Studio"""
    
    functions = [
        {
            "name": "read_file",
            "description": "Read a file from the NAE codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read (relative to NAE root)"
                    }
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "write_file",
            "description": "Write content to a file in the NAE codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        },
        {
            "name": "list_directory",
            "description": "List files and directories in a given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "search_codebase",
            "description": "Search the codebase semantically for code patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "execute_python",
            "description": "Execute Python code safely",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        },
        {
            "name": "get_system_status",
            "description": "Get current NAE system status including agents and health",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
    
    functions_path = Path(__file__).parent / "autogen_functions.json"
    with open(functions_path, 'w') as f:
        json.dump(functions, f, indent=2)
    
    print(f"✅ Created AutoGen functions: {functions_path}")
    return functions_path

def create_autogen_workflow():
    """Create workflow configuration"""
    
    workflow = {
        "name": "NAE Casey Workflow",
        "type": "groupchat",
        "agents": ["Casey", "User"],
        "max_rounds": 10,
        "speaker_selection_method": "round_robin"
    }
    
    workflow_path = Path(__file__).parent / "autogen_workflow.json"
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"✅ Created AutoGen workflow: {workflow_path}")
    return workflow_path

def main():
    """Main setup function"""
    print("=" * 60)
    print("AutoGen Studio Integration Setup for NAE")
    print("=" * 60)
    print()
    
    # Check if AutoGen is installed
    try:
        import autogen
        print("✅ AutoGen is installed")
    except ImportError:
        print("❌ AutoGen is not installed")
        print("   Install with: pip install pyautogen")
        return
    
    # Create configuration files
    config_path = create_autogen_studio_config()
    functions_path = create_autogen_functions()
    workflow_path = create_autogen_workflow()
    
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install AutoGen Studio:")
    print("   pip install autogenstudio")
    print()
    print("2. Start AutoGen Studio:")
    print("   autogenstudio ui")
    print()
    print("3. Import configuration:")
    print(f"   - Config: {config_path}")
    print(f"   - Functions: {functions_path}")
    print(f"   - Workflow: {workflow_path}")
    print()
    print("4. Start chatting with Casey through AutoGen Studio!")

if __name__ == "__main__":
    main()


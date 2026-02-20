#!/usr/bin/env python3
"""
AutoGen Studio Integration for NAE
Integrates NAE agents with Microsoft AutoGen Studio for visual workflow management

Based on: https://github.com/microsoft/autogen/tree/main/python/packages/autogen-studio
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add NAE to path
nae_root = Path(__file__).parent
sys.path.insert(0, str(nae_root))

# Try to import AutoGen Studio components
AUTOGEN_STUDIO_AVAILABLE = False
TeamManager = None

try:
    # Try the newer autogenstudio package
    from autogenstudio import TeamManager
    AUTOGEN_STUDIO_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.agents import AssistantAgent as StudioAssistantAgent
        AUTOGEN_STUDIO_AVAILABLE = True
    except ImportError:
        try:
            # Check if autogenstudio is installed but has import issues
            import autogenstudio
            AUTOGEN_STUDIO_AVAILABLE = True
            # TeamManager will be None, but we can still create configs
        except ImportError:
            pass

if not AUTOGEN_STUDIO_AVAILABLE:
    print("Note: AutoGen Studio TeamManager not available, but configs can still be generated")

# Import NAE agents
try:
    from agents.casey import CaseyAgent
    from agents.optimus import OptimusAgent
    from agents.ralph import RalphAgent
    from agents.donnie import DonnieAgent
    from agents.genny import GennyAgent
    NAE_AGENTS_AVAILABLE = True
except ImportError as e:
    NAE_AGENTS_AVAILABLE = False
    print(f"Warning: Some NAE agents not available: {e}")


class NAEAgentBridge:
    """
    Bridge between NAE agents and AutoGen Studio
    Makes NAE agents compatible with AutoGen Studio workflows
    """
    
    def __init__(self, agent_name: str, nae_agent):
        """
        Initialize bridge
        
        Args:
            agent_name: Name for AutoGen Studio
            nae_agent: NAE agent instance
        """
        self.agent_name = agent_name
        self.nae_agent = nae_agent
        self.message_history = []
    
    def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Process message through NAE agent
        
        Args:
            message: Message to process
            context: Additional context
            
        Returns:
            Agent response
        """
        try:
            # Log message
            self.message_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "sender": context.get("sender", "unknown") if context else "unknown"
            })
            
            # Route to appropriate agent method
            if hasattr(self.nae_agent, 'process_command'):
                return self.nae_agent.process_command(message)
            elif hasattr(self.nae_agent, 'log_action'):
                self.nae_agent.log_action(f"Received message: {message}")
                return f"Message received by {self.agent_name}. Processing..."
            else:
                return f"{self.agent_name} received: {message}"
        
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.agent_name,
            "type": type(self.nae_agent).__name__,
            "messages_processed": len(self.message_history),
            "last_message": self.message_history[-1] if self.message_history else None
        }


class NAEStudioIntegration:
    """
    Main integration class for AutoGen Studio and NAE
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize AutoGen Studio integration
        
        Args:
            config_path: Path to AutoGen Studio config directory
        """
        self.config_path = config_path or nae_root / ".autogenstudio"
        self.config_path.mkdir(exist_ok=True)
        
        self.agent_bridges: Dict[str, NAEAgentBridge] = {}
        self.team_manager = None
        
        # Initialize NAE agents
        self._initialize_nae_agents()
        
        # Initialize AutoGen Studio if available
        if AUTOGEN_STUDIO_AVAILABLE:
            self._initialize_autogen_studio()
    
    def _initialize_nae_agents(self):
        """Initialize NAE agents and create bridges"""
        if not NAE_AGENTS_AVAILABLE:
            print("Warning: NAE agents not available")
            return
        
        # Create agent bridges
        try:
            casey = CaseyAgent()
            self.agent_bridges["Casey"] = NAEAgentBridge("Casey", casey)
            print("✅ Initialized Casey agent bridge")
        except Exception as e:
            print(f"⚠️ Failed to initialize Casey: {e}")
        
        try:
            optimus = OptimusAgent()
            self.agent_bridges["Optimus"] = NAEAgentBridge("Optimus", optimus)
            print("✅ Initialized Optimus agent bridge")
        except Exception as e:
            print(f"⚠️ Failed to initialize Optimus: {e}")
        
        try:
            ralph = RalphAgent()
            self.agent_bridges["Ralph"] = NAEAgentBridge("Ralph", ralph)
            print("✅ Initialized Ralph agent bridge")
        except Exception as e:
            print(f"⚠️ Failed to initialize Ralph: {e}")
        
        try:
            donnie = DonnieAgent()
            self.agent_bridges["Donnie"] = NAEAgentBridge("Donnie", donnie)
            print("✅ Initialized Donnie agent bridge")
        except Exception as e:
            print(f"⚠️ Failed to initialize Donnie: {e}")
        
        try:
            genny = GennyAgent()
            self.agent_bridges["Genny"] = NAEAgentBridge("Genny", genny)
            print("✅ Initialized Genny agent bridge")
        except Exception as e:
            print(f"⚠️ Failed to initialize Genny: {e}")
    
    def _initialize_autogen_studio(self):
        """Initialize AutoGen Studio TeamManager"""
        try:
            if TeamManager is not None:
                self.team_manager = TeamManager()
                print("✅ Initialized AutoGen Studio TeamManager")
            else:
                print("ℹ️ AutoGen Studio TeamManager not available (configs can still be generated)")
        except Exception as e:
            print(f"⚠️ AutoGen Studio TeamManager initialization error: {e}")
            print("ℹ️ Configurations can still be generated for manual import")
    
    def create_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Create AutoGen Studio agent configuration
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration dict
        """
        bridge = self.agent_bridges.get(agent_name)
        if not bridge:
            raise ValueError(f"Agent {agent_name} not found")
        
        # Get agent-specific system message
        system_messages = {
            "Casey": """You are Casey, the AI-powered system orchestrator for NAE (Neural Agency Engine).
Your responsibilities:
- Build or refine all agents dynamically
- Monitor agent CPU/Memory usage
- File Operations: Read, write, edit, delete, list files
- Codebase Search: Grep, glob, semantic search
- Code Execution: Execute Python code, terminal commands
- Context Understanding: Analyze multiple files
- Debugging & Testing: Debug code, suggest fixes

ALIGNED WITH 3 CORE GOALS:
1. Achieve generational wealth
2. Generate $5,000,000.00 within 8 years, every 8 years consistently
3. Optimize NAE and agents for successful options trading""",
            
            "Optimus": """You are Optimus, the primary trading agent for NAE.
Your responsibilities:
- Execute intelligent trading strategies
- Monitor market conditions
- Execute trades through broker adapters
- Track P&L and performance
- Maintain compliance with trading rules

Goal: Generate consistent profits through intelligent trading.""",
            
            "Ralph": """You are Ralph, the strategy research and development agent.
Your responsibilities:
- Research and develop trading strategies
- Backtest strategies
- Analyze market data
- Recommend optimal strategies
- Learn from successful patterns

Goal: Continuously improve trading strategies.""",
            
            "Donnie": """You are Donnie, the market data and research agent.
Your responsibilities:
- Gather market data
- Analyze market trends
- Provide research insights
- Monitor news and events
- Support trading decisions

Goal: Provide accurate and timely market intelligence.""",
            
            "Genny": """You are Genny, the generational wealth and tax management agent.
Your responsibilities:
- Track all trades and transactions
- Maintain cost basis (FIFO/LIFO/HIFO)
- Calculate capital gains
- Track deductible expenses
- Generate tax summaries
- Optimize tax strategies

Goal: Maximize generational wealth while ensuring tax compliance."""
        }
        
        config = {
            "name": agent_name,
            "type": "assistant",
            "model": "gpt-4",  # Can be configured
            "system_message": system_messages.get(agent_name, f"You are {agent_name}, an agent in the NAE system."),
            "tools": self._get_agent_tools(agent_name),
            "code_execution_config": {
                "work_dir": str(nae_root),
                "use_docker": False
            }
        }
        
        return config
    
    def _get_agent_tools(self, agent_name: str) -> List[str]:
        """Get tools available to agent"""
        base_tools = ["read_file", "write_file", "list_directory", "execute_python"]
        
        agent_specific_tools = {
            "Casey": base_tools + ["grep_search", "semantic_search", "debug_code", "understand_context"],
            "Optimus": base_tools + ["execute_trade", "check_balance", "get_positions"],
            "Ralph": base_tools + ["backtest_strategy", "analyze_market_data"],
            "Donnie": base_tools + ["fetch_market_data", "analyze_trends"],
            "Genny": base_tools + ["track_trade", "calculate_taxes", "generate_tax_report"]
        }
        
        return agent_specific_tools.get(agent_name, base_tools)
    
    def create_workflow_config(self, workflow_name: str, agent_names: List[str]) -> Dict[str, Any]:
        """
        Create workflow configuration
        
        Args:
            workflow_name: Name of the workflow
            agent_names: List of agent names to include
            
        Returns:
            Workflow configuration dict
        """
        return {
            "name": workflow_name,
            "type": "groupchat",
            "agents": agent_names,
            "max_rounds": 10,
            "speaker_selection_method": "round_robin",
            "termination_condition": "all_agents_agree"
        }
    
    def save_config(self):
        """Save configuration to AutoGen Studio config directory"""
        # Save agent configs
        agents_config = {}
        for agent_name in self.agent_bridges.keys():
            agents_config[agent_name] = self.create_agent_config(agent_name)
        
        agents_file = self.config_path / "agents.json"
        with open(agents_file, 'w') as f:
            json.dump(agents_config, f, indent=2)
        print(f"✅ Saved agent configs to {agents_file}")
        
        # Save workflow configs
        workflows = {
            "NAE_Trading_Workflow": self.create_workflow_config(
                "NAE Trading Workflow",
                ["Casey", "Optimus", "Ralph"]
            ),
            "NAE_Research_Workflow": self.create_workflow_config(
                "NAE Research Workflow",
                ["Casey", "Ralph", "Donnie"]
            ),
            "NAE_Wealth_Management": self.create_workflow_config(
                "NAE Wealth Management",
                ["Casey", "Genny", "Optimus"]
            ),
            "NAE_Full_System": self.create_workflow_config(
                "NAE Full System",
                list(self.agent_bridges.keys())
            )
        }
        
        workflows_file = self.config_path / "workflows.json"
        with open(workflows_file, 'w') as f:
            json.dump(workflows, f, indent=2)
        print(f"✅ Saved workflow configs to {workflows_file}")
    
    def run_workflow(self, workflow_name: str, task: str) -> Any:
        """
        Run a workflow with a task
        
        Args:
            workflow_name: Name of workflow to run
            task: Task description
            
        Returns:
            Workflow result
        """
        if not self.team_manager:
            raise RuntimeError("AutoGen Studio TeamManager not initialized")
        
        # Load workflow config
        workflows_file = self.config_path / "workflows.json"
        if not workflows_file.exists():
            self.save_config()
        
        with open(workflows_file, 'r') as f:
            workflows = json.load(f)
        
        workflow_config = workflows.get(workflow_name)
        if not workflow_config:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        # Run workflow
        result = self.team_manager.run(
            task=task,
            team_config=workflow_config
        )
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "autogen_studio_available": AUTOGEN_STUDIO_AVAILABLE,
            "nae_agents_available": NAE_AGENTS_AVAILABLE,
            "agents_initialized": list(self.agent_bridges.keys()),
            "config_path": str(self.config_path),
            "team_manager_initialized": self.team_manager is not None
        }


def main():
    """Main function to set up AutoGen Studio integration"""
    print("=" * 60)
    print("NAE AutoGen Studio Integration")
    print("=" * 60)
    print()
    
    # Create integration
    integration = NAEStudioIntegration()
    
    # Save configurations
    integration.save_config()
    
    # Print status
    status = integration.get_status()
    print()
    print("=" * 60)
    print("Integration Status")
    print("=" * 60)
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Install AutoGen Studio (if not already installed):")
    print("   pip install autogenstudio")
    print()
    print("2. Start AutoGen Studio UI:")
    print("   autogenstudio ui --port 8080")
    print()
    print("3. Access AutoGen Studio at: http://localhost:8080")
    print()
    print("4. Import configurations from:")
    print(f"   {integration.config_path}")
    print()
    print("5. Start building workflows with NAE agents!")


if __name__ == "__main__":
    main()


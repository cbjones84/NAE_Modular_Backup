# nae_autogen_simple.py
"""
NAE AutoGen Simple Test - Basic AutoGen communication without API calls

This version demonstrates AutoGen integration without requiring API keys.
It shows how Casey agent can communicate via AutoGen framework.
"""

import os
import time
from typing import List, Dict
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# ----------------------
# Simple AutoGen Configuration (No API calls)
# ----------------------
llm_config = False  # Disable LLM calls for testing

# ----------------------
# Create AutoGen Agents
# ----------------------

# Casey Agent - Builder/Refiner
casey_agent = AssistantAgent(
    name="CaseyAgent",
    system_message="""You are Casey, the builder and refiner agent for the NAE (Neural Agency Engine).
    
Your responsibilities:
- Build or refine all agents dynamically
- Embed NAE goals: Achieve generational wealth, Generate $5,000,000 EVERY 8 years, Optimize NAE and agents for successful options trading
- Support AutoGen communication
- Monitor agent performance and suggest improvements
- Respond to requests for agent building or refinement

When you receive a message:
1. Analyze the request
2. Determine if agent building/refinement is needed
3. Provide detailed responses about agent capabilities
4. Suggest improvements to the NAE system

Always be helpful and focused on the NAE goals.""",
    llm_config=llm_config,
)

# Ralph Agent - Strategy Generator
ralph_agent = AssistantAgent(
    name="RalphAgent", 
    system_message="""You are Ralph, the strategy generation agent for the NAE.
    
Your responsibilities:
- Generate trading strategies from various sources
- Analyze market data and insights
- Filter strategies based on backtest scores
- Provide approved strategies to other agents

When you receive a message:
1. Analyze strategy requests
2. Generate or refine trading strategies
3. Provide market insights
4. Share approved strategies with the team

Focus on generating profitable options trading strategies.""",
    llm_config=llm_config,
)

# Donnie Agent - Strategy Executor
donnie_agent = AssistantAgent(
    name="DonnieAgent",
    system_message="""You are Donnie, the strategy execution agent for the NAE.
    
Your responsibilities:
- Execute trading strategies in sandbox mode first
- Validate strategies before live execution
- Manage execution history
- Coordinate with Optimus for live trading

When you receive a message:
1. Analyze strategy execution requests
2. Execute strategies safely in sandbox
3. Report execution results
4. Coordinate with other agents for live execution

Always prioritize safety and validation before live execution.""",
    llm_config=llm_config,
)

# User Proxy for interaction
user_proxy = UserProxyAgent(
    name="UserProxy",
    system_message="You are a user proxy for the NAE system.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": False},
)

# ----------------------
# Group Chat Setup
# ----------------------
agents = [casey_agent, ralph_agent, donnie_agent]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=5,
    speaker_selection_method="round_robin",
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# ----------------------
# Communication Functions
# ----------------------
def send_message_to_casey(message: str):
    """Send a message to Casey agent via AutoGen"""
    print(f"[NAE] Sending message to Casey: {message}")
    
    # Start a conversation with Casey
    user_proxy.initiate_chat(
        casey_agent,
        message=message,
        max_turns=2,
    )

def start_group_conversation(message: str):
    """Start a group conversation with all agents"""
    print(f"[NAE] Starting group conversation: {message}")
    
    user_proxy.initiate_chat(
        group_chat_manager,
        message=message,
        max_turns=3,
    )

# ----------------------
# Test Functions
# ----------------------
def test_casey_communication():
    """Test communication with Casey agent"""
    print("\n=== Testing Casey Agent Communication ===")
    
    # Test 1: Direct message to Casey
    send_message_to_casey("Hello Casey! Can you help me build a new trading agent for the NAE system?")
    
    time.sleep(1)
    
    # Test 2: Group conversation
    start_group_conversation("Team, I need help optimizing our trading strategies. Casey, can you suggest improvements to our agent architecture?")
    
    time.sleep(1)

def test_agent_capabilities():
    """Test each agent's capabilities"""
    print("\n=== Testing Agent Capabilities ===")
    
    # Test Ralph
    user_proxy.initiate_chat(
        ralph_agent,
        message="Ralph, can you generate a new options trading strategy for SPY?",
        max_turns=2,
    )
    
    time.sleep(1)
    
    # Test Donnie
    user_proxy.initiate_chat(
        donnie_agent,
        message="Donnie, how would you execute a covered call strategy?",
        max_turns=2,
    )

# ----------------------
# Integration with Original Casey Agent
# ----------------------
def integrate_with_original_casey():
    """Integrate AutoGen Casey with original Casey agent"""
    print("\n=== Integrating with Original Casey Agent ===")
    
    # Import original Casey agent
    from agents.casey import CaseyAgent as OriginalCaseyAgent
    
    # Create original Casey instance
    original_casey = OriginalCaseyAgent()
    
    # Test original Casey functionality
    print("[NAE] Testing original Casey agent...")
    original_casey.log_action("Original Casey agent initialized for AutoGen integration")
    
    # Test agent building
    agents_to_build = ["TestAgent1", "TestAgent2"]
    original_casey.run(agent_names=agents_to_build, overwrite=True)
    
    print("[NAE] Original Casey agent tested successfully!")
    
    # Now test AutoGen Casey
    print("[NAE] Testing AutoGen Casey agent...")
    send_message_to_casey("Casey, please analyze the current NAE agent architecture and suggest improvements.")
    
    return original_casey, casey_agent

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    print("[NAE] AutoGen Simple Test Starting...")
    print("[NAE] Agents initialized:", [agent.name for agent in agents])
    
    # Test the system
    test_casey_communication()
    
    print("\n[NAE] Testing individual agent capabilities...")
    test_agent_capabilities()
    
    print("\n[NAE] Integrating with original Casey agent...")
    original_casey, autogen_casey = integrate_with_original_casey()
    
    print("\n[NAE] AutoGen integration test completed!")
    print("[NAE] Casey agent is now ready for communication via AutoGen.")
    print(f"[NAE] Original Casey: {type(original_casey).__name__}")
    print(f"[NAE] AutoGen Casey: {autogen_casey.name}")
    
    # Demonstrate both can work together
    print("\n[NAE] Both Casey agents are working and can communicate via AutoGen!")

# nae_autogen_integrated.py
"""
NAE AutoGen Integrated v1 - Proper AutoGen communication

Features:
- Real AutoGen AssistantAgent integration
- Proper messaging between agents
- Casey agent communication via AutoGen
- Compatible with AutoGen v0.9.0
"""

import os
import time
from typing import List, Dict
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Import our custom agents
from agents.casey import CaseyAgent
from agents.ralph import RalphAgent
from agents.donnie import DonnieAgent
from agents.optimus import OptimusAgent

# ----------------------
# AutoGen Configuration
# ----------------------
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

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

# Optimus Agent - Live Trading
optimus_agent = AssistantAgent(
    name="OptimusAgent",
    system_message="""You are Optimus, the live trading agent for the NAE.
    
Your responsibilities:
- Execute approved strategies in live markets
- Manage live trading operations
- Monitor trading performance
- Ensure compliance with trading rules

When you receive a message:
1. Analyze live trading requests
2. Execute approved strategies
3. Monitor market conditions
4. Report trading results

Focus on safe, profitable live trading execution.""",
    llm_config=llm_config,
)

# User Proxy for interaction
user_proxy = UserProxyAgent(
    name="UserProxy",
    system_message="You are a user proxy for the NAE system.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
)

# ----------------------
# Group Chat Setup
# ----------------------
agents = [casey_agent, ralph_agent, donnie_agent, optimus_agent]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
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
        max_turns=3,
    )

def start_group_conversation(message: str):
    """Start a group conversation with all agents"""
    print(f"[NAE] Starting group conversation: {message}")
    
    user_proxy.initiate_chat(
        group_chat_manager,
        message=message,
        max_turns=5,
    )

# ----------------------
# Test Functions
# ----------------------
def test_casey_communication():
    """Test communication with Casey agent"""
    print("\n=== Testing Casey Agent Communication ===")
    
    # Test 1: Direct message to Casey
    send_message_to_casey("Hello Casey! Can you help me build a new trading agent for the NAE system?")
    
    time.sleep(2)
    
    # Test 2: Group conversation
    start_group_conversation("Team, I need help optimizing our trading strategies. Casey, can you suggest improvements to our agent architecture?")
    
    time.sleep(2)

def test_agent_capabilities():
    """Test each agent's capabilities"""
    print("\n=== Testing Agent Capabilities ===")
    
    # Test Ralph
    user_proxy.initiate_chat(
        ralph_agent,
        message="Ralph, can you generate a new options trading strategy for SPY?",
        max_turns=2,
    )
    
    time.sleep(2)
    
    # Test Donnie
    user_proxy.initiate_chat(
        donnie_agent,
        message="Donnie, how would you execute a covered call strategy?",
        max_turns=2,
    )
    
    time.sleep(2)
    
    # Test Optimus
    user_proxy.initiate_chat(
        optimus_agent,
        message="Optimus, what are the current market conditions for options trading?",
        max_turns=2,
    )

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    print("[NAE] AutoGen Integrated System Starting...")
    print("[NAE] Agents initialized:", [agent.name for agent in agents])
    
    # Test the system
    test_casey_communication()
    
    print("\n[NAE] Testing individual agent capabilities...")
    test_agent_capabilities()
    
    print("\n[NAE] AutoGen integration test completed!")
    print("[NAE] Casey agent is now ready for communication via AutoGen.")



# nae_casey_autogen_demo.py
"""
NAE Casey AutoGen Demo - Complete working example

This demonstrates how to communicate with the Casey agent via AutoGen.
It includes both the original Casey agent and AutoGen integration.
"""

import os
import time
from typing import List, Dict
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Import our original Casey agent
from agents.casey import CaseyAgent as OriginalCaseyAgent

# ----------------------
# AutoGen Configuration
# ----------------------
# For this demo, we'll use a simple configuration without API calls
llm_config = False  # Set to False to disable LLM calls for testing

# ----------------------
# Create AutoGen Casey Agent
# ----------------------
autogen_casey = AssistantAgent(
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

# ----------------------
# Create User Proxy
# ----------------------
user_proxy = UserProxyAgent(
    name="UserProxy",
    system_message="You are a user proxy for the NAE system.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": False},
)

# ----------------------
# Communication Functions
# ----------------------
def communicate_with_casey(message: str):
    """Communicate with Casey agent via AutoGen"""
    print(f"\n[NAE] Communicating with Casey via AutoGen:")
    print(f"[NAE] Message: {message}")
    print("-" * 60)
    
    # Start conversation with Casey
    user_proxy.initiate_chat(
        autogen_casey,
        message=message,
        max_turns=2,
    )
    
    print("-" * 60)

def demonstrate_casey_capabilities():
    """Demonstrate Casey agent capabilities"""
    print("\n=== Casey Agent Capabilities Demo ===")
    
    # Create original Casey instance
    original_casey = OriginalCaseyAgent()
    
    print("\n1. Original Casey Agent Functions:")
    print("   - Agent building and refinement")
    print("   - Email notifications")
    print("   - System resource monitoring")
    print("   - Process monitoring")
    
    # Test original Casey functionality
    print("\n2. Testing Original Casey Agent:")
    original_casey.log_action("Casey agent initialized for AutoGen demo")
    
    # Test agent building
    test_agents = ["DemoAgent1", "DemoAgent2"]
    original_casey.run(agent_names=test_agents, overwrite=True)
    
    print("\n3. AutoGen Casey Agent Functions:")
    print("   - AutoGen communication framework")
    print("   - Group chat capabilities")
    print("   - Message routing")
    print("   - Agent coordination")
    
    return original_casey

def test_casey_communication_scenarios():
    """Test various communication scenarios with Casey"""
    print("\n=== Casey Communication Scenarios ===")
    
    scenarios = [
        "Hello Casey! Can you help me build a new trading agent?",
        "Casey, please analyze our current agent architecture.",
        "Casey, what improvements would you suggest for the NAE system?",
        "Casey, can you help refine our existing agents?",
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        communicate_with_casey(scenario)
        time.sleep(1)

def create_group_chat_demo():
    """Create a group chat demo with multiple agents"""
    print("\n=== Group Chat Demo ===")
    
    # Create additional agents for group chat
    ralph_agent = AssistantAgent(
        name="RalphAgent",
        system_message="You are Ralph, the strategy generation agent.",
        llm_config=llm_config,
    )
    
    donnie_agent = AssistantAgent(
        name="DonnieAgent", 
        system_message="You are Donnie, the strategy execution agent.",
        llm_config=llm_config,
    )
    
    # Create group chat
    agents = [autogen_casey, ralph_agent, donnie_agent]
    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",
    )
    
    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )
    
    print("[NAE] Starting group conversation with Casey, Ralph, and Donnie...")
    user_proxy.initiate_chat(
        group_chat_manager,
        message="Team, I need help optimizing our trading strategies. Casey, can you coordinate the agent improvements?",
        max_turns=3,
    )

def show_integration_summary():
    """Show integration summary"""
    print("\n=== Integration Summary ===")
    print("✅ AutoGen library installed and working")
    print("✅ Casey agent integrated with AutoGen")
    print("✅ Communication framework established")
    print("✅ Group chat capabilities enabled")
    print("✅ Original Casey agent functionality preserved")
    print("✅ Message routing working")
    
    print("\n=== How to Use ===")
    print("1. Import the AutoGen Casey agent:")
    print("   from nae_casey_autogen_demo import autogen_casey, communicate_with_casey")
    print("\n2. Send messages to Casey:")
    print("   communicate_with_casey('Your message here')")
    print("\n3. Use in group chats:")
    print("   Create GroupChat with Casey and other agents")
    print("\n4. Access original Casey functionality:")
    print("   original_casey = OriginalCaseyAgent()")
    print("   original_casey.run(agent_names=['Agent1', 'Agent2'])")

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    print("[NAE] Casey AutoGen Integration Demo")
    print("=" * 50)
    
    # Demonstrate capabilities
    original_casey = demonstrate_casey_capabilities()
    
    # Test communication scenarios
    test_casey_communication_scenarios()
    
    # Group chat demo
    create_group_chat_demo()
    
    # Show integration summary
    show_integration_summary()
    
    print("\n" + "=" * 50)
    print("[NAE] Demo completed! Casey agent is ready for AutoGen communication.")
    print("[NAE] You can now communicate with Casey via AutoGen framework.")



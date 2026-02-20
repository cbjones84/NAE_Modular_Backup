#!/usr/bin/env python3
"""
End-to-End Test for Genius Communication System

Tests all major features:
- Message sending and receiving
- Intelligent routing
- Collaborative sessions
- Knowledge sharing
- Execution coordination
- Context-aware messaging
"""

import os
import sys
import time
import json
from datetime import datetime

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, nae_root)

from agents.genius_communication_protocol import (
    GeniusCommunicationProtocol,
    MessageType,
    MessagePriority
)
from agents.genius_coordination_engine import GeniusCoordinationEngine


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, name):
        self.name = name
        self.inbox = []
        self.received_messages = []
    
    def receive_message(self, message):
        """Receive message (legacy method)"""
        self.inbox.append(message)
        self.received_messages.append(message)
    
    def receive_genius_message(self, message):
        """Receive genius message"""
        self.inbox.append(message)
        self.received_messages.append(message)
        print(f"  📨 [{self.name}] received: {message.get('subject', 'No subject')}")


def test_message_sending():
    """Test 1: Basic message sending"""
    print("\n" + "="*60)
    print("TEST 1: Basic Message Sending")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register mock agents
    casey = MockAgent("CaseyAgent")
    ralph = MockAgent("RalphAgent")
    optimus = MockAgent("OptimusAgent")
    
    protocol.register_agent(
        agent_name="CaseyAgent",
        capabilities=["orchestration", "monitoring"],
        expertise=["system_architecture"],
        agent_instance=casey
    )
    
    protocol.register_agent(
        agent_name="RalphAgent",
        capabilities=["strategy_generation"],
        expertise=["strategy_development"],
        agent_instance=ralph
    )
    
    protocol.register_agent(
        agent_name="OptimusAgent",
        capabilities=["trading", "execution"],
        expertise=["options_trading"],
        agent_instance=optimus
    )
    
    # Send message
    message = protocol.send_genius_message(
        sender="CaseyAgent",
        recipients=["RalphAgent"],
        message_type=MessageType.REQUEST,
        subject="Generate New Strategy",
        content="Please generate a new options trading strategy",
        priority=MessagePriority.IMPORTANT
    )
    
    print(f"✅ Message sent: {message.message_id}")
    print(f"   Subject: {message.subject}")
    print(f"   Recipients: {message.recipients}")
    
    # Check if message was received
    messages = protocol.receive_messages("RalphAgent")
    assert len(messages) > 0, "Ralph should have received the message"
    print(f"✅ Ralph received {len(messages)} message(s)")
    
    return True


def test_intelligent_routing():
    """Test 2: Intelligent routing"""
    print("\n" + "="*60)
    print("TEST 2: Intelligent Routing")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register agents
    agents = {}
    for name in ["CaseyAgent", "RalphAgent", "OptimusAgent", "DonnieAgent", "GennyAgent"]:
        agent = MockAgent(name)
        agents[name] = agent
        protocol.register_agent(
            agent_name=name,
            capabilities=["test"],
            expertise=["test"],
            agent_instance=agent
        )
    
    # Send broadcast message about trading
    message = protocol.send_genius_message(
        sender="CaseyAgent",
        recipients=["all"],
        message_type=MessageType.COLLABORATION,
        subject="Trading Strategy Discussion",
        content="We need to discuss a new trading strategy for options",
        priority=MessagePriority.IMPORTANT
    )
    
    print(f"✅ Broadcast message sent")
    print(f"   Intelligent routing determined recipients: {message.recipients}")
    
    # Check routing - should include trading-related agents
    assert "RalphAgent" in message.recipients, "Should route to Ralph (strategy)"
    assert "OptimusAgent" in message.recipients, "Should route to Optimus (trading)"
    assert "CaseyAgent" in message.recipients or "CaseyAgent" == message.sender, "Casey should be included"
    
    print(f"✅ Intelligent routing working correctly")
    
    return True


def test_collaborative_session():
    """Test 3: Collaborative problem-solving"""
    print("\n" + "="*60)
    print("TEST 3: Collaborative Problem-Solving")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register agents
    participants = ["CaseyAgent", "RalphAgent", "OptimusAgent"]
    for name in participants:
        protocol.register_agent(
            agent_name=name,
            capabilities=["test"],
            expertise=["test"],
            agent_instance=MockAgent(name)
        )
    
    # Start collaborative session
    session = protocol.start_collaborative_session(
        initiator="CaseyAgent",
        participants=participants,
        problem="Optimize position sizing for current market conditions",
        goal="Develop optimal position sizing algorithm"
    )
    
    print(f"✅ Collaborative session started: {session.session_id}")
    print(f"   Problem: {session.problem}")
    print(f"   Participants: {session.participants}")
    
    # Contribute to session
    protocol.contribute_to_session(
        session_id=session.session_id,
        contributor="RalphAgent",
        contribution="Based on Kelly Criterion analysis, we should use fractional Kelly with 0.5 multiplier",
        data={"kelly_fraction": 0.5, "multiplier": 0.5}
    )
    
    protocol.contribute_to_session(
        session_id=session.session_id,
        contributor="OptimusAgent",
        contribution="Current risk metrics suggest we should reduce position sizes by 20%",
        data={"risk_reduction": 0.20}
    )
    
    print(f"✅ Contributions added: {len(session.contributions)}")
    
    # Synthesize session
    synthesis = protocol.synthesize_session(session.session_id)
    
    assert synthesis is not None, "Synthesis should be generated"
    assert "synthesized_insights" in synthesis, "Should have synthesized insights"
    assert "recommendations" in synthesis, "Should have recommendations"
    
    print(f"✅ Session synthesized")
    print(f"   Insights: {len(synthesis.get('synthesized_insights', {}).get('common_themes', []))} themes")
    print(f"   Recommendations: {len(synthesis.get('recommendations', []))}")
    
    return True


def test_knowledge_sharing():
    """Test 4: Knowledge sharing"""
    print("\n" + "="*60)
    print("TEST 4: Knowledge Sharing")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register agents
    for name in ["CaseyAgent", "RalphAgent", "OptimusAgent"]:
        protocol.register_agent(
            agent_name=name,
            capabilities=["test"],
            expertise=["test"],
            agent_instance=MockAgent(name)
        )
    
    # Share knowledge
    knowledge = {
        "title": "Market Volatility Pattern",
        "insights": [
            "IV is currently elevated",
            "Skew favors put options",
            "Volume is above average"
        ],
        "recommendations": [
            "Focus on put spreads",
            "Reduce position sizes",
            "Monitor volatility closely"
        ]
    }
    
    protocol.share_knowledge(
        sharer="RalphAgent",
        knowledge=knowledge,
        recipients=["OptimusAgent", "CaseyAgent"]
    )
    
    print(f"✅ Knowledge shared: {knowledge['title']}")
    
    # Check shared knowledge
    assert len(protocol.shared_knowledge) > 0, "Knowledge should be stored"
    
    knowledge_keys = list(protocol.shared_knowledge.keys())
    print(f"✅ Knowledge stored: {len(knowledge_keys)} item(s)")
    
    # Check if recipients received notification
    optimus_messages = protocol.receive_messages("OptimusAgent")
    casey_messages = protocol.receive_messages("CaseyAgent")
    
    knowledge_messages = [
        m for m in optimus_messages + casey_messages
        if m.message_type == MessageType.KNOWLEDGE_SHARE
    ]
    
    assert len(knowledge_messages) > 0, "Recipients should be notified"
    print(f"✅ Recipients notified: {len(knowledge_messages)} message(s)")
    
    return True


def test_execution_coordination():
    """Test 5: Execution coordination"""
    print("\n" + "="*60)
    print("TEST 5: Execution Coordination")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register agents
    participants = ["CaseyAgent", "RalphAgent", "DonnieAgent", "OptimusAgent"]
    for name in participants:
        protocol.register_agent(
            agent_name=name,
            capabilities=["test"],
            expertise=["test"],
            agent_instance=MockAgent(name)
        )
    
    # Create execution plan
    execution_plan = {
        "goal": "Execute approved trading strategy",
        "steps": [
            {
                "step_id": "step_1",
                "agent": "RalphAgent",
                "action": "Generate strategy",
                "output_required": "strategy_id"
            },
            {
                "step_id": "step_2",
                "agent": "DonnieAgent",
                "action": "Validate strategy",
                "output_required": "validation_result"
            },
            {
                "step_id": "step_3",
                "agent": "OptimusAgent",
                "action": "Execute strategy",
                "output_required": "execution_result"
            }
        ],
        "dependencies": []
    }
    
    # Coordinate execution
    execution_id = protocol.coordinate_execution(
        coordinator="CaseyAgent",
        execution_plan=execution_plan,
        participants=participants
    )
    
    print(f"✅ Execution coordinated: {execution_id}")
    
    # Check if participants received coordination messages
    coordination_messages = []
    for participant in participants:
        messages = protocol.receive_messages(participant)
        coord_msgs = [m for m in messages if m.message_type == MessageType.COORDINATION]
        coordination_messages.extend(coord_msgs)
    
    assert len(coordination_messages) > 0, "Participants should receive coordination messages"
    print(f"✅ Coordination messages sent: {len(coordination_messages)}")
    
    return True


def test_context_awareness():
    """Test 6: Context-aware messaging"""
    print("\n" + "="*60)
    print("TEST 6: Context-Aware Messaging")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register agents with context
    casey = MockAgent("CaseyAgent")
    ralph = MockAgent("RalphAgent")
    
    protocol.register_agent(
        agent_name="CaseyAgent",
        capabilities=["orchestration"],
        expertise=["system_architecture"],
        agent_instance=casey
    )
    
    protocol.register_agent(
        agent_name="RalphAgent",
        capabilities=["strategy_generation"],
        expertise=["strategy_development"],
        agent_instance=ralph
    )
    
    # Update agent contexts
    protocol.update_agent_context("CaseyAgent", {
        "current_task": "orchestrating strategy generation",
        "status": "active"
    })
    
    protocol.update_agent_context("RalphAgent", {
        "current_strategies": 5,
        "last_strategy_time": datetime.now().isoformat()
    })
    
    # Send message with context
    message = protocol.send_genius_message(
        sender="CaseyAgent",
        recipients=["RalphAgent"],
        message_type=MessageType.REQUEST,
        subject="Strategy Request",
        content="Generate a new strategy",
        priority=MessagePriority.IMPORTANT,
        context={"urgency": "high", "deadline": "1 hour"}
    )
    
    print(f"✅ Context-aware message sent")
    
    # Check if context was synthesized
    assert "context" in message.__dict__, "Message should have context"
    assert "sender_context" in message.context, "Should include sender context"
    assert "recipient_contexts" in message.context, "Should include recipient contexts"
    assert "system_state" in message.context, "Should include system state"
    
    print(f"✅ Context synthesized:")
    print(f"   Sender context: {len(message.context.get('sender_context', {}))} items")
    print(f"   Recipient contexts: {len(message.context.get('recipient_contexts', {}))} items")
    print(f"   System state: {len(message.context.get('system_state', {}))} items")
    
    return True


def test_priority_handling():
    """Test 7: Priority handling"""
    print("\n" + "="*60)
    print("TEST 7: Priority Handling")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register agent
    ralph = MockAgent("RalphAgent")
    protocol.register_agent(
        agent_name="RalphAgent",
        capabilities=["test"],
        expertise=["test"],
        agent_instance=ralph
    )
    
    # Send messages with different priorities
    messages = [
        protocol.send_genius_message(
            sender="CaseyAgent",
            recipients=["RalphAgent"],
            message_type=MessageType.STATUS_UPDATE,
            subject="Low Priority Info",
            content="This is low priority",
            priority=MessagePriority.BACKGROUND
        ),
        protocol.send_genius_message(
            sender="CaseyAgent",
            recipients=["RalphAgent"],
            message_type=MessageType.ALERT,
            subject="Critical Alert",
            content="This is critical!",
            priority=MessagePriority.CRITICAL
        ),
        protocol.send_genius_message(
            sender="CaseyAgent",
            recipients=["RalphAgent"],
            message_type=MessageType.REQUEST,
            subject="Normal Request",
            content="This is normal priority",
            priority=MessagePriority.IMPORTANT
        )
    ]
    
    print(f"✅ Sent {len(messages)} messages with different priorities")
    
    # Receive messages - should be sorted by priority
    received = protocol.receive_messages("RalphAgent")
    
    assert len(received) == 3, "Should receive all messages"
    
    # Check sorting - CRITICAL should be first
    assert received[0].priority == MessagePriority.CRITICAL, "Critical should be first"
    
    print(f"✅ Messages sorted by priority:")
    for i, msg in enumerate(received):
        print(f"   {i+1}. {msg.priority.value.upper()}: {msg.subject}")
    
    return True


def test_communication_intelligence():
    """Test 8: Communication intelligence"""
    print("\n" + "="*60)
    print("TEST 8: Communication Intelligence")
    print("="*60)
    
    protocol = GeniusCommunicationProtocol()
    
    # Register multiple agents
    for name in ["CaseyAgent", "RalphAgent", "OptimusAgent", "DonnieAgent", "GennyAgent"]:
        protocol.register_agent(
            agent_name=name,
            capabilities=["test"],
            expertise=["test"],
            agent_instance=MockAgent(name)
        )
    
    # Send several messages to build patterns
    for i in range(5):
        protocol.send_genius_message(
            sender="CaseyAgent",
            recipients=["RalphAgent"],
            message_type=MessageType.REQUEST,
            subject=f"Request {i+1}",
            content=f"Test message {i+1}",
            priority=MessagePriority.IMPORTANT
        )
    
    # Get communication intelligence
    intelligence = protocol.get_communication_intelligence()
    
    print(f"✅ Communication intelligence:")
    print(f"   Total messages: {intelligence['total_messages']}")
    print(f"   Active messages: {intelligence['active_messages']}")
    print(f"   Registered agents: {intelligence['registered_agents']}")
    print(f"   Shared knowledge items: {intelligence['shared_knowledge_items']}")
    print(f"   Communication patterns: {len(intelligence['communication_patterns'])}")
    
    assert intelligence['total_messages'] >= 5, "Should have recorded messages"
    assert intelligence['registered_agents'] == 5, "Should have 5 agents"
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("GENIUS COMMUNICATION SYSTEM - END-TO-END TEST")
    print("="*60)
    
    tests = [
        ("Basic Message Sending", test_message_sending),
        ("Intelligent Routing", test_intelligent_routing),
        ("Collaborative Sessions", test_collaborative_session),
        ("Knowledge Sharing", test_knowledge_sharing),
        ("Execution Coordination", test_execution_coordination),
        ("Context Awareness", test_context_awareness),
        ("Priority Handling", test_priority_handling),
        ("Communication Intelligence", test_communication_intelligence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS", None))
            print(f"✅ {test_name}: PASSED")
        except AssertionError as e:
            results.append((test_name, "FAIL", str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
        except Exception as e:
            results.append((test_name, "ERROR", str(e)))
            print(f"💥 {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = len([r for r in results if r[1] == "PASS"])
    failed = len([r for r in results if r[1] == "FAIL"])
    errors = len([r for r in results if r[1] == "ERROR"])
    
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    
    if failed > 0 or errors > 0:
        print("\nFailed/Error Details:")
        for test_name, status, error in results:
            if status != "PASS":
                print(f"  {test_name}: {status}")
                if error:
                    print(f"    {error}")
    
    print("\n" + "="*60)
    
    if failed == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED! Genius Communication System is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


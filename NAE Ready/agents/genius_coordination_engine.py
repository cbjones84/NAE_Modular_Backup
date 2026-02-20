#!/usr/bin/env python3
"""
Genius Coordination Engine

Orchestrates genius-level coordination and execution across all NAE agents.
Enables collaborative problem-solving and efficient execution.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, nae_root)

from agents.genius_communication_protocol import (
    GeniusCommunicationProtocol,
    GeniusMessage,
    MessageType,
    MessagePriority,
    CollaborativeSession
)

logger = logging.getLogger(__name__)


class GeniusCoordinationEngine:
    """
    Genius coordination engine for NAE
    
    Enables:
    - Anticipatory communication
    - Collaborative problem-solving
    - Efficient execution orchestration
    - Knowledge synthesis
    - Context-aware coordination
    """
    
    def __init__(self):
        """Initialize coordination engine"""
        self.protocol = GeniusCommunicationProtocol()
        self.agent_instances: Dict[str, Any] = {}
        self.coordination_active = False
        self.coordination_thread = None
        
        # Execution tracking
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info("🎯 Genius Coordination Engine initialized")
    
    def register_all_agents(self, agents: Dict[str, Any]):
        """Register all NAE agents"""
        # Agent capability definitions
        agent_capabilities = {
            "CaseyAgent": {
                "capabilities": [
                    "orchestration", "monitoring", "system_improvement",
                    "agent_coordination", "file_operations", "codebase_search"
                ],
                "expertise": ["system_architecture", "agent_management", "code_analysis"]
            },
            "OptimusAgent": {
                "capabilities": [
                    "live_trading", "order_execution", "risk_management",
                    "position_sizing", "strategy_execution"
                ],
                "expertise": ["options_trading", "execution", "risk_controls"]
            },
            "RalphAgent": {
                "capabilities": [
                    "strategy_generation", "backtesting", "market_analysis",
                    "strategy_filtering", "performance_analysis"
                ],
                "expertise": ["strategy_development", "market_research", "backtesting"]
            },
            "DonnieAgent": {
                "capabilities": [
                    "strategy_validation", "sandbox_execution", "execution_coordination",
                    "strategy_approval"
                ],
                "expertise": ["strategy_validation", "execution_management", "safety"]
            },
            "GennyAgent": {
                "capabilities": [
                    "wealth_management", "tax_preparation", "financial_planning",
                    "generational_wealth_tracking"
                ],
                "expertise": ["wealth_management", "tax_optimization", "financial_planning"]
            },
            "BebopAgent": {
                "capabilities": [
                    "compliance_monitoring", "pdt_prevention", "risk_tracking",
                    "regulatory_compliance"
                ],
                "expertise": ["compliance", "risk_management", "regulations"]
            },
            "PhisherAgent": {
                "capabilities": [
                    "security_monitoring", "threat_detection", "vulnerability_scanning",
                    "security_improvements"
                ],
                "expertise": ["cybersecurity", "threat_intelligence", "security"]
            },
            "RocksteadyAgent": {
                "capabilities": [
                    "threat_response", "security_fixes", "incident_response"
                ],
                "expertise": ["security_response", "threat_mitigation"]
            },
            "SplinterAgent": {
                "capabilities": [
                    "master_scheduling", "agent_orchestration", "workflow_management"
                ],
                "expertise": ["orchestration", "scheduling", "workflow"]
            }
        }
        
        # Register each agent
        for agent_name, agent_instance in agents.items():
            if agent_name in agent_capabilities:
                caps = agent_capabilities[agent_name]
                self.protocol.register_agent(
                    agent_name=agent_name,
                    capabilities=caps["capabilities"],
                    expertise=caps["expertise"],
                    agent_instance=agent_instance
                )
                self.agent_instances[agent_name] = agent_instance
        
        logger.info(f"✅ Registered {len(agents)} agents with genius protocol")
    
    def start_coordination(self):
        """Start continuous coordination"""
        if self.coordination_active:
            return
        
        self.coordination_active = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        logger.info("🚀 Genius coordination started")
    
    def stop_coordination(self):
        """Stop coordination"""
        self.coordination_active = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        logger.info("🛑 Genius coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.coordination_active:
            try:
                # Process messages for all agents
                self._process_agent_messages()
                
                # Coordinate active executions
                self._coordinate_executions()
                
                # Synthesize knowledge
                self._synthesize_knowledge()
                
                # Anticipate needs
                self._anticipate_agent_needs()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                time.sleep(5)
    
    def _process_agent_messages(self):
        """Process messages for all registered agents"""
        for agent_name in self.protocol.agent_registry.keys():
            messages = self.protocol.receive_messages(agent_name)
            
            if messages and agent_name in self.agent_instances:
                agent = self.agent_instances[agent_name]
                
                for message in messages:
                    try:
                        # Deliver message to agent
                        self._deliver_message_to_agent(agent, message)
                    except Exception as e:
                        logger.error(f"Error delivering message to {agent_name}: {e}")
    
    def _deliver_message_to_agent(self, agent: Any, message: GeniusMessage):
        """Deliver genius message to agent"""
        # Convert genius message to agent-compatible format
        agent_message = {
            "message_id": message.message_id,
            "sender": message.sender,
            "subject": message.subject,
            "content": message.content,
            "message_type": message.message_type.value,
            "priority": message.priority.value,
            "context": message.context,
            "intent": message.intent,
            "execution_plan": message.execution_plan,
            "timestamp": message.timestamp.isoformat()
        }
        
        # Try different message delivery methods
        if hasattr(agent, 'receive_genius_message'):
            agent.receive_genius_message(agent_message)
        elif hasattr(agent, 'receive_message'):
            # Try with sender name
            try:
                agent.receive_message(message.sender, agent_message)
            except:
                # Fallback to single arg
                agent.receive_message(agent_message)
        elif hasattr(agent, 'inbox'):
            agent.inbox.append(agent_message)
    
    def _coordinate_executions(self):
        """Coordinate active executions"""
        for execution_id, execution in list(self.active_executions.items()):
            # Check execution status
            # Coordinate steps
            # Handle dependencies
            pass
    
    def _synthesize_knowledge(self):
        """Synthesize knowledge from communications"""
        # Periodically synthesize knowledge from message history
        # Store in shared knowledge base
        pass
    
    def _anticipate_agent_needs(self):
        """Anticipate what agents need and proactively communicate"""
        # Analyze agent states
        # Predict needs
        # Send anticipatory messages
        pass
    
    def orchestrate_collaborative_execution(
        self,
        goal: str,
        participants: List[str],
        execution_steps: List[Dict[str, Any]]
    ) -> str:
        """Orchestrate collaborative execution across agents"""
        execution_plan = {
            "goal": goal,
            "steps": execution_steps,
            "participants": participants,
            "dependencies": self._analyze_dependencies(execution_steps)
        }
        
        execution_id = self.protocol.coordinate_execution(
            coordinator="CaseyAgent",
            execution_plan=execution_plan,
            participants=participants
        )
        
        self.active_executions[execution_id] = {
            "id": execution_id,
            "plan": execution_plan,
            "status": "active",
            "started_at": datetime.now().isoformat()
        }
        
        return execution_id
    
    def _analyze_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze dependencies between execution steps"""
        dependencies = []
        
        for i, step in enumerate(steps):
            step_deps = []
            
            # Check if step depends on previous steps
            if i > 0:
                prev_step = steps[i-1]
                if prev_step.get("output_required"):
                    step_deps.append({
                        "depends_on": prev_step.get("step_id", f"step_{i-1}"),
                        "requires": prev_step.get("output_required")
                    })
            
            if step_deps:
                dependencies.append({
                    "step_id": step.get("step_id", f"step_{i}"),
                    "dependencies": step_deps
                })
        
        return dependencies
    
    def facilitate_collaboration(
        self,
        initiator: str,
        problem: str,
        goal: str,
        suggested_participants: Optional[List[str]] = None
    ) -> CollaborativeSession:
        """Facilitate collaborative problem-solving"""
        # Determine best participants if not provided
        if not suggested_participants:
            suggested_participants = self._determine_best_participants(problem, goal)
        
        # Start collaborative session
        session = self.protocol.start_collaborative_session(
            initiator=initiator,
            participants=suggested_participants,
            problem=problem,
            goal=goal
        )
        
        return session
    
    def _determine_best_participants(self, problem: str, goal: str) -> List[str]:
        """Determine best agents to collaborate on a problem"""
        participants = ["CaseyAgent"]  # Always include Casey
        
        problem_lower = (problem + " " + goal).lower()
        
        # Trading-related
        if any(word in problem_lower for word in ["trade", "strategy", "execution", "order"]):
            participants.extend(["OptimusAgent", "RalphAgent", "DonnieAgent"])
        
        # Risk/compliance
        if any(word in problem_lower for word in ["risk", "compliance", "pdt"]):
            participants.append("BebopAgent")
        
        # Security
        if any(word in problem_lower for word in ["security", "threat", "vulnerability"]):
            participants.extend(["PhisherAgent", "RocksteadyAgent"])
        
        # Wealth management
        if any(word in problem_lower for word in ["wealth", "tax", "financial"]):
            participants.append("GennyAgent")
        
        return list(set(participants))
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination status"""
        return {
            "active": self.coordination_active,
            "registered_agents": len(self.protocol.agent_registry),
            "active_executions": len(self.active_executions),
            "communication_intelligence": self.protocol.get_communication_intelligence()
        }


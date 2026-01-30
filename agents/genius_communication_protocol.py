#!/usr/bin/env python3
"""
Genius Communication Protocol

Enables genius-level communication and coordination between all NAE agents.
Inspired by how the most brilliant individuals collaborate:
- Perfect context understanding
- Anticipatory communication
- Collaborative problem-solving
- Knowledge synthesis
- Efficient execution orchestration
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import hashlib

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = "critical"  # Immediate action required
    URGENT = "urgent"  # High priority, act soon
    IMPORTANT = "important"  # Normal priority
    INFORMATIONAL = "informational"  # FYI
    BACKGROUND = "background"  # Low priority


class MessageType(Enum):
    """Message types for intelligent routing"""
    COMMAND = "command"  # Direct command/instruction
    REQUEST = "request"  # Request for action
    RESPONSE = "response"  # Response to request
    COLLABORATION = "collaboration"  # Collaborative problem-solving
    KNOWLEDGE_SHARE = "knowledge_share"  # Sharing knowledge/insights
    STATUS_UPDATE = "status_update"  # Status/progress update
    ALERT = "alert"  # Alert/warning
    COORDINATION = "coordination"  # Coordination message
    SYNTHESIS = "synthesis"  # Synthesized knowledge
    EXECUTION = "execution"  # Execution instruction


class CommunicationMode(Enum):
    """Communication modes"""
    DIRECT = "direct"  # Direct agent-to-agent
    BROADCAST = "broadcast"  # Broadcast to all
    GROUP = "group"  # Group communication
    CONTEXT_AWARE = "context_aware"  # Context-aware routing
    COLLABORATIVE = "collaborative"  # Collaborative mode


@dataclass
class GeniusMessage:
    """
    Genius-level message with full context and intelligence
    """
    message_id: str
    sender: str
    recipients: List[str]  # Can be specific or ["all"] for broadcast
    message_type: MessageType
    priority: MessagePriority
    subject: str
    content: str
    context: Dict[str, Any] = field(default_factory=dict)  # Full context
    intent: str = ""  # What the sender wants to achieve
    expected_response: Optional[str] = None  # What response is expected
    dependencies: List[str] = field(default_factory=list)  # Message IDs this depends on
    related_messages: List[str] = field(default_factory=list)  # Related message IDs
    knowledge_synthesis: Optional[Dict[str, Any]] = None  # Synthesized knowledge
    execution_plan: Optional[Dict[str, Any]] = None  # Execution plan if applicable
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_name: str
    capabilities: List[str]  # What this agent can do
    expertise: List[str]  # Areas of expertise
    current_context: Dict[str, Any] = field(default_factory=dict)
    availability: str = "available"  # available, busy, unavailable
    response_time_avg: float = 0.0  # Average response time
    success_rate: float = 1.0  # Success rate for tasks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborativeSession:
    """Represents a collaborative problem-solving session"""
    session_id: str
    participants: List[str]
    problem: str
    goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    contributions: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: Optional[Dict[str, Any]] = None
    solution: Optional[Dict[str, Any]] = None
    started_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, completed, abandoned


class GeniusCommunicationProtocol:
    """
    Genius-level communication protocol for NAE agents
    
    Features:
    - Context-aware messaging
    - Anticipatory communication
    - Collaborative problem-solving
    - Knowledge synthesis
    - Intelligent routing
    - Efficient execution orchestration
    """
    
    def __init__(self):
        """Initialize genius communication protocol"""
        # Message storage
        self.message_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.message_history: deque = deque(maxlen=10000)
        self.active_messages: Dict[str, GeniusMessage] = {}
        
        # Agent registry
        self.agent_registry: Dict[str, AgentCapability] = {}
        self.agent_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Collaborative sessions
        self.collaborative_sessions: Dict[str, CollaborativeSession] = {}
        
        # Knowledge base
        self.shared_knowledge: Dict[str, Any] = {}
        self.knowledge_synthesis_cache: Dict[str, Any] = {}
        
        # Communication patterns
        self.communication_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Routing intelligence
        self.routing_intelligence: Dict[str, float] = {}  # Agent -> routing score
        
        # Execution coordination
        self.execution_coordination: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🧠 Genius Communication Protocol initialized")
    
    def register_agent(
        self,
        agent_name: str,
        capabilities: List[str],
        expertise: List[str],
        agent_instance: Any = None
    ):
        """Register an agent with the protocol"""
        capability = AgentCapability(
            agent_name=agent_name,
            capabilities=capabilities,
            expertise=expertise,
            metadata={"instance": agent_instance}
        )
        
        self.agent_registry[agent_name] = capability
        self.agent_contexts[agent_name] = {}
        self.message_queue[agent_name] = deque(maxlen=1000)
        
        logger.info(f"✅ Registered agent: {agent_name} with {len(capabilities)} capabilities")
    
    def send_genius_message(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType,
        subject: str,
        content: str,
        priority: MessagePriority = MessagePriority.IMPORTANT,
        context: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        expected_response: Optional[str] = None,
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> GeniusMessage:
        """
        Send a genius-level message with full context
        
        This is the main method agents use to communicate intelligently
        """
        # Generate message ID
        message_id = f"msg_{hashlib.md5(f'{sender}:{time.time()}:{subject}'.encode()).hexdigest()[:12]}"
        
        # Intelligent recipient routing
        if "all" in recipients or recipients == ["all"]:
            recipients = self._intelligent_broadcast_routing(sender, message_type, subject, content)
        else:
            recipients = self._intelligent_recipient_routing(sender, recipients, message_type, subject, content)
        
        # Synthesize context
        full_context = self._synthesize_context(sender, recipients, message_type, context or {})
        
        # Determine intent if not provided
        if not intent:
            intent = self._determine_intent(subject, content, message_type)
        
        # Create message
        message = GeniusMessage(
            message_id=message_id,
            sender=sender,
            recipients=recipients,
            message_type=message_type,
            priority=priority,
            subject=subject,
            content=content,
            context=full_context,
            intent=intent,
            expected_response=expected_response,
            execution_plan=execution_plan,
            metadata={
                "routing_score": self._calculate_routing_score(sender, recipients, message_type),
                "urgency": self._calculate_urgency(priority, message_type, content)
            }
        )
        
        # Store message
        self.active_messages[message_id] = message
        self.message_history.append(message)
        
        # Route to recipients
        for recipient in recipients:
            if recipient in self.message_queue:
                self.message_queue[recipient].append(message)
            else:
                logger.warning(f"Recipient {recipient} not registered")
        
        # Update communication patterns
        self._update_communication_patterns(sender, recipients, message_type)
        
        # Log
        logger.info(f"📨 [{sender} → {', '.join(recipients)}] {subject} ({message_type.value})")
        
        return message
    
    def _intelligent_broadcast_routing(
        self,
        sender: str,
        message_type: MessageType,
        subject: str,
        content: str
    ) -> List[str]:
        """Intelligently determine who should receive a broadcast"""
        recipients = []
        
        # Analyze message to determine relevant agents
        content_lower = (subject + " " + content).lower()
        
        # Trading-related → Optimus, Donnie, Ralph
        if any(word in content_lower for word in ["trade", "order", "strategy", "execution"]):
            recipients.extend(["OptimusAgent", "DonnieAgent", "RalphAgent"])
        
        # Strategy-related → Ralph, Donnie
        if any(word in content_lower for word in ["strategy", "backtest", "generate"]):
            recipients.extend(["RalphAgent", "DonnieAgent"])
        
        # Risk/compliance → Bebop, Casey
        if any(word in content_lower for word in ["risk", "compliance", "pdt", "regulation"]):
            recipients.extend(["BebopAgent", "CaseyAgent"])
        
        # Security → Phisher, Rocksteady
        if any(word in content_lower for word in ["security", "threat", "vulnerability"]):
            recipients.extend(["PhisherAgent", "RocksteadyAgent"])
        
        # Wealth management → Genny
        if any(word in content_lower for word in ["wealth", "tax", "financial", "capital"]):
            recipients.append("GennyAgent")
        
        # System/architecture → Casey, Splinter
        if any(word in content_lower for word in ["system", "architecture", "improve", "optimize"]):
            recipients.extend(["CaseyAgent", "SplinterAgent"])
        
        # Always include Casey for orchestration
        if "CaseyAgent" not in recipients:
            recipients.append("CaseyAgent")
        
        # Remove duplicates
        return list(set(recipients))
    
    def _intelligent_recipient_routing(
        self,
        sender: str,
        requested_recipients: List[str],
        message_type: MessageType,
        subject: str,
        content: str
    ) -> List[str]:
        """Intelligently route to requested recipients and suggest additional ones"""
        recipients = list(requested_recipients)
        
        # Suggest additional recipients based on message content
        content_lower = (subject + " " + content).lower()
        
        # If message mentions coordination, include Casey
        if message_type == MessageType.COORDINATION and "CaseyAgent" not in recipients:
            recipients.append("CaseyAgent")
        
        # If message is about execution, ensure Optimus is aware
        if message_type == MessageType.EXECUTION and "OptimusAgent" not in recipients:
            if any(word in content_lower for word in ["trade", "order", "execute"]):
                recipients.append("OptimusAgent")
        
        return list(set(recipients))
    
    def _synthesize_context(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType,
        provided_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize full context for the message"""
        context = provided_context.copy()
        
        # Add sender context
        if sender in self.agent_contexts:
            context["sender_context"] = self.agent_contexts[sender]
        
        # Add recipient contexts
        context["recipient_contexts"] = {
            recipient: self.agent_contexts.get(recipient, {})
            for recipient in recipients
        }
        
        # Add shared knowledge relevant to this message
        context["relevant_knowledge"] = self._get_relevant_knowledge(sender, recipients, message_type)
        
        # Add recent related messages
        context["related_messages"] = self._find_related_messages(sender, recipients, message_type)
        
        # Add system state
        context["system_state"] = {
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.collaborative_sessions),
            "message_queue_sizes": {
                agent: len(queue) for agent, queue in self.message_queue.items()
            }
        }
        
        return context
    
    def _get_relevant_knowledge(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType
    ) -> Dict[str, Any]:
        """Get relevant knowledge from shared knowledge base"""
        relevant = {}
        
        # Get knowledge relevant to sender and recipients
        for key, value in self.shared_knowledge.items():
            if any(agent in str(value) for agent in [sender] + recipients):
                relevant[key] = value
        
        return relevant
    
    def _find_related_messages(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType
    ) -> List[Dict[str, Any]]:
        """Find related messages from history"""
        related = []
        
        # Look at last 100 messages
        for message in list(self.message_history)[-100:]:
            # Check if related by sender/recipient
            if (message.sender == sender or any(r in message.recipients for r in recipients)):
                related.append({
                    "message_id": message.message_id,
                    "subject": message.subject,
                    "timestamp": message.timestamp.isoformat()
                })
        
        return related[:5]  # Return top 5 related
    
    def _determine_intent(self, subject: str, content: str, message_type: MessageType) -> str:
        """Determine the intent of the message"""
        content_lower = (subject + " " + content).lower()
        
        if message_type == MessageType.COMMAND:
            if "execute" in content_lower or "run" in content_lower:
                return "execute_action"
            elif "generate" in content_lower or "create" in content_lower:
                return "create_resource"
            else:
                return "perform_task"
        
        elif message_type == MessageType.REQUEST:
            if "help" in content_lower or "assist" in content_lower:
                return "request_assistance"
            elif "information" in content_lower or "data" in content_lower:
                return "request_information"
            else:
                return "request_action"
        
        elif message_type == MessageType.COLLABORATION:
            return "collaborate_on_problem"
        
        elif message_type == MessageType.KNOWLEDGE_SHARE:
            return "share_knowledge"
        
        else:
            return "communicate"
    
    def _calculate_routing_score(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType
    ) -> float:
        """Calculate routing intelligence score"""
        score = 1.0
        
        # Boost score if recipients match their expertise
        for recipient in recipients:
            if recipient in self.agent_registry:
                capability = self.agent_registry[recipient]
                # Higher score if agent is available and capable
                if capability.availability == "available":
                    score += 0.1
                if capability.success_rate > 0.8:
                    score += 0.1
        
        return min(1.0, score)
    
    def _calculate_urgency(
        self,
        priority: MessagePriority,
        message_type: MessageType,
        content: str
    ) -> float:
        """Calculate urgency score"""
        urgency = 0.5  # Base urgency
        
        # Priority-based
        if priority == MessagePriority.CRITICAL:
            urgency = 1.0
        elif priority == MessagePriority.URGENT:
            urgency = 0.8
        elif priority == MessagePriority.IMPORTANT:
            urgency = 0.6
        elif priority == MessagePriority.INFORMATIONAL:
            urgency = 0.4
        else:
            urgency = 0.2
        
        # Message type adjustments
        if message_type == MessageType.ALERT:
            urgency = min(1.0, urgency + 0.2)
        elif message_type == MessageType.EXECUTION:
            urgency = min(1.0, urgency + 0.1)
        
        # Content-based urgency indicators
        content_lower = content.lower()
        if any(word in content_lower for word in ["urgent", "immediate", "critical", "now"]):
            urgency = min(1.0, urgency + 0.2)
        
        return urgency
    
    def _update_communication_patterns(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType
    ):
        """Update communication patterns for learning"""
        pattern_key = f"{sender}:{','.join(sorted(recipients))}"
        
        self.communication_patterns[pattern_key].append({
            "message_type": message_type.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 patterns per key
        if len(self.communication_patterns[pattern_key]) > 100:
            self.communication_patterns[pattern_key] = self.communication_patterns[pattern_key][-100:]
    
    def receive_messages(self, agent_name: str) -> List[GeniusMessage]:
        """Get messages for an agent (genius-level)"""
        if agent_name not in self.message_queue:
            return []
        
        messages = list(self.message_queue[agent_name])
        self.message_queue[agent_name].clear()
        
        # Sort by priority and urgency
        messages.sort(key=lambda m: (
            m.priority.value == "critical",
            m.priority.value == "urgent",
            m.metadata.get("urgency", 0.5),
            m.timestamp
        ), reverse=True)
        
        return messages
    
    def start_collaborative_session(
        self,
        initiator: str,
        participants: List[str],
        problem: str,
        goal: str
    ) -> CollaborativeSession:
        """Start a collaborative problem-solving session"""
        session_id = f"collab_{hashlib.md5(f'{initiator}:{time.time()}:{problem}'.encode()).hexdigest()[:12]}"
        
        session = CollaborativeSession(
            session_id=session_id,
            participants=participants,
            problem=problem,
            goal=goal,
            context={
                "initiator": initiator,
                "started_at": datetime.now().isoformat()
            }
        )
        
        self.collaborative_sessions[session_id] = session
        
        # Notify all participants
        for participant in participants:
            self.send_genius_message(
                sender=initiator,
                recipients=[participant],
                message_type=MessageType.COLLABORATION,
                subject=f"Collaborative Session: {problem}",
                content=f"Goal: {goal}\n\nLet's work together to solve this.",
                priority=MessagePriority.IMPORTANT,
                context={"session_id": session_id}
            )
        
        logger.info(f"🤝 Started collaborative session: {session_id} with {len(participants)} participants")
        
        return session
    
    def contribute_to_session(
        self,
        session_id: str,
        contributor: str,
        contribution: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Contribute to a collaborative session"""
        if session_id not in self.collaborative_sessions:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.collaborative_sessions[session_id]
        
        contribution_record = {
            "contributor": contributor,
            "contribution": contribution,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        session.contributions.append(contribution_record)
        
        # Broadcast contribution to other participants
        other_participants = [p for p in session.participants if p != contributor]
        if other_participants:
            self.send_genius_message(
                sender=contributor,
                recipients=other_participants,
                message_type=MessageType.COLLABORATION,
                subject=f"Contribution to: {session.problem}",
                content=contribution,
                priority=MessagePriority.IMPORTANT,
                context={"session_id": session_id, "contribution": contribution_record}
            )
        
        logger.info(f"💡 [{contributor}] contributed to session {session_id}")
    
    def synthesize_session(self, session_id: str) -> Dict[str, Any]:
        """Synthesize knowledge from a collaborative session"""
        if session_id not in self.collaborative_sessions:
            return {}
        
        session = self.collaborative_sessions[session_id]
        
        # Synthesize all contributions
        synthesis = {
            "session_id": session_id,
            "problem": session.problem,
            "goal": session.goal,
            "participants": session.participants,
            "contributions_summary": [
                {
                    "contributor": c["contributor"],
                    "key_points": self._extract_key_points(c["contribution"])
                }
                for c in session.contributions
            ],
            "synthesized_insights": self._synthesize_contributions(session.contributions),
            "recommendations": self._generate_recommendations(session),
            "timestamp": datetime.now().isoformat()
        }
        
        session.synthesis = synthesis
        session.status = "completed"
        
        # Store in knowledge base
        knowledge_key = f"collaboration_{session_id}"
        self.shared_knowledge[knowledge_key] = synthesis
        
        logger.info(f"🧠 Synthesized session {session_id}: {len(session.contributions)} contributions")
        
        return synthesis
    
    def _extract_key_points(self, contribution: str) -> List[str]:
        """Extract key points from a contribution"""
        # Simple extraction - in production, use NLP
        sentences = contribution.split('.')
        key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
        return key_points
    
    def _synthesize_contributions(self, contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize multiple contributions into insights"""
        # Combine all contributions
        all_text = " ".join([c["contribution"] for c in contributions])
        
        # Extract common themes (simplified)
        themes = {}
        for contribution in contributions:
            # Simple theme extraction
            words = contribution["contribution"].lower().split()
            for word in words:
                if len(word) > 4:  # Meaningful words
                    themes[word] = themes.get(word, 0) + 1
        
        # Top themes
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "common_themes": [theme for theme, count in top_themes],
            "contributor_count": len(contributions),
            "synthesis": f"Synthesized {len(contributions)} contributions into actionable insights"
        }
    
    def _generate_recommendations(self, session: CollaborativeSession) -> List[str]:
        """Generate recommendations from session"""
        recommendations = []
        
        # Analyze contributions for recommendations
        if session.contributions:
            recommendations.append(f"Consider implementing solutions from {len(session.contributions)} contributions")
            recommendations.append("Review synthesized insights for actionable items")
        
        return recommendations
    
    def share_knowledge(
        self,
        sharer: str,
        knowledge: Dict[str, Any],
        recipients: Optional[List[str]] = None
    ):
        """Share knowledge with other agents"""
        knowledge_id = f"knowledge_{hashlib.md5(f'{sharer}:{time.time()}'.encode()).hexdigest()[:12]}"
        
        # Store in shared knowledge base
        self.shared_knowledge[knowledge_id] = {
            "id": knowledge_id,
            "sharer": sharer,
            "knowledge": knowledge,
            "timestamp": datetime.now().isoformat()
        }
        
        # Notify recipients
        if recipients:
            self.send_genius_message(
                sender=sharer,
                recipients=recipients,
                message_type=MessageType.KNOWLEDGE_SHARE,
                subject="Knowledge Shared",
                content=f"New knowledge available: {knowledge.get('title', 'Untitled')}",
                priority=MessagePriority.INFORMATIONAL,
                context={"knowledge_id": knowledge_id, "knowledge": knowledge}
            )
        else:
            # Broadcast to all if no specific recipients
            self.send_genius_message(
                sender=sharer,
                recipients=["all"],
                message_type=MessageType.KNOWLEDGE_SHARE,
                subject="Knowledge Shared",
                content=f"New knowledge available: {knowledge.get('title', 'Untitled')}",
                priority=MessagePriority.INFORMATIONAL,
                context={"knowledge_id": knowledge_id, "knowledge": knowledge}
            )
        
        logger.info(f"📚 [{sharer}] shared knowledge: {knowledge_id}")
    
    def coordinate_execution(
        self,
        coordinator: str,
        execution_plan: Dict[str, Any],
        participants: List[str]
    ) -> str:
        """Coordinate execution across multiple agents"""
        execution_id = f"exec_{hashlib.md5(f'{coordinator}:{time.time()}'.encode()).hexdigest()[:12]}"
        
        # Store execution plan
        self.execution_coordination[execution_id] = {
            "id": execution_id,
            "coordinator": coordinator,
            "plan": execution_plan,
            "participants": participants,
            "status": "initiated",
            "started_at": datetime.now().isoformat(),
            "steps": execution_plan.get("steps", [])
        }
        
        # Send coordination messages to participants
        for participant in participants:
            participant_steps = [
                step for step in execution_plan.get("steps", [])
                if step.get("agent") == participant
            ]
            
            self.send_genius_message(
                sender=coordinator,
                recipients=[participant],
                message_type=MessageType.COORDINATION,
                subject=f"Execution Coordination: {execution_plan.get('goal', 'Execute Plan')}",
                content=f"Your role in execution:\n{json.dumps(participant_steps, indent=2)}",
                priority=MessagePriority.IMPORTANT,
                execution_plan={
                    "execution_id": execution_id,
                    "steps": participant_steps,
                    "dependencies": execution_plan.get("dependencies", [])
                }
            )
        
        logger.info(f"🎯 [{coordinator}] coordinating execution: {execution_id} with {len(participants)} participants")
        
        return execution_id
    
    def update_agent_context(self, agent_name: str, context: Dict[str, Any]):
        """Update agent's current context"""
        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = {}
        
        self.agent_contexts[agent_name].update(context)
        self.agent_contexts[agent_name]["last_updated"] = datetime.now().isoformat()
    
    def get_agent_capabilities(self, agent_name: str) -> Optional[AgentCapability]:
        """Get agent capabilities"""
        return self.agent_registry.get(agent_name)
    
    def get_communication_intelligence(self) -> Dict[str, Any]:
        """Get communication intelligence report"""
        return {
            "total_messages": len(self.message_history),
            "active_messages": len(self.active_messages),
            "registered_agents": len(self.agent_registry),
            "active_sessions": len([s for s in self.collaborative_sessions.values() if s.status == "active"]),
            "shared_knowledge_items": len(self.shared_knowledge),
            "communication_patterns": {
                key: len(patterns) for key, patterns in list(self.communication_patterns.items())[:10]
            },
            "agent_capabilities": {
                name: {
                    "capabilities": cap.capabilities,
                    "expertise": cap.expertise,
                    "availability": cap.availability
                }
                for name, cap in self.agent_registry.items()
            }
        }


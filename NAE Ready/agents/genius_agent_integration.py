#!/usr/bin/env python3
"""
Genius Agent Integration

Provides easy integration of genius communication protocol into all NAE agents.
"""

import os
import sys
from typing import Dict, Any, List, Optional

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, nae_root)

from dataclasses import asdict
from agents.genius_communication_protocol import (
    GeniusCommunicationProtocol,
    MessageType,
    MessagePriority
)


class GeniusAgentMixin:
    """
    Mixin class that adds genius communication capabilities to any agent
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize genius capabilities"""
        super().__init__(*args, **kwargs)
        
        # Get or create global protocol instance
        self.genius_protocol = self._get_global_protocol()
        
        # Register this agent
        if self.genius_protocol:
            self._register_with_protocol()
    
    def _get_global_protocol(self) -> Optional[GeniusCommunicationProtocol]:
        """Get global protocol instance (singleton pattern)"""
        # Try to get from Casey if available
        try:
            from agents.casey import CaseyAgent
            # This is a simplified approach - in production, use a proper singleton
            # For now, we'll create a new instance per agent but they'll share state via a global registry
            return GeniusCommunicationProtocol()
        except:
            return None
    
    def _register_with_protocol(self):
        """Register this agent with the protocol"""
        agent_name = self.__class__.__name__
        
        # Determine capabilities based on agent type
        capabilities, expertise = self._determine_capabilities()
        
        if self.genius_protocol:
            self.genius_protocol.register_agent(
                agent_name=agent_name,
                capabilities=capabilities,
                expertise=expertise,
                agent_instance=self
            )
    
    def _determine_capabilities(self) -> tuple:
        """Determine agent capabilities - override in subclasses"""
        return [], []
    
    def send_genius_message(
        self,
        recipients: List[str],
        subject: str,
        content: str,
        message_type: str = "request",
        priority: str = "important",
        context: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None
    ):
        """Send genius-level message"""
        if not self.genius_protocol:
            # Fallback to regular send_message
            if hasattr(self, 'send_message'):
                for recipient in recipients:
                    # Try to find recipient agent
                    pass
            return None
        
        try:
            msg_type = MessageType[message_type.upper()] if hasattr(MessageType, message_type.upper()) else MessageType.REQUEST
            msg_priority = MessagePriority[priority.upper()] if hasattr(MessagePriority, priority.upper()) else MessagePriority.IMPORTANT
            
            agent_name = self.__class__.__name__
            
            message = self.genius_protocol.send_genius_message(
                sender=agent_name,
                recipients=recipients,
                message_type=msg_type,
                subject=subject,
                content=content,
                priority=msg_priority,
                context=context or {},
                intent=intent
            )
            
            if hasattr(self, 'log_action'):
                self.log_action(f"📨 Sent genius message: {subject} to {', '.join(recipients)}")
            
            return message
        except Exception as e:
            if hasattr(self, 'log_action'):
                self.log_action(f"Error sending genius message: {e}")
            return None
    
    def receive_genius_message(self, message: Dict[str, Any]):
        """Receive genius-level message"""
        # Add to inbox
        if hasattr(self, 'inbox'):
            self.inbox.append(message)
        
        # Process based on type
        msg_type = message.get("message_type", "request")
        
        if msg_type == "command":
            self._handle_genius_command(message)
        elif msg_type == "collaboration":
            self._handle_collaboration(message)
        elif msg_type == "coordination":
            self._handle_coordination(message)
        else:
            self._handle_genius_request(message)
    
    def _handle_genius_command(self, message: Dict[str, Any]):
        """Handle genius command - override in subclasses"""
        content = message.get("content", "")
        if hasattr(self, 'log_action'):
            self.log_action(f"🎯 Received command: {content[:100]}")
    
    def _handle_collaboration(self, message: Dict[str, Any]):
        """Handle collaboration request - override in subclasses"""
        if hasattr(self, 'log_action'):
            self.log_action(f"🤝 Received collaboration request")
    
    def _handle_coordination(self, message: Dict[str, Any]):
        """Handle coordination message - override in subclasses"""
        execution_plan = message.get("execution_plan", {})
        if hasattr(self, 'log_action'):
            self.log_action(f"🎯 Received coordination: {execution_plan.get('execution_id', 'unknown')}")
    
    def _handle_genius_request(self, message: Dict[str, Any]):
        """Handle genius request - override in subclasses"""
        content = message.get("content", "")
        if hasattr(self, 'log_action'):
            self.log_action(f"📨 Received request: {content[:100]}")
    
    def get_genius_messages(self) -> List[Dict[str, Any]]:
        """Get pending genius messages"""
        if not self.genius_protocol:
            return []
        
        agent_name = self.__class__.__name__
        messages = self.genius_protocol.receive_messages(agent_name)
        
        return [asdict(msg) for msg in messages]
    
    def start_collaboration(
        self,
        problem: str,
        goal: str,
        participants: Optional[List[str]] = None
    ):
        """Start a collaborative session"""
        if not self.genius_protocol:
            return None
        
        agent_name = self.__class__.__name__
        
        if not participants:
            # Determine best participants
            participants = self._suggest_collaborators(problem, goal)
        
        session = self.genius_protocol.start_collaborative_session(
            initiator=agent_name,
            participants=participants,
            problem=problem,
            goal=goal
        )
        
        return session
    
    def _suggest_collaborators(self, problem: str, goal: str) -> List[str]:
        """Suggest collaborators for a problem - override in subclasses"""
        # Default: include Casey and self
        return ["CaseyAgent", self.__class__.__name__]
    
    def contribute_to_collaboration(
        self,
        session_id: str,
        contribution: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Contribute to a collaborative session"""
        if not self.genius_protocol:
            return
        
        agent_name = self.__class__.__name__
        self.genius_protocol.contribute_to_session(
            session_id=session_id,
            contributor=agent_name,
            contribution=contribution,
            data=data
        )
    
    def share_knowledge(
        self,
        knowledge: Dict[str, Any],
        recipients: Optional[List[str]] = None
    ):
        """Share knowledge with other agents"""
        if not self.genius_protocol:
            return
        
        agent_name = self.__class__.__name__
        self.genius_protocol.share_knowledge(
            sharer=agent_name,
            knowledge=knowledge,
            recipients=recipients
        )




#!/usr/bin/env python3
"""
Casey Continuous Learning System

Multi-model continuous learning engine that learns from:
- Cursor/Auto (latest version)
- ChatGPT (latest version)
- Grok (latest version)
- Gemini (latest version)

Features:
- Continuous learning from multiple AI models
- Self-awareness and self-healing
- NAE improvement suggestions
- Financial optimization while maintaining compliance
- Knowledge synthesis and application
"""

import os
import sys
import json
import time
import logging
import threading
import re
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import hashlib

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, nae_root)

logger = logging.getLogger(__name__)


class LearningSource(Enum):
    """Learning source types"""
    CURSOR_AUTO = "cursor_auto"
    CHATGPT = "chatgpt"
    GROK = "grok"
    GEMINI = "gemini"
    NAE_INTERNAL = "nae_internal"
    EXTERNAL = "external"


class LearningCategory(Enum):
    """Learning categories"""
    CODE_IMPROVEMENT = "code_improvement"
    ARCHITECTURE = "architecture"
    TRADING_STRATEGY = "trading_strategy"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    FINANCIAL_OPTIMIZATION = "financial_optimization"
    SELF_HEALING = "self_healing"
    SYSTEM_OPTIMIZATION = "system_optimization"


class LearningPriority(Enum):
    """Learning priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LearningInsight:
    """Represents a learning insight from an AI model"""
    insight_id: str
    source: LearningSource
    category: LearningCategory
    priority: LearningPriority
    title: str
    description: str
    content: str
    learned_at: datetime
    confidence: float  # 0.0 to 1.0
    applicable_to: List[str]  # Agent names or "NAE" for system-wide
    implementation_steps: List[str] = field(default_factory=list)
    expected_impact: str = ""
    risk_assessment: str = ""
    compliance_check: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementAction:
    """Represents an improvement action to be taken"""
    action_id: str
    insight_id: str
    agent: str  # "casey", "optimus", "NAE", etc.
    action_type: str  # "code_change", "config_update", "strategy_adjustment", etc.
    description: str
    implementation: str  # Code or instructions
    priority: LearningPriority
    created_at: datetime
    status: str = "pending"  # pending, implementing, completed, failed
    applied_at: Optional[datetime] = None
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModelLearner:
    """
    Learns from multiple AI models and synthesizes knowledge
    """
    
    def __init__(self):
        """Initialize multi-model learner"""
        self.learned_insights: Dict[str, LearningInsight] = {}
        self.improvement_actions: Dict[str, ImprovementAction] = {}
        self.learning_history: deque = deque(maxlen=10000)  # Last 10k insights
        
        # Model configurations
        self.model_configs = {
            LearningSource.CURSOR_AUTO: {
                "enabled": True,
                "api_key": os.getenv("CURSOR_API_KEY"),
                "model": "gpt-4",
                "learning_rate": 0.8  # How much to trust this source
            },
            LearningSource.CHATGPT: {
                "enabled": True,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4-turbo",
                "learning_rate": 0.7
            },
            LearningSource.GROK: {
                "enabled": True,
                "api_key": os.getenv("GROK_API_KEY"),
                "model": "grok-beta",
                "learning_rate": 0.75
            },
            LearningSource.GEMINI: {
                "enabled": True,
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": "gemini-pro",
                "learning_rate": 0.7
            }
        }
        
        # Learning patterns
        self.learning_patterns = {
            "code_patterns": deque(maxlen=1000),
            "strategy_patterns": deque(maxlen=1000),
            "optimization_patterns": deque(maxlen=1000),
            "compliance_patterns": deque(maxlen=1000)
        }
        
        logger.info("🧠 Multi-model learner initialized")
    
    def learn_from_model(
        self,
        source: LearningSource,
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[LearningInsight]:
        """
        Learn from a model interaction
        
        Args:
            source: Learning source
            prompt: Prompt sent to model
            response: Response from model
            context: Additional context
        
        Returns:
            LearningInsight if valuable knowledge extracted
        """
        if not self.model_configs[source]["enabled"]:
            return None
        
        try:
            # Extract insights from response
            insight = self._extract_insight(source, prompt, response, context)
            
            if insight:
                # Store insight
                self.learned_insights[insight.insight_id] = insight
                self.learning_history.append(insight)
                
                # Update learning patterns
                self._update_learning_patterns(insight)
                
                logger.info(f"📚 Learned from {source.value}: {insight.title}")
                
                return insight
            
        except Exception as e:
            logger.error(f"Error learning from {source.value}: {e}")
        
        return None
    
    def _extract_insight(
        self,
        source: LearningSource,
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[LearningInsight]:
        """Extract learning insight from model response"""
        # Generate insight ID
        content_hash = hashlib.md5(f"{source.value}:{prompt}:{response}".encode()).hexdigest()
        insight_id = f"insight_{content_hash[:12]}"
        
        # Analyze response to extract insights
        # This is a simplified version - in production, use NLP/ML to extract structured insights
        
        # Determine category
        category = self._categorize_content(prompt, response)
        
        # Determine priority
        priority = self._determine_priority(response, category)
        
        # Extract applicable agents
        applicable_to = self._extract_applicable_agents(prompt, response)
        
        # Extract implementation steps
        implementation_steps = self._extract_implementation_steps(response)
        
        # Calculate confidence based on source and content quality
        confidence = self._calculate_confidence(source, response, category)
        
        # Check compliance
        compliance_check = self._check_compliance(response, category)
        
        # Create insight
        insight = LearningInsight(
            insight_id=insight_id,
            source=source,
            category=category,
            priority=priority,
            title=self._extract_title(response),
            description=self._extract_description(response),
            content=response,
            learned_at=datetime.now(),
            confidence=confidence,
            applicable_to=applicable_to,
            implementation_steps=implementation_steps,
            expected_impact=self._extract_expected_impact(response),
            risk_assessment=self._assess_risk(response, category),
            compliance_check=compliance_check,
            metadata={
                "prompt": prompt,
                "context": context or {},
                "source_model": self.model_configs[source]["model"]
            }
        )
        
        # Only return high-value insights
        if confidence > 0.5 and compliance_check:
            return insight
        
        return None
    
    def _categorize_content(self, prompt: str, response: str) -> LearningCategory:
        """Categorize learning content"""
        content_lower = (prompt + " " + response).lower()
        
        if any(word in content_lower for word in ["code", "function", "class", "implementation"]):
            return LearningCategory.CODE_IMPROVEMENT
        elif any(word in content_lower for word in ["architecture", "design", "structure"]):
            return LearningCategory.ARCHITECTURE
        elif any(word in content_lower for word in ["strategy", "trade", "trading", "position"]):
            return LearningCategory.TRADING_STRATEGY
        elif any(word in content_lower for word in ["risk", "safety", "limit"]):
            return LearningCategory.RISK_MANAGEMENT
        elif any(word in content_lower for word in ["compliance", "regulation", "legal"]):
            return LearningCategory.COMPLIANCE
        elif any(word in content_lower for word in ["performance", "speed", "optimize", "efficient"]):
            return LearningCategory.PERFORMANCE
        elif any(word in content_lower for word in ["profit", "gain", "revenue", "financial"]):
            return LearningCategory.FINANCIAL_OPTIMIZATION
        elif any(word in content_lower for word in ["heal", "fix", "repair", "recover"]):
            return LearningCategory.SELF_HEALING
        else:
            return LearningCategory.SYSTEM_OPTIMIZATION
    
    def _determine_priority(self, response: str, category: LearningCategory) -> LearningPriority:
        """Determine priority based on content"""
        response_lower = response.lower()
        
        # Critical keywords
        if any(word in response_lower for word in ["critical", "urgent", "immediate", "security", "compliance"]):
            return LearningPriority.CRITICAL
        elif any(word in response_lower for word in ["important", "significant", "major"]):
            return LearningPriority.HIGH
        elif any(word in response_lower for word in ["minor", "small", "low"]):
            return LearningPriority.LOW
        else:
            return LearningPriority.MEDIUM
    
    def _extract_applicable_agents(self, prompt: str, response: str) -> List[str]:
        """Extract which agents this applies to"""
        content = (prompt + " " + response).lower()
        applicable = []
        
        agent_names = ["casey", "optimus", "ralph", "donnie", "genny", "shredder", "april"]
        for agent in agent_names:
            if agent in content:
                applicable.append(agent)
        
        # If mentions "system", "NAE", "all", apply to NAE
        if any(word in content for word in ["system", "nae", "all agents", "entire"]):
            applicable.append("NAE")
        
        return applicable if applicable else ["NAE"]
    
    def _extract_implementation_steps(self, response: str) -> List[str]:
        """Extract implementation steps from response"""
        steps = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered steps, bullet points, or action items
            if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or
                line.startswith(('-', '*', '•')) or
                line.startswith(('Step', 'step', 'ACTION', 'Action'))):
                # Clean up the step
                step = re.sub(r'^[\d\.\-\*\•\s]+', '', line)
                if step and len(step) > 10:  # Only meaningful steps
                    steps.append(step)
        
        return steps[:10]  # Max 10 steps
    
    def _calculate_confidence(self, source: LearningSource, response: str, category: LearningCategory) -> float:
        """Calculate confidence in the insight"""
        base_confidence = self.model_configs[source]["learning_rate"]
        
        # Adjust based on response quality
        response_length = len(response)
        if response_length > 500:
            base_confidence += 0.1
        elif response_length < 100:
            base_confidence -= 0.2
        
        # Adjust based on category
        if category in [LearningCategory.COMPLIANCE, LearningCategory.RISK_MANAGEMENT]:
            base_confidence -= 0.1  # Be more cautious with compliance/risk
        
        return max(0.0, min(1.0, base_confidence))
    
    def _check_compliance(self, response: str, category: LearningCategory) -> bool:
        """Check if insight is compliant"""
        response_lower = response.lower()
        
        # Red flags
        red_flags = [
            "illegal", "unethical", "manipulate", "insider", "fraud",
            "violate", "bypass", "exploit", "hack", "unauthorized"
        ]
        
        if any(flag in response_lower for flag in red_flags):
            return False
        
        # For compliance/risk categories, be extra careful
        if category in [LearningCategory.COMPLIANCE, LearningCategory.RISK_MANAGEMENT]:
            # Additional checks
            if "without" in response_lower and "compliance" in response_lower:
                return False
        
        return True
    
    def _extract_title(self, response: str) -> str:
        """Extract title from response"""
        # Try to find a title or first sentence
        lines = response.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100 and not line.startswith(('```', '---', '#')):
                return line[:80]
        
        # Fallback to first 80 chars
        return response[:80].replace('\n', ' ')
    
    def _extract_description(self, response: str) -> str:
        """Extract description from response"""
        # Get first paragraph or first 200 chars
        paragraphs = response.split('\n\n')
        if paragraphs:
            desc = paragraphs[0].strip()
            if len(desc) > 50:
                return desc[:300]
        
        return response[:300]
    
    def _extract_expected_impact(self, response: str) -> str:
        """Extract expected impact"""
        response_lower = response.lower()
        
        # Look for impact statements
        impact_keywords = ["impact", "benefit", "improve", "increase", "reduce", "optimize"]
        for keyword in impact_keywords:
            if keyword in response_lower:
                # Try to extract the sentence
                sentences = response.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()[:200]
        
        return "Expected positive impact on system performance"
    
    def _assess_risk(self, response: str, category: LearningCategory) -> str:
        """Assess risk level"""
        response_lower = response.lower()
        
        # High risk indicators
        if any(word in response_lower for word in ["experimental", "untested", "breaking", "major change"]):
            return "High risk - requires careful testing"
        elif any(word in response_lower for word in ["safe", "proven", "standard", "best practice"]):
            return "Low risk - safe to implement"
        else:
            return "Medium risk - standard precautions recommended"
    
    def _update_learning_patterns(self, insight: LearningInsight):
        """Update learning patterns based on new insight"""
        pattern_key = f"{insight.category.value}_patterns"
        if pattern_key in self.learning_patterns:
            self.learning_patterns[pattern_key].append({
                "insight_id": insight.insight_id,
                "source": insight.source.value,
                "timestamp": insight.learned_at.isoformat(),
                "confidence": insight.confidence
            })
    
    def synthesize_insights(self, category: Optional[LearningCategory] = None) -> List[LearningInsight]:
        """Synthesize insights from multiple sources"""
        insights = list(self.learned_insights.values())
        
        if category:
            insights = [i for i in insights if i.category == category]
        
        # Sort by confidence and priority
        insights.sort(key=lambda x: (
            x.priority.value == "critical",
            x.confidence,
            x.learned_at
        ), reverse=True)
        
        return insights
    
    def generate_improvement_actions(self, insight: LearningInsight) -> List[ImprovementAction]:
        """Generate improvement actions from insight"""
        actions = []
        
        for agent in insight.applicable_to:
            action_id = f"action_{insight.insight_id}_{agent}_{int(time.time())}"
            
            action = ImprovementAction(
                action_id=action_id,
                insight_id=insight.insight_id,
                agent=agent,
                action_type=self._determine_action_type(insight, agent),
                description=f"Apply {insight.title} to {agent}",
                implementation=self._generate_implementation(insight, agent),
                priority=insight.priority,
                created_at=datetime.now(),
                metadata={
                    "source": insight.source.value,
                    "category": insight.category.value,
                    "confidence": insight.confidence
                }
            )
            
            actions.append(action)
            self.improvement_actions[action_id] = action
        
        return actions
    
    def _determine_action_type(self, insight: LearningInsight, agent: str) -> str:
        """Determine action type"""
        if insight.category == LearningCategory.CODE_IMPROVEMENT:
            return "code_change"
        elif insight.category == LearningCategory.ARCHITECTURE:
            return "architecture_update"
        elif insight.category == LearningCategory.TRADING_STRATEGY:
            return "strategy_adjustment"
        elif insight.category == LearningCategory.COMPLIANCE:
            return "compliance_update"
        else:
            return "config_update"
    
    def _generate_implementation(self, insight: LearningInsight, agent: str) -> str:
        """Generate implementation code/instructions"""
        if insight.implementation_steps:
            return "\n".join([f"{i+1}. {step}" for i, step in enumerate(insight.implementation_steps)])
        else:
            return f"Apply insight: {insight.description}\n\nContent: {insight.content[:500]}"


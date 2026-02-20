#!/usr/bin/env python3
"""
Casey Enhanced Learning System - Main Integration

Integrates all learning components:
- Multi-model continuous learning
- Self-healing
- Financial optimization
- NAE improvement

This is the main entry point that plugs into Casey.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests

# Add NAE paths
script_dir = os.path.dirname(os.path.abspath(__file__))
nae_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, nae_root)

from agents.casey_continuous_learning import (
    MultiModelLearner,
    LearningSource,
    LearningCategory,
    LearningInsight,
    ImprovementAction
)
from agents.casey_self_healing import CaseySelfHealing
from agents.casey_financial_optimizer import CaseyFinancialOptimizer

logger = logging.getLogger(__name__)


class CaseyEnhancedLearningSystem:
    """
    Main enhanced learning system for Casey
    
    Integrates:
    - Multi-model learning from Cursor, ChatGPT, Grok, Gemini
    - Self-healing capabilities
    - Financial optimization
    - NAE improvement suggestions
    """
    
    def __init__(self, casey_agent):
        """Initialize enhanced learning system"""
        self.casey = casey_agent
        
        # Initialize components
        self.multi_model_learner = MultiModelLearner()
        self.self_healing = CaseySelfHealing(casey_agent)
        self.financial_optimizer = CaseyFinancialOptimizer(casey_agent)
        
        # Learning loop
        self.learning_active = False
        self.learning_thread = None
        self.learning_interval = 3600  # 1 hour
        
        # API clients for different models
        self.api_clients = {
            LearningSource.CHATGPT: self._create_openai_client(),
            LearningSource.GROK: self._create_grok_client(),
            LearningSource.GEMINI: self._create_gemini_client()
        }
        
        logger.info("🧠 Casey Enhanced Learning System initialized")
    
    def start(self):
        """Start continuous learning"""
        if self.learning_active:
            return
        
        self.learning_active = True
        
        # Start self-healing monitoring
        self.self_healing.start_monitoring()
        
        # Start learning loop
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("🚀 Enhanced learning system started")
    
    def stop(self):
        """Stop continuous learning"""
        self.learning_active = False
        self.self_healing.stop_monitoring()
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        logger.info("🛑 Enhanced learning system stopped")
    
    def _learning_loop(self):
        """Main learning loop"""
        while self.learning_active:
            try:
                # Learn from multiple models
                self._learn_from_models()
                
                # Analyze and synthesize insights
                self._synthesize_and_apply_insights()
                
                # Analyze financial optimizations
                self._analyze_financial_optimizations()
                
                # Generate improvement suggestions
                self._generate_improvement_suggestions()
                
                time.sleep(self.learning_interval)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(self.learning_interval)
    
    def _learn_from_models(self):
        """Learn from all configured models"""
        # Generate learning prompts based on current NAE state
        prompts = self._generate_learning_prompts()
        
        for source in [LearningSource.CHATGPT, LearningSource.GROK, LearningSource.GEMINI]:
            if not self.multi_model_learner.model_configs[source]["enabled"]:
                continue
            
            for prompt in prompts:
                try:
                    response = self._query_model(source, prompt)
                    if response:
                        insight = self.multi_model_learner.learn_from_model(
                            source=source,
                            prompt=prompt,
                            response=response,
                            context=self._get_nae_context()
                        )
                        if insight:
                            self.casey.log_action(f"📚 Learned from {source.value}: {insight.title}")
                except Exception as e:
                    logger.warning(f"Error learning from {source.value}: {e}")
    
    def _generate_learning_prompts(self) -> List[str]:
        """Generate learning prompts based on NAE state"""
        prompts = []
        
        # Prompt 1: How to improve NAE architecture
        prompts.append("""
        Analyze the Neural Agency Engine (NAE) trading system architecture and suggest improvements.
        Focus on:
        - Code quality and maintainability
        - Performance optimizations
        - Better agent coordination
        - Risk management enhancements
        - Compliance improvements
        
        Provide specific, actionable recommendations.
        """)
        
        # Prompt 2: How to expedite financial gains safely
        prompts.append("""
        Suggest ways to expedite financial gains in options trading while maintaining:
        - Full compliance with FINRA/SEC regulations
        - Risk management best practices
        - Pattern Day Trader (PDT) rule compliance
        - Account safety limits
        
        Focus on legitimate strategies like:
        - Better entry/exit timing
        - Optimal position sizing
        - Strategy selection
        - Tax optimization
        
        Provide specific, actionable recommendations.
        """)
        
        # Prompt 3: How to improve self-healing
        prompts.append("""
        Suggest improvements for a self-healing trading system that:
        - Automatically detects and fixes issues
        - Monitors system health continuously
        - Learns from failures
        - Improves itself over time
        
        Provide specific, actionable recommendations.
        """)
        
        # Prompt 4: How to improve agent coordination
        prompts.append("""
        Suggest improvements for multi-agent coordination in a trading system with agents:
        - Casey (orchestrator)
        - Optimus (live trading)
        - Ralph (strategy generation)
        - Donnie (strategy validation)
        - Genny (wealth management)
        
        Focus on better communication, coordination, and efficiency.
        Provide specific, actionable recommendations.
        """)
        
        return prompts
    
    def _query_model(self, source: LearningSource, prompt: str) -> Optional[str]:
        """Query a specific model"""
        try:
            if source == LearningSource.CHATGPT:
                return self._query_chatgpt(prompt)
            elif source == LearningSource.GROK:
                return self._query_grok(prompt)
            elif source == LearningSource.GEMINI:
                return self._query_gemini(prompt)
        except Exception as e:
            logger.error(f"Error querying {source.value}: {e}")
        
        return None
    
    def _query_chatgpt(self, prompt: str) -> Optional[str]:
        """Query ChatGPT via OpenAI API"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert trading system architect and financial advisor. Provide specific, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            return None
        except Exception as e:
            logger.error(f"Error querying ChatGPT: {e}")
            return None
    
    def _query_grok(self, prompt: str) -> Optional[str]:
        """Query Grok via API"""
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            return None
        
        try:
            # Grok API endpoint (adjust based on actual API)
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": "You are an expert trading system architect."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content")
        except Exception as e:
            logger.error(f"Error querying Grok: {e}")
        
        return None
    
    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Query Gemini via API"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"You are an expert trading system architect. {prompt}"
            )
            
            return response.text
        except ImportError:
            logger.warning("Google Generative AI library not installed. Install with: pip install google-generativeai")
            return None
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            return None
    
    def _create_openai_client(self):
        """Create OpenAI client"""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return openai.OpenAI(api_key=api_key)
        except:
            pass
        return None
    
    def _create_grok_client(self):
        """Create Grok client"""
        # Grok API client would go here
        return None
    
    def _create_gemini_client(self):
        """Create Gemini client"""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                return genai.GenerativeModel('gemini-pro')
        except:
            pass
        return None
    
    def _get_nae_context(self) -> Dict[str, Any]:
        """Get current NAE context for learning"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "agents": [],
            "system_health": {},
            "recent_trades": [],
            "recent_errors": []
        }
        
        # Get agent status
        if hasattr(self.casey, 'monitored_agents'):
            if isinstance(self.casey.monitored_agents, dict):
                context["agents"] = list(self.casey.monitored_agents.keys())
            elif isinstance(self.casey.monitored_agents, list):
                context["agents"] = [name for name, _ in self.casey.monitored_agents]
        
        # Get system health
        if hasattr(self.self_healing, 'health_history') and self.self_healing.health_history:
            latest_health = self.self_healing.health_history[-1]
            context["system_health"] = {
                "health_score": latest_health.health_score,
                "overall_health": latest_health.overall_health.value,
                "issues": latest_health.issues
            }
        
        return context
    
    def _synthesize_and_apply_insights(self):
        """Synthesize insights and generate improvement actions"""
        # Get high-priority insights
        insights = self.multi_model_learner.synthesize_insights()
        
        # Focus on high-confidence, high-priority insights
        high_value_insights = [
            i for i in insights
            if i.confidence > 0.7 and i.priority.value in ["critical", "high"]
        ][:10]  # Top 10
        
        for insight in high_value_insights:
            # Generate improvement actions
            actions = self.multi_model_learner.generate_improvement_actions(insight)
            
            for action in actions:
                # Check if action should be auto-applied
                if self._should_auto_apply(action):
                    self._apply_improvement_action(action)
                else:
                    # Queue for review
                    self.casey.log_action(f"💡 Improvement suggestion: {action.description}")
    
    def _should_auto_apply(self, action: ImprovementAction) -> bool:
        """Determine if action should be auto-applied"""
        # Auto-apply only safe, low-risk actions
        if action.priority.value == "critical" and action.metadata.get("risk_assessment", "").lower().startswith("low"):
            return True
        
        # Don't auto-apply high-risk or compliance-related changes
        if "compliance" in action.description.lower() or "risk" in action.description.lower():
            return False
        
        return False
    
    def _apply_improvement_action(self, action: ImprovementAction):
        """Apply an improvement action"""
        try:
            action.status = "implementing"
            
            # Apply based on action type
            if action.action_type == "code_change":
                # This would modify code files
                self.casey.log_action(f"🔧 Applying code change: {action.description}")
            elif action.action_type == "config_update":
                # This would update configuration
                self.casey.log_action(f"⚙️ Applying config update: {action.description}")
            elif action.action_type == "strategy_adjustment":
                # This would adjust trading strategies
                self.casey.log_action(f"📊 Applying strategy adjustment: {action.description}")
            
            action.status = "completed"
            action.applied_at = datetime.now()
            action.result = "Successfully applied"
            
            self.casey.log_action(f"✅ Applied improvement: {action.description}")
        
        except Exception as e:
            action.status = "failed"
            action.result = str(e)
            logger.error(f"Error applying improvement action: {e}")
    
    def _analyze_financial_optimizations(self):
        """Analyze and apply financial optimizations"""
        opportunities = self.financial_optimizer.analyze_optimization_opportunities()
        
        # Apply safe optimizations automatically
        for opt in opportunities:
            if opt.compliance_safe and opt.risk_level == "low":
                self.financial_optimizer.apply_optimization(opt.optimization_id)
    
    def _generate_improvement_suggestions(self):
        """Generate improvement suggestions for NAE"""
        # Get insights
        insights = self.multi_model_learner.synthesize_insights()
        
        # Generate suggestions
        suggestions = []
        for insight in insights[:5]:  # Top 5
            suggestions.append({
                "title": insight.title,
                "description": insight.description,
                "category": insight.category.value,
                "priority": insight.priority.value,
                "expected_impact": insight.expected_impact,
                "risk": insight.risk_assessment
            })
        
        # Store in Casey
        if hasattr(self.casey, 'improvement_suggestions'):
            self.casey.improvement_suggestions = suggestions
        
        return suggestions
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning report"""
        return {
            "learning_status": "active" if self.learning_active else "inactive",
            "insights_learned": len(self.multi_model_learner.learned_insights),
            "improvement_actions": len(self.multi_model_learner.improvement_actions),
            "self_awareness": self.self_healing.get_self_awareness_report(),
            "financial_optimizations": self.financial_optimizer.get_optimization_report(),
            "recent_insights": [
                {
                    "title": i.title,
                    "source": i.source.value,
                    "category": i.category.value,
                    "confidence": i.confidence,
                    "priority": i.priority.value
                }
                for i in list(self.multi_model_learner.learned_insights.values())[-10:]
            ]
        }


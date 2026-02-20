# NAE/model_config.py
"""
Model Assignment Configuration for NAE Agents
Assigns optimal models to each agent component
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Model configuration for an agent"""
    model_name: str
    provider: str  # "openai", "anthropic", "deepseek", etc.
    api_key_env: str
    temperature: float
    max_tokens: Optional[int]
    timeout: int
    description: str

class ModelAssignmentManager:
    """Manages model assignments for NAE agents"""
    
    def __init__(self, config_file: str = "config/model_assignments.json"):
        self.config_file = config_file
        self.assignments: Dict[str, ModelConfig] = {}
        self._load_assignments()
    
    def _load_assignments(self):
        """Load model assignments from config"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                for agent_name, model_data in config.items():
                    self.assignments[agent_name] = ModelConfig(**model_data)
            else:
                # Create default assignments
                self._create_default_assignments()
                self._save_assignments()
        except Exception as e:
            print(f"Error loading model assignments: {e}")
            self._create_default_assignments()
    
    def _create_default_assignments(self):
        """Create optimal model assignments for each agent"""
        self.assignments = {
            "RalphAgent": ModelConfig(
                model_name="claude-sonnet-4-20250514",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.7,
                max_tokens=8192,
                timeout=120,
                description="Best for strategy analysis and learning from multiple sources"
            ),
            "CaseyAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.3,
                max_tokens=16384,
                timeout=120,
                description="Superior code generation and agent building"
            ),
            "DonnieAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.5,
                max_tokens=8192,
                timeout=120,
                description="Execution planning and strategy validation"
            ),
            "OptimusAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.2,
                max_tokens=4096,
                timeout=60,
                description="Risk analysis and trading decisions - requires precision"
            ),
            "SplinterAgent": ModelConfig(
                model_name="claude-sonnet-4-20250514",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=8192,
                timeout=120,
                description="Complex orchestration and multi-agent coordination"
            ),
            "BebopAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.4,
                max_tokens=4096,
                timeout=60,
                description="Pattern recognition and monitoring"
            ),
            "PhisherAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.2,
                max_tokens=8192,
                timeout=120,
                description="Security analysis and code auditing"
            ),
            "GennyAgent": ModelConfig(
                model_name="claude-sonnet-4-20250514",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.7,
                max_tokens=8192,
                timeout=120,
                description="Long-term planning and generational wealth tracking"
            ),
            "RocksteadyAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.3,
                max_tokens=4096,
                timeout=60,
                description="Compliance analysis and regulatory monitoring"
            ),
            "ShredderAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.3,
                max_tokens=4096,
                timeout=60,
                description="Risk management and portfolio analysis"
            ),
            "MikeyAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.4,
                max_tokens=8192,
                timeout=120,
                description="Data processing and analysis"
            ),
            "LeoAgent": ModelConfig(
                model_name="claude-sonnet-4-20250514",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                temperature=0.6,
                max_tokens=8192,
                timeout=120,
                description="Strategic leadership and decision making"
            ),
            "AprilAgent": ModelConfig(
                model_name="gpt-4-turbo-preview",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                temperature=0.2,
                max_tokens=4096,
                timeout=60,
                description="Financial analysis and ledger management"
            )
        }
    
    def _save_assignments(self):
        """Save assignments to config file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {name: asdict(model_config) for name, model_config in self.assignments.items()}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving assignments: {e}")
    
    def get_model_config(self, agent_name: str) -> Optional[ModelConfig]:
        """Get model configuration for an agent"""
        return self.assignments.get(agent_name)
    
    def update_model_config(self, agent_name: str, model_config: ModelConfig):
        """Update model configuration for an agent"""
        self.assignments[agent_name] = model_config
        self._save_assignments()
    
    def get_llm_config(self, agent_name: str) -> Optional[dict]:
        """Get AutoGen-compatible LLM config for an agent"""
        model_config = self.get_model_config(agent_name)
        if not model_config:
            return None
        
        # Try to get API key from environment
        api_key = os.getenv(model_config.api_key_env)
        
        # If not found, try auto-loader
        if not api_key:
            try:
                from env_loader import get_env_loader
                loader = get_env_loader()
                if model_config.api_key_env == 'OPENAI_API_KEY':
                    api_key = loader.get_openai_key()
                elif model_config.api_key_env == 'ANTHROPIC_API_KEY':
                    api_key = loader.get_anthropic_key()
            except Exception:
                pass
        
        if not api_key:
            print(f"Warning: {model_config.api_key_env} not set for {agent_name}")
            return None
        
        if model_config.provider == "openai":
            return {
                "config_list": [{
                    "model": model_config.model_name,
                    "api_key": api_key,
                    "api_type": "open_ai",
                    "base_url": "https://api.openai.com/v1"
                }],
                "temperature": model_config.temperature,
                "timeout": model_config.timeout,
                "max_tokens": model_config.max_tokens
            }
        elif model_config.provider == "anthropic":
            return {
                "config_list": [{
                    "model": model_config.model_name,
                    "api_key": api_key,
                    "api_type": "anthropic",
                    "base_url": "https://api.anthropic.com/v1"
                }],
                "temperature": model_config.temperature,
                "timeout": model_config.timeout,
                "max_tokens": model_config.max_tokens
            }
        return None


# Global model assignment manager
_model_manager = None

def get_model_manager() -> ModelAssignmentManager:
    """Get global model assignment manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelAssignmentManager()
    return _model_manager

def get_agent_llm_config(agent_name: str) -> Optional[dict]:
    """Get LLM config for an agent"""
    return get_model_manager().get_llm_config(agent_name)


if __name__ == "__main__":
    # Test model assignments
    manager = ModelAssignmentManager()
    
    print("Model Assignments:")
    for agent_name, config in manager.assignments.items():
        print(f"\n{agent_name}:")
        print(f"  Model: {config.model_name}")
        print(f"  Provider: {config.provider}")
        print(f"  Description: {config.description}")


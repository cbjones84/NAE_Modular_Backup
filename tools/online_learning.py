# NAE/tools/online_learning.py
"""
Online Learning Framework for NAE

Implements incremental learning with catastrophic forgetting prevention:
- Elastic Weight Consolidation (EWC)
- Replay buffers
- Incremental model updates
- Distribution drift adaptation
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
import os


@dataclass
class ReplayBuffer:
    """Replay buffer for online learning"""
    max_size: int = 10000
    buffer: deque = field(default_factory=lambda: deque(maxlen=10000))
    
    def add(self, sample: Dict[str, Any]):
        """Add sample to buffer"""
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.buffer)


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
    
    Preserves important weights from previous tasks while learning new ones
    """
    
    def __init__(self, lambda_ewc: float = 0.4):
        """
        Args:
            lambda_ewc: EWC regularization strength
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_information: Dict[str, np.ndarray] = {}
        self.optimal_weights: Dict[str, np.ndarray] = {}
    
    def compute_fisher_information(
        self,
        model,
        data: List[Dict[str, Any]],
        num_samples: int = 100
    ):
        """
        Compute Fisher information matrix for EWC
        
        Measures importance of each parameter for the task
        """
        # Simplified Fisher information computation
        # In practice, would compute gradients of log-likelihood
        fisher = {}
        
        # Sample from data
        samples = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
        
        # Compute importance (simplified - would use actual gradients)
        for param_name, param_value in model.get_parameters().items():
            # Fisher information approximates parameter importance
            fisher[param_name] = np.ones_like(param_value) * 0.1  # Placeholder
        
        self.fisher_information = fisher
    
    def save_optimal_weights(self, model):
        """Save optimal weights from previous task"""
        self.optimal_weights = model.get_parameters()
    
    def ewc_loss(
        self,
        current_weights: Dict[str, np.ndarray],
        base_loss: float
    ) -> float:
        """
        Compute EWC loss (base loss + regularization)
        
        Penalizes changes to important weights
        """
        ewc_penalty = 0.0
        
        for param_name in current_weights:
            if param_name in self.fisher_information and param_name in self.optimal_weights:
                fisher = self.fisher_information[param_name]
                optimal = self.optimal_weights[param_name]
                current = current_weights[param_name]
                
                # EWC penalty: lambda * Fisher * (current - optimal)^2
                penalty = self.lambda_ewc * np.sum(fisher * (current - optimal) ** 2)
                ewc_penalty += penalty
        
        return base_loss + ewc_penalty


class OnlineLearner:
    """
    Online learning framework for incremental model updates
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        use_ewc: bool = True,
        use_replay: bool = True,
        replay_buffer_size: int = 10000
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.use_ewc = use_ewc
        self.use_replay = use_replay
        
        self.ewc = ElasticWeightConsolidation() if use_ewc else None
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size) if use_replay else None
        
        self.update_count = 0
        self.performance_history: List[float] = []
    
    def update(
        self,
        new_data: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Incremental model update with new data
        
        Args:
            new_data: New training samples
            batch_size: Batch size for updates
        
        Returns:
            Update statistics
        """
        self.update_count += 1
        
        # Add to replay buffer
        if self.replay_buffer:
            for sample in new_data:
                self.replay_buffer.add(sample)
        
        # Combine new data with replay samples
        if self.replay_buffer and self.replay_buffer.size() > 0:
            replay_samples = self.replay_buffer.sample(batch_size // 2)
            training_data = new_data[:batch_size // 2] + replay_samples
        else:
            training_data = new_data[:batch_size]
        
        # Compute Fisher information if using EWC and first update
        if self.use_ewc and self.ewc and self.update_count == 1:
            self.ewc.compute_fisher_information(self.model, training_data)
            self.ewc.save_optimal_weights(self.model)
        
        # Perform incremental update
        try:
            # Simplified update (would use actual model training)
            loss = self._incremental_update(training_data)
            
            # Compute EWC loss if enabled
            if self.use_ewc and self.ewc:
                current_weights = self.model.get_parameters()
                loss = self.ewc.ewc_loss(current_weights, loss)
            
            return {
                "update_count": self.update_count,
                "loss": loss,
                "samples_used": len(training_data),
                "replay_samples": len(replay_samples) if self.replay_buffer else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "update_count": self.update_count,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _incremental_update(self, data: List[Dict[str, Any]]) -> float:
        """Perform incremental model update"""
        # Placeholder for actual model update logic
        # Would call model.fit() or model.partial_fit() depending on model type
        return 0.1  # Placeholder loss
    
    def detect_drift(
        self,
        recent_performance: List[float],
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Detect performance drift
        
        Returns:
            (has_drifted, drift_score)
        """
        if len(recent_performance) < 10:
            return False, 0.0
        
        baseline = np.mean(recent_performance[:len(recent_performance)//2])
        recent = np.mean(recent_performance[len(recent_performance)//2:])
        
        drift_score = abs(recent - baseline) / (baseline + 1e-8)
        has_drifted = drift_score > threshold
        
        return has_drifted, drift_score
    
    def adapt_to_drift(
        self,
        new_data: List[Dict[str, Any]],
        adaptation_rate: float = 0.5
    ):
        """
        Adapt model to distribution drift
        
        Args:
            new_data: New data from drifted distribution
            adaptation_rate: How aggressively to adapt (0-1)
        """
        # Increase learning rate temporarily
        original_lr = self.learning_rate
        self.learning_rate = original_lr * (1 + adaptation_rate)
        
        # Update with new data
        result = self.update(new_data)
        
        # Restore learning rate
        self.learning_rate = original_lr
        
        return result


class MetaLearner:
    """
    Meta-learner for model selection and weighting
    
    Chooses which model to train/weight based on recent performance
    """
    
    def __init__(self):
        self.model_performance: Dict[str, List[float]] = {}
        self.model_weights: Dict[str, float] = {}
    
    def update_performance(self, model_id: str, performance: float):
        """Update performance history for model"""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = []
        
        self.model_performance[model_id].append(performance)
        
        # Keep only recent history
        if len(self.model_performance[model_id]) > 100:
            self.model_performance[model_id] = self.model_performance[model_id][-100:]
    
    def select_model(self, available_models: List[str]) -> str:
        """
        Select best model based on recent performance
        
        Returns:
            Model ID of best performing model
        """
        if not available_models:
            return None
        
        best_model = available_models[0]
        best_performance = 0.0
        
        for model_id in available_models:
            if model_id in self.model_performance:
                recent_perf = np.mean(self.model_performance[model_id][-10:])
                if recent_perf > best_performance:
                    best_performance = recent_perf
                    best_model = model_id
        
        return best_model
    
    def calculate_weights(self, model_ids: List[str]) -> Dict[str, float]:
        """
        Calculate weights for ensemble based on performance
        
        Returns:
            Dictionary of model_id -> weight
        """
        weights = {}
        total_performance = 0.0
        
        # Calculate total performance
        for model_id in model_ids:
            if model_id in self.model_performance:
                perf = np.mean(self.model_performance[model_id][-10:])
                weights[model_id] = max(0.0, perf)
                total_performance += weights[model_id]
            else:
                weights[model_id] = 1.0
                total_performance += 1.0
        
        # Normalize weights
        if total_performance > 0:
            for model_id in weights:
                weights[model_id] /= total_performance
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(model_ids)
            weights = {mid: equal_weight for mid in model_ids}
        
        self.model_weights = weights
        return weights


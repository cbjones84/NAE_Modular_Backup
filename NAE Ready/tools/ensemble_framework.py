# NAE/tools/ensemble_framework.py
"""
Ensemble & Mixture-of-Experts Framework

Supports:
- Multiple model types (statistical, ML, neural, EBM/PGM)
- Performance-weighted ensemble
- Bayesian model averaging
- Stacking with meta-learner
- Regime-aware weighting
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class ModelType(Enum):
    STATISTICAL = "statistical"  # GARCH, HMM
    ML = "ml"  # XGBoost, LightGBM
    NEURAL = "neural"  # LSTM, Transformer
    EBM = "ebm"  # Energy-based models
    PGM = "pgm"  # Probabilistic graphical models


@dataclass
class EnsembleMember:
    """Ensemble member model"""
    model_id: str
    model_type: ModelType
    model_object: Any
    weight: float = 1.0
    performance_history: List[float] = field(default_factory=list)
    last_prediction: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnsembleFramework:
    """
    Ensemble framework for combining multiple models
    """
    
    def __init__(self, weighting_method: str = "performance_weighted"):
        """
        Initialize ensemble
        
        Args:
            weighting_method: "performance_weighted", "bayesian", "equal", "stacking"
        """
        self.weighting_method = weighting_method
        self.members: List[EnsembleMember] = []
        self.meta_learner: Optional[Any] = None  # For stacking
    
    def add_member(
        self,
        model_id: str,
        model_type: ModelType,
        model_object: Any,
        initial_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add model to ensemble"""
        member = EnsembleMember(
            model_id=model_id,
            model_type=model_type,
            model_object=model_object,
            weight=initial_weight,
            metadata=metadata or {}
        )
        self.members.append(member)
    
    def predict(
        self,
        features: np.ndarray,
        regime: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Make ensemble prediction
        
        Args:
            features: Input features
            regime: Current market regime (optional)
        
        Returns:
            (prediction, details)
        """
        if not self.members:
            raise ValueError("No ensemble members")
        
        predictions = []
        weights = []
        member_details = []
        
        for member in self.members:
            try:
                # Get prediction from member
                if hasattr(member.model_object, 'predict'):
                    pred = member.model_object.predict(features)
                elif callable(member.model_object):
                    pred = member.model_object(features)
                else:
                    pred = 0.0
                
                predictions.append(pred)
                
                # Calculate weight
                weight = self._calculate_weight(member, regime)
                weights.append(weight)
                
                member_details.append({
                    "model_id": member.model_id,
                    "model_type": member.model_type.value,
                    "prediction": float(pred),
                    "weight": float(weight)
                })
            except Exception as e:
                # Skip failed predictions
                print(f"Warning: Model {member.model_id} prediction failed: {e}")
                continue
        
        if not predictions:
            return 0.0, {"error": "All models failed"}
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        # Weighted average
        ensemble_prediction = np.average(predictions, weights=weights)
        
        details = {
            "ensemble_prediction": float(ensemble_prediction),
            "members": member_details,
            "weighting_method": self.weighting_method,
            "regime": regime
        }
        
        return float(ensemble_prediction), details
    
    def _calculate_weight(self, member: EnsembleMember, regime: Optional[str] = None) -> float:
        """Calculate weight for ensemble member"""
        if self.weighting_method == "equal":
            return 1.0
        
        elif self.weighting_method == "performance_weighted":
            # Weight by recent performance
            if member.performance_history:
                recent_performance = np.mean(member.performance_history[-10:])  # Last 10
                return max(0.0, recent_performance)  # Ensure non-negative
            return member.weight
        
        elif self.weighting_method == "bayesian":
            # Bayesian model averaging (simplified)
            if member.performance_history:
                # Use performance as likelihood proxy
                likelihood = np.mean(member.performance_history[-10:])
                prior = member.weight
                # Simple Bayesian update
                return prior * likelihood
            return member.weight
        
        elif self.weighting_method == "regime_aware" and regime:
            # Regime-specific weighting
            regime_weights = member.metadata.get("regime_weights", {})
            return regime_weights.get(regime, member.weight)
        
        else:
            return member.weight
    
    def update_performance(self, model_id: str, performance: float):
        """Update performance history for model"""
        member = next((m for m in self.members if m.model_id == model_id), None)
        if member:
            member.performance_history.append(performance)
            # Keep only recent history
            if len(member.performance_history) > 100:
                member.performance_history = member.performance_history[-100:]
    
    def reweight_members(self):
        """Recalculate weights based on performance"""
        if self.weighting_method == "performance_weighted":
            total_performance = sum(
                max(0.0, np.mean(m.performance_history[-10:])) if m.performance_history else 0.0
                for m in self.members
            )
            
            if total_performance > 0:
                for member in self.members:
                    if member.performance_history:
                        performance = np.mean(member.performance_history[-10:])
                        member.weight = max(0.0, performance) / total_performance
                    else:
                        member.weight = 1.0 / len(self.members)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble summary"""
        return {
            "num_members": len(self.members),
            "weighting_method": self.weighting_method,
            "members": [
                {
                    "model_id": m.model_id,
                    "model_type": m.model_type.value,
                    "weight": m.weight,
                    "avg_performance": np.mean(m.performance_history) if m.performance_history else 0.0,
                    "num_predictions": len(m.performance_history)
                }
                for m in self.members
            ]
        }


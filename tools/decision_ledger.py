# NAE/tools/decision_ledger.py
"""
Decision Ledger & Explainability System

Tracks all trading decisions with:
- Model(s) used
- Inputs and features
- Confidence scores
- Expected PnL
- Pre-trade checks
- Human overrides
- Post-trade analysis
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib


class DecisionType(Enum):
    TRADE = "trade"
    HOLD = "hold"
    EXIT = "exit"
    HEDGE = "hedge"


class OverrideType(Enum):
    NONE = "none"
    HUMAN = "human"
    CIRCUIT_BREAKER = "circuit_breaker"
    RISK_LIMIT = "risk_limit"


@dataclass
class ModelDecision:
    """Individual model decision"""
    model_id: str
    model_type: str
    prediction: float
    confidence: float
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class DecisionRecord:
    """Complete decision record"""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    symbol: str
    action: str  # "buy", "sell", "hold", etc.
    
    # Model information
    models_used: List[ModelDecision] = field(default_factory=list)
    ensemble_prediction: Optional[float] = None
    ensemble_confidence: Optional[float] = None
    
    # Inputs
    market_data: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    
    # Risk & validation
    pre_trade_checks: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = "unknown"
    position_size: Optional[float] = None
    
    # Expected outcomes
    expected_pnl: Optional[float] = None
    expected_probability: Optional[float] = None
    
    # Execution
    executed: bool = False
    execution_price: Optional[float] = None
    execution_timestamp: Optional[datetime] = None
    
    # Overrides
    override_type: OverrideType = OverrideType.NONE
    override_reason: Optional[str] = None
    
    # Post-trade
    actual_pnl: Optional[float] = None
    performance_attribution: Dict[str, float] = field(default_factory=dict)


class DecisionLedger:
    """
    Immutable decision ledger for audit and explainability
    """
    
    def __init__(self, ledger_path: str = "logs/decision_ledger"):
        self.ledger_path = ledger_path
        os.makedirs(ledger_path, exist_ok=True)
        self.decisions: List[DecisionRecord] = []
    
    def record_decision(
        self,
        decision_type: DecisionType,
        symbol: str,
        action: str,
        models_used: List[ModelDecision],
        market_data: Dict[str, Any],
        features: Dict[str, float],
        pre_trade_checks: List[Dict[str, Any]],
        risk_level: str,
        position_size: Optional[float] = None,
        expected_pnl: Optional[float] = None,
        expected_probability: Optional[float] = None,
        ensemble_prediction: Optional[float] = None,
        ensemble_confidence: Optional[float] = None
    ) -> DecisionRecord:
        """
        Record a trading decision
        
        Returns decision record with unique ID
        """
        decision_id = self._generate_decision_id()
        timestamp = datetime.now()
        
        decision = DecisionRecord(
            decision_id=decision_id,
            timestamp=timestamp,
            decision_type=decision_type,
            symbol=symbol,
            action=action,
            models_used=models_used,
            ensemble_prediction=ensemble_prediction,
            ensemble_confidence=ensemble_confidence,
            market_data=market_data,
            features=features,
            pre_trade_checks=pre_trade_checks,
            risk_level=risk_level,
            position_size=position_size,
            expected_pnl=expected_pnl,
            expected_probability=expected_probability
        )
        
        # Save to ledger
        self._save_decision(decision)
        
        self.decisions.append(decision)
        
        return decision
    
    def record_execution(
        self,
        decision_id: str,
        execution_price: float,
        actual_pnl: Optional[float] = None
    ):
        """Record execution of a decision"""
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)
        if not decision:
            # Try to load from file
            decision = self._load_decision(decision_id)
        
        if decision:
            decision.executed = True
            decision.execution_price = execution_price
            decision.execution_timestamp = datetime.now()
            if actual_pnl is not None:
                decision.actual_pnl = actual_pnl
            
            # Update saved decision
            self._save_decision(decision)
    
    def record_override(
        self,
        decision_id: str,
        override_type: OverrideType,
        reason: str
    ):
        """Record human or system override"""
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)
        if not decision:
            decision = self._load_decision(decision_id)
        
        if decision:
            decision.override_type = override_type
            decision.override_reason = reason
            self._save_decision(decision)
    
    def record_performance_attribution(
        self,
        decision_id: str,
        attribution: Dict[str, float]
    ):
        """Record performance attribution to models"""
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)
        if not decision:
            decision = self._load_decision(decision_id)
        
        if decision:
            decision.performance_attribution = attribution
            self._save_decision(decision)
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Get decision by ID"""
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)
        if not decision:
            decision = self._load_decision(decision_id)
        return decision
    
    def get_decisions(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        executed_only: bool = False
    ) -> List[DecisionRecord]:
        """Get decisions with filters"""
        decisions = self.decisions.copy()
        
        if symbol:
            decisions = [d for d in decisions if d.symbol == symbol]
        
        if start_date:
            decisions = [d for d in decisions if d.timestamp >= start_date]
        
        if end_date:
            decisions = [d for d in decisions if d.timestamp <= end_date]
        
        if executed_only:
            decisions = [d for d in decisions if d.executed]
        
        return sorted(decisions, key=lambda x: x.timestamp)
    
    def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        """Generate explanation for a decision"""
        decision = self.get_decision(decision_id)
        if not decision:
            return {"error": "Decision not found"}
        
        explanation = {
            "decision_id": decision_id,
            "timestamp": decision.timestamp.isoformat(),
            "symbol": decision.symbol,
            "action": decision.action,
            "decision_type": decision.decision_type.value,
            "explanation": {
                "models_used": len(decision.models_used),
                "model_details": [
                    {
                        "model_id": m.model_id,
                        "model_type": m.model_type,
                        "prediction": m.prediction,
                        "confidence": m.confidence,
                        "top_features": m.top_features[:5]  # Top 5 features
                    }
                    for m in decision.models_used
                ],
                "ensemble": {
                    "prediction": decision.ensemble_prediction,
                    "confidence": decision.ensemble_confidence
                },
                "key_features": dict(sorted(decision.features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]),
                "pre_trade_checks": decision.pre_trade_checks,
                "risk_assessment": decision.risk_level,
                "expected_outcome": {
                    "expected_pnl": decision.expected_pnl,
                    "expected_probability": decision.expected_probability
                }
            }
        }
        
        if decision.executed:
            explanation["execution"] = {
                "executed": True,
                "execution_price": decision.execution_price,
                "actual_pnl": decision.actual_pnl,
                "performance_attribution": decision.performance_attribution
            }
        
        if decision.override_type != OverrideType.NONE:
            explanation["override"] = {
                "type": decision.override_type.value,
                "reason": decision.override_reason
            }
        
        return explanation
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        timestamp_str = datetime.now().isoformat()
        hash_str = hashlib.sha256(timestamp_str.encode()).hexdigest()[:12]
        return f"DEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_str}"
    
    def _save_decision(self, decision: DecisionRecord):
        """Save decision to ledger file"""
        filename = f"{decision.decision_id}.json"
        filepath = os.path.join(self.ledger_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(decision), f, indent=2, default=str)
    
    def _load_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Load decision from file"""
        filename = f"{decision_id}.json"
        filepath = os.path.join(self.ledger_path, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            # Convert enums
            data["decision_type"] = DecisionType(data["decision_type"])
            data["override_type"] = OverrideType(data["override_type"])
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            if data.get("execution_timestamp"):
                data["execution_timestamp"] = datetime.fromisoformat(data["execution_timestamp"])
            
            # Convert model decisions
            for m in data["models_used"]:
                m["top_features"] = [tuple(f) for f in m["top_features"]]
            
            return DecisionRecord(**data)
        except Exception as e:
            print(f"Error loading decision {decision_id}: {e}")
            return None


# Global ledger instance
_decision_ledger: Optional[DecisionLedger] = None


def get_decision_ledger() -> DecisionLedger:
    """Get or create global decision ledger"""
    global _decision_ledger
    if _decision_ledger is None:
        _decision_ledger = DecisionLedger()
    return _decision_ledger


# NAE/tools/robustness_integration.py
"""
Robustness Systems Integration

Ensures all robustness systems are properly initialized and integrated
across all NAE agents for holistic enhancement.
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# Import all robustness systems
try:
    from tools.metrics_collector import get_metrics_collector
    from tools.risk_controls import RiskControlSystem, CircuitBreakerConfig, PositionLimit
    from tools.decision_ledger import get_decision_ledger
    from tools.ensemble_framework import EnsembleFramework
    from tools.regime_detection import RegimeDetector
    from tools.data_quality import get_data_lake, get_data_validator
    from tools.model_registry import get_model_registry
    from tools.backtest_engine import BacktestEngine, BacktestConfig
    ROBUSTNESS_AVAILABLE = True
except ImportError as e:
    ROBUSTNESS_AVAILABLE = False
    print(f"Warning: Robustness systems not fully available: {e}")


class RobustnessIntegrator:
    """
    Integrates all robustness systems across NAE
    """
    
    def __init__(self):
        self.systems_initialized = False
        self.initialization_status: Dict[str, bool] = {}
        
    def initialize_all_systems(self, portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """
        Initialize all robustness systems
        
        Returns status of each system
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Metrics Collector
        try:
            metrics = get_metrics_collector()
            self.initialization_status["metrics"] = True
            status["systems"]["metrics"] = {"status": "initialized", "port": 8000}
        except Exception as e:
            self.initialization_status["metrics"] = False
            status["systems"]["metrics"] = {"status": "failed", "error": str(e)}
        
        # Risk Controls
        try:
            risk_system = RiskControlSystem(
                portfolio_value=portfolio_value,
                circuit_breaker_config=CircuitBreakerConfig(),
                position_limits=PositionLimit()
            )
            self.initialization_status["risk_controls"] = True
            status["systems"]["risk_controls"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["risk_controls"] = False
            status["systems"]["risk_controls"] = {"status": "failed", "error": str(e)}
        
        # Decision Ledger
        try:
            ledger = get_decision_ledger()
            self.initialization_status["decision_ledger"] = True
            status["systems"]["decision_ledger"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["decision_ledger"] = False
            status["systems"]["decision_ledger"] = {"status": "failed", "error": str(e)}
        
        # Ensemble Framework
        try:
            ensemble = EnsembleFramework(weighting_method="performance_weighted")
            self.initialization_status["ensemble"] = True
            status["systems"]["ensemble"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["ensemble"] = False
            status["systems"]["ensemble"] = {"status": "failed", "error": str(e)}
        
        # Regime Detection
        try:
            regime_detector = RegimeDetector()
            self.initialization_status["regime_detection"] = True
            status["systems"]["regime_detection"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["regime_detection"] = False
            status["systems"]["regime_detection"] = {"status": "failed", "error": str(e)}
        
        # Data Quality
        try:
            data_lake = get_data_lake()
            data_validator = get_data_validator()
            self.initialization_status["data_quality"] = True
            status["systems"]["data_quality"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["data_quality"] = False
            status["systems"]["data_quality"] = {"status": "failed", "error": str(e)}
        
        # Model Registry
        try:
            registry = get_model_registry()
            self.initialization_status["model_registry"] = True
            status["systems"]["model_registry"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["model_registry"] = False
            status["systems"]["model_registry"] = {"status": "failed", "error": str(e)}
        
        # Backtest Engine
        try:
            backtest_config = BacktestConfig()
            backtest_engine = BacktestEngine(backtest_config)
            self.initialization_status["backtest_engine"] = True
            status["systems"]["backtest_engine"] = {"status": "initialized"}
        except Exception as e:
            self.initialization_status["backtest_engine"] = False
            status["systems"]["backtest_engine"] = {"status": "failed", "error": str(e)}
        
        self.systems_initialized = all(self.initialization_status.values())
        status["overall_status"] = "initialized" if self.systems_initialized else "partial"
        
        return status
    
    def verify_agent_integration(self, agent_name: str) -> Dict[str, Any]:
        """
        Verify that an agent has integrated robustness systems
        
        Checks for:
        - Metrics collection
        - Risk controls
        - Decision ledger
        - Ensemble framework
        - Regime detection
        """
        verification = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "integrations": {}
        }
        
        # This would check agent attributes
        # For now, return structure
        verification["integrations"] = {
            "metrics": "check_agent_has_metrics_collector",
            "risk_controls": "check_agent_has_risk_system",
            "decision_ledger": "check_agent_has_decision_ledger",
            "ensemble": "check_agent_has_ensemble",
            "regime_detection": "check_agent_has_regime_detector"
        }
        
        return verification
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all robustness systems"""
        return {
            "timestamp": datetime.now().isoformat(),
            "systems": self.initialization_status,
            "overall_health": "healthy" if self.systems_initialized else "degraded"
        }


# Global integrator
_robustness_integrator: RobustnessIntegrator = None


def get_robustness_integrator() -> RobustnessIntegrator:
    """Get or create global robustness integrator"""
    global _robustness_integrator
    if _robustness_integrator is None:
        _robustness_integrator = RobustnessIntegrator()
    return _robustness_integrator


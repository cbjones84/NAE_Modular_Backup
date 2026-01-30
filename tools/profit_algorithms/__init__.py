# NAE/tools/profit_algorithms/__init__.py
"""
Profit Enhancement Algorithms Module
Contains implementations of algorithms to improve NAE profitability

VERY_AGGRESSIVE MODE - Optimized for Growth Milestones:
Year 1: $9,411 | Year 5: $982,500
Year 2: $44,110 | Year 6: $2,477,897
Year 3: $152,834 | Year 7: $6,243,561 (TARGET)
Year 4: $388,657 | Year 8: $15,726,144 (STRETCH)
"""

from .meta_labeling import MetaLabelingModel
from .kelly_criterion import KellyCriterion
from .smart_order_routing import (
    SmartOrderRouter,
    ExecutionVenue,
    create_default_venues,
)
from .universal_portfolio import UniversalPortfolio
from .timing_strategies import (
    TimingStrategyEngine,
    create_timing_engine,
    EntryAnalysis,
    ExitAnalysis,
    TechnicalIndicators,
    MarketCondition,
    EntrySignal,
    ExitReason
)
from .iv_surface_model import (
    IVSurfaceForecaster,
    IVSurfaceSnapshot,
    IVForecastResult,
    build_surface_from_chain,
)
from .volatility_ensemble import (
    VolatilityEnsembleForecaster,
    EnsembleForecast,
    GARCHNotAvailableError,
)
from .dispersion_engine import DispersionEngine, DispersionSignal, implied_correlation
from .hedging_optimizer import HedgingOptimizer, HedgingDecision, GreekExposure
from .position_sizing import HybridKellySizer, KellyInput, KellyResult
from .rl_trading_agent import RLTradingAgent, RLState, RLAction, RLExperience
from .execution_costs import ExecutionCostModel, ExecutionInputs, ExecutionCost

try:
    from .lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTMPredictor = None
    LSTM_AVAILABLE = False

try:
    from .quant_agent_framework import (
        QuantAgentFramework,
        IndicatorAgent,
        PatternAgent,
        TrendAgent,
        RiskAgent,
        MarketSignal,
    )
    QUANT_AGENT_AVAILABLE = True
except ImportError:
    QuantAgentFramework = None
    IndicatorAgent = None
    PatternAgent = None
    TrendAgent = None
    RiskAgent = None
    MarketSignal = None
    QUANT_AGENT_AVAILABLE = False

try:
    from .enhanced_rl_agent import EnhancedRLTradingAgent
    ENHANCED_RL_AVAILABLE = True
except ImportError:
    EnhancedRLTradingAgent = None
    ENHANCED_RL_AVAILABLE = False

try:
    from .rl_execution_optimizer import RLExecutionOptimizer, ExecutionDecision, MarketMicrostructure
    RL_EXECUTION_AVAILABLE = True
except ImportError:
    RLExecutionOptimizer = None
    ExecutionDecision = None
    MarketMicrostructure = None
    RL_EXECUTION_AVAILABLE = False

# Milestone Accelerator - Dynamic growth optimization
try:
    from .milestone_accelerator import (
        MilestoneAccelerator,
        get_accelerator,
        calculate_accelerated_position,
        MilestoneStatus,
        AccelerationProfile,
        AccountPhase as AcceleratorPhase,
        GROWTH_MILESTONES,
        MONTHLY_MILESTONES,
    )
    MILESTONE_ACCELERATOR_AVAILABLE = True
except ImportError:
    MilestoneAccelerator = None
    get_accelerator = None
    calculate_accelerated_position = None
    MilestoneStatus = None
    AccelerationProfile = None
    AcceleratorPhase = None
    GROWTH_MILESTONES = None
    MONTHLY_MILESTONES = None
    MILESTONE_ACCELERATOR_AVAILABLE = False

__all__ = [
    'MetaLabelingModel',
    'KellyCriterion',
    'SmartOrderRouter',
    'ExecutionVenue',
    'create_default_venues',
    'UniversalPortfolio',
    'LSTMPredictor',
    'LSTM_AVAILABLE',
    'TimingStrategyEngine',
    'create_timing_engine',
    "EntryAnalysis",
    "ExitAnalysis",
    "TechnicalIndicators",
    "MarketCondition",
    "EntrySignal",
    "ExitReason",
    "IVSurfaceForecaster",
    "IVSurfaceSnapshot",
    "IVForecastResult",
    "build_surface_from_chain",
    "VolatilityEnsembleForecaster",
    "EnsembleForecast",
    "GARCHNotAvailableError",
    "DispersionEngine",
    "DispersionSignal",
    "implied_correlation",
    "HedgingOptimizer",
    "HedgingDecision",
    "GreekExposure",
    "HybridKellySizer",
    "KellyInput",
    "KellyResult",
    "RLTradingAgent",
    "RLState",
    "RLAction",
    "RLExperience",
    "ExecutionCostModel",
    "ExecutionInputs",
    "ExecutionCost",
    "QuantAgentFramework",
    "IndicatorAgent",
    "PatternAgent",
    "TrendAgent",
    "RiskAgent",
    "MarketSignal",
    "QUANT_AGENT_AVAILABLE",
    "EnhancedRLTradingAgent",
    "ENHANCED_RL_AVAILABLE",
    "RLExecutionOptimizer",
    "ExecutionDecision",
    "MarketMicrostructure",
    "RL_EXECUTION_AVAILABLE",
    # Milestone Accelerator
    "MilestoneAccelerator",
    "get_accelerator",
    "calculate_accelerated_position",
    "MilestoneStatus",
    "AccelerationProfile",
    "AcceleratorPhase",
    "GROWTH_MILESTONES",
    "MONTHLY_MILESTONES",
    "MILESTONE_ACCELERATOR_AVAILABLE",
]
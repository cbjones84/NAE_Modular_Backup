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

try:
    from .meta_labeling import MetaLabelingModel
except ImportError:
    MetaLabelingModel = None

try:
    from .kelly_criterion import KellyCriterion
except ImportError:
    KellyCriterion = None

try:
    from .smart_order_routing import (
        SmartOrderRouter,
        ExecutionVenue,
        create_default_venues,
    )
except ImportError:
    SmartOrderRouter = None
    ExecutionVenue = None
    create_default_venues = None

try:
    from .universal_portfolio import UniversalPortfolio
except ImportError:
    UniversalPortfolio = None
try:
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
except ImportError:
    TimingStrategyEngine = None
    create_timing_engine = None
    EntryAnalysis = None
    ExitAnalysis = None
    TechnicalIndicators = None
    MarketCondition = None
    EntrySignal = None
    ExitReason = None

try:
    from .iv_surface_model import (
        IVSurfaceForecaster,
        IVSurfaceSnapshot,
        IVForecastResult,
        build_surface_from_chain,
    )
except ImportError:
    IVSurfaceForecaster = None
    IVSurfaceSnapshot = None
    IVForecastResult = None
    build_surface_from_chain = None

try:
    from .volatility_ensemble import (
        VolatilityEnsembleForecaster,
        EnsembleForecast,
        GARCHNotAvailableError,
    )
except ImportError:
    VolatilityEnsembleForecaster = None
    EnsembleForecast = None
    GARCHNotAvailableError = None

try:
    from .dispersion_engine import DispersionEngine, DispersionSignal, implied_correlation
except ImportError:
    DispersionEngine = None
    DispersionSignal = None
    implied_correlation = None

try:
    from .hedging_optimizer import HedgingOptimizer, HedgingDecision, GreekExposure
except ImportError:
    HedgingOptimizer = None
    HedgingDecision = None
    GreekExposure = None

try:
    from .position_sizing import HybridKellySizer, KellyInput, KellyResult
except ImportError:
    HybridKellySizer = None
    KellyInput = None
    KellyResult = None

try:
    from .rl_trading_agent import RLTradingAgent, RLState, RLAction, RLExperience
except ImportError:
    RLTradingAgent = None
    RLState = None
    RLAction = None
    RLExperience = None

try:
    from .execution_costs import ExecutionCostModel, ExecutionInputs, ExecutionCost
except ImportError:
    ExecutionCostModel = None
    ExecutionInputs = None
    ExecutionCost = None

try:
    from .lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTMPredictor = None
    LSTM_AVAILABLE = False

# Lightweight ML Suite (replaces TensorFlow/PyTorch — ~60 MB total)
try:
    from .lightweight_ml import (
        LightGBMPredictor,
        ContextualBandit,
        OnlineMetaLabeler,
        WeekendRetrainer,
        LIGHTGBM_AVAILABLE,
    )
    LIGHTWEIGHT_ML_AVAILABLE = True
except ImportError:
    LightGBMPredictor = None
    ContextualBandit = None
    OnlineMetaLabeler = None
    WeekendRetrainer = None
    LIGHTGBM_AVAILABLE = False
    LIGHTWEIGHT_ML_AVAILABLE = False

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

# TD3 Stock Agent — ported from austin-starks/Deep-RL-Stocks (arxiv:1802.09477)
try:
    from .td3_stock_agent import TD3StockAgent, TD3Config, TD3Signal, create_td3_agent, check_system_resources_ok, TORCH_AVAILABLE as TD3_TORCH_AVAILABLE
    TD3_AVAILABLE = True
except ImportError:
    TD3StockAgent = None
    TD3Config = None
    TD3Signal = None
    create_td3_agent = None
    check_system_resources_ok = None
    TD3_TORCH_AVAILABLE = False
    TD3_AVAILABLE = False

try:
    from .stock_trading_env import StockTradingEnv, EnvConfig, run_training_episode
    STOCK_ENV_AVAILABLE = True
except ImportError:
    StockTradingEnv = None
    EnvConfig = None
    run_training_episode = None
    STOCK_ENV_AVAILABLE = False

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
    # Lightweight ML Suite
    "LightGBMPredictor",
    "ContextualBandit",
    "OnlineMetaLabeler",
    "WeekendRetrainer",
    "LIGHTGBM_AVAILABLE",
    "LIGHTWEIGHT_ML_AVAILABLE",
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
    # TD3 Stock Agent (Deep-RL-Stocks port)
    "TD3StockAgent",
    "TD3Config",
    "TD3Signal",
    "create_td3_agent",
    "check_system_resources_ok",
    "TD3_AVAILABLE",
    "TD3_TORCH_AVAILABLE",
    "StockTradingEnv",
    "EnvConfig",
    "run_training_episode",
    "STOCK_ENV_AVAILABLE",
]
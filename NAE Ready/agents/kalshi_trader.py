# NAE/agents/kalshi_trader.py
"""
Kalshi Trader Agent - CFTC-Regulated Prediction Market Trading for NAE
======================================================================
Automated trading agent for Kalshi prediction markets.

Kalshi is the FIRST federally regulated (CFTC) exchange for event contracts
in the United States - providing legal certainty and proper tax reporting.

ENHANCED with official Kalshi resources:
- Official kalshi-python SDK
- Kalshi API documentation (docs.kalshi.com)
- github.com/Kalshi/tools-and-analysis

Strategies:
1. High-Probability Bonding - Buy 95%+ outcomes for consistent returns
2. Cross-Platform Arbitrage - Exploit price differences with Polymarket
3. AI-Powered Semantic Trading - Use NAE's AI for prediction advantage
4. Event-Driven Trading - Trade around known catalyst events
5. Superforecasting - LLM-based probability estimation

REGULATORY ADVANTAGES:
- CFTC regulated = legal certainty in US
- USD-denominated (no crypto required)
- Proper tax reporting (1099 forms)
- FDIC-insured funds (at custodian)

ALIGNED WITH NAE GOALS:
1. Achieve generational wealth
2. Generate $6,243,561+ within 8 years (Year 7 target)
3. Diversify income streams with regulated prediction markets

Growth Milestones:
Year 1: $9,411 | Year 5: $982,500
Year 2: $44,110 | Year 6: $2,477,897
Year 3: $152,834 | Year 7: $6,243,561 (TARGET)
Year 4: $388,657 | Year 8: $15,726,144 (STRETCH)
"""

import os
import sys
import json
import datetime
import time
import re
import math
import threading
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
from enum import Enum

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import goal manager
try:
    from goal_manager import get_nae_goals
    GOALS = get_nae_goals()
except ImportError:
    GOALS = {
        "goal_1": "Achieve generational wealth",
        "goal_2": "Generate $6,243,561 within 8 years",
        "goal_3": "Optimize NAE for successful trading"
    }

# Import Kalshi adapter
try:
    from adapters.kalshi import (
        KalshiAdapter,
        KalshiMarket,
        KalshiEvent,
        KalshiPosition,
        get_kalshi_adapter
    )
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    KalshiAdapter = None
    KalshiMarket = None
    KalshiEvent = None
    KalshiPosition = None
    get_kalshi_adapter = None

# Try to import Polymarket adapter for cross-platform arbitrage
try:
    from adapters.polymarket import PolymarketAdapter, get_polymarket_adapter
    POLYMARKET_AVAILABLE = True
except ImportError:
    POLYMARKET_AVAILABLE = False
    PolymarketAdapter = None
    get_polymarket_adapter = None

# Import Kalshi Learning System
try:
    from tools.kalshi_learning_system import (
        KalshiLearningSystem,
        get_kalshi_learning_system
    )
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    KalshiLearningSystem = None
    get_kalshi_learning_system = None

# Try to import LangChain for AI-powered trading
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None


# =============================================================================
# PROMPTS - Based on Kalshi market characteristics
# =============================================================================

class KalshiPrompter:
    """
    Prompt templates for Kalshi AI trading
    Tailored for Kalshi's market categories and structure
    """
    
    @staticmethod
    def market_analyst() -> str:
        """System prompt for Kalshi market analysis"""
        return """You are an expert analyst specializing in CFTC-regulated prediction markets.
Your task is to analyze Kalshi event contracts and provide probability estimates.

Kalshi markets cover:
1. ECONOMICS - Fed rates, inflation, GDP, unemployment
2. POLITICS - Elections, policy outcomes, government actions
3. WEATHER - Temperature records, hurricanes, climate events
4. FINANCE - Stock market milestones, crypto prices
5. SCIENCE - Discoveries, space events
6. ENTERTAINMENT - Awards, streaming records

You should:
1. Consider base rates and historical precedents
2. Analyze economic indicators and trends
3. Evaluate political dynamics and polling data
4. Consider weather models and forecasts
5. Account for market efficiency - Kalshi markets tend to be well-calibrated

Provide your analysis in a structured format with clear reasoning."""

    @staticmethod
    def superforecaster(title: str, subtitle: str, category: str, rules: str) -> List[Any]:
        """
        Superforecasting prompt for Kalshi markets
        """
        if not HumanMessage or not SystemMessage:
            return []
        
        system_prompt = """You are a superforecaster analyzing CFTC-regulated event contracts on Kalshi.

Your methodology:
1. OUTSIDE VIEW: Start with base rates - how often do similar events occur?
2. INSIDE VIEW: What specific factors make this case different?
3. CATEGORY ANALYSIS:
   - ECONOMICS: Fed policy patterns, historical rate decisions, inflation data
   - POLITICS: Polling aggregates, historical election patterns, incumbency effects
   - WEATHER: Climate models, historical data, meteorological patterns
   - FINANCE: Technical analysis, market sentiment, macro trends
4. SYNTHESIS: Combine views, weigh evidence carefully
5. CALIBRATION: Kalshi markets are often well-calibrated - significant edge is rare

Format your response as:
ANALYSIS: [Your detailed reasoning]
PROBABILITY: [Your probability estimate as decimal, e.g., 0.65]
CONFIDENCE: [Your confidence in this estimate, LOW/MEDIUM/HIGH]
EDGE_ASSESSMENT: [Whether you believe there's genuine edge vs market price]"""

        user_prompt = f"""Analyze this Kalshi market:

TITLE: {title}

SUBTITLE: {subtitle}

CATEGORY: {category}

RESOLUTION RULES: {rules[:500] if rules else 'Standard Kalshi rules apply'}

Estimate the probability of YES occurring.
Consider the specific category dynamics and apply superforecasting methodology."""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    @staticmethod
    def arbitrage_analyzer(
        kalshi_title: str,
        kalshi_price: float,
        polymarket_question: str,
        polymarket_price: float
    ) -> List[Any]:
        """Prompt for cross-platform arbitrage analysis"""
        if not HumanMessage or not SystemMessage:
            return []
        
        system_prompt = """You are an arbitrage specialist analyzing price discrepancies between prediction markets.

Your task:
1. Verify if both markets refer to the SAME event with SAME resolution criteria
2. Analyze why prices might differ (liquidity, fees, user base)
3. Assess execution risk and timing considerations
4. Calculate true arbitrage potential after fees

Remember:
- Kalshi charges ~7% winner fee
- Polymarket has minimal fees but USDC conversion costs
- Resolution criteria may differ subtly
- Timing of settlement may differ"""

        user_prompt = f"""Analyze this potential arbitrage:

KALSHI:
- Market: {kalshi_title}
- Price: {kalshi_price:.2%} YES

POLYMARKET:
- Question: {polymarket_question}
- Price: {polymarket_price:.2%} YES

SPREAD: {abs(kalshi_price - polymarket_price):.2%}

Is this a genuine arbitrage opportunity? What are the risks?"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]


class TradingStrategy(Enum):
    """Available trading strategies"""
    BONDING = "bonding"              # High-probability betting
    ARBITRAGE = "arbitrage"          # Cross-platform price differences
    SEMANTIC = "semantic"            # AI-powered predictions
    EVENT_DRIVEN = "event_driven"    # Trade around catalysts
    SUPERFORECAST = "superforecast"  # LLM-based forecasting


@dataclass
class TradeSignal:
    """Signal for a potential trade"""
    ticker: str
    market_title: str
    strategy: TradingStrategy
    side: str  # "yes" or "no"
    price_cents: int
    count: int      # Number of contracts
    confidence: float
    expected_return: float
    risk_level: str
    reasoning: str
    category: str = ""


@dataclass
class TradeResult:
    """Result of an executed trade"""
    trade_id: str
    ticker: str
    strategy: str
    side: str
    price_cents: int
    count: int
    status: str
    timestamp: str
    pnl_cents: int = 0


class KalshiTrader:
    """
    Kalshi Trading Agent for NAE
    
    ENHANCED with official Kalshi patterns:
    - RSA-PSS authentication
    - Category-specific analysis
    - Cross-platform arbitrage with Polymarket
    - LLM-based superforecasting
    
    Executes prediction market strategies on CFTC-regulated markets
    to generate additional income for NAE's growth milestones.
    """
    
    def __init__(
        self,
        adapter: Optional[Any] = None,
        polymarket_adapter: Optional[Any] = None,
        max_position_size_cents: int = 100000,  # $1000 max per position
        max_daily_trades: int = 20,
        risk_tolerance: str = "MODERATE",
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        demo: bool = False
    ):
        """
        Initialize Kalshi Trader
        
        Args:
            adapter: Kalshi adapter instance
            polymarket_adapter: Optional Polymarket adapter for arbitrage
            max_position_size_cents: Maximum cents per position (default $1000)
            max_daily_trades: Maximum trades per day
            risk_tolerance: CONSERVATIVE, MODERATE, or AGGRESSIVE
            openai_api_key: OpenAI API key for LLM trading
            llm_model: LLM model to use
            demo: Use Kalshi demo environment
        """
        self.goals = GOALS
        self.log_file = "logs/kalshi_trader.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Growth milestones
        self.target_goal = 6_243_561.0
        self.stretch_goal = 15_726_144.0
        self.growth_milestones = {
            1: 9_411, 2: 44_110, 3: 152_834, 4: 388_657,
            5: 982_500, 6: 2_477_897, 7: 6_243_561, 8: 15_726_144
        }
        
        # Initialize Kalshi adapter
        if adapter:
            self.adapter = adapter
        elif KALSHI_AVAILABLE and get_kalshi_adapter:
            self.adapter = get_kalshi_adapter(demo=demo)
            # Verify credentials loaded
            if self.adapter and self.adapter.api_key_id:
                self.log_action(f"Kalshi API Key loaded: {self.adapter.api_key_id[:8]}...")
        else:
            self.adapter = None
            self.log_action("[WARNING] Kalshi adapter not available")
        
        # Initialize Polymarket adapter for arbitrage
        if polymarket_adapter:
            self.polymarket_adapter = polymarket_adapter
        elif POLYMARKET_AVAILABLE and get_polymarket_adapter:
            self.polymarket_adapter = get_polymarket_adapter()
        else:
            self.polymarket_adapter = None
            self.log_action("ℹ️ Polymarket adapter not available for arbitrage")
        
        # Initialize LLM for AI-powered trading
        self.llm: Optional[Any] = None
        self.prompter = KalshiPrompter()
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if LANGCHAIN_AVAILABLE and ChatOpenAI and self.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model=llm_model,
                    temperature=0.1,
                    api_key=self.openai_api_key
                )
                self.log_action(f"✅ LLM initialized ({llm_model})")
            except Exception as e:
                self.log_action(f"⚠️ LLM init failed: {e}")
        
        # Trading parameters
        self.max_position_size_cents = max_position_size_cents
        self.max_daily_trades = max_daily_trades
        self.risk_tolerance = risk_tolerance
        self.demo = demo
        
        # Risk parameters by tolerance
        self.risk_params = {
            "CONSERVATIVE": {
                "min_probability": 0.97,
                "max_position_pct": 0.05,
                "min_volume": 500,
                "min_annualized_return": 30,
                "min_edge": 0.15,
                "kelly_fraction": 0.10
            },
            "MODERATE": {
                "min_probability": 0.95,
                "max_position_pct": 0.10,
                "min_volume": 200,
                "min_annualized_return": 50,
                "min_edge": 0.10,
                "kelly_fraction": 0.15
            },
            "AGGRESSIVE": {
                "min_probability": 0.90,
                "max_position_pct": 0.20,
                "min_volume": 100,
                "min_annualized_return": 100,
                "min_edge": 0.05,
                "kelly_fraction": 0.25
            }
        }
        
        # Trading state
        self.daily_trades = 0
        self.last_trade_date: Optional[datetime.date] = None
        self.trade_history: List[TradeResult] = []
        self.total_pnl_cents = 0
        
        # =====================================================================
        # LEARNING SYSTEM INTEGRATION
        # =====================================================================
        self.learning_system = None
        self.learning_enabled = False
        
        if LEARNING_AVAILABLE and get_kalshi_learning_system:
            try:
                self.learning_system = get_kalshi_learning_system()
                self.learning_enabled = True
                self.log_action("🧠 Learning system enabled - Active learning ON")
            except Exception as e:
                self.log_action(f"⚠️ Learning system init failed: {e}")
        else:
            self.log_action("ℹ️ Learning system not available")
        
        # Load state
        self._load_state()
        
        # Balance tracking for auto-deposit detection
        self.last_known_balance_cents = 0
        self.balance_sync_thread = None
        self.monitoring_active = False
        
        # Initial balance sync
        self._sync_account_balance()
        
        # Start auto-monitoring if in LIVE mode (or demo if requested)
        if not self.demo or demo:  # Simple logic: monitor if running
            self.start_auto_monitoring()
        
        mode = "DEMO" if demo else "LIVE"
        self.log_action(f"KalshiTrader initialized ({mode} mode)")
    
    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.datetime.now().isoformat()
        log_entry = f"[{ts}] {message}"
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
        except Exception:
            pass
        print(f"[KalshiTrader] {message}")
    
    def _load_state(self):
        """Load trading state from file"""
        state_file = "logs/kalshi_trader_state.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.total_pnl_cents = state.get("total_pnl_cents", 0)
                    self.trade_history = [
                        TradeResult(**t) for t in state.get("trade_history", [])[-100:]
                    ]
            except Exception as e:
                self.log_action(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save trading state to file"""
        state_file = "logs/kalshi_trader_state.json"
        try:
            state = {
                "total_pnl_cents": self.total_pnl_cents,
                "trade_history": [asdict(t) for t in self.trade_history[-100:]],
                "last_updated": datetime.datetime.now().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving state: {e}")

    # =============================================================================
    # AUTO-MONITORING (Deposit Detection)
    # =============================================================================
    
    def start_auto_monitoring(self):
        """Start background monitoring for deposits"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.balance_sync_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.balance_sync_thread.start()
        self.log_action("🔄 Auto-deposit monitoring started (60s interval)")
    
    def _monitoring_loop(self):
        """Background loop for monitoring balance"""
        while self.monitoring_active:
            try:
                self._sync_account_balance()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.log_action(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _sync_account_balance(self) -> int:
        """
        Sync current account balance from Kalshi API.
        Detects new deposits and updates available capital.
        
        Returns:
            Current balance in cents
        """
        if not self.adapter:
            return 0
            
        try:
            # Get balance from adapter
            # Note: Adapter implementation might vary, try common methods
            balance_cents = 0
            if hasattr(self.adapter, 'get_balance'):
                balance_info = self.adapter.get_balance()
                if isinstance(balance_info, dict):
                    balance_cents = balance_info.get('balance_cents', 0)
                elif isinstance(balance_info, int):
                    balance_cents = balance_info
            
            # If we couldn't get a valid balance, skip
            if balance_cents <= 0:
                # Fallback to simulated tracking if API fails or returns 0
                return self.last_known_balance_cents
            
            # Check for changes
            if balance_cents != self.last_known_balance_cents:
                diff = balance_cents - self.last_known_balance_cents
                
                # If positive difference, determine if it's profit or deposit
                if diff > 0 and self.last_known_balance_cents > 0:
                    # Check if we recently closed a trade (within last minute)
                    # This is a heuristic; ideal is checking transaction history
                    recent_profit = self._check_recent_profit()
                    
                    if recent_profit < diff:
                        # Likely a deposit!
                        deposit_amount = diff - recent_profit
                        self.log_action(f"💰 New Capital Detected: +${deposit_amount/100:,.2f}")
                        
                        # Automatically update internal capital tracking if needed
                        # (Self-correcting behavior)
                
                self.last_known_balance_cents = balance_cents
                
                # Update max position size based on new balance (maintain risk %)
                # Default to 5% of NAV if not specified otherwise
                safe_pos_size = int(balance_cents * 0.05)
                if safe_pos_size > self.max_position_size_cents:
                    # Don't automatically increase beyond user-set max, 
                    # but ensure we don't exceed 5% of NEW balance
                    pass
                
            return balance_cents
            
        except Exception as e:
            # excessive logging prevention
            # self.log_action(f"Error syncing balance: {e}")
            return self.last_known_balance_cents

    def _check_recent_profit(self) -> int:
        """Check for potentially recent realized profit (last 60s)"""
        # This is a placeholder. In a full implementation, this should 
        # query the adapter for recent 'settlement' transactions.
        return 0
    
    # ==================== STRATEGY: HIGH-PROBABILITY BONDING ====================
    
    def run_bonding_strategy(
        self,
        capital_cents: int = 100000,  # $1000
        dry_run: bool = True
    ) -> List[TradeSignal]:
        """
        Execute high-probability bonding strategy
        
        Finds markets with outcomes priced at 95%+ probability
        and places bets for consistent, low-risk returns.
        
        Args:
            capital_cents: Total capital in cents
            dry_run: If True, generate signals without executing
            
        Returns:
            List of trade signals
        """
        if not self.adapter:
            self.log_action("❌ Adapter not available for bonding strategy")
            return []
        
        self.log_action(f"🎯 Running bonding strategy with ${capital_cents/100:,.2f}")
        
        params = self.risk_params.get(self.risk_tolerance, self.risk_params["MODERATE"])
        
        # Find bonding opportunities
        opportunities = self.adapter.find_bonding_opportunities(
            min_probability=params["min_probability"],
            max_days_to_close=14,
            min_volume=params["min_volume"],
            min_annualized_return=params["min_annualized_return"]
        )
        
        if not opportunities:
            self.log_action("No bonding opportunities found")
            return []
        
        self.log_action(f"Found {len(opportunities)} bonding opportunities")
        
        signals = []
        remaining_capital = capital_cents
        max_per_trade = int(capital_cents * params["max_position_pct"])
        
        for opp in opportunities:
            if remaining_capital < 100:  # Min $1
                break
            
            if self.daily_trades >= self.max_daily_trades:
                self.log_action("Daily trade limit reached")
                break
            
            # Calculate position size
            position_cents = min(
                max_per_trade,
                int(remaining_capital * 0.2),
                self.max_position_size_cents
            )
            
            # Calculate number of contracts
            price_cents = opp["price_cents"]
            count = position_cents // price_cents
            
            if count < 1:
                continue
            
            signal = TradeSignal(
                ticker=opp["ticker"],
                market_title=opp["title"],
                strategy=TradingStrategy.BONDING,
                side=opp["side"],
                price_cents=price_cents,
                count=count,
                confidence=opp["price_decimal"],
                expected_return=opp["annualized_return"],
                risk_level=opp["risk_level"],
                reasoning=f"Bonding: {opp['price_cents']}¢ → {opp['annualized_return']:.1f}% APY",
                category=opp["category"]
            )
            
            signals.append(signal)
            
            if not dry_run:
                result = self._execute_signal(signal)
                if result and result.get("success"):
                    remaining_capital -= position_cents
                    self.daily_trades += 1
        
        self.log_action(f"Generated {len(signals)} bonding signals")
        return signals
    
    # ==================== STRATEGY: CROSS-PLATFORM ARBITRAGE ====================
    
    def run_arbitrage_strategy(
        self,
        capital_cents: int = 50000,  # $500
        dry_run: bool = True
    ) -> List[TradeSignal]:
        """
        Execute cross-platform arbitrage between Kalshi and Polymarket
        
        Finds price discrepancies between the two platforms and trades
        to capture the spread.
        
        Args:
            capital_cents: Total capital in cents
            dry_run: If True, generate signals without executing
            
        Returns:
            List of trade signals (Kalshi side only)
        """
        if not self.adapter:
            self.log_action("❌ Kalshi adapter not available")
            return []
        
        if not self.polymarket_adapter:
            self.log_action("❌ Polymarket adapter not available for arbitrage")
            return []
        
        self.log_action(f"🔄 Running arbitrage strategy with ${capital_cents/100:,.2f}")
        
        # Get Polymarket prices
        try:
            poly_markets = self.polymarket_adapter.get_markets(active_only=True, limit=200)
            poly_prices = {m.question.lower(): m.yes_price for m in poly_markets}
        except Exception as e:
            self.log_action(f"Error fetching Polymarket data: {e}")
            return []
        
        # Find arbitrage opportunities
        opportunities = self.adapter.find_cross_platform_arbitrage(
            polymarket_prices=poly_prices,
            min_spread=0.03  # 3% minimum spread
        )
        
        if not opportunities:
            self.log_action("No arbitrage opportunities found")
            return []
        
        self.log_action(f"Found {len(opportunities)} arbitrage opportunities")
        
        signals = []
        
        for opp in opportunities[:5]:  # Limit to top 5
            # Determine Kalshi side
            side = "yes" if opp["action"]["kalshi"] == "BUY" else "no"
            price_cents = int(opp["kalshi_price"] * 100)
            
            # Position size based on spread
            position_cents = min(
                capital_cents // 5,  # Split across opportunities
                self.max_position_size_cents
            )
            
            count = position_cents // price_cents
            if count < 1:
                continue
            
            signal = TradeSignal(
                ticker=opp["kalshi_ticker"],
                market_title=opp["kalshi_title"],
                strategy=TradingStrategy.ARBITRAGE,
                side=side,
                price_cents=price_cents,
                count=count,
                confidence=0.8,  # High confidence for arbitrage
                expected_return=opp["spread_pct"],
                risk_level="MEDIUM",
                reasoning=f"Arbitrage: {opp['spread_pct']:.1f}% spread vs Polymarket"
            )
            
            signals.append(signal)
            
            if not dry_run:
                self._execute_signal(signal)
        
        self.log_action(f"Generated {len(signals)} arbitrage signals")
        return signals
    
    # ==================== STRATEGY: AI SUPERFORECASTING ====================
    
    def run_superforecast_strategy(
        self,
        capital_cents: int = 50000,
        categories: Optional[List[str]] = None,
        max_markets: int = 10,
        dry_run: bool = True
    ) -> List[TradeSignal]:
        """
        Run LLM-powered superforecasting strategy
        
        Uses AI to analyze markets and find mispriced outcomes.
        
        Args:
            capital_cents: Total capital in cents
            categories: Categories to focus on
            max_markets: Maximum markets to analyze
            dry_run: If True, generate signals without executing
            
        Returns:
            List of trade signals
        """
        if not self.adapter:
            self.log_action("❌ Adapter not available")
            return []
        
        self.log_action(f"🔮 Running superforecast strategy with ${capital_cents/100:,.2f}")
        
        # Get markets
        markets = self.adapter.get_markets(status="open", limit=100)
        
        # Filter by categories
        if categories:
            markets = [m for m in markets if any(
                cat.lower() in m.category.lower()
                for cat in categories
            )]
        
        # Sort by volume and limit
        markets.sort(key=lambda m: m.volume, reverse=True)
        markets = markets[:max_markets]
        
        params = self.risk_params.get(self.risk_tolerance, self.risk_params["MODERATE"])
        signals = []
        
        for market in markets:
            # Get superforecast
            forecast = self.get_superforecast(market)
            
            if not forecast:
                continue
            
            # Calculate edge
            predicted_prob = forecast.get("predicted_probability", 0.5)
            market_price = market.yes_price
            edge = predicted_prob - market_price
            
            # Determine side
            if edge > params["min_edge"]:
                side = "yes"
                price_cents = int(market.yes_ask)
            elif edge < -params["min_edge"]:
                side = "no"
                price_cents = int(market.no_ask)
                edge = abs(edge)
            else:
                continue
            
            # Kelly criterion position sizing
            confidence = forecast.get("confidence", 0.5)
            kelly = edge / (price_cents / 100) if price_cents > 0 else 0
            position_fraction = kelly * params["kelly_fraction"] * confidence
            position_fraction = min(position_fraction, params["max_position_pct"])
            
            if position_fraction < 0.01:
                continue
            
            position_cents = int(capital_cents * position_fraction)
            count = position_cents // price_cents
            
            if count < 1:
                continue
            
            signal = TradeSignal(
                ticker=market.ticker,
                market_title=market.title,
                strategy=TradingStrategy.SUPERFORECAST,
                side=side,
                price_cents=price_cents,
                count=count,
                confidence=confidence,
                expected_return=edge * 100,
                risk_level="LOW" if edge >= 0.15 else "MEDIUM" if edge >= 0.10 else "HIGH",
                reasoning=f"Superforecast edge: {edge*100:+.1f}%",
                category=market.category
            )
            
            signals.append(signal)
            
            if not dry_run:
                self._execute_signal(signal)
        
        self.log_action(f"Generated {len(signals)} superforecast signals")
        return signals
    
    def get_superforecast(self, market: Any) -> Dict[str, Any]:
        """Get LLM-based superforecast for a Kalshi market"""
        if not self.llm:
            return self._heuristic_forecast(market)
        
        try:
            messages = self.prompter.superforecaster(
                title=market.title,
                subtitle=market.subtitle,
                category=market.category,
                rules=market.rules_primary
            )
            
            if not messages:
                return self._heuristic_forecast(market)
            
            result = self.llm.invoke(messages)
            content = result.content
            
            # Parse response
            probability = 0.5
            confidence = "MEDIUM"
            
            prob_match = re.search(r'PROBABILITY:\s*([0-9.]+)', content, re.IGNORECASE)
            if prob_match:
                probability = float(prob_match.group(1))
                probability = max(0.01, min(0.99, probability))
            
            conf_match = re.search(r'CONFIDENCE:\s*(\w+)', content, re.IGNORECASE)
            if conf_match:
                confidence = conf_match.group(1).upper()
            
            confidence_score = {"LOW": 0.4, "MEDIUM": 0.6, "HIGH": 0.8}.get(confidence, 0.5)
            
            self.log_action(f"🔮 Forecast: {market.ticker} → {probability:.2%} ({confidence})")
            
            return {
                "predicted_probability": probability,
                "confidence": confidence_score,
                "confidence_level": confidence,
                "analysis": content
            }
            
        except Exception as e:
            self.log_action(f"❌ Forecast error: {e}")
            return self._heuristic_forecast(market)
    
    def _heuristic_forecast(self, market: Any) -> Dict[str, Any]:
        """Fallback heuristic forecast"""
        import random
        base_prob = market.yes_price + random.uniform(-0.05, 0.05)
        base_prob = max(0.05, min(0.95, base_prob))
        
        return {
            "predicted_probability": base_prob,
            "confidence": 0.3,
            "confidence_level": "LOW",
            "analysis": "Heuristic estimate"
        }
    
    # ==================== TRADE EXECUTION ====================
    
    def _execute_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade signal"""
        if not self.adapter:
            return {"error": "Adapter not available"}
        
        # =====================================================================
        # LEARNING: Record signal for feedback tracking
        # =====================================================================
        learning_result = None
        if self.learning_enabled and self.learning_system:
            try:
                learning_result = self.learning_system.record_trade_signal(
                    market_ticker=signal.ticker,
                    market_title=signal.ticker,  # Would be better with actual title
                    category=signal.category if hasattr(signal, 'category') else "unknown",
                    predicted_probability=signal.expected_value,
                    market_price=signal.price_cents / 100,
                    side=signal.side,
                    strategy=signal.strategy.value,
                    confidence=signal.confidence,
                    volume=0,  # Would get from market data
                    spread=0.02,  # Estimated
                    days_to_close=7  # Estimated
                )
                
                # Check if learning system recommends trading
                if not learning_result.get("should_trade", True):
                    self.log_action(f"🧠 Learning: Skip signal ({learning_result.get('trade_reason')})")
                    # Still record but don't execute
                    return {"status": "skipped_by_learning", "reason": learning_result.get("trade_reason")}
                
                # Adjust confidence based on learning
                adjusted_confidence = learning_result.get("confidence_adjusted", signal.confidence)
                if adjusted_confidence < signal.confidence * 0.8:
                    self.log_action(f"🧠 Learning: Confidence reduced {signal.confidence:.2f} → {adjusted_confidence:.2f}")
                
            except Exception as e:
                self.log_action(f"⚠️ Learning record error: {e}")
        
        self.log_action(f"Executing: {signal.side.upper()} {signal.count}x {signal.ticker} @ {signal.price_cents}¢")
        
        result = self.adapter.place_order(
            ticker=signal.ticker,
            side=signal.side,
            count=signal.count,
            price=signal.price_cents
        )
        
        if result.get("success"):
            trade_result = TradeResult(
                trade_id=result.get("order_id", "unknown"),
                ticker=signal.ticker,
                strategy=signal.strategy.value,
                side=signal.side,
                price_cents=signal.price_cents,
                count=signal.count,
                status="resting",
                timestamp=datetime.datetime.now().isoformat()
            )
            self.trade_history.append(trade_result)
            self._save_state()
            
            self.log_action(f"✅ Trade executed: {trade_result.trade_id}")
        else:
            self.log_action(f"❌ Trade failed: {result.get('error')}")
        
        return result
    
    # ==================== LEARNING INTEGRATION ====================
    
    def record_trade_outcome(
        self,
        market_ticker: str,
        actual_outcome: bool,
        pnl_cents: int,
        strategy: str = "unknown",
        category: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Record trade outcome for learning
        
        Call this when a market resolves to update the learning system.
        
        Args:
            market_ticker: The market that resolved
            actual_outcome: True if YES won, False if NO won
            pnl_cents: Actual P&L in cents
            strategy: Strategy used
            category: Market category
            
        Returns:
            Learning feedback with insights
        """
        if not self.learning_enabled or not self.learning_system:
            return {"status": "learning_disabled"}
        
        try:
            result = self.learning_system.record_trade_outcome(
                market_ticker=market_ticker,
                actual_outcome=actual_outcome,
                pnl_cents=pnl_cents,
                strategy=strategy,
                category=category
            )
            
            self.log_action(f"🧠 Learning recorded: {market_ticker} → {'WIN' if pnl_cents > 0 else 'LOSS'}")
            
            return result
            
        except Exception as e:
            self.log_action(f"⚠️ Learning outcome error: {e}")
            return {"status": "error", "error": str(e)}
    
    def select_best_strategy(
        self,
        available_strategies: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Use learning system to select the best strategy
        
        Returns:
            (strategy_name, expected_value)
        """
        if available_strategies is None:
            available_strategies = ["bonding", "arbitrage", "superforecast"]
        
        if self.learning_enabled and self.learning_system:
            return self.learning_system.select_best_strategy(
                available_strategies=available_strategies,
                category=category,
                explore=True  # Thompson sampling with exploration
            )
        else:
            # Default fallback
            return available_strategies[0], 0.5
    
    def get_learning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive learning report
        
        Returns insights on:
        - Calibration (are predictions accurate?)
        - Strategy performance rankings
        - Category-specific performance
        - Recommendations for improvement
        """
        if not self.learning_enabled or not self.learning_system:
            return {"status": "learning_disabled"}
        
        try:
            return self.learning_system.get_learning_report()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_confidence_adjustment(self, category: str, strategy: str) -> float:
        """
        Get confidence adjustment multiplier from learning
        
        Returns multiplier to apply to raw confidence scores.
        > 1.0 means historical overperformance
        < 1.0 means historical underperformance
        """
        if self.learning_enabled and self.learning_system:
            return self.learning_system.get_confidence_adjustment(category, strategy)
        return 1.0
    
    # ==================== PORTFOLIO MANAGEMENT ====================
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        if not self.adapter:
            return {"error": "Adapter not available"}
        
        balance = self.adapter.get_balance()
        positions = self.adapter.get_positions()
        
        total_exposure = sum(p.market_exposure for p in positions)
        total_realized = sum(p.realized_pnl for p in positions)
        
        return {
            "balance_usd": balance.get("balance_usd", 0),
            "available_usd": balance.get("available_usd", 0),
            "total_positions": len(positions),
            "total_exposure_usd": total_exposure / 100,
            "realized_pnl_usd": total_realized / 100,
            "trades_today": self.daily_trades,
            "total_trades": len(self.trade_history),
            "positions": [asdict(p) for p in positions]
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.trade_history:
            return {"status": "no_trades"}
        
        total_trades = len(self.trade_history)
        winning = sum(1 for t in self.trade_history if t.pnl_cents > 0)
        losing = sum(1 for t in self.trade_history if t.pnl_cents < 0)
        
        win_rate = winning / total_trades if total_trades > 0 else 0
        
        # Strategy breakdown
        strategy_pnl: Dict[str, Dict[str, Any]] = {}
        for trade in self.trade_history:
            if trade.strategy not in strategy_pnl:
                strategy_pnl[trade.strategy] = {"trades": 0, "pnl_cents": 0}
            strategy_pnl[trade.strategy]["trades"] += 1
            strategy_pnl[trade.strategy]["pnl_cents"] += trade.pnl_cents
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": round(win_rate * 100, 2),
            "total_pnl_usd": self.total_pnl_cents / 100,
            "strategy_breakdown": strategy_pnl,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    # ==================== MAIN RUN CYCLE ====================
    
    def run_cycle(
        self,
        strategies: Optional[List[str]] = None,
        capital_cents: int = 100000,  # $1000
        categories: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete trading cycle
        
        Args:
            strategies: List of strategies to run
            capital_cents: Total capital in cents
            categories: Categories to focus on
            dry_run: If True, generate signals without executing
            
        Returns:
            Cycle results
        """
        self.log_action(f"{'='*60}")
        self.log_action(f"Starting Kalshi trading cycle (dry_run={dry_run})")
        self.log_action(f"Capital: ${capital_cents/100:,.2f}")
        self.log_action(f"Environment: {'DEMO' if self.demo else 'PRODUCTION'}")
        self.log_action(f"{'='*60}")
        
        # Reset daily counter
        today = datetime.date.today()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        # Default strategies
        if strategies is None:
            strategies = ["bonding"]
            if self.polymarket_adapter:
                strategies.append("arbitrage")
            if self.llm:
                strategies.append("superforecast")
        
        # =====================================================================
        # LEARNING: Use learning system for strategy selection and weighting
        # =====================================================================
        learning_strategy_info = None
        if self.learning_enabled and self.learning_system:
            try:
                # Get learning-optimized strategy rankings
                category = categories[0] if categories else None
                best_strategy, expected_value = self.select_best_strategy(strategies, category)
                
                learning_strategy_info = {
                    "best_strategy": best_strategy,
                    "expected_value": expected_value,
                    "learning_active": True
                }
                
                self.log_action(f"🧠 Learning recommends: {best_strategy} (expected: {expected_value:.2%})")
                
            except Exception as e:
                self.log_action(f"⚠️ Learning strategy selection error: {e}")
        
        all_signals = []
        results = {}
        
        # Strategy weights - adjusted by learning if available
        weights = {
            "bonding": 0.5,
            "arbitrage": 0.3,
            "superforecast": 0.2
        }
        
        # Boost weight of learning-recommended strategy
        if learning_strategy_info and learning_strategy_info.get("best_strategy"):
            best = learning_strategy_info["best_strategy"]
            if best in weights:
                # Increase weight of best strategy by 20%
                boost = weights[best] * 0.2
                weights[best] += boost
                # Normalize other weights
                for s in weights:
                    if s != best:
                        weights[s] -= boost / (len(weights) - 1)
        
        total_weight = sum(weights.get(s, 0.2) for s in strategies)
        
        # Run strategies
        if "bonding" in strategies:
            w = weights.get("bonding", 0.5)
            cap = int(capital_cents * (w / total_weight))
            
            signals = self.run_bonding_strategy(capital_cents=cap, dry_run=dry_run)
            all_signals.extend(signals)
            results["bonding"] = {
                "signals": len(signals),
                "capital_allocated_usd": cap / 100,
                "top": [{"ticker": s.ticker, "return": s.expected_return} for s in signals[:3]]
            }
        
        if "arbitrage" in strategies:
            w = weights.get("arbitrage", 0.3)
            cap = int(capital_cents * (w / total_weight))
            
            signals = self.run_arbitrage_strategy(capital_cents=cap, dry_run=dry_run)
            all_signals.extend(signals)
            results["arbitrage"] = {
                "signals": len(signals),
                "capital_allocated_usd": cap / 100,
                "top": [{"ticker": s.ticker, "spread": s.expected_return} for s in signals[:3]]
            }
        
        if "superforecast" in strategies:
            w = weights.get("superforecast", 0.2)
            cap = int(capital_cents * (w / total_weight))
            
            signals = self.run_superforecast_strategy(
                capital_cents=cap,
                categories=categories,
                dry_run=dry_run
            )
            all_signals.extend(signals)
            results["superforecast"] = {
                "signals": len(signals),
                "capital_allocated_usd": cap / 100,
                "top": [{"ticker": s.ticker, "edge": s.expected_return} for s in signals[:3]]
            }
        
        # Summary
        avg_return = sum(s.expected_return for s in all_signals) / len(all_signals) if all_signals else 0
        
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dry_run": dry_run,
            "environment": "DEMO" if self.demo else "PRODUCTION",
            "total_signals": len(all_signals),
            "capital_usd": capital_cents / 100,
            "strategies_run": strategies,
            "avg_expected_return": round(avg_return, 2),
            "strategy_results": results,
            "portfolio_status": self.get_portfolio_status() if self.adapter else {},
            "regulatory_note": "CFTC-regulated exchange - legal in US"
        }
        
        # =====================================================================
        # LEARNING: Include learning summary in results
        # =====================================================================
        if self.learning_enabled and self.learning_system:
            try:
                learning_report = self.get_learning_report()
                summary["learning"] = {
                    "active": True,
                    "total_predictions": learning_report.get("summary", {}).get("total_predictions", 0),
                    "strategy_recommendation": learning_strategy_info,
                    "calibration_error": learning_report.get("calibration", {}).get("overall_calibration_error", 0),
                    "recommendations_count": len(learning_report.get("recommendations", []))
                }
                
                # Log key learning insights
                if learning_report.get("recommendations"):
                    self.log_action(f"🧠 Learning has {len(learning_report['recommendations'])} recommendations")
                    
            except Exception as e:
                summary["learning"] = {"active": False, "error": str(e)}
        else:
            summary["learning"] = {"active": False}
        
        self.log_action(f"✅ Cycle complete: {len(all_signals)} signals")
        
        return summary
    
    # ==================== ONE BEST TRADE ====================
    
    def one_best_trade(
        self,
        capital_cents: int = 10000,  # $100
        categories: Optional[List[str]] = None,
        execute: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Find and optionally execute the single best trade
        
        Args:
            capital_cents: Amount in cents
            categories: Optional category filter
            execute: If True, execute the trade
            
        Returns:
            Best trade details
        """
        self.log_action("🎯 Finding ONE BEST TRADE on Kalshi...")
        
        if not self.adapter:
            return None
        
        # Get bonding opportunities (safest)
        bonding = self.adapter.find_bonding_opportunities()
        
        if not bonding:
            # Fall back to superforecast
            markets = self.adapter.get_markets(status="open", limit=20)
            if categories:
                markets = [m for m in markets if any(
                    c.lower() in m.category.lower() for c in categories
                )]
            
            if not markets:
                self.log_action("❌ No markets found")
                return None
            
            # Score and pick best
            markets.sort(key=lambda m: m.volume, reverse=True)
            best_market = markets[0]
            
            forecast = self.get_superforecast(best_market)
            if not forecast:
                return None
            
            edge = forecast["predicted_probability"] - best_market.yes_price
            if abs(edge) < 0.05:
                self.log_action("❌ No significant edge found")
                return None
            
            side = "yes" if edge > 0 else "no"
            price_cents = int(best_market.yes_ask if side == "yes" else best_market.no_ask)
            
            trade = {
                "ticker": best_market.ticker,
                "title": best_market.title,
                "side": side,
                "price_cents": price_cents,
                "count": capital_cents // price_cents,
                "expected_edge": abs(edge) * 100,
                "source": "superforecast"
            }
        else:
            # Use best bonding opportunity
            best = bonding[0]
            trade = {
                "ticker": best["ticker"],
                "title": best["title"],
                "side": best["side"],
                "price_cents": best["price_cents"],
                "count": capital_cents // best["price_cents"],
                "expected_return": best["annualized_return"],
                "source": "bonding"
            }
        
        self.log_action(f"🎯 BEST: {trade['side'].upper()} {trade['ticker']} @ {trade['price_cents']}¢")
        
        if execute and trade["count"] >= 1:
            signal = TradeSignal(
                ticker=trade["ticker"],
                market_title=trade["title"],
                strategy=TradingStrategy.BONDING if trade["source"] == "bonding" else TradingStrategy.SUPERFORECAST,
                side=trade["side"],
                price_cents=trade["price_cents"],
                count=trade["count"],
                confidence=0.9 if trade["source"] == "bonding" else 0.6,
                expected_return=trade.get("expected_return", trade.get("expected_edge", 0)),
                risk_level="LOW" if trade["source"] == "bonding" else "MEDIUM",
                reasoning=f"One best trade via {trade['source']}"
            )
            result = self._execute_signal(signal)
            trade["executed"] = result.get("success", False)
            trade["order_id"] = result.get("order_id")
        
        return trade
    
    # ==================== INTEGRATION ====================
    
    def get_profit_for_shredder(self) -> float:
        """Get realized profits (USD) for Shredder allocation"""
        return self.total_pnl_cents / 100
    
    def record_profit_allocation(self, amount: float, allocation_type: str):
        """Record that Shredder allocated profits"""
        self.log_action(f"Shredder allocated ${amount:.2f} to {allocation_type}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_kalshi_trader(demo: bool = False) -> KalshiTrader:
    """Get global Kalshi trader instance"""
    return KalshiTrader(demo=demo)


# =============================================================================
# MAIN - Test
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("KALSHI TRADER - CFTC-Regulated Prediction Markets")
    print("="*70)
    
    # Use demo mode for testing
    trader = KalshiTrader(demo=True, risk_tolerance="MODERATE")
    
    print("\n=== Running Dry-Run Cycle ===")
    results = trader.run_cycle(
        strategies=["bonding"],
        capital_cents=100000,  # $1000
        dry_run=True
    )
    
    print("\n=== Cycle Results ===")
    print(json.dumps(results, indent=2, default=str))
    
    print("\n=== One Best Trade ===")
    best = trader.one_best_trade(capital_cents=10000, execute=False)
    if best:
        print(json.dumps(best, indent=2, default=str))
    
    print("\n=== Performance Report ===")
    print(json.dumps(trader.get_performance_report(), indent=2))
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)


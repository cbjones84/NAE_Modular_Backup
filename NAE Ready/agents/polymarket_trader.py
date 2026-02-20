# NAE/agents/polymarket_trader.py
"""
Polymarket Trader Agent - Prediction Market Trading for NAE
============================================================
Automated trading agent for Polymarket prediction markets.

ENHANCED with official code patterns from:
- github.com/Polymarket/agents (Official AI agents framework)
- github.com/Polymarket/py-clob-client (Official Python SDK)

Strategies:
1. High-Probability Bonding - Buy 95%+ outcomes for consistent returns
2. Cross-Platform Arbitrage - Exploit price differences across platforms
3. AI-Powered Semantic Trading - Use NAE's AI for prediction advantage
4. Superforecasting - LLM-based probability estimation
5. Liquidity Provision - Earn fees by providing market liquidity

ALIGNED WITH NAE GOALS:
1. Achieve generational wealth
2. Generate $6,243,561+ within 8 years (Year 7 target)
3. Diversify income streams beyond traditional stock/options trading

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

# Import Polymarket adapter
try:
    from adapters.polymarket import (
        PolymarketAdapter,
        PolymarketMarket,
        PolymarketEvent,
        GammaMarketClient,
        get_polymarket_adapter
    )
    POLYMARKET_AVAILABLE = True
except ImportError:
    POLYMARKET_AVAILABLE = False
    PolymarketAdapter = None
    PolymarketMarket = None
    PolymarketEvent = None
    GammaMarketClient = None
    get_polymarket_adapter = None

# Try to import LangChain for AI-powered trading (official Polymarket agents pattern)
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
# PROMPTS - Based on official Polymarket agents framework
# github.com/Polymarket/agents/agents/application/prompts.py
# =============================================================================

class PolymarketPrompter:
    """
    Prompt templates for Polymarket AI trading
    Based on official Polymarket agents prompts.py
    """
    
    @staticmethod
    def market_analyst() -> str:
        """System prompt for market analysis"""
        return """You are an expert prediction market analyst and superforecaster.
Your task is to analyze prediction markets and provide probability estimates.

You should:
1. Consider base rates and historical precedents
2. Identify key factors that could affect the outcome
3. Weigh evidence for and against each outcome
4. Consider timing and time to resolution
5. Account for market efficiency and liquidity

Provide your analysis in a structured format with clear reasoning."""

    @staticmethod
    def superforecaster(question: str, description: str, outcomes: List[str]) -> List[Any]:
        """
        Superforecasting prompt for probability estimation
        Based on official Polymarket agents superforecaster methodology
        """
        if not HumanMessage or not SystemMessage:
            return []
        
        system_prompt = """You are a superforecaster with exceptional ability to estimate probabilities.

Your methodology:
1. OUTSIDE VIEW: Start with base rates - how often do similar events occur?
2. INSIDE VIEW: What specific factors make this case different?
3. SYNTHESIS: Combine outside and inside views, weighing evidence
4. CALIBRATION: Avoid overconfidence, express genuine uncertainty
5. UPDATE: How might new information change your estimate?

Format your response as:
ANALYSIS: [Your detailed reasoning]
PROBABILITY: [Your probability estimate as decimal, e.g., 0.65]
CONFIDENCE: [Your confidence in this estimate, LOW/MEDIUM/HIGH]"""

        user_prompt = f"""Analyze this prediction market:

QUESTION: {question}

DESCRIPTION: {description}

OUTCOMES: {', '.join(outcomes)}

Estimate the probability of the first outcome occurring.
Consider all available information and apply superforecasting methodology."""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    @staticmethod
    def one_best_trade(analysis: str, outcomes: List[str], prices: List[float]) -> List[Any]:
        """
        Determine optimal trade based on analysis
        Based on official Polymarket agents trade selection
        """
        if not HumanMessage or not SystemMessage:
            return []
        
        system_prompt = """You are a quantitative trader optimizing prediction market positions.

Given an analysis and current market prices, determine the optimal trade.

Consider:
1. Edge = Your probability estimate - Market price
2. Kelly criterion for position sizing
3. Risk-adjusted expected value
4. Market liquidity constraints

Format your response as:
SIDE: [BUY_YES / BUY_NO / NO_TRADE]
SIZE_FRACTION: [fraction of capital to deploy, 0.0-0.25]
REASONING: [Brief explanation]"""

        user_prompt = f"""Based on this analysis:

{analysis}

Current market prices:
- {outcomes[0]}: {prices[0]:.2f}
- {outcomes[1] if len(outcomes) > 1 else 'Other'}: {prices[1] if len(prices) > 1 else 1-prices[0]:.2f}

What is the optimal trade?"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    @staticmethod
    def filter_events() -> str:
        """Prompt for filtering events to trade"""
        return """Identify events that are:
1. Likely to resolve within 30 days
2. Have clear, verifiable resolution criteria
3. Are not subject to regulatory restrictions
4. Have sufficient information for analysis

Return the IDs of events worth analyzing further."""


class TradingStrategy(Enum):
    """Available trading strategies"""
    BONDING = "bonding"              # High-probability betting
    ARBITRAGE = "arbitrage"          # Cross-platform price differences
    SEMANTIC = "semantic"            # AI-powered predictions
    LIQUIDITY = "liquidity"          # Market making


@dataclass
class TradeSignal:
    """Signal for a potential trade"""
    market_id: str
    market_question: str
    strategy: TradingStrategy
    side: str  # BUY or SELL
    outcome: str  # YES or NO
    price: float
    size: float
    confidence: float
    expected_return: float
    risk_level: str
    reasoning: str


@dataclass
class TradeResult:
    """Result of an executed trade"""
    trade_id: str
    market_id: str
    strategy: str
    side: str
    outcome: str
    price: float
    size: float
    status: str
    timestamp: str
    pnl: float = 0.0


class PolymarketTrader:
    """
    Polymarket Trading Agent for NAE
    
    ENHANCED with official Polymarket agents patterns:
    - LLM-based superforecasting
    - Gamma API market discovery
    - Event filtering and RAG
    - Optimal trade selection
    
    Executes prediction market strategies to generate additional
    income streams for NAE's growth milestones.
    """
    
    def __init__(
        self,
        adapter: Optional[Any] = None,  # PolymarketAdapter instance
        max_position_size: float = 1000.0,
        max_daily_trades: int = 20,
        risk_tolerance: str = "MODERATE",
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize Polymarket Trader
        
        Args:
            adapter: Polymarket adapter instance
            max_position_size: Maximum USDC per position
            max_daily_trades: Maximum trades per day
            risk_tolerance: CONSERVATIVE, MODERATE, or AGGRESSIVE
            openai_api_key: OpenAI API key for LLM-based trading
            llm_model: LLM model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.goals = GOALS
        self.log_file = "logs/polymarket_trader.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Growth milestones
        self.target_goal = 6_243_561.0
        self.stretch_goal = 15_726_144.0
        self.growth_milestones = {
            1: 9_411, 2: 44_110, 3: 152_834, 4: 388_657,
            5: 982_500, 6: 2_477_897, 7: 6_243_561, 8: 15_726_144
        }
        
        # Initialize adapter
        if adapter:
            self.adapter = adapter
        elif POLYMARKET_AVAILABLE and get_polymarket_adapter:
            self.adapter = get_polymarket_adapter()
        else:
            self.adapter = None
            self.log_action("⚠️ Polymarket adapter not available")
        
        # Initialize Gamma client for direct market access
        self.gamma: Optional[Any] = None
        if POLYMARKET_AVAILABLE and GammaMarketClient:
            self.gamma = GammaMarketClient()
        
        # Initialize LLM for AI-powered trading (official Polymarket agents pattern)
        self.llm: Optional[Any] = None
        self.prompter = PolymarketPrompter()
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if LANGCHAIN_AVAILABLE and ChatOpenAI and self.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model=llm_model,
                    temperature=0.1,  # Low temperature for consistent predictions
                    api_key=self.openai_api_key
                )
                self.log_action(f"✅ LLM initialized ({llm_model}) for AI-powered trading")
            except Exception as e:
                self.log_action(f"⚠️ LLM initialization failed: {e}")
        else:
            self.log_action("⚠️ LLM not available (install langchain-openai and set OPENAI_API_KEY)")
        
        # Trading parameters
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.risk_tolerance = risk_tolerance
        
        # Risk parameters by tolerance
        self.risk_params = {
            "CONSERVATIVE": {
                "min_probability": 0.97,
                "max_position_pct": 0.05,
                "min_liquidity": 5000,
                "min_annualized_return": 30,
                "min_edge": 0.15,
                "kelly_fraction": 0.10
            },
            "MODERATE": {
                "min_probability": 0.95,
                "max_position_pct": 0.10,
                "min_liquidity": 2000,
                "min_annualized_return": 50,
                "min_edge": 0.10,
                "kelly_fraction": 0.15
            },
            "AGGRESSIVE": {
                "min_probability": 0.90,
                "max_position_pct": 0.20,
                "min_liquidity": 1000,
                "min_annualized_return": 100,
                "min_edge": 0.05,
                "kelly_fraction": 0.25
            }
        }
        
        # Trading state
        self.daily_trades = 0
        self.last_trade_date = None
        self.open_positions: List[Dict[str, Any]] = []
        self.trade_history: List[TradeResult] = []
        self.total_pnl = 0.0
        
        # Load state
        self._load_state()
        
        self.log_action("PolymarketTrader initialized (Enhanced with official Polymarket patterns)")
    
    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.datetime.now().isoformat()
        log_entry = f"[{ts}] {message}"
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        print(f"[PolymarketTrader] {message}")
    
    def _load_state(self):
        """Load trading state from file"""
        state_file = "logs/polymarket_trader_state.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.total_pnl = state.get("total_pnl", 0.0)
                    self.trade_history = [
                        TradeResult(**t) for t in state.get("trade_history", [])[-100:]
                    ]
            except Exception as e:
                self.log_action(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save trading state to file"""
        state_file = "logs/polymarket_trader_state.json"
        try:
            state = {
                "total_pnl": self.total_pnl,
                "trade_history": [asdict(t) for t in self.trade_history[-100:]],
                "last_updated": datetime.datetime.now().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.log_action(f"Error saving state: {e}")
    
    # ==================== STRATEGY: HIGH-PROBABILITY BONDING ====================
    
    def run_bonding_strategy(
        self,
        capital: float = 1000.0,
        dry_run: bool = True
    ) -> List[TradeSignal]:
        """
        Execute high-probability bonding strategy
        
        Finds markets with outcomes priced at 95%+ probability
        and places bets for consistent, low-risk returns.
        
        Args:
            capital: Total capital to deploy
            dry_run: If True, generate signals without executing
            
        Returns:
            List of trade signals (executed if not dry_run)
        """
        if not self.adapter:
            self.log_action("❌ Adapter not available for bonding strategy")
            return []
        
        self.log_action(f"🎯 Running bonding strategy with ${capital:,.2f}")
        
        # Get risk parameters
        params = self.risk_params.get(self.risk_tolerance, self.risk_params["MODERATE"])
        
        # Find bonding opportunities
        opportunities = self.adapter.find_bonding_opportunities(
            min_probability=params["min_probability"],
            max_days_to_resolution=14,
            min_liquidity=params["min_liquidity"],
            min_annualized_return=params["min_annualized_return"]
        )
        
        if not opportunities:
            self.log_action("No bonding opportunities found")
            return []
        
        self.log_action(f"Found {len(opportunities)} bonding opportunities")
        
        signals = []
        remaining_capital = capital
        max_per_trade = capital * params["max_position_pct"]
        
        for opp in opportunities:
            if remaining_capital < 10:  # Minimum $10 per trade
                break
            
            if self.daily_trades >= self.max_daily_trades:
                self.log_action("Daily trade limit reached")
                break
            
            # Calculate position size
            position_size = min(
                max_per_trade,
                remaining_capital * 0.2,  # Max 20% of remaining per trade
                opp["liquidity"] * 0.1,   # Max 10% of market liquidity
                self.max_position_size
            )
            
            # Create signal
            signal = TradeSignal(
                market_id=opp["market_id"],
                market_question=opp["question"],
                strategy=TradingStrategy.BONDING,
                side="BUY",
                outcome=opp["side"],
                price=opp["price"],
                size=position_size / opp["price"],  # Convert to shares
                confidence=opp["price"],  # Use price as confidence proxy
                expected_return=opp["annualized_return"],
                risk_level=opp["risk_level"],
                reasoning=f"High-probability bonding: {opp['price']*100:.1f}% → {opp['annualized_return']:.1f}% APY"
            )
            
            signals.append(signal)
            
            if not dry_run:
                result = self._execute_signal(signal)
                if result and result.get("success"):
                    remaining_capital -= position_size
                    self.daily_trades += 1
        
        self.log_action(f"Generated {len(signals)} bonding signals")
        return signals
    
    # ==================== STRATEGY: SEMANTIC/AI TRADING ====================
    
    def run_semantic_strategy(
        self,
        ralph_agent=None,
        topics: Optional[List[str]] = None,
        capital: float = 500.0,
        dry_run: bool = True
    ) -> List[TradeSignal]:
        """
        Execute AI-powered semantic trading strategy
        
        Uses NAE's AI capabilities (Ralph) to analyze prediction
        markets and find mispriced outcomes.
        
        Args:
            ralph_agent: RalphAgent instance for AI analysis
            topics: List of topics to focus on (e.g., ["politics", "crypto"])
            capital: Total capital to deploy
            dry_run: If True, generate signals without executing
            
        Returns:
            List of trade signals
        """
        if not self.adapter:
            self.log_action("❌ Adapter not available for semantic strategy")
            return []
        
        self.log_action(f"🧠 Running semantic strategy with ${capital:,.2f}")
        
        # Get markets
        markets = self.adapter.get_markets(active_only=True, limit=100)
        
        if topics:
            # Filter by topics
            markets = [m for m in markets if any(
                t.lower() in m.category.lower() or t.lower() in m.question.lower()
                for t in topics
            )]
        
        signals = []
        
        for market in markets[:20]:  # Analyze top 20 markets
            # Get AI prediction
            if ralph_agent and hasattr(ralph_agent, 'analyze_prediction_market'):
                analysis = ralph_agent.analyze_prediction_market(market.question)
            else:
                # Fallback: simple heuristic analysis
                analysis = self._heuristic_analysis(market)
            
            if not analysis:
                continue
            
            predicted_prob = analysis.get("predicted_probability", 0.5)
            confidence = analysis.get("confidence", 0.0)
            
            # Check for edge
            market_price = market.yes_price
            edge = predicted_prob - market_price
            
            # Only trade if significant edge and confidence
            if abs(edge) >= 0.10 and confidence >= 0.6:
                signal = TradeSignal(
                    market_id=market.id,
                    market_question=market.question,
                    strategy=TradingStrategy.SEMANTIC,
                    side="BUY" if edge > 0 else "SELL",
                    outcome="YES" if edge > 0 else "NO",
                    price=market_price if edge > 0 else market.no_price,
                    size=min(capital * 0.1, 100) / market_price,
                    confidence=confidence,
                    expected_return=abs(edge) * 100,
                    risk_level="MEDIUM" if confidence >= 0.7 else "HIGH",
                    reasoning=f"AI edge: {edge*100:+.1f}% (confidence: {confidence*100:.0f}%)"
                )
                signals.append(signal)
                
                if not dry_run and len(signals) <= 5:
                    self._execute_signal(signal)
        
        self.log_action(f"Generated {len(signals)} semantic signals")
        return signals
    
    def _heuristic_analysis(self, market: Any) -> Dict[str, Any]:  # PolymarketMarket
        """Simple heuristic analysis when Ralph/LLM not available"""
        # Basic analysis based on market characteristics
        question_lower = market.question.lower()
        
        # Look for keywords that suggest outcome
        bullish_keywords = ["will", "pass", "win", "approve", "succeed", "reach"]
        bearish_keywords = ["won't", "fail", "lose", "reject", "below"]
        
        bullish_score = sum(1 for k in bullish_keywords if k in question_lower)
        bearish_score = sum(1 for k in bearish_keywords if k in question_lower)
        
        # Volume/liquidity suggests market efficiency
        efficiency = min(market.liquidity / 10000, 1.0)
        
        # Base probability on market price with slight random adjustment
        import random
        noise = random.uniform(-0.05, 0.05)
        predicted_prob = market.yes_price + noise
        
        # Adjust for sentiment
        if bullish_score > bearish_score:
            predicted_prob += 0.03
        elif bearish_score > bullish_score:
            predicted_prob -= 0.03
        
        predicted_prob = max(0.05, min(0.95, predicted_prob))
        
        return {
            "predicted_probability": predicted_prob,
            "confidence": 0.4 + (0.3 * efficiency),  # Higher liquidity = more confidence
            "reasoning": "Heuristic analysis"
        }
    
    # ==================== LLM-POWERED SUPERFORECASTING ====================
    # Based on official Polymarket agents executor pattern
    
    def get_superforecast(
        self,
        question: str,
        description: str,
        outcomes: List[str]
    ) -> Dict[str, Any]:
        """
        Get LLM-based superforecast for a market
        Based on official Polymarket agents superforecaster methodology
        
        Args:
            question: Market question
            description: Market description
            outcomes: List of possible outcomes
            
        Returns:
            Dict with probability, confidence, and analysis
        """
        if not self.llm:
            return self._heuristic_superforecast(question, outcomes)
        
        try:
            messages = self.prompter.superforecaster(question, description, outcomes)
            if not messages:
                return self._heuristic_superforecast(question, outcomes)
            
            result = self.llm.invoke(messages)
            content = result.content
            
            # Parse response
            probability = 0.5
            confidence = "MEDIUM"
            analysis = content
            
            # Extract probability
            prob_match = re.search(r'PROBABILITY:\s*([0-9.]+)', content, re.IGNORECASE)
            if prob_match:
                probability = float(prob_match.group(1))
                probability = max(0.01, min(0.99, probability))
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\w+)', content, re.IGNORECASE)
            if conf_match:
                confidence = conf_match.group(1).upper()
            
            # Extract analysis
            analysis_match = re.search(r'ANALYSIS:\s*(.+?)(?=PROBABILITY:|$)', content, re.DOTALL | re.IGNORECASE)
            if analysis_match:
                analysis = analysis_match.group(1).strip()
            
            confidence_score = {"LOW": 0.4, "MEDIUM": 0.6, "HIGH": 0.8}.get(confidence, 0.5)
            
            self.log_action(f"🔮 Superforecast: {question[:50]}... → {probability:.2%} ({confidence})")
            
            return {
                "predicted_probability": probability,
                "confidence": confidence_score,
                "confidence_level": confidence,
                "analysis": analysis,
                "reasoning": f"LLM superforecast: {confidence} confidence"
            }
            
        except Exception as e:
            self.log_action(f"❌ Superforecast error: {e}")
            return self._heuristic_superforecast(question, outcomes)
    
    def _heuristic_superforecast(self, question: str, outcomes: List[str]) -> Dict[str, Any]:
        """Fallback heuristic superforecast when LLM unavailable"""
        import random
        base_prob = 0.5 + random.uniform(-0.1, 0.1)
        return {
            "predicted_probability": base_prob,
            "confidence": 0.3,
            "confidence_level": "LOW",
            "analysis": "Heuristic estimate (LLM unavailable)",
            "reasoning": "Heuristic superforecast"
        }
    
    def get_optimal_trade(
        self,
        market: Any,
        analysis: Dict[str, Any]
    ) -> Optional[TradeSignal]:
        """
        Determine optimal trade based on superforecast
        Based on official Polymarket agents trade selection
        
        Uses Kelly criterion for position sizing:
        f* = (bp - q) / b
        where b = odds, p = probability of winning, q = 1-p
        """
        params = self.risk_params.get(self.risk_tolerance, self.risk_params["MODERATE"])
        
        predicted_prob = analysis.get("predicted_probability", 0.5)
        confidence = analysis.get("confidence", 0.5)
        
        # Calculate edge for YES and NO
        yes_price = market.yes_price
        no_price = market.no_price
        
        yes_edge = predicted_prob - yes_price
        no_edge = (1 - predicted_prob) - no_price
        
        # Choose the side with better edge
        if abs(yes_edge) > abs(no_edge) and yes_edge > 0:
            side = "BUY"
            outcome = "YES"
            edge = yes_edge
            price = yes_price
        elif abs(no_edge) > abs(yes_edge) and no_edge > 0:
            side = "BUY"
            outcome = "NO"
            edge = no_edge
            price = no_price
        else:
            # No positive edge
            return None
        
        # Check minimum edge threshold
        if edge < params["min_edge"]:
            return None
        
        # Kelly criterion position sizing
        # f* = edge / odds
        odds = (1 - price) / price  # Implied odds
        kelly = edge / price if price > 0 else 0
        
        # Apply Kelly fraction based on risk tolerance
        kelly_fraction = params["kelly_fraction"]
        position_fraction = kelly * kelly_fraction * confidence
        
        # Cap position size
        position_fraction = min(position_fraction, params["max_position_pct"])
        
        if position_fraction < 0.01:  # Minimum 1% position
            return None
        
        return TradeSignal(
            market_id=market.id,
            market_question=market.question,
            strategy=TradingStrategy.SEMANTIC,
            side=side,
            outcome=outcome,
            price=price,
            size=position_fraction,  # Fraction of capital
            confidence=confidence,
            expected_return=edge * 100,
            risk_level="LOW" if edge >= 0.15 else "MEDIUM" if edge >= 0.10 else "HIGH",
            reasoning=f"Superforecast edge: {edge*100:+.1f}% (Kelly: {kelly*100:.1f}%)"
        )
    
    def run_superforecast_strategy(
        self,
        capital: float = 500.0,
        max_markets: int = 10,
        categories: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> List[TradeSignal]:
        """
        Run LLM-powered superforecasting strategy
        Based on official Polymarket agents one_best_trade methodology
        
        Args:
            capital: Total capital to deploy
            max_markets: Maximum markets to analyze
            categories: Categories to focus on (e.g., ["politics", "crypto"])
            dry_run: If True, generate signals without executing
            
        Returns:
            List of trade signals
        """
        if not self.adapter:
            self.log_action("❌ Adapter not available for superforecast strategy")
            return []
        
        self.log_action(f"🔮 Running superforecast strategy with ${capital:,.2f}")
        
        # Get tradeable markets
        markets = self.adapter.get_markets(active_only=True, limit=100)
        
        # Filter by categories if specified
        if categories:
            markets = [m for m in markets if any(
                cat.lower() in (m.category or "").lower() or 
                cat.lower() in m.question.lower()
                for cat in categories
            )]
        
        # Sort by liquidity (prefer liquid markets)
        markets.sort(key=lambda m: m.liquidity, reverse=True)
        markets = markets[:max_markets]
        
        signals = []
        remaining_capital = capital
        
        for market in markets:
            if remaining_capital < 10:
                break
            
            if self.daily_trades >= self.max_daily_trades:
                break
            
            # Get superforecast
            forecast = self.get_superforecast(
                question=market.question,
                description=market.description,
                outcomes=market.outcomes
            )
            
            # Get optimal trade
            signal = self.get_optimal_trade(market, forecast)
            
            if signal:
                # Convert fraction to dollar amount
                position_size = min(
                    remaining_capital * signal.size,
                    self.max_position_size
                )
                
                # Update signal with actual size
                signal = TradeSignal(
                    market_id=signal.market_id,
                    market_question=signal.market_question,
                    strategy=signal.strategy,
                    side=signal.side,
                    outcome=signal.outcome,
                    price=signal.price,
                    size=position_size / signal.price,  # Convert to shares
                    confidence=signal.confidence,
                    expected_return=signal.expected_return,
                    risk_level=signal.risk_level,
                    reasoning=signal.reasoning
                )
                
                signals.append(signal)
                
                if not dry_run:
                    result = self._execute_signal(signal)
                    if result and result.get("success"):
                        remaining_capital -= position_size
                        self.daily_trades += 1
        
        self.log_action(f"Generated {len(signals)} superforecast signals")
        return signals
    
    # ==================== TRADE EXECUTION ====================
    
    def _execute_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade signal"""
        if not self.adapter:
            return {"error": "Adapter not available"}
        
        self.log_action(f"Executing: {signal.side} {signal.outcome} @ {signal.price}")
        
        result = self.adapter.place_order(
            market_id=signal.market_id,
            outcome=signal.outcome,
            side=signal.side.lower(),
            price=signal.price,
            size=signal.size
        )
        
        if result.get("success"):
            trade_result = TradeResult(
                trade_id=result.get("order_id", "unknown"),
                market_id=signal.market_id,
                strategy=signal.strategy.value,
                side=signal.side,
                outcome=signal.outcome,
                price=signal.price,
                size=signal.size,
                status="open",
                timestamp=datetime.datetime.now().isoformat()
            )
            self.trade_history.append(trade_result)
            self._save_state()
            
            self.log_action(f"✅ Trade executed: {trade_result.trade_id}")
        else:
            self.log_action(f"❌ Trade failed: {result.get('error')}")
        
        return result
    
    # ==================== PORTFOLIO MANAGEMENT ====================
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        positions = self.adapter.get_positions() if self.adapter else []
        
        total_value = sum(p.shares * p.current_price for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        return {
            "total_positions": len(positions),
            "total_value": total_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": self.total_pnl,
            "total_pnl": self.total_pnl + total_unrealized_pnl,
            "trades_today": self.daily_trades,
            "total_trades": len(self.trade_history),
            "positions": [asdict(p) for p in positions] if positions else []
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.trade_history:
            return {"status": "no_trades", "message": "No trading history"}
        
        # Calculate metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.pnl > 0)
        losing_trades = sum(1 for t in self.trade_history if t.pnl < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Strategy breakdown
        strategy_pnl = {}
        for trade in self.trade_history:
            if trade.strategy not in strategy_pnl:
                strategy_pnl[trade.strategy] = {"trades": 0, "pnl": 0}
            strategy_pnl[trade.strategy]["trades"] += 1
            strategy_pnl[trade.strategy]["pnl"] += trade.pnl
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate * 100, 2),
            "total_pnl": round(self.total_pnl, 2),
            "strategy_breakdown": strategy_pnl,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    # ==================== MAIN RUN LOOP ====================
    
    def run_cycle(
        self,
        strategies: Optional[List[str]] = None,
        capital: float = 1000.0,
        categories: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete trading cycle
        
        Enhanced with official Polymarket agents patterns:
        - Superforecasting strategy with LLM
        - Multiple strategy orchestration
        - Event filtering by category
        
        Args:
            strategies: List of strategies to run (default: bonding, superforecast)
            capital: Total capital to deploy
            categories: Categories to focus on (e.g., ["politics", "crypto"])
            dry_run: If True, generate signals without executing
            
        Returns:
            Cycle results with signals and performance
        """
        self.log_action(f"{'='*60}")
        self.log_action(f"Starting trading cycle (dry_run={dry_run})")
        self.log_action(f"Capital: ${capital:,.2f}")
        self.log_action(f"Risk Tolerance: {self.risk_tolerance}")
        self.log_action(f"LLM Available: {'✅' if self.llm else '❌'}")
        self.log_action(f"{'='*60}")
        
        # Reset daily counter if new day
        today = datetime.date.today()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        # Default strategies - prioritize bonding (low risk) and superforecast (AI-powered)
        if strategies is None:
            strategies = ["bonding", "superforecast"] if self.llm else ["bonding", "semantic"]
        
        all_signals = []
        results = {}
        
        # Split capital among strategies (weighted by risk)
        strategy_weights = {
            "bonding": 0.6,      # 60% - safest
            "superforecast": 0.3,  # 30% - AI-powered
            "semantic": 0.2,    # 20% - heuristic
            "arbitrage": 0.1    # 10% - opportunistic
        }
        
        total_weight = sum(strategy_weights.get(s, 0.2) for s in strategies)
        
        # Run each strategy
        if "bonding" in strategies:
            weight = strategy_weights.get("bonding", 0.5)
            strat_capital = capital * (weight / total_weight)
            
            bonding_signals = self.run_bonding_strategy(
                capital=strat_capital,
                dry_run=dry_run
            )
            all_signals.extend(bonding_signals)
            results["bonding"] = {
                "signals": len(bonding_signals),
                "capital_allocated": strat_capital,
                "top_opportunities": [
                    {"question": s.market_question[:50], "return": s.expected_return}
                    for s in bonding_signals[:3]
                ]
            }
        
        if "superforecast" in strategies:
            weight = strategy_weights.get("superforecast", 0.3)
            strat_capital = capital * (weight / total_weight)
            
            superforecast_signals = self.run_superforecast_strategy(
                capital=strat_capital,
                max_markets=10,
                categories=categories,
                dry_run=dry_run
            )
            all_signals.extend(superforecast_signals)
            results["superforecast"] = {
                "signals": len(superforecast_signals),
                "capital_allocated": strat_capital,
                "llm_model": "gpt-4o-mini" if self.llm else "N/A",
                "top_opportunities": [
                    {"question": s.market_question[:50], "edge": s.expected_return, "confidence": s.confidence}
                    for s in superforecast_signals[:3]
                ]
            }
        
        if "semantic" in strategies:
            weight = strategy_weights.get("semantic", 0.2)
            strat_capital = capital * (weight / total_weight)
            
            semantic_signals = self.run_semantic_strategy(
                capital=strat_capital,
                topics=categories,
                dry_run=dry_run
            )
            all_signals.extend(semantic_signals)
            results["semantic"] = {
                "signals": len(semantic_signals),
                "capital_allocated": strat_capital,
                "top_opportunities": [
                    {"question": s.market_question[:50], "edge": s.expected_return}
                    for s in semantic_signals[:3]
                ]
            }
        
        # Calculate totals
        total_potential_return = sum(s.expected_return for s in all_signals) / len(all_signals) if all_signals else 0
        
        # Summary
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dry_run": dry_run,
            "total_signals": len(all_signals),
            "capital_allocated": capital if not dry_run else 0,
            "strategies_run": strategies,
            "categories": categories,
            "avg_expected_return": round(total_potential_return, 2),
            "strategy_results": results,
            "portfolio_status": self.get_portfolio_status(),
            "nae_goals_alignment": {
                "target_goal": self.target_goal,
                "contribution_potential": f"${capital * total_potential_return / 100:.2f}" if all_signals else "$0"
            }
        }
        
        self.log_action(f"✅ Cycle complete: {len(all_signals)} signals generated")
        self.log_action(f"📊 Average expected return: {total_potential_return:.1f}%")
        
        return summary
    
    # ==================== ONE BEST TRADE ====================
    # Direct implementation of official Polymarket agents pattern
    
    def one_best_trade(
        self,
        capital: float = 100.0,
        categories: Optional[List[str]] = None,
        execute: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Find and optionally execute the single best trade
        Based on official Polymarket agents one_best_trade methodology
        
        This is the core pattern from:
        github.com/Polymarket/agents/agents/application/trade.py
        
        Args:
            capital: Amount to deploy on this trade
            categories: Optional categories to filter
            execute: If True, execute the trade (requires credentials)
            
        Returns:
            Best trade details or None if no good opportunity
        """
        self.log_action("🎯 Finding ONE BEST TRADE...")
        
        if not self.adapter:
            self.log_action("❌ Adapter not available")
            return None
        
        # Step 1: Get active events/markets
        markets = self.adapter.get_all_tradeable_markets() if hasattr(self.adapter, 'get_all_tradeable_markets') else self.adapter.get_markets(active_only=True, limit=100)
        
        if categories:
            markets = [m for m in markets if any(
                cat.lower() in (m.category or "").lower()
                for cat in categories
            )]
        
        self.log_action(f"📊 Found {len(markets)} markets to analyze")
        
        # Step 2: Score and filter markets
        scored_markets = []
        for market in markets[:50]:  # Limit analysis
            score = self._score_market(market)
            if score > 0:
                scored_markets.append((market, score))
        
        scored_markets.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_markets:
            self.log_action("❌ No suitable markets found")
            return None
        
        # Step 3: Get superforecast for top market
        best_market, score = scored_markets[0]
        
        forecast = self.get_superforecast(
            question=best_market.question,
            description=best_market.description,
            outcomes=best_market.outcomes
        )
        
        # Step 4: Calculate optimal trade
        signal = self.get_optimal_trade(best_market, forecast)
        
        if not signal:
            self.log_action("❌ No positive edge found")
            return None
        
        # Step 5: Format result
        trade = {
            "market_id": signal.market_id,
            "question": signal.market_question,
            "side": signal.side,
            "outcome": signal.outcome,
            "price": signal.price,
            "amount_usd": min(capital, self.max_position_size),
            "expected_edge": signal.expected_return,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "forecast": forecast
        }
        
        self.log_action(f"🎯 BEST TRADE: {signal.side} {signal.outcome} @ {signal.price:.2f}")
        self.log_action(f"   {signal.market_question[:60]}...")
        self.log_action(f"   Expected edge: {signal.expected_return:.1f}%")
        
        # Step 6: Execute if requested
        if execute:
            result = self._execute_signal(signal)
            trade["execution_result"] = result
            trade["executed"] = result.get("success", False)
        
        return trade
    
    def _score_market(self, market: Any) -> float:
        """Score a market for trading potential"""
        score = 0.0
        
        # Liquidity score (prefer liquid markets)
        if market.liquidity >= 5000:
            score += 3
        elif market.liquidity >= 1000:
            score += 2
        elif market.liquidity >= 500:
            score += 1
        
        # Volume score (prefer active markets)
        if market.volume >= 10000:
            score += 2
        elif market.volume >= 1000:
            score += 1
        
        # Time to resolution (prefer soon but not too soon)
        days = market.days_to_resolution
        if 3 <= days <= 30:
            score += 2
        elif 1 <= days <= 60:
            score += 1
        
        # Price score (prefer markets with prices not at extremes)
        price = market.yes_price
        if 0.20 <= price <= 0.80:
            score += 2  # Good for finding edge
        elif 0.10 <= price <= 0.90:
            score += 1
        
        # Spread score (lower spread = better)
        if market.spread <= 0.02:
            score += 2
        elif market.spread <= 0.05:
            score += 1
        
        return score
    
    # ==================== INTEGRATION METHODS ====================
    
    def get_profit_for_shredder(self) -> float:
        """Get realized profits for Shredder allocation"""
        return self.total_pnl
    
    def record_profit_allocation(self, amount: float, allocation_type: str):
        """Record that Shredder allocated profits"""
        self.log_action(f"Shredder allocated ${amount:.2f} to {allocation_type}")


# Convenience function
def get_polymarket_trader() -> PolymarketTrader:
    """Get global Polymarket trader instance"""
    return PolymarketTrader()


if __name__ == "__main__":
    # Test the trader
    trader = PolymarketTrader(risk_tolerance="MODERATE")
    
    print("\n=== Polymarket Trader Test ===")
    
    # Run a dry-run cycle
    results = trader.run_cycle(
        strategies=["bonding"],
        capital=1000.0,
        dry_run=True
    )
    
    print("\n=== Cycle Results ===")
    print(json.dumps(results, indent=2, default=str))
    
    print("\n=== Performance Report ===")
    print(json.dumps(trader.get_performance_report(), indent=2))


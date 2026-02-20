# NAE/tools/rl_framework.py
"""
Reinforcement Learning Framework for Position Sizing & Strategy Design

Implements:
- Risk-aware RL (PPO with risk penalty)
- Position sizing optimization
- Strategy switching
- Shadow trading
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    HEDGE = "hedge"


@dataclass
class RLState:
    """RL state representation"""
    portfolio_value: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    market_features: np.ndarray
    risk_metrics: Dict[str, float]
    recent_returns: List[float]
    timestamp: datetime


@dataclass
class RLAction:
    """RL action representation"""
    action_type: ActionType
    symbol: str
    quantity: float
    confidence: float
    expected_return: float
    risk_score: float


@dataclass
class RLExperience:
    """RL experience tuple"""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool
    timestamp: datetime


class TradingEnvironment:
    """
    Trading environment for RL
    
    Simulates trading and provides rewards based on performance
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.history: List[RLExperience] = []
        self.current_state: Optional[RLState] = None
    
    def reset(self) -> RLState:
        """Reset environment to initial state"""
        self.capital = self.initial_capital
        self.positions = {}
        
        self.current_state = RLState(
            portfolio_value=self.capital,
            cash=self.capital,
            positions={},
            market_features=np.zeros(10),  # Placeholder
            risk_metrics={},
            recent_returns=[],
            timestamp=datetime.now()
        )
        
        return self.current_state
    
    def step(self, action: RLAction, market_price: float) -> Tuple[RLState, float, bool]:
        """
        Execute action and return next state, reward, done
        
        Args:
            action: Action to execute
            market_price: Current market price
        
        Returns:
            (next_state, reward, done)
        """
        reward = 0.0
        
        # Execute action
        if action.action_type == ActionType.BUY:
            cost = action.quantity * market_price
            if cost <= self.capital:
                self.capital -= cost
                if action.symbol in self.positions:
                    self.positions[action.symbol] += action.quantity
                else:
                    self.positions[action.symbol] = action.quantity
        
        elif action.action_type == ActionType.SELL:
            if action.symbol in self.positions and self.positions[action.symbol] >= action.quantity:
                proceeds = action.quantity * market_price
                self.capital += proceeds
                self.positions[action.symbol] -= action.quantity
                if self.positions[action.symbol] <= 0:
                    del self.positions[action.symbol]
        
        # Calculate portfolio value
        portfolio_value = self.capital
        for symbol, qty in self.positions.items():
            portfolio_value += qty * market_price  # Simplified
        
        # Calculate reward
        prev_value = self.current_state.portfolio_value if self.current_state else self.initial_capital
        return_pct = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        
        # Risk-adjusted reward
        risk_penalty = action.risk_score * 0.1  # Penalize high risk
        reward = return_pct - risk_penalty
        
        # Update state
        next_state = RLState(
            portfolio_value=portfolio_value,
            cash=self.capital,
            positions=self.positions.copy(),
            market_features=np.zeros(10),  # Would compute from market data
            risk_metrics={"drawdown": 0.0, "volatility": 0.0},
            recent_returns=[return_pct],
            timestamp=datetime.now()
        )
        
        # Check if done (e.g., max steps, bankruptcy)
        done = portfolio_value < self.initial_capital * 0.5  # Stop if lose 50%
        
        # Store experience
        if self.current_state:
            experience = RLExperience(
                state=self.current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                timestamp=datetime.now()
            )
            self.history.append(experience)
        
        self.current_state = next_state
        
        return next_state, reward, done


class RiskAwarePPO:
    """
    Risk-aware Proximal Policy Optimization (PPO) for trading
    
    Optimizes position sizing and strategy selection with risk penalties
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        learning_rate: float = 0.0003,
        risk_penalty_weight: float = 0.1
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            risk_penalty_weight: Weight for risk penalty in reward
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.risk_penalty_weight = risk_penalty_weight
        
        # Simplified policy network (would use actual neural network)
        self.policy_weights = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.value_weights = np.random.normal(0, 0.1, (state_dim, 1))
        
        self.training_history: List[Dict[str, Any]] = []
    
    def select_action(self, state: RLState) -> RLAction:
        """
        Select action using current policy
        
        Returns:
            Action to take
        """
        # Extract state features
        state_features = self._extract_features(state)
        
        # Policy network output
        logits = state_features @ self.policy_weights
        
        # Sample action (simplified - would use proper policy distribution)
        action_probs = self._softmax(logits)
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        # Map to action
        action_types = [ActionType.BUY, ActionType.SELL, ActionType.HOLD, ActionType.HEDGE]
        action_type = action_types[action_idx]
        
        # Determine quantity (simplified)
        quantity = abs(state_features[0]) * 10  # Placeholder
        
        return RLAction(
            action_type=action_type,
            symbol="SPY",  # Placeholder
            quantity=quantity,
            confidence=action_probs[action_idx],
            expected_return=logits[action_idx],
            risk_score=0.5  # Would compute from state
        )
    
    def update(self, experiences: List[RLExperience]):
        """
        Update policy using PPO algorithm
        
        Args:
            experiences: List of experiences for training
        """
        if not experiences:
            return
        
        # Extract states, actions, rewards
        states = [self._extract_features(e.state) for e in experiences]
        rewards = [e.reward for e in experiences]
        
        # Compute advantages (simplified)
        returns = self._compute_returns(rewards)
        advantages = returns - np.mean(returns)
        
        # Policy gradient update (simplified PPO)
        # In practice, would use clipped objective and value function
        
        # Simple gradient update
        for i, state in enumerate(states):
            if i < len(advantages):
                # Simplified policy update
                gradient = state[:, np.newaxis] * advantages[i]
                self.policy_weights += self.learning_rate * gradient
        
        # Record training
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "num_experiences": len(experiences),
            "avg_reward": np.mean(rewards),
            "avg_advantage": np.mean(advantages)
        })
    
    def _extract_features(self, state: RLState) -> np.ndarray:
        """Extract features from state"""
        # Simplified feature extraction
        features = np.zeros(self.state_dim)
        
        if state.market_features.size > 0:
            features[:min(len(state.market_features), self.state_dim)] = state.market_features[:self.state_dim]
        
        # Add portfolio features
        if len(features) > 10:
            features[10] = state.portfolio_value / self.state_dim  # Normalized
            features[11] = state.cash / self.state_dim  # Normalized
        
        return features
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _compute_returns(self, rewards: List[float], gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns"""
        returns = np.zeros(len(rewards))
        G = 0
        
        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            returns[i] = G
        
        return returns


class RLPositionSizer:
    """
    VERY_AGGRESSIVE RL-based position sizing system
    
    Optimized for $6.2M target / $15.7M stretch goal
    Uses RL to optimize position sizes with milestone acceleration
    
    Growth Milestones:
    Year 1: $9,411 | Year 5: $982,500
    Year 2: $44,110 | Year 6: $2,477,897
    Year 3: $152,834 | Year 7: $6,243,561 (TARGET)
    Year 4: $388,657 | Year 8: $15,726,144 (STRETCH)
    """
    
    def __init__(self, initial_capital: float = 100000.0, aggressiveness: str = "VERY_AGGRESSIVE"):
        self.environment = TradingEnvironment(initial_capital)
        self.agent = RiskAwarePPO()
        self.shadow_mode = False  # VERY_AGGRESSIVE: Start in LIVE mode
        self.shadow_trades: List[Dict[str, Any]] = []
        self.aggressiveness = aggressiveness
        
        # VERY_AGGRESSIVE: Higher base sizing and caps
        self.aggression_config = {
            "CONSERVATIVE": {"base_pct": 0.01, "max_pct": 0.05, "risk_penalty": 0.3},
            "MODERATE": {"base_pct": 0.02, "max_pct": 0.10, "risk_penalty": 0.2},
            "MODERATE_PLUS": {"base_pct": 0.03, "max_pct": 0.15, "risk_penalty": 0.15},
            "AGGRESSIVE": {"base_pct": 0.05, "max_pct": 0.25, "risk_penalty": 0.1},
            "VERY_AGGRESSIVE": {"base_pct": 0.10, "max_pct": 0.40, "risk_penalty": 0.05},
        }
        self.config = self.aggression_config.get(aggressiveness, self.aggression_config["VERY_AGGRESSIVE"])
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        market_state: Dict[str, Any],
        strategy_confidence: float = 0.5
    ) -> Tuple[float, Dict[str, Any]]:
        """
        VERY_AGGRESSIVE: Calculate optimal position size using RL
        
        Args:
            symbol: Trading symbol
            price: Current price
            market_state: Market state features
            strategy_confidence: Strategy confidence score
        
        Returns:
            (position_size, details)
        """
        # Build RL state
        state = RLState(
            portfolio_value=self.environment.capital,
            cash=self.environment.capital,
            positions=self.environment.positions.copy(),
            market_features=np.array(list(market_state.values())[:10]) if market_state else np.zeros(10),
            risk_metrics={},
            recent_returns=[],
            timestamp=datetime.now()
        )
        
        # Get action from RL agent
        action = self.agent.select_action(state)
        
        # VERY_AGGRESSIVE position sizing
        if action.action_type == ActionType.BUY:
            # AGGRESSIVE: Higher base size (10% vs 2%)
            base_size = self.environment.capital * self.config["base_pct"]
            
            # VERY_AGGRESSIVE: Reduced risk penalty impact
            risk_penalty = action.risk_score * self.config["risk_penalty"]
            rl_multiplier = action.confidence * (1 - risk_penalty)
            
            # Apply strategy confidence boost
            confidence_boost = 0.5 + (strategy_confidence * 0.5)  # 0.5-1.0 multiplier
            position_size = base_size * rl_multiplier * confidence_boost
            
            # AGGRESSIVE: Higher cap (40% vs 10%)
            position_size = min(position_size, self.environment.capital * self.config["max_pct"])
        else:
            position_size = 0.0
        
        details = {
            "rl_action": action.action_type.value,
            "rl_confidence": action.confidence,
            "rl_risk_score": action.risk_score,
            "position_size": position_size,
            "shadow_mode": self.shadow_mode,
            "aggressiveness": self.aggressiveness,
            "base_pct": self.config["base_pct"],
            "max_pct": self.config["max_pct"]
        }
        
        # Record shadow trade if in shadow mode
        if self.shadow_mode:
            self.shadow_trades.append({
                "symbol": symbol,
                "price": price,
                "position_size": position_size,
                "action": action.action_type.value,
                "timestamp": datetime.now().isoformat(),
                "details": details
            })
        
        return position_size, details
    
    def train_on_experience(self, experiences: List[RLExperience]):
        """Train RL agent on experiences"""
        self.agent.update(experiences)
    
    def enable_live_trading(self):
        """Enable live trading (exit shadow mode)"""
        self.shadow_mode = False
    
    def get_shadow_performance(self) -> Dict[str, Any]:
        """Get shadow trading performance"""
        if not self.shadow_trades:
            return {"status": "no_trades", "total_trades": 0}
        
        # Calculate performance (simplified)
        total_trades = len(self.shadow_trades)
        avg_confidence = np.mean([t["details"]["rl_confidence"] for t in self.shadow_trades])
        
        return {
            "status": "shadow_mode",
            "total_trades": total_trades,
            "avg_confidence": avg_confidence,
            "ready_for_live": avg_confidence > 0.7 and total_trades > 50
        }


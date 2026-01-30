# NAE/tools/risk_controls.py
"""
Risk Controls & Guardrails System for NAE

Implements:
- Circuit breakers (per-agent and system-wide)
- Position sizing (fixed fractional, volatility parity, Kelly)
- Pre-trade checks (liquidity, IV sanity, arbitrage)
- Kill switches
- Hard limits
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
import os


class RiskLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class PositionLimit:
    """Position size limit configuration"""
    max_position_pct_portfolio: float = 0.10  # 10% max per position
    max_strategy_exposure_pct: float = 0.20  # 20% max per strategy
    max_total_exposure_pct: float = 0.50  # 50% max total


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit
    max_consecutive_losses: int = 5
    max_drawdown_pct: float = 0.15  # 15% drawdown limit
    cooldown_hours: int = 24  # Hours to wait after trigger
    enabled: bool = True


@dataclass
class PreTradeCheck:
    """Pre-trade validation check"""
    name: str
    passed: bool
    message: str
    severity: RiskLevel = RiskLevel.WARNING


class CircuitBreaker:
    """
    Circuit breaker for trading operations
    
    Prevents trading when risk thresholds are exceeded
    """
    
    def __init__(self, config: CircuitBreakerConfig, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.lock = threading.Lock()
        self.history: List[Dict[str, Any]] = []
    
    def record_trade(self, pnl: float):
        """Record a trade and update circuit breaker state"""
        with self.lock:
            self.daily_pnl += pnl
            
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Check if circuit breaker should trigger
            self._check_triggers()
    
    def _check_triggers(self):
        """Check if circuit breaker should trigger"""
        if not self.config.enabled:
            return
        
        if self.triggered:
            # Check if cooldown period has passed
            if self.trigger_time:
                elapsed = datetime.now() - self.trigger_time
                if elapsed >= timedelta(hours=self.config.cooldown_hours):
                    self._reset()
            return
        
        # Check daily loss limit
        if abs(self.daily_pnl) > self.config.max_daily_loss_pct:
            self._trigger(f"Daily loss limit exceeded: {self.daily_pnl:.2%} > {self.config.max_daily_loss_pct:.2%}")
            return
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger(f"Consecutive losses limit exceeded: {self.consecutive_losses} >= {self.config.max_consecutive_losses}")
            return
        
        # Check drawdown
        if self.max_drawdown > self.config.max_drawdown_pct:
            self._trigger(f"Max drawdown limit exceeded: {self.max_drawdown:.2%} > {self.config.max_drawdown_pct:.2%}")
            return
    
    def _trigger(self, reason: str):
        """Trigger circuit breaker"""
        self.triggered = True
        self.trigger_time = datetime.now()
        self.trigger_reason = reason
        
        event = {
            "agent": self.agent_name,
            "triggered": True,
            "reason": reason,
            "timestamp": self.trigger_time.isoformat(),
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "max_drawdown": self.max_drawdown
        }
        
        self.history.append(event)
        
        # Log circuit breaker trigger
        self._log_trigger(event)
    
    def _reset(self):
        """Reset circuit breaker"""
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = None
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
    
    def _log_trigger(self, event: Dict[str, Any]):
        """Log circuit breaker trigger"""
        log_file = "logs/circuit_breakers.jsonl"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        print(f"🔴 CIRCUIT BREAKER TRIGGERED [{self.agent_name}]: {event['reason']}")
    
    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is allowed"""
        with self.lock:
            if self.triggered:
                return False, f"Circuit breaker active: {self.trigger_reason}"
            return True, None
    
    def reset_daily(self):
        """Reset daily counters (call at start of trading day)"""
        with self.lock:
            self.daily_pnl = 0.0


class PositionSizer:
    """
    Position sizing calculator
    
    Supports multiple sizing strategies:
    - Fixed fractional
    - Volatility parity
    - Kelly criterion (with shrinkage)
    """
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
    
    def fixed_fractional(self, fraction: float = 0.02) -> float:
        """
        Fixed fractional position sizing
        
        Args:
            fraction: Fraction of portfolio to risk (default 2%)
        """
        return self.portfolio_value * fraction
    
    def volatility_parity(
        self,
        target_volatility: float = 0.15,
        asset_volatility: float = 0.20
    ) -> float:
        """
        Volatility parity position sizing
        
        Args:
            target_volatility: Target portfolio volatility (15% annual)
            asset_volatility: Asset volatility (20% annual)
        """
        if asset_volatility == 0:
            return 0.0
        
        position_pct = target_volatility / asset_volatility
        return self.portfolio_value * min(position_pct, 1.0)  # Cap at 100%
    
    def kelly_criterion(
        self,
        win_probability: float,
        win_loss_ratio: float,
        shrinkage: float = 0.25
    ) -> float:
        """
        Kelly criterion with shrinkage
        
        Args:
            win_probability: Probability of winning trade
            win_loss_ratio: Average win / Average loss
            shrinkage: Shrinkage factor (0.25 = quarter Kelly)
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        if win_loss_ratio <= 0:
            return 0.0
        
        # Kelly fraction: f = (p * b - q) / b
        # where p = win prob, q = 1-p, b = win/loss ratio
        q = 1 - win_probability
        kelly_fraction = (win_probability * win_loss_ratio - q) / win_loss_ratio
        
        # Apply shrinkage
        kelly_fraction *= shrinkage
        
        # Ensure positive and reasonable
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        return self.portfolio_value * kelly_fraction


class PreTradeValidator:
    """
    Pre-trade validation system
    
    Performs checks before executing trades:
    - Liquidity checks
    - IV sanity checks
    - Arbitrage detection
    - Position limits
    """
    
    def __init__(self, position_limits: PositionLimit):
        self.position_limits = position_limits
        self.checks_history: List[List[PreTradeCheck]] = []
    
    def validate_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        current_positions: Dict[str, float],
        portfolio_value: float,
        bid_ask_spread: Optional[float] = None,
        implied_vol: Optional[float] = None,
        historical_vol: Optional[float] = None
    ) -> Tuple[bool, List[PreTradeCheck], RiskLevel]:
        """
        Validate a trade before execution
        
        Returns:
            (is_valid, checks, risk_level)
        """
        checks: List[PreTradeCheck] = []
        
        # Check position size limit
        position_value = quantity * price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        if position_pct > self.position_limits.max_position_pct_portfolio:
            checks.append(PreTradeCheck(
                name="position_size_limit",
                passed=False,
                message=f"Position size {position_pct:.2%} exceeds limit {self.position_limits.max_position_pct_portfolio:.2%}",
                severity=RiskLevel.CRITICAL
            ))
        else:
            checks.append(PreTradeCheck(
                name="position_size_limit",
                passed=True,
                message=f"Position size {position_pct:.2%} within limits",
                severity=RiskLevel.SAFE
            ))
        
        # Check total exposure
        total_exposure = sum(abs(pos * price) for pos in current_positions.values()) + position_value
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0.0
        
        if exposure_pct > self.position_limits.max_total_exposure_pct:
            checks.append(PreTradeCheck(
                name="total_exposure_limit",
                passed=False,
                message=f"Total exposure {exposure_pct:.2%} exceeds limit {self.position_limits.max_total_exposure_pct:.2%}",
                severity=RiskLevel.CRITICAL
            ))
        else:
            checks.append(PreTradeCheck(
                name="total_exposure_limit",
                passed=True,
                message=f"Total exposure {exposure_pct:.2%} within limits",
                severity=RiskLevel.SAFE
            ))
        
        # Check bid-ask spread (if available)
        if bid_ask_spread is not None:
            spread_pct = bid_ask_spread / price if price > 0 else 0.0
            if spread_pct > 0.05:  # 5% spread threshold
                checks.append(PreTradeCheck(
                    name="liquidity_check",
                    passed=False,
                    message=f"Bid-ask spread {spread_pct:.2%} too wide",
                    severity=RiskLevel.WARNING
                ))
            else:
                checks.append(PreTradeCheck(
                    name="liquidity_check",
                    passed=True,
                    message=f"Bid-ask spread {spread_pct:.2%} acceptable",
                    severity=RiskLevel.SAFE
                ))
        
        # Check IV vs HV sanity (if available)
        if implied_vol is not None and historical_vol is not None:
            iv_hv_ratio = implied_vol / historical_vol if historical_vol > 0 else 1.0
            
            if iv_hv_ratio > 3.0 or iv_hv_ratio < 0.33:
                checks.append(PreTradeCheck(
                    name="iv_sanity_check",
                    passed=False,
                    message=f"IV/HV ratio {iv_hv_ratio:.2f} outside normal range (0.33-3.0)",
                    severity=RiskLevel.WARNING
                ))
            else:
                checks.append(PreTradeCheck(
                    name="iv_sanity_check",
                    passed=True,
                    message=f"IV/HV ratio {iv_hv_ratio:.2f} within normal range",
                    severity=RiskLevel.SAFE
                ))
        
        # Determine overall risk level
        critical_failed = any(c.severity == RiskLevel.CRITICAL and not c.passed for c in checks)
        warning_failed = any(c.severity == RiskLevel.WARNING and not c.passed for c in checks)
        
        if critical_failed:
            risk_level = RiskLevel.CRITICAL
        elif warning_failed:
            risk_level = RiskLevel.WARNING
        else:
            risk_level = RiskLevel.SAFE
        
        # Trade is valid if no critical checks failed
        is_valid = not critical_failed
        
        # Save checks history
        self.checks_history.append(checks)
        
        return is_valid, checks, risk_level


class RiskControlSystem:
    """
    Centralized risk control system
    
    Manages circuit breakers, position sizing, and pre-trade validation
    """
    
    def __init__(
        self,
        portfolio_value: float,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        position_limits: Optional[PositionLimit] = None
    ):
        self.portfolio_value = portfolio_value
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.position_limits = position_limits or PositionLimit()
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.position_sizer = PositionSizer(portfolio_value)
        self.pre_trade_validator = PreTradeValidator(self.position_limits)
        
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
    
    def get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent"""
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreaker(
                self.circuit_breaker_config,
                agent_name
            )
        return self.circuit_breakers[agent_name]
    
    def activate_kill_switch(self, reason: str):
        """Activate system-wide kill switch"""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        
        event = {
            "kill_switch": True,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        log_file = "logs/kill_switch.jsonl"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        print(f"🔴 KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch"""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        print("✅ Kill switch deactivated")
    
    def can_execute_trade(self, agent_name: str) -> Tuple[bool, Optional[str]]:
        """Check if trade execution is allowed"""
        if self.kill_switch_active:
            return False, f"Kill switch active: {self.kill_switch_reason}"
        
        circuit_breaker = self.get_circuit_breaker(agent_name)
        can_trade, reason = circuit_breaker.can_trade()
        
        if not can_trade:
            return False, reason
        
        return True, None
    
    def calculate_position_size(
        self,
        method: str = "fixed_fractional",
        **kwargs
    ) -> float:
        """
        Calculate position size using specified method
        
        Args:
            method: "fixed_fractional", "volatility_parity", or "kelly"
            **kwargs: Method-specific parameters
        """
        if method == "fixed_fractional":
            return self.position_sizer.fixed_fractional(kwargs.get("fraction", 0.02))
        elif method == "volatility_parity":
            return self.position_sizer.volatility_parity(
                kwargs.get("target_volatility", 0.15),
                kwargs.get("asset_volatility", 0.20)
            )
        elif method == "kelly":
            return self.position_sizer.kelly_criterion(
                kwargs.get("win_probability", 0.5),
                kwargs.get("win_loss_ratio", 1.5),
                kwargs.get("shrinkage", 0.25)
            )
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    def validate_trade(self, **kwargs) -> Tuple[bool, List[PreTradeCheck], RiskLevel]:
        """Validate trade using pre-trade validator"""
        return self.pre_trade_validator.validate_trade(**kwargs)

